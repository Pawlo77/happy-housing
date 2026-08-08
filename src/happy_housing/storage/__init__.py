"""DuckDB + GeoParquet storage helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_geoparquet(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Write listings with WKT geometry, bbox columns, and Z-order locality sort.

    Uses Morton/Z-order on lon/lat ranks as a fast Hilbert-family locality key
    (true Hilbert curves need heavier deps; Z-order still clusters neighbors).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gdf.copy()
    if not frame.empty and "geometry" in frame.columns:
        xs = frame.geometry.x
        ys = frame.geometry.y
        bounds = frame.geometry.bounds
        frame["lon"] = xs
        frame["lat"] = ys
        frame["bbox_minx"] = bounds.minx
        frame["bbox_miny"] = bounds.miny
        frame["bbox_maxx"] = bounds.maxx
        frame["bbox_maxy"] = bounds.maxy
        rx = xs.rank(method="average").astype("uint32")
        ry = ys.rank(method="average").astype("uint32")
        frame["_hilbert"] = _morton_interleave16(rx.to_numpy(), ry.to_numpy())
        frame = frame.sort_values("_hilbert").drop(columns=["_hilbert"])
        frame["geometry_wkt"] = frame.geometry.to_wkt()
    table = pd.DataFrame(frame.drop(columns=["geometry"], errors="ignore"))
    table.to_parquet(path, index=False)
    logger.info("Wrote %s rows to %s", len(table), path)
    return path


def _morton_interleave16(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Interleave lower 16 bits of two integer arrays (Z-order key)."""
    x = np.asarray(xs, dtype=np.uint32) & np.uint32(0xFFFF)
    y = np.asarray(ys, dtype=np.uint32) & np.uint32(0xFFFF)

    def _part1by1(n: np.ndarray) -> np.ndarray:
        n = (n | (n << 8)) & np.uint32(0x00FF00FF)
        n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
        n = (n | (n << 2)) & np.uint32(0x33333333)
        n = (n | (n << 1)) & np.uint32(0x55555555)
        return n

    return np.asarray(
        _part1by1(x) | (_part1by1(y) << 1),
        dtype=np.uint32,
    )


def read_listings_parquet(path: Path) -> pd.DataFrame:
    """Read enriched listings parquet."""
    return pd.read_parquet(path)


def connect_duckdb(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open DuckDB and load the spatial extension when available."""
    conn = duckdb.connect(database)
    try:
        conn.execute("INSTALL spatial; LOAD spatial;")
    except duckdb.Error as exc:  # pragma: no cover - env-dependent
        logger.warning("DuckDB spatial extension unavailable: %s", exc)
    return conn


def query_listings(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    *,
    max_price_per_m2: float | None = None,
    district: str | None = None,
    in_oou: int | None = None,
) -> pd.DataFrame:
    """Filter enriched listings via DuckDB over a parquet file."""
    clauses: list[str] = ["1=1"]
    params: list[object] = [str(parquet_path)]
    if max_price_per_m2 is not None:
        clauses.append("price_per_m2 <= ?")
        params.append(max_price_per_m2)
    if district is not None:
        clauses.append("district = ?")
        params.append(district)
    if in_oou is not None:
        clauses.append("in_oou = ?")
        params.append(in_oou)
    # Clauses are fixed literals; values bound via params (not user SQL).
    where_sql = " AND ".join(clauses)
    sql = f"SELECT * FROM read_parquet(?) WHERE {where_sql}"  # nosec B608
    return conn.execute(sql, params).df()


def query_listings_bbox(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> pd.DataFrame:
    """BBox prune using covering columns written by ``write_geoparquet``."""
    sql = """
        SELECT * FROM read_parquet(?)
        WHERE bbox_maxx >= ?
          AND bbox_minx <= ?
          AND bbox_maxy >= ?
          AND bbox_miny <= ?
    """
    return conn.execute(
        sql,
        [str(parquet_path), west, east, south, north],
    ).df()
