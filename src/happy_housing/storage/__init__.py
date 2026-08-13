"""DuckDB + GeoParquet storage helpers."""

import logging
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HILBERT_BITS: int = 16
"""Bits per axis for Hilbert keys (grid size ``2**bits``)."""


def xy_to_hilbert(x: int, y: int, *, bits: int = _HILBERT_BITS) -> int:
    """Map integer grid coordinates to a Hilbert curve index.

    Args:
        x: X cell in ``0 .. 2**bits - 1``.
        y: Y cell in ``0 .. 2**bits - 1``.
        bits: Bits per axis (default 16 → 65536² grid).

    Returns:
        Hilbert distance along the curve.
    """
    x_i = int(x) & ((1 << bits) - 1)
    y_i = int(y) & ((1 << bits) - 1)
    index = 0
    for level in range(bits - 1, -1, -1):
        mask = 1 << level
        rx = 1 if (x_i & mask) else 0
        ry = 1 if (y_i & mask) else 0
        index = (index << 2) | ((ry << 1) | (rx ^ ry))
        if ry == 0:
            if rx == 1:
                x_i = mask - 1 - x_i
                y_i = mask - 1 - y_i
            x_i, y_i = y_i, x_i
    return index


def _lonlat_to_hilbert_keys(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    bits: int = _HILBERT_BITS,
) -> np.ndarray:
    """Normalize lon/lat to a square grid then encode Hilbert keys."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = 1 << bits
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)
    gx = np.clip(((x - xmin) / dx * (n - 1)).astype(np.int64), 0, n - 1)
    gy = np.clip(((y - ymin) / dy * (n - 1)).astype(np.int64), 0, n - 1)
    return np.fromiter(
        (xy_to_hilbert(int(a), int(b), bits=bits) for a, b in zip(gx, gy, strict=True)),
        dtype=np.int64,
        count=len(gx),
    )


def write_geoparquet(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Write Hilbert-sorted GeoParquet with lon/lat + bbox covering columns.

    Uses a true Hilbert curve on a lon/lat grid (not Morton/Z-order). Output is
    GeoPandas GeoParquet (geometry + CRS metadata) so spatial tools round-trip;
    ``bbox_*`` / ``lon`` / ``lat`` stay plain columns for DuckDB attribute prune.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gdf.copy()
    if frame.empty or "geometry" not in frame.columns:
        empty = gpd.GeoDataFrame(frame, geometry="geometry", crs=gdf.crs or "EPSG:4326")
        empty.to_parquet(path, index=False)
        logger.info("Wrote empty GeoParquet → %s", path)
        return path

    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif str(frame.crs) != "EPSG:4326":
        frame = frame.to_crs("EPSG:4326")

    xs = frame.geometry.x
    ys = frame.geometry.y
    bounds = frame.geometry.bounds
    frame["lon"] = xs
    frame["lat"] = ys
    frame["bbox_minx"] = bounds.minx
    frame["bbox_miny"] = bounds.miny
    frame["bbox_maxx"] = bounds.maxx
    frame["bbox_maxy"] = bounds.maxy
    frame["_hilbert"] = _lonlat_to_hilbert_keys(xs.to_numpy(), ys.to_numpy())
    frame = frame.sort_values("_hilbert", kind="mergesort").drop(columns=["_hilbert"])
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
    frame.to_parquet(path, index=False)
    logger.info("Wrote %s GeoParquet rows → %s", len(frame), path)
    return path


def read_listings_parquet(path: Path) -> pd.DataFrame:
    """Read enriched listings parquet (attributes; geometry optional)."""
    try:
        return gpd.read_parquet(path).drop(columns=["geometry"], errors="ignore")
    except (ValueError, TypeError, OSError):
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
