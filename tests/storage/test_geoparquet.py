"""Tests for Hilbert locality keys and GeoParquet writes."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from happy_housing.gis import listings_to_gdf
from happy_housing.ingest import load_listings_json
from happy_housing.storage import (
    connect_duckdb,
    query_listings_bbox,
    write_geoparquet,
    xy_to_hilbert,
)


def test_xy_to_hilbert_is_deterministic() -> None:
    """Same grid cell always maps to the same Hilbert index."""
    assert xy_to_hilbert(0, 0) == xy_to_hilbert(0, 0)
    assert xy_to_hilbert(1, 2) != xy_to_hilbert(200, 200)


def test_hilbert_keeps_neighbors_closer_than_far_points() -> None:
    """Adjacent cells have smaller |ΔH| than distant cells (order-16 curve)."""
    near = abs(xy_to_hilbert(100, 100) - xy_to_hilbert(101, 100))
    far = abs(xy_to_hilbert(100, 100) - xy_to_hilbert(20_000, 20_000))
    assert near < far


def test_write_geoparquet_roundtrips_geometry(tmp_path: Path) -> None:
    """Writer emits GeoParquet readable by GeoPandas with CRS intact."""
    gdf = gpd.GeoDataFrame(
        {"district": ["wola"], "price_per_m2": [15000.0]},
        geometry=[Point(20.98, 52.23)],
        crs="EPSG:4326",
    )
    path = tmp_path / "geo.parquet"
    write_geoparquet(gdf, path)
    back = gpd.read_parquet(path)
    assert not back.empty
    assert back.crs is not None
    assert "bbox_minx" in back.columns
    assert abs(float(back.geometry.x.iloc[0]) - 20.98) < 1e-6


def test_write_geoparquet_duckdb_bbox_still_works(tmp_path: Path) -> None:
    """Attribute bbox columns remain queryable via DuckDB after GeoParquet write."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    gdf = listings_to_gdf(listings)
    path = tmp_path / "listings.parquet"
    write_geoparquet(gdf, path)
    conn = connect_duckdb()
    lat0 = float(listings["lat"].iloc[0])
    lon0 = float(listings["lon"].iloc[0])
    hit = query_listings_bbox(
        conn,
        path,
        west=lon0 - 0.01,
        south=lat0 - 0.01,
        east=lon0 + 0.01,
        north=lat0 + 0.01,
    )
    assert not hit.empty
    # Flat pandas read still sees covering columns for app pipelines.
    flat = pd.read_parquet(path)
    assert {"bbox_minx", "lon", "lat"} <= set(flat.columns)
