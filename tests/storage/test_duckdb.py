"""Tests for DuckDB parquet filtering."""

from pathlib import Path

from happy_housing.gis import listings_to_gdf
from happy_housing.ingest import load_listings_json
from happy_housing.storage import (
    connect_duckdb,
    query_listings,
    query_listings_bbox,
    write_geoparquet,
)


def test_duckdb_filter_district(tmp_path: Path) -> None:
    """DuckDB query returns only the selected district."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    gdf = listings_to_gdf(listings)
    gdf["in_oou"] = 0
    path = tmp_path / "listings.parquet"
    write_geoparquet(gdf, path)
    conn = connect_duckdb()
    district = str(listings["district"].iloc[0])
    out = query_listings(conn, path, district=district)
    assert not out.empty
    assert (out["district"] == district).all()


def test_duckdb_bbox_prune(tmp_path: Path) -> None:
    """BBox query uses covering columns from write_geoparquet."""
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
    miss = query_listings_bbox(
        conn,
        path,
        west=0.0,
        south=0.0,
        east=1.0,
        north=1.0,
    )
    assert miss.empty
