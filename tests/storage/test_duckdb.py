"""Tests for DuckDB parquet filtering."""

from pathlib import Path

from happy_housing.gis import listings_to_gdf
from happy_housing.ingest import load_listings_json
from happy_housing.storage import connect_duckdb, query_listings, write_geoparquet


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
