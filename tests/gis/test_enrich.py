"""Tests for GIS enrichment on fixture listings."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from happy_housing.gis import enrich_listings, load_or_create_oou_polygon
from happy_housing.ingest import load_listings_json


def test_oou_flag_and_distances(tmp_path: Path) -> None:
    """OOU join and nearest distances populate expected columns."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    oou_path = tmp_path / "oou.geojson"
    load_or_create_oou_polygon(oou_path)
    amenities = gpd.GeoDataFrame(
        {"amenity": ["hospital", "fire_station", "police", "subway"]},
        geometry=gpd.points_from_xy(
            [21.0, 20.98, 21.02, 21.01],
            [52.22, 52.21, 52.23, 52.229],
        ),
        crs="EPSG:4326",
    )
    gios = pd.DataFrame({
        "station_id": [1],
        "lat": [52.23],
        "lon": [21.01],
        "name": ["Test"],
        "city": ["Warszawa"],
    })
    stops = pd.DataFrame({"stop_id": ["A"], "stop_lat": [52.23], "stop_lon": [21.00]})
    enriched = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        oou_path=oou_path,
        amenities=amenities,
        gios_stations=gios,
        gtfs_stops=stops,
        commute_lon=20.983,
        commute_lat=52.23,
    )
    assert "in_oou" in enriched.columns
    assert enriched["in_oou"].isin([0, 1]).all()
    assert enriched["dist_transit_stop_m"].notna().all()
    assert enriched["dist_metro_m"].notna().all()
    assert enriched["dist_daszynskiego_m"].notna().all()
    assert int(enriched["in_oou"].sum()) >= 1
