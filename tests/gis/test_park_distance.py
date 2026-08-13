"""Tests for park-proximity enrichment."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from happy_housing.gis import enrich_listings
from happy_housing.ingest import load_listings_json


def test_enrich_listings_dist_park_m(tmp_path: Path) -> None:
    """Nearest park polygon populates dist_park_m for every listing."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    parks = gpd.GeoDataFrame(
        {"name": ["Park Bliski"]},
        geometry=[
            Polygon([
                (21.00, 52.22),
                (21.01, 52.22),
                (21.01, 52.23),
                (21.00, 52.23),
                (21.00, 52.22),
            ])
        ],
        crs="EPSG:4326",
    )
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
        amenities=amenities,
        gios_stations=gios,
        gtfs_stops=stops,
        commute_lon=20.983,
        commute_lat=52.23,
        parks=parks,
    )
    assert "dist_park_m" in enriched.columns
    assert enriched["dist_park_m"].notna().all()
    assert float(enriched["dist_park_m"].min()) < 5_000.0
