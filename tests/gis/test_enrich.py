"""Tests for GIS enrichment on fixture listings."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from happy_housing.gis import enrich_listings
from happy_housing.ingest import load_listings_json


def test_aircraft_noise_and_distances(tmp_path: Path) -> None:
    """Aircraft Lden join and nearest distances populate expected columns."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    lden_path = tmp_path / "lden.geojson"
    gpd.GeoDataFrame(
        {
            "lden_db": [55],
            "geometry": [
                Polygon([
                    (20.90, 52.14),
                    (21.02, 52.14),
                    (21.02, 52.20),
                    (20.90, 52.20),
                    (20.90, 52.14),
                ])
            ],
        },
        crs="EPSG:4326",
    ).to_file(lden_path, driver="GeoJSON")
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
        aircraft_lden_path=lden_path,
    )
    assert "aircraft_lden_db" in enriched.columns
    assert "in_oou" in enriched.columns
    assert enriched["in_oou"].isin([0, 1]).all()
    assert enriched["dist_transit_stop_m"].notna().all()
    assert enriched["dist_metro_m"].notna().all()
    assert enriched["dist_daszynskiego_m"].notna().all()
    assert float(enriched["aircraft_lden_db"].max()) >= 55.0
    assert int(enriched["in_oou"].sum()) >= 1
