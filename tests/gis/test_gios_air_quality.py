"""Tests: nearest GIOŚ station attaches AQ values, not only distance."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from happy_housing.gis import enrich_listings


def test_enrich_listings_attaches_nearest_station_air_quality() -> None:
    listings = pd.DataFrame({
        "listing_id": ["near_good", "near_poor"],
        "lat": [52.230, 52.141],
        "lon": [21.012, 21.034],
        "price_pln": [800_000.0, 700_000.0],
        "area_m2": [50.0, 45.0],
        "rooms": [2, 2],
        "year_built": [2000, 1995],
        "district": ["srodmiescie", "ursynow"],
        "listed_at": ["2026-01-01", "2026-01-01"],
    })
    amenities = gpd.GeoDataFrame(
        {"amenity": ["hospital"]},
        geometry=gpd.points_from_xy([21.0], [52.22]),
        crs="EPSG:4326",
    )
    gios = pd.DataFrame({
        "station_id": [1, 2],
        "lat": [52.2297, 52.1410],
        "lon": [21.0122, 21.0340],
        "name": ["good", "poor"],
        "city": ["Warszawa", "Warszawa"],
        "aq_index": [1.0, 4.0],
        "pm10_ug_m3": [10.0, 80.0],
        "pm25_ug_m3": [5.0, 50.0],
        "no2_ug_m3": [12.0, 55.0],
    })
    stops = pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon"])
    enriched = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        amenities=amenities,
        gios_stations=gios,
        gtfs_stops=stops,
        commute_lon=20.983,
        commute_lat=52.23,
    )
    by_id = enriched.set_index("listing_id")
    assert float(by_id.loc["near_good", "air_aq_index"]) == 1.0
    assert float(by_id.loc["near_good", "air_pm10_ug_m3"]) == 10.0
    assert float(by_id.loc["near_poor", "air_aq_index"]) == 4.0
    assert float(by_id.loc["near_poor", "air_pm25_ug_m3"]) == 50.0
    assert by_id["dist_air_station_m"].notna().all()
