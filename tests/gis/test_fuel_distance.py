"""Regression: enrich attaches dist_fuel_m from OSM amenity=fuel."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from happy_housing.gis import enrich_listings


def test_enrich_listings_attaches_fuel_distance() -> None:
    listings = pd.DataFrame({
        "listing_id": ["near", "far"],
        "lat": [52.230, 52.30],
        "lon": [21.010, 21.10],
        "price_pln": [800_000.0, 700_000.0],
        "area_m2": [50.0, 45.0],
        "rooms": [2, 2],
        "year_built": [2000, 1995],
        "district": ["srodmiescie", "bialoleka"],
        "listed_at": ["2026-01-01", "2026-01-01"],
    })
    amenities = gpd.GeoDataFrame(
        {
            "amenity": ["fuel", "hospital"],
            "name": ["Orlen", "Szpital"],
        },
        geometry=[Point(21.011, 52.231), Point(21.02, 52.24)],
        crs="EPSG:4326",
    )
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    stops = pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon"])
    out = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        amenities=amenities,
        gios_stations=pd.DataFrame(columns=["lon", "lat"]),
        gtfs_stops=stops,
        commute_lon=21.0,
        commute_lat=52.23,
        parks=empty,
    )
    assert "dist_fuel_m" in out.columns
    by_id = out.set_index("listing_id")
    assert float(by_id.loc["near", "dist_fuel_m"]) < float(
        by_id.loc["far", "dist_fuel_m"]
    )
