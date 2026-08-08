"""Zone-polygon proximity features for the price model."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from happy_housing.gis import enrich_listings
from happy_housing.ingest import load_listings_json


def test_enrich_listings_zone_distances(tmp_path: Path) -> None:
    """Rail / military / airport polygons populate nearest-edge distances."""
    del tmp_path
    listings = load_listings_json(Path("data/fixtures/listings.json")).head(2)
    # Pin listing 0 near a rail yard; listing 1 farther away.
    listings = listings.copy()
    listings.loc[listings.index[0], "lon"] = 21.0
    listings.loc[listings.index[0], "lat"] = 52.23
    listings.loc[listings.index[1], "lon"] = 21.08
    listings.loc[listings.index[1], "lat"] = 52.23

    rail = gpd.GeoDataFrame(
        {"name": ["Yard"]},
        geometry=[
            Polygon([
                (20.999, 52.229),
                (21.001, 52.229),
                (21.001, 52.231),
                (20.999, 52.231),
                (20.999, 52.229),
            ])
        ],
        crs="EPSG:4326",
    )
    military = gpd.GeoDataFrame(
        {"name": ["Barracks"]},
        geometry=[
            Polygon([
                (21.079, 52.229),
                (21.081, 52.229),
                (21.081, 52.231),
                (21.079, 52.231),
                (21.079, 52.229),
            ])
        ],
        crs="EPSG:4326",
    )
    airport = gpd.GeoDataFrame(
        {"name": ["Field"]},
        geometry=[
            Polygon([
                (20.95, 52.15),
                (20.96, 52.15),
                (20.96, 52.16),
                (20.95, 52.16),
                (20.95, 52.15),
            ])
        ],
        crs="EPSG:4326",
    )
    amenities = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gios = pd.DataFrame(columns=["station_id", "lon", "lat"])
    stops = pd.DataFrame(columns=["stop_id", "stop_lon", "stop_lat"])

    enriched = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        amenities=amenities,
        gios_stations=gios,
        gtfs_stops=stops,
        commute_lon=20.98,
        commute_lat=52.23,
        rail_areas=rail,
        military=military,
        airport_areas=airport,
    )
    assert "dist_rail_area_m" in enriched.columns
    assert "dist_military_m" in enriched.columns
    assert "dist_airport_area_m" in enriched.columns
    assert float(enriched.iloc[0]["dist_rail_area_m"]) < float(
        enriched.iloc[1]["dist_rail_area_m"]
    )
    assert float(enriched.iloc[1]["dist_military_m"]) < float(
        enriched.iloc[0]["dist_military_m"]
    )
    assert enriched["dist_airport_area_m"].notna().all()


def test_enrich_listings_traffic_sample_features() -> None:
    """Traffic road sample yields nearest-road distance and local weight sum."""
    frame = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lon": [21.0, 21.05],
        "lat": [52.23, 52.23],
        "price_pln": [500_000.0, 500_000.0],
        "area_m2": [50.0, 50.0],
        "price_per_m2": [10_000.0, 10_000.0],
        "rooms": [2, 2],
        "district": ["Wola", "Wola"],
        "source": ["fixture", "fixture"],
    })
    traffic = pd.DataFrame({
        "lon": [21.0001, 21.0002, 21.06],
        "lat": [52.2301, 52.2302, 52.23],
        "base_weight": [5.0, 5.0, 1.0],
    })
    amenities = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    enriched = enrich_listings(
        frame,
        projected_crs="EPSG:2180",
        amenities=amenities,
        gios_stations=pd.DataFrame(columns=["station_id", "lon", "lat"]),
        gtfs_stops=pd.DataFrame(columns=["stop_id", "stop_lon", "stop_lat"]),
        commute_lon=20.98,
        commute_lat=52.23,
        traffic_sample=traffic,
    )
    assert "dist_traffic_road_m" in enriched.columns
    assert "traffic_weight_nearby" in enriched.columns
    assert float(enriched.iloc[0]["dist_traffic_road_m"]) < float(
        enriched.iloc[1]["dist_traffic_road_m"]
    )
    assert float(enriched.iloc[0]["traffic_weight_nearby"]) > float(
        enriched.iloc[1]["traffic_weight_nearby"]
    )
