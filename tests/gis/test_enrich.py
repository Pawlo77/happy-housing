"""Tests for GIS enrichment on fixture listings."""

import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from happy_housing.gis import enrich_listings
from happy_housing.gis.frame_util import assign_columns
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
    assert "dist_school_m" in enriched.columns
    assert "dist_shop_m" in enriched.columns
    assert "road_lden_db" in enriched.columns
    assert "rail_lden_db" in enriched.columns


def test_enrich_listings_school_shop_and_road_rail_lden(tmp_path: Path) -> None:
    """UM schools, shops, and road/rail Lden contours attach expected features."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    schools = gpd.GeoDataFrame(
        {"name": ["SP Bliska"]},
        geometry=gpd.points_from_xy([21.00], [52.225]),
        crs="EPSG:4326",
    )
    shops = gpd.GeoDataFrame(
        {"shop": ["supermarket"], "name": ["Sklep Bliski"]},
        geometry=gpd.points_from_xy([21.01], [52.23]),
        crs="EPSG:4326",
    )
    road_path = tmp_path / "road_lden.geojson"
    rail_path = tmp_path / "rail_lden.geojson"
    gpd.GeoDataFrame(
        {
            "lden_db": [65],
            "geometry": [
                Polygon([
                    (20.97, 52.22),
                    (20.99, 52.22),
                    (20.99, 52.24),
                    (20.97, 52.24),
                    (20.97, 52.22),
                ])
            ],
        },
        crs="EPSG:4326",
    ).to_file(road_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "lden_db": [55],
            "geometry": [
                Polygon([
                    (21.00, 52.22),
                    (21.03, 52.22),
                    (21.03, 52.24),
                    (21.00, 52.24),
                    (21.00, 52.22),
                ])
            ],
        },
        crs="EPSG:4326",
    ).to_file(rail_path, driver="GeoJSON")
    amenities = gpd.GeoDataFrame(
        {"amenity": ["hospital", "fire_station", "police", "subway", "school"]},
        geometry=gpd.points_from_xy(
            [21.0, 20.98, 21.02, 21.01, 20.90],
            [52.22, 52.21, 52.23, 52.229, 52.10],
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
        schools=schools,
        shops=shops,
        road_lden_path=road_path,
        rail_lden_path=rail_path,
    )
    assert enriched["dist_school_m"].notna().all()
    assert float(enriched["dist_school_m"].min()) < 5_000.0
    assert enriched["dist_shop_m"].notna().all()
    assert float(enriched["road_lden_db"].max()) >= 65.0
    assert float(enriched["rail_lden_db"].max()) >= 55.0


def test_enrich_listings_school_falls_back_to_osm_amenity() -> None:
    """Empty UM schools → nearest OSM amenity=school distance."""
    listings = pd.DataFrame({
        "listing_id": ["a"],
        "lon": [21.0],
        "lat": [52.23],
        "price_pln": [500_000],
        "area_m2": [50.0],
    })
    amenities = gpd.GeoDataFrame(
        {"amenity": ["school"]},
        geometry=gpd.points_from_xy([21.001], [52.231]),
        crs="EPSG:4326",
    )
    enriched = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        amenities=amenities,
        gios_stations=pd.DataFrame(columns=["lon", "lat"]),
        gtfs_stops=pd.DataFrame(columns=["stop_lon", "stop_lat"]),
        commute_lon=21.0,
        commute_lat=52.23,
        schools=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    )
    assert enriched["dist_school_m"].notna().all()
    assert float(enriched["dist_school_m"].iloc[0]) < 1_000.0


def test_assign_columns_joins_without_fragmentation() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    out = assign_columns(frame, {"b": [3, 4], "c": [5, 6]})
    assert list(out.columns) == ["a", "b", "c"]
    assert out["b"].tolist() == [3, 4]


def test_enrich_listings_no_fragmentation_warning(tmp_path: Path) -> None:
    """Many amenity distance columns must not trigger pandas PerformanceWarning."""
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    amenities = gpd.GeoDataFrame(
        {"amenity": ["hospital", "fire_station", "police", "subway"]},
        geometry=gpd.points_from_xy(
            [21.0, 20.98, 21.02, 21.01],
            [52.22, 52.21, 52.23, 52.229],
        ),
        crs="EPSG:4326",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=pd.errors.PerformanceWarning)
        enriched = enrich_listings(
            listings,
            projected_crs="EPSG:2180",
            amenities=amenities,
            gios_stations=pd.DataFrame(columns=["lon", "lat"]),
            gtfs_stops=pd.DataFrame({
                "stop_id": ["A"],
                "stop_lat": [52.23],
                "stop_lon": [21.00],
            }),
            commute_lon=20.983,
            commute_lat=52.23,
        )
    perf = [w for w in caught if issubclass(w.category, pd.errors.PerformanceWarning)]
    assert not perf
    assert "dist_transit_stop_m" in enriched.columns
    assert "dist_gym_m" in enriched.columns
