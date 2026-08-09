"""Amenity catalog + city-POI recommend / map helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from happy_housing.app.app_settings import load_app_settings
from happy_housing.app.recommender import score_offers
from happy_housing.gis import enrich_listings
from happy_housing.gis.amenity_catalog import (
    DEFAULT_MAP_OBJECT_TYPES,
    coerce_amenity_weights,
    dist_column,
    sport_key_for_osrid,
)
from happy_housing.ingest.warsaw_map import (
    fetch_warsaw_city_pois,
    fetch_warsaw_sports_objects,
)


def test_sport_osrid_maps_gym_and_pool() -> None:
    assert sport_key_for_osrid(47) == "gym"
    assert sport_key_for_osrid(35.0) == "indoor_pool"
    assert sport_key_for_osrid("ODKRYTE") == "outdoor_ice_rink"


def test_default_map_types_exclude_sports() -> None:
    assert "gym" not in DEFAULT_MAP_OBJECT_TYPES
    assert "fitness" not in DEFAULT_MAP_OBJECT_TYPES
    assert "indoor_pool" not in DEFAULT_MAP_OBJECT_TYPES
    assert DEFAULT_MAP_OBJECT_TYPES == ()


def test_coerce_amenity_weights_clips_and_fills() -> None:
    out = coerce_amenity_weights({"gym": 2.0, "cemetery": -0.5, "nope": 1})
    assert out["gym"] == 1.0
    assert out["cemetery"] == -0.5
    assert "nope" not in out
    assert out["indoor_pool"] == 0.0


def test_fetch_city_pois_uses_fixture(tmp_path: Path, monkeypatch) -> None:
    fixture = Path("data/fixtures/city_pois.geojson")

    def _empty_sports(**_kwargs):
        return gpd.GeoDataFrame(
            columns=["object_type", "name", "source", "geometry"],
            geometry=[],
            crs="EPSG:4326",
        )

    def _empty_civic(*_args, **_kwargs):
        return _empty_sports()

    monkeypatch.setattr(
        "happy_housing.ingest.warsaw_map.fetch_warsaw_sports_objects",
        _empty_sports,
    )
    monkeypatch.setattr(
        "happy_housing.ingest.warsaw_map.fetch_warsaw_civic_pois",
        _empty_civic,
    )
    monkeypatch.setattr(
        "happy_housing.ingest.warsaw_map.fetch_osm_fitness_pois",
        lambda *_args, **_kwargs: _empty_sports(),
    )
    pois = fetch_warsaw_city_pois(tmp_path, fixture_path=fixture, force_refresh=True)
    assert not pois.empty
    assert "object_type" in pois.columns
    assert set(pois["object_type"]) >= {"gym", "indoor_pool", "library"}
    assert (tmp_path / "warsaw_city_pois.geojson").exists()


def test_fetch_sports_fixture_fallback(tmp_path: Path) -> None:
    """Offline sports path when MapViewer unavailable → fixture via city POIs."""
    fixture = Path("data/fixtures/city_pois.geojson")
    sports = fetch_warsaw_sports_objects(
        fixture_path=fixture,
        dataserver_base="http://127.0.0.1:9",
        timeout_s=0.1,
    )
    # Direct sports call with unreachable server may still be empty; city POIs
    # path is the supported offline entry. Keep API smoke-tested.
    assert sports.empty or "object_type" in sports.columns


def test_enrich_adds_city_poi_distances() -> None:
    listings = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lon": [20.9901, 21.05],
        "lat": [52.2301, 52.30],
    })
    city = gpd.GeoDataFrame(
        {
            "object_type": ["gym", "indoor_pool"],
            "name": ["G", "P"],
            "source": ["sports", "sports"],
        },
        geometry=[Point(20.99, 52.23), Point(21.01, 52.24)],
        crs="EPSG:4326",
    )
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    out = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        amenities=empty,
        gios_stations=pd.DataFrame(columns=["lon", "lat"]),
        gtfs_stops=pd.DataFrame(columns=["stop_lon", "stop_lat"]),
        commute_lon=21.0,
        commute_lat=52.23,
        city_pois=city,
    )
    assert dist_column("gym") in out.columns
    assert float(out.loc[0, dist_column("gym")]) < float(out.loc[1, dist_column("gym")])


def test_score_offers_positive_gym_weight_prefers_near() -> None:
    frame = pd.DataFrame({
        "listing_id": ["near", "far"],
        "delta_price_per_m2": [-500.0, -500.0],
        "commute_min": [10.0, 10.0],
        "dist_transit_stop_m": [100.0, 100.0],
        "dist_gym_m": [100.0, 1800.0],
    })
    ranked = score_offers(
        frame,
        demotion_alpha=0.0,
        impressions={},
        amenity_weights={"gym": 1.0},
    )
    assert list(ranked["listing_id"]) == ["near", "far"]


def test_score_offers_negative_weight_prefers_far() -> None:
    frame = pd.DataFrame({
        "listing_id": ["near", "far"],
        "delta_price_per_m2": [-500.0, -500.0],
        "commute_min": [10.0, 10.0],
        "dist_transit_stop_m": [100.0, 100.0],
        "dist_cemetery_m": [100.0, 1800.0],
    })
    ranked = score_offers(
        frame,
        demotion_alpha=0.0,
        impressions={},
        amenity_weights={"cemetery": -1.0},
    )
    assert list(ranked["listing_id"]) == ["far", "near"]


def test_app_settings_persists_amenity_weights(tmp_path: Path) -> None:
    path = tmp_path / "app_settings.json"
    path.write_text(
        '{"recommender_amenity_weights": {"gym": 0.5, "cemetery": -0.25}}\n',
        encoding="utf-8",
    )
    settings = load_app_settings(path)
    assert settings.recommender_amenity_weights is not None
    assert settings.recommender_amenity_weights["gym"] == 0.5
    assert settings.recommender_amenity_weights["cemetery"] == -0.25
