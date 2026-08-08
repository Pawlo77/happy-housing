"""Tests for OSM commercial fitness centres merged into city POIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import Point

from happy_housing.gis import (
    CITY_POIS_SCHEMA_VERSION,
    DEFAULT_OSM_FITNESS_TAGS,
    fetch_osm_fitness_pois,
    osm_fitness_to_city_pois,
)
from happy_housing.gis.amenity_catalog import DEFAULT_MAP_OBJECT_TYPES, amenity_type
from happy_housing.ingest.warsaw_map import fetch_warsaw_city_pois


def test_default_map_includes_fitness_studio() -> None:
    """UM sala fitness + gyms should be on by default (commercial clubs via OSM)."""
    assert "gym" in DEFAULT_MAP_OBJECT_TYPES
    assert "fitness" in DEFAULT_MAP_OBJECT_TYPES
    fit = amenity_type("fitness")
    assert fit is not None
    assert fit.default_map_visible is True


def test_osm_fitness_tags_target_commercial_clubs() -> None:
    assert DEFAULT_OSM_FITNESS_TAGS.get("leisure") == "fitness_centre"
    assert DEFAULT_OSM_FITNESS_TAGS.get("amenity") == "gym"
    assert CITY_POIS_SCHEMA_VERSION >= 2


def test_osm_fitness_to_city_pois_maps_zdrofit_as_gym() -> None:
    raw = gpd.GeoDataFrame(
        {
            "leisure": ["fitness_centre", "fitness_centre"],
            "amenity": [None, "gym"],
            "name": ["Zdrofit Westfield Mokotów", None],
            "brand": [None, "CityFit"],
            "geometry": [Point(21.003, 52.180), Point(21.01, 52.181)],
        },
        crs="EPSG:4326",
    )
    out = osm_fitness_to_city_pois(raw)
    assert list(out["object_type"]) == ["gym", "gym"]
    assert list(out["source"]) == ["osm", "osm"]
    assert "Zdrofit" in str(out.iloc[0]["name"])
    assert "CityFit" in str(out.iloc[1]["name"])


def test_fetch_osm_fitness_pois_caches(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fresh = gpd.GeoDataFrame(
        {
            "leisure": ["fitness_centre"],
            "amenity": [None],
            "name": ["Zdrofit"],
            "brand": [None],
            "geometry": [Point(21.0, 52.18)],
        },
        crs="EPSG:4326",
    )
    mock_ox = MagicMock()
    mock_ox.features_from_bbox.return_value = fresh
    mock_ox.settings = MagicMock()
    real_import = __import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "osmnx":
            return mock_ox
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        out = fetch_osm_fitness_pois((20.8, 52.0, 21.3, 52.4), cache_dir)

    assert len(out) == 1
    assert out.iloc[0]["object_type"] == "gym"
    assert (cache_dir / "osm_fitness.geojson").exists()
    tags = mock_ox.features_from_bbox.call_args.kwargs["tags"]
    assert tags["leisure"] == "fitness_centre"


def test_fetch_warsaw_city_pois_merges_osm_fitness(tmp_path: Path) -> None:
    """Combined city POI cache includes OSM commercial gyms."""
    sports = gpd.GeoDataFrame(
        {
            "object_type": ["outdoor_gym"],
            "name": ["Siłownia plenerowa"],
            "source": ["sports"],
            "geometry": [Point(21.0, 52.18)],
        },
        crs="EPSG:4326",
    )
    osm = gpd.GeoDataFrame(
        {
            "object_type": ["gym"],
            "name": ["Zdrofit"],
            "source": ["osm"],
            "geometry": [Point(21.003, 52.180)],
        },
        crs="EPSG:4326",
    )
    with (
        patch(
            "happy_housing.ingest.warsaw_map.fetch_warsaw_sports_objects",
            return_value=sports,
        ),
        patch(
            "happy_housing.ingest.warsaw_map.fetch_warsaw_civic_pois",
            return_value=gpd.GeoDataFrame(
                columns=["object_type", "name", "source", "geometry"],
                geometry=[],
                crs="EPSG:4326",
            ),
        ),
        patch(
            "happy_housing.ingest.warsaw_map.fetch_osm_fitness_pois",
            return_value=osm,
        ),
    ):
        out = fetch_warsaw_city_pois(tmp_path, force_refresh=True)

    assert set(out["source"]) >= {"sports", "osm"}
    assert "Zdrofit" in set(out["name"].astype(str))
    assert (tmp_path / "warsaw_city_pois.geojson").exists()
    meta = tmp_path / "warsaw_city_pois.meta.json"
    assert meta.exists()
