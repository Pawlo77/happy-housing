"""Tests for city map object key dedupe (no OSM checkbox overlaps)."""

from __future__ import annotations

from happy_housing.gis.amenity_catalog import (
    CITY_MAP_OBJECT_EXCLUDE_KEYS,
    amenity_keys,
    amenity_type,
    city_map_object_keys,
)


def test_city_map_objects_exclude_osm_checkbox_duplicates() -> None:
    keys = city_map_object_keys()
    for dup in (
        "hospital",
        "fire_station",
        "police",
        "shop",
        "metro",
        "charging_station",
        "fuel",
    ):
        assert dup in CITY_MAP_OBJECT_EXCLUDE_KEYS
        assert dup not in keys
    # Still in full catalog for recommender / distances.
    assert "hospital" in amenity_keys()
    assert "fuel" in amenity_keys()
    assert amenity_type("fuel") is not None
    assert amenity_type("fuel").source == "osm"


def test_city_map_objects_keep_um_and_sports() -> None:
    keys = city_map_object_keys()
    assert "gym" in keys
    assert "outdoor_gym" in keys
    assert "library" in keys
    assert "police_um" in keys
    assert "pharmacy" in keys
