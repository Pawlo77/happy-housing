"""Tests for city map object key dedupe (no checkbox overlaps)."""

from __future__ import annotations

from happy_housing.gis.amenity_catalog import (
    CITY_MAP_OBJECT_EXCLUDE_KEYS,
    amenity_keys,
    amenity_type,
    city_map_object_keys,
    city_map_objects_by_group,
)


def test_city_map_objects_exclude_checkbox_duplicates() -> None:
    keys = city_map_object_keys()
    for dup in (
        "hospital",
        "fire_station",
        "police",
        "police_um",
        "shop",
        "metro",
        "charging_station",
        "fuel",
        "park",
    ):
        assert dup in CITY_MAP_OBJECT_EXCLUDE_KEYS
        assert dup not in keys
    # Still in full catalog for recommender / distances.
    assert "hospital" in amenity_keys()
    assert "fuel" in amenity_keys()
    assert "police_um" in amenity_keys()
    assert amenity_type("fuel") is not None
    assert amenity_type("fuel").source == "osm"


def test_city_map_objects_keep_um_and_sports() -> None:
    keys = city_map_object_keys()
    assert "gym" in keys
    assert "outdoor_gym" in keys
    assert "library" in keys
    assert "pharmacy" in keys
    assert "police_um" not in keys


def test_city_map_objects_by_group_splits_sports_and_civic() -> None:
    sports = city_map_objects_by_group("Sports")
    civic = city_map_objects_by_group("Civic")
    assert "gym" in sports
    assert "indoor_pool" in sports
    assert "library" in civic
    assert "hotel" in civic
    assert set(sports).isdisjoint(civic)
    assert set(sports) | set(civic) == set(city_map_object_keys())
    assert "police_um" not in civic
    assert "hospital" not in civic
    assert not any(
        (t := amenity_type(k)) is not None and t.group == "OSM"
        for k in city_map_object_keys()
    )
