"""Tests for full-scope OSM amenity/shop Overpass fetches and cache invalidation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import Point

from happy_housing.gis import (
    DEFAULT_OSM_AMENITY_TAGS,
    DEFAULT_OSM_SHOP_TAGS,
    OSM_CACHE_SCHEMA_VERSION,
    fetch_osm_amenities,
    fetch_osm_shops,
)


def test_default_osm_tags_request_entire_keys() -> None:
    """Defaults pull every amenity/shop in the bbox, not a tiny allow-list."""
    assert DEFAULT_OSM_AMENITY_TAGS.get("amenity") is True
    assert DEFAULT_OSM_AMENITY_TAGS.get("station") == "subway"
    assert DEFAULT_OSM_SHOP_TAGS.get("shop") is True
    assert OSM_CACHE_SCHEMA_VERSION >= 1


def test_stale_narrow_amenity_cache_refetches(tmp_path: Path) -> None:
    """Old amenity GeoJSON without matching meta must not short-circuit."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale = gpd.GeoDataFrame(
        {"amenity": ["hospital"]},
        geometry=[Point(21.0, 52.23)],
        crs="EPSG:4326",
    )
    stale.to_file(cache_dir / "osm_amenities.geojson", driver="GeoJSON")

    fresh = gpd.GeoDataFrame(
        {
            "amenity": ["hospital", "cafe", "school"],
            "station": [None, None, None],
            "name": ["H", "C", "S"],
            "geometry": [Point(21.0, 52.23), Point(21.01, 52.24), Point(21.02, 52.22)],
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
        out = fetch_osm_amenities(
            (20.8, 52.0, 21.3, 52.4),
            cache_dir,
        )

    assert len(out) == 3
    mock_ox.features_from_bbox.assert_called_once()
    tags = mock_ox.features_from_bbox.call_args.kwargs["tags"]
    assert tags["amenity"] is True
    meta = cache_dir / "osm_amenities.meta.json"
    assert meta.exists()


def test_valid_amenity_cache_skips_overpass(tmp_path: Path) -> None:
    """Matching meta + geojson reuse cache without calling OSMnx."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = gpd.GeoDataFrame(
        {"amenity": ["cafe", "hospital"]},
        geometry=[Point(21.0, 52.23), Point(21.01, 52.24)],
        crs="EPSG:4326",
    )
    cache_path = cache_dir / "osm_amenities.geojson"
    cached.to_file(cache_path, driver="GeoJSON")
    # Write meta via a force_refresh=False path after seeding — call helper
    # by fetching once with patched ox, then second call must not hit ox.
    mock_ox = MagicMock()
    mock_ox.features_from_bbox.return_value = cached
    mock_ox.settings = MagicMock()
    real_import = __import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "osmnx":
            return mock_ox
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import):
        # First call with force_refresh writes meta for current defaults.
        fetch_osm_amenities(
            (20.8, 52.0, 21.3, 52.4),
            cache_dir,
            force_refresh=True,
        )
        mock_ox.features_from_bbox.reset_mock()
        out = fetch_osm_amenities((20.8, 52.0, 21.3, 52.4), cache_dir)

    mock_ox.features_from_bbox.assert_not_called()
    assert len(out) == 2


def test_force_refresh_shops_ignores_cache(tmp_path: Path) -> None:
    """force_refresh=True always re-queries shops."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale = gpd.GeoDataFrame(
        {"shop": ["bakery"]},
        geometry=[Point(21.0, 52.23)],
        crs="EPSG:4326",
    )
    stale.to_file(cache_dir / "osm_shops.geojson", driver="GeoJSON")
    fresh = gpd.GeoDataFrame(
        {
            "shop": ["supermarket", "clothes"],
            "name": ["A", "B"],
            "geometry": [Point(21.0, 52.23), Point(21.01, 52.24)],
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
        out = fetch_osm_shops(
            (20.8, 52.0, 21.3, 52.4),
            cache_dir,
            force_refresh=True,
        )

    assert len(out) == 2
    tags = mock_ox.features_from_bbox.call_args.kwargs["tags"]
    assert tags["shop"] is True


def test_sample_rcn_default_is_unlimited() -> None:
    """Legacy RCN sample helper must not silently cap at 2000 rows."""
    import inspect

    from happy_housing.ingest import sample_rcn_from_zip

    default = inspect.signature(sample_rcn_from_zip).parameters["max_rows"].default
    assert default is None
