"""Tests for shared Streamlit mtime-keyed I/O caches."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from happy_housing.app import io_cache
from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.data import clear_data_caches, load_shops
from happy_housing.app.metadata import (
    RunMetadata,
    load_run_metadata,
    write_run_metadata,
)
from happy_housing.app.stars import load_stars, save_stars, stars_path


def test_read_json_is_mtime_keyed(tmp_path: Path) -> None:
    path = tmp_path / "blob.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    first = io_cache.read_json(str(path), path.stat().st_mtime)
    assert first == {"a": 1}
    path.write_text(json.dumps({"a": 2}), encoding="utf-8")
    second = io_cache.read_json(str(path), path.stat().st_mtime)
    assert second == {"a": 2}


def test_load_stars_and_settings_use_disk_cache(tmp_path: Path) -> None:
    stars = stars_path(tmp_path)
    save_stars(stars, frozenset({"A", "B"}))
    assert load_stars(stars) == frozenset({"A", "B"})

    settings_file = settings_path(tmp_path)
    save_app_settings(settings_file, AppSettings(recommender_top_n=9))
    loaded = load_app_settings(settings_file)
    assert loaded.recommender_top_n == 9


def test_load_run_metadata_cached(tmp_path: Path) -> None:
    path = tmp_path / "run_metadata.json"
    write_run_metadata(
        path,
        RunMetadata(
            refreshed_at="2026-08-08T12:00:00+00:00",
            duration_s=1.5,
            n_offers=10,
        ),
    )
    meta = load_run_metadata(path)
    assert meta is not None
    assert meta.n_offers == 10


def test_load_shops_from_geojson_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shops = cache_dir / "osm_shops.geojson"
    gdf = gpd.GeoDataFrame(
        {"name": ["A"], "shop": ["bakery"]},
        geometry=[Point(21.0, 52.2)],
        crs="EPSG:4326",
    )
    gdf.to_file(shops, driver="GeoJSON")

    class _Data:
        def cache_path(self) -> Path:
            return cache_dir

        def fixtures_path(self) -> Path:
            return tmp_path / "fixtures"

    class _Cfg:
        data = _Data()

    monkeypatch.setattr("happy_housing.app.data.load_config", lambda: _Cfg())
    frame = load_shops()
    assert len(frame) == 1
    assert frame.iloc[0]["name"] == "A"


def test_clear_data_caches_runs() -> None:
    clear_data_caches()
