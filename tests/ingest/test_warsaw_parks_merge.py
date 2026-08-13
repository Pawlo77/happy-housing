"""Tests for UM + OSM park merge and WFS truncation detection."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from happy_housing.ingest.warsaw_map import (
    PARKS_CACHE_SCHEMA,
    fetch_warsaw_parks,
    merge_park_frames,
    wfs_page_looks_truncated,
)


def test_wfs_page_looks_truncated() -> None:
    assert wfs_page_looks_truncated(10_000, max_features=10_000) is True
    assert wfs_page_looks_truncated(427, max_features=10_000) is False


def test_merge_park_frames_tags_sources() -> None:
    um = gpd.GeoDataFrame(
        {"name": ["Park UM"]},
        geometry=[Polygon([(21.0, 52.2), (21.01, 52.2), (21.01, 52.21), (21.0, 52.2)])],
        crs="EPSG:4326",
    )
    osm = gpd.GeoDataFrame(
        {"name": ["Park OSM"]},
        geometry=[
            Polygon([(21.02, 52.2), (21.03, 52.2), (21.03, 52.21), (21.02, 52.2)])
        ],
        crs="EPSG:4326",
    )
    out = merge_park_frames(um, osm)
    assert len(out) == 2
    assert set(out["park_source"].astype(str)) == {"um_place_skwery", "osm"}


def test_fetch_warsaw_parks_merges_osm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    um = gpd.GeoDataFrame(
        {"name": ["Skwer A"]},
        geometry=[Polygon([(21.0, 52.2), (21.01, 52.2), (21.01, 52.21), (21.0, 52.2)])],
        crs="EPSG:4326",
    )
    osm = gpd.GeoDataFrame(
        {"name": ["Las Kabacki"], "leisure": ["park"]},
        geometry=[
            Polygon([(21.05, 52.15), (21.08, 52.15), (21.08, 52.18), (21.05, 52.15)])
        ],
        crs="EPSG:4326",
    )

    def fake_wfs(*_a: object, **_k: object) -> gpd.GeoDataFrame:
        return um.copy()

    def fake_osm(*_a: object, **_k: object) -> gpd.GeoDataFrame:
        return osm.copy()

    monkeypatch.setattr(
        "happy_housing.ingest.warsaw_map.fetch_warsaw_wfs_layer", fake_wfs
    )
    monkeypatch.setattr(
        "happy_housing.ingest.warsaw_map.fetch_osm_green_areas", fake_osm
    )

    out = fetch_warsaw_parks(
        tmp_path,
        force_refresh=True,
        bbox=(20.85, 52.10, 21.28, 52.37),
    )
    assert len(out) == 2
    assert PARKS_CACHE_SCHEMA >= 2
    cached = tmp_path / "warsaw_parks.geojson"
    assert cached.exists()
    assert (tmp_path / "warsaw_parks.meta.json").exists()
    assert "Las Kabacki" in set(out["name"].astype(str))
