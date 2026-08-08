"""Tests for OSM bike-path fetch, nearest distance, and neighbourhood coverage."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point

from happy_housing.gis.bike_paths import (
    BIKE_PATH_COVERAGE_RADIUS_M,
    attach_bike_path_features,
    densify_bike_paths,
    fetch_bike_paths,
)


def test_densify_bike_paths_emits_points_along_line() -> None:
    roads = gpd.GeoDataFrame(
        {"highway": ["cycleway"]},
        geometry=[LineString([(21.0, 52.23), (21.01, 52.23)])],
        crs="EPSG:4326",
    )
    pts = densify_bike_paths(roads, step_m=100.0)
    assert len(pts) >= 5
    assert {"lon", "lat", "step_m"}.issubset(pts.columns)


def test_attach_bike_path_features_distance_and_coverage() -> None:
    listings = gpd.GeoDataFrame(
        {"listing_id": ["a", "b"]},
        geometry=[Point(21.0, 52.23), Point(21.05, 52.23)],
        crs="EPSG:4326",
    )
    # Path near listing a only.
    paths = gpd.GeoDataFrame(
        {"highway": ["cycleway"]},
        geometry=[LineString([(20.999, 52.2305), (21.001, 52.2305)])],
        crs="EPSG:4326",
    )
    out = attach_bike_path_features(
        listings,
        paths,
        projected_crs="EPSG:2180",
        coverage_radius_m=300.0,
        sample_step_m=25.0,
    )
    assert "dist_bike_path_m" in out.columns
    assert "bike_path_coverage_m" in out.columns
    assert float(out.loc[0, "dist_bike_path_m"]) < float(out.loc[1, "dist_bike_path_m"])
    assert float(out.loc[0, "bike_path_coverage_m"]) > float(
        out.loc[1, "bike_path_coverage_m"]
    )
    assert float(out.loc[0, "bike_path_coverage_m"]) > 0.0


def test_attach_bike_path_features_empty_paths() -> None:
    listings = gpd.GeoDataFrame(
        {"listing_id": ["a"]},
        geometry=[Point(21.0, 52.23)],
        crs="EPSG:4326",
    )
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    out = attach_bike_path_features(listings, empty, projected_crs="EPSG:2180")
    assert out["dist_bike_path_m"].isna().all()
    assert (out["bike_path_coverage_m"] == 0.0).all()


def test_fetch_bike_paths_uses_fixture(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "bike_paths.geojson"
    gpd.GeoDataFrame(
        {"highway": ["cycleway"], "name": ["Test"]},
        geometry=[LineString([(21.0, 52.23), (21.01, 52.23)])],
        crs="EPSG:4326",
    ).to_file(fixture, driver="GeoJSON")

    def _fail(*_a, **_k):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "happy_housing.gis.bike_paths._download_osm_bike_paths",
        _fail,
    )
    out = fetch_bike_paths(
        tmp_path,
        bbox=(20.9, 52.1, 21.2, 52.4),
        fixture_path=fixture,
        force_refresh=True,
    )
    assert not out.empty
    assert (tmp_path / "bike_paths.geojson").exists()
    assert BIKE_PATH_COVERAGE_RADIUS_M == 500.0
