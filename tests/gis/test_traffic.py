"""Tests for OSM diurnal traffic-proxy road sampling."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from happy_housing.gis.traffic import (
    HIGHWAY_BASE_WEIGHT,
    _cap_sample,
    densify_road_lines,
    fetch_traffic_road_sample,
    highway_weight,
    normalize_highway_key,
)


def test_highway_weight_ranks_motorway_above_residential() -> None:
    assert highway_weight("motorway") > highway_weight("primary")
    assert highway_weight("primary") > highway_weight("residential")
    assert highway_weight("unknown_tag") == HIGHWAY_BASE_WEIGHT["unclassified"]


def test_normalize_highway_key_picks_heavier_from_list() -> None:
    assert normalize_highway_key(["residential", "primary"]) == "primary"
    assert normalize_highway_key("secondary;service") == "secondary"


def test_densify_road_lines_emits_points_along_segment() -> None:
    roads = gpd.GeoDataFrame(
        {"highway": ["primary"], "name": ["Test"]},
        geometry=[
            LineString([(21.0, 52.23), (21.01, 52.23)]),  # ~680 m east
        ],
        crs="EPSG:4326",
    )
    pts = densify_road_lines(roads, fine_step_m=100.0, coarse_step_m=100.0)
    assert len(pts) >= 5
    assert set(pts.columns) >= {"lon", "lat", "base_weight", "name", "highway"}
    assert pts["base_weight"].iloc[0] == HIGHWAY_BASE_WEIGHT["primary"]
    assert pts["lon"].between(21.0, 21.01).all()


def test_densify_skips_service_highways() -> None:
    roads = gpd.GeoDataFrame(
        {"highway": ["service", "residential"], "name": ["a", "b"]},
        geometry=[
            LineString([(21.0, 52.23), (21.005, 52.23)]),
            LineString([(21.0, 52.24), (21.005, 52.24)]),
        ],
        crs="EPSG:4326",
    )
    pts = densify_road_lines(roads, fine_step_m=200.0, coarse_step_m=200.0)
    assert set(pts["highway"]) == {"residential"}


def test_cap_sample_keeps_residential_share() -> None:
    frame = pd.DataFrame({
        "lon": [21.0] * 1000,
        "lat": [52.2] * 1000,
        "base_weight": [10.0] * 100 + [1.6] * 900,
        "name": ["x"] * 1000,
        "highway": ["motorway"] * 100 + ["residential"] * 900,
    })
    capped = _cap_sample(frame, max_points=200)
    assert len(capped) == 200
    share = (capped["highway"] == "residential").mean()
    assert share >= 0.5


def test_fetch_traffic_road_sample_uses_fixture(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "traffic_fixture.json"
    pd.DataFrame({
        "name": ["a", "b"],
        "lon": [21.0, 21.01],
        "lat": [52.23, 52.24],
        "base_weight": [4.0, 2.0],
        "highway": ["primary", "residential"],
    }).to_json(fixture, orient="records")

    def _fail(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "happy_housing.gis.traffic._download_osm_drive_edges",
        _fail,
    )
    out = fetch_traffic_road_sample(
        tmp_path,
        bbox=(20.9, 52.1, 21.2, 52.4),
        fixture_path=fixture,
        force_refresh=True,
    )
    assert not out.empty
    assert {"lon", "lat", "base_weight"}.issubset(out.columns)
    assert (tmp_path / "traffic_roads.parquet").exists()


def test_attach_traffic_features_adds_major_and_nearest_highway() -> None:
    from shapely.geometry import Point

    from happy_housing.gis.traffic import attach_traffic_features

    listings = gpd.GeoDataFrame(
        {"listing_id": ["near-primary", "near-res"]},
        geometry=[Point(21.0, 52.23), Point(21.02, 52.24)],
        crs="EPSG:4326",
    )
    sample = pd.DataFrame({
        "lon": [21.0001, 21.02],
        "lat": [52.23, 52.24],
        "base_weight": [7.0, 1.6],
        "name": ["Aleja", "Boczna"],
        "highway": ["primary", "residential"],
    })
    out = attach_traffic_features(
        listings,
        sample,
        projected_crs="EPSG:2180",
        neighbourhood_radius_m=500.0,
    )
    assert out.loc[0, "nearest_highway"] == "primary"
    assert out.loc[1, "nearest_highway"] == "residential"
    assert float(out.loc[0, "dist_major_road_m"]) < float(
        out.loc[1, "dist_major_road_m"]
    )
    assert float(out.loc[0, "major_traffic_weight_nearby"]) >= float(
        out.loc[1, "major_traffic_weight_nearby"]
    )
    assert float(out.loc[0, "traffic_weight_nearby"]) > 0.0
