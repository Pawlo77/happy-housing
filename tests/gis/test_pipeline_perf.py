"""Regression tests for pipeline enrich / HPO / OSRM performance helpers."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from happy_housing.gis import nearest_distance_meters
from happy_housing.gis.bike_paths import attach_bike_path_features
from happy_housing.gis.routing import osrm_table_durations_seconds
from happy_housing.gis.traffic import attach_traffic_features


def test_nearest_distance_reuses_projected_left_without_full_copy() -> None:
    """Projected left + copy=False mutates the same frame (no reproject thrash)."""
    listings = gpd.GeoDataFrame(
        {"listing_id": ["a", "b"]},
        geometry=[Point(21.0, 52.23), Point(21.01, 52.23)],
        crs="EPSG:4326",
    )
    left_proj = listings.to_crs("EPSG:2180")
    points = gpd.GeoDataFrame(
        geometry=[Point(21.0, 52.2305)],
        crs="EPSG:4326",
    )
    out = nearest_distance_meters(
        listings,
        points,
        "EPSG:2180",
        "dist_x_m",
        left_projected=left_proj,
        copy=False,
    )
    assert out is listings
    assert "dist_x_m" in listings.columns
    assert float(listings.loc[0, "dist_x_m"]) < float(listings.loc[1, "dist_x_m"])


def test_bike_coverage_accepts_precomputed_sample() -> None:
    """Cached densified sample skips re-densify and still yields coverage."""
    listings = gpd.GeoDataFrame(
        {"listing_id": ["a"]},
        geometry=[Point(21.0, 52.23)],
        crs="EPSG:4326",
    )
    paths = gpd.GeoDataFrame(
        {"highway": ["cycleway"]},
        geometry=[LineString([(20.999, 52.2305), (21.001, 52.2305)])],
        crs="EPSG:4326",
    )
    sample = pd.DataFrame({
        "lon": [21.0, 21.0005],
        "lat": [52.2305, 52.2305],
        "step_m": [25.0, 25.0],
    })
    with patch(
        "happy_housing.gis.bike_paths.densify_bike_paths",
        side_effect=AssertionError("must not densify"),
    ):
        out = attach_bike_path_features(
            listings,
            paths,
            projected_crs="EPSG:2180",
            coverage_radius_m=300.0,
            sample_points=sample,
        )
    assert float(out.loc[0, "bike_path_coverage_m"]) >= 25.0


def test_traffic_features_balltree_matches_near_and_far() -> None:
    listings = gpd.GeoDataFrame(
        {"listing_id": ["near", "far"]},
        geometry=[Point(21.0, 52.23), Point(21.05, 52.23)],
        crs="EPSG:4326",
    )
    sample = pd.DataFrame({
        "lon": [21.0001, 21.0002],
        "lat": [52.2301, 52.2302],
        "base_weight": [5.0, 5.0],
    })
    out = attach_traffic_features(
        listings, sample, projected_crs="EPSG:2180", neighbourhood_radius_m=300.0
    )
    assert float(out.loc[0, "dist_traffic_road_m"]) < float(
        out.loc[1, "dist_traffic_road_m"]
    )
    assert float(out.loc[0, "traffic_weight_nearby"]) > float(
        out.loc[1, "traffic_weight_nearby"]
    )


def test_osrm_table_chunks_large_origin_lists() -> None:
    origins = [(21.0 + i * 0.001, 52.23) for i in range(5)]
    calls: list[int] = []

    def fake_get(url: str, timeout: float = 0):
        del timeout
        # Count sources in the query string.
        sources = url.split("sources=")[1].split("&")[0]
        n = len(sources.split(";"))
        calls.append(n)
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"durations": [[float(i * 60)] for i in range(n)]}
        return mock

    with (
        patch("happy_housing.gis.routing.requests.get", side_effect=fake_get),
        patch("happy_housing.gis.routing.OSRM_CHUNK_SIZE", 2),
    ):
        out = osrm_table_durations_seconds(
            origins, (20.98, 52.23), base_url="http://osrm.test"
        )
    assert out is not None
    assert len(out) == 5
    assert calls == [2, 2, 1]
    assert np.isfinite(out).all()
