"""Tests for commute minutes with OSRM unavailable (distance proxy)."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from happy_housing.gis.routing import (
    attach_commute_minutes,
    correct_active_mode_duration_s,
    duration_seconds_from_distance_m,
    resolve_osrm_base_url,
)


def test_resolve_osrm_base_url_by_profile() -> None:
    assert resolve_osrm_base_url("driving") == "http://127.0.0.1:5001"
    assert resolve_osrm_base_url("foot") == "http://127.0.0.1:5002"
    assert resolve_osrm_base_url("bike") == "http://127.0.0.1:5003"
    assert (
        resolve_osrm_base_url(
            "foot",
            osrm_url="http://car.test",
            osrm_foot_url="http://foot.test",
        )
        == "http://foot.test"
    )
    assert (
        resolve_osrm_base_url(
            "bicycle",
            osrm_bike_url="http://bike.test/",
        )
        == "http://bike.test"
    )


def test_attach_commute_minutes_distance_fallback() -> None:
    """Without OSRM, commute_min derives from dist_daszynskiego_m at 30 km/h."""
    frame = pd.DataFrame({
        "lon": [21.0],
        "lat": [52.23],
        "dist_daszynskiego_m": [3000.0],
    })
    out = attach_commute_minutes(
        frame,
        dest_lon=20.983,
        dest_lat=52.23,
        osrm_url="http://127.0.0.1:9",
    )
    # 3 km at 30 km/h → 6 minutes.
    assert float(out["commute_min"].iloc[0]) == 6.0


def test_attach_commute_minutes_skips_filled_rows() -> None:
    frame = pd.DataFrame({
        "lon": [21.0, 21.01],
        "lat": [52.23, 52.24],
        "dist_daszynskiego_m": [3000.0, 1500.0],
        "commute_min": [7.0, float("nan")],
    })
    with patch(
        "happy_housing.gis.routing.osrm_table_durations_seconds",
        return_value=np.array([600.0]),
    ) as table_mock:
        out = attach_commute_minutes(
            frame,
            dest_lon=20.983,
            dest_lat=52.23,
            osrm_url="http://osrm.test",
        )
    origins = table_mock.call_args[0][0]
    assert origins == [(21.01, 52.24)]
    assert float(out["commute_min"].iloc[0]) == 7.0
    assert float(out["commute_min"].iloc[1]) == 10.0


def test_duration_seconds_from_distance_m_walk_pace() -> None:
    # 2.182 km at 5 km/h → 26.184 min
    assert abs(duration_seconds_from_distance_m(2182.0, 5.0) - 1571.04) < 0.1


def test_correct_active_mode_duration_s_rescales_car_like_foot() -> None:
    corrected = correct_active_mode_duration_s(
        2182.0,
        240.0,
        profile="foot",
        proxy_speed_kmh=5.0,
    )
    assert abs(corrected - 1571.04) < 1.0


def test_correct_active_mode_duration_s_keeps_plausible_foot() -> None:
    walk_s = duration_seconds_from_distance_m(1000.0, 5.0)
    kept = correct_active_mode_duration_s(
        1000.0,
        walk_s,
        profile="foot",
        proxy_speed_kmh=5.0,
    )
    assert abs(kept - walk_s) < 1e-6


def test_attach_commute_minutes_foot_rescales_car_table() -> None:
    """Foot profile against car OSRM table → walk pace via distance column."""
    frame = pd.DataFrame({
        "lon": [21.0],
        "lat": [52.23],
        "dist_daszynskiego_m": [2182.0],
    })
    with patch(
        "happy_housing.gis.routing.osrm_table_durations_seconds",
        return_value=np.array([240.0]),
    ):
        out = attach_commute_minutes(
            frame,
            dest_lon=20.983,
            dest_lat=52.23,
            osrm_url="http://osrm.test",
            profile="foot",
            out_col="walk_commute_min",
            proxy_speed_kmh=5.0,
        )
    assert abs(float(out["walk_commute_min"].iloc[0]) - 26.184) < 0.05
