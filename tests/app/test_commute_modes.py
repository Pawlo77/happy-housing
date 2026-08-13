"""Tests for multi-mode commute helpers (car / transit / bike / walk)."""

import pandas as pd

from happy_housing.app.commute_modes import (
    COMMUTE_DISPLAY_DEFAULT,
    COMMUTE_MODE_COLUMN,
    coerce_commute_display_mode,
    commute_column_for_mode,
    commute_series,
    commute_tooltip_label,
)
from happy_housing.gis.routing import attach_commute_minutes


def test_default_display_mode_is_public_transport() -> None:
    assert COMMUTE_DISPLAY_DEFAULT == "transit"
    assert coerce_commute_display_mode(None) == "transit"
    assert coerce_commute_display_mode("") == "transit"
    assert coerce_commute_display_mode("bogus") == "transit"
    assert coerce_commute_display_mode("CAR") == "car"
    assert coerce_commute_display_mode("walk") == "walk"


def test_commute_column_for_mode_maps_modes() -> None:
    assert commute_column_for_mode("car") == "commute_min"
    assert commute_column_for_mode("transit") == "transit_commute_min"
    assert commute_column_for_mode("bike") == "bike_commute_min"
    assert commute_column_for_mode("walk") == "walk_commute_min"
    assert set(COMMUTE_MODE_COLUMN) == {"car", "transit", "bike", "walk"}


def test_commute_series_picks_mode_column() -> None:
    frame = pd.DataFrame({
        "commute_min": [10.0, 20.0],
        "transit_commute_min": [25.0, 40.0],
        "bike_commute_min": [15.0, 30.0],
        "walk_commute_min": [45.0, 60.0],
    })
    assert float(commute_series(frame, "transit").iloc[0]) == 25.0
    assert float(commute_series(frame, "bike").iloc[1]) == 30.0
    assert commute_tooltip_label("transit") == "Transit"


def test_attach_bike_and_walk_use_speed_proxy_when_osrm_down() -> None:
    frame = pd.DataFrame({
        "lon": [21.0],
        "lat": [52.23],
        "dist_daszynskiego_m": [3000.0],
    })
    bike = attach_commute_minutes(
        frame,
        dest_lon=20.983,
        dest_lat=52.23,
        osrm_url="http://127.0.0.1:9",
        profile="bike",
        out_col="bike_commute_min",
        proxy_speed_kmh=15.0,
    )
    walk = attach_commute_minutes(
        frame,
        dest_lon=20.983,
        dest_lat=52.23,
        osrm_url="http://127.0.0.1:9",
        profile="foot",
        out_col="walk_commute_min",
        proxy_speed_kmh=5.0,
    )
    # 3 km @ 15 km/h → 12 min; @ 5 km/h → 36 min.
    assert float(bike["bike_commute_min"].iloc[0]) == 12.0
    assert float(walk["walk_commute_min"].iloc[0]) == 36.0


def test_osrm_table_uses_profile_in_url() -> None:
    from unittest.mock import MagicMock, patch

    from happy_housing.gis.routing import osrm_table_durations_seconds

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"durations": [[120.0]]}
    with patch("happy_housing.gis.routing.requests.get", return_value=response) as get:
        out = osrm_table_durations_seconds(
            [(21.0, 52.23)],
            (20.98, 52.23),
            base_url="http://osrm.test",
            profile="bike",
        )
    assert out is not None
    assert float(out[0]) == 120.0
    url = str(get.call_args.args[0])
    assert "/table/v1/bike/" in url
