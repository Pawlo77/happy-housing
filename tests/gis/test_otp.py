"""Tests for optional OpenTripPlanner transit commute attach."""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from happy_housing.gis.routing import (
    attach_transit_commute_minutes,
    otp_plan_duration_seconds,
)


def test_otp_plan_duration_parses_itinerary() -> None:
    """OTP plan JSON duration is already seconds."""
    payload = {
        "plan": {
            "itineraries": [
                {"duration": 1200},
                {"duration": 900},
            ]
        }
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    with patch("happy_housing.gis.routing.requests.get", return_value=mock_resp):
        seconds = otp_plan_duration_seconds(
            (21.0, 52.23),
            (20.983, 52.23),
            base_url="http://127.0.0.1:8080",
        )
    assert seconds == 900.0


def test_otp_unavailable_returns_none() -> None:
    """Down OTP yields None (caller keeps GTFS/distance proxies)."""
    with patch(
        "happy_housing.gis.routing.requests.get",
        side_effect=ConnectionError("down"),
    ):
        assert (
            otp_plan_duration_seconds(
                (21.0, 52.23),
                (20.983, 52.23),
                base_url="http://127.0.0.1:9",
            )
            is None
        )


def test_attach_transit_commute_minutes_fallback_nan() -> None:
    """When OTP is down, transit_commute_min is NaN (GTFS dist stays elsewhere)."""
    frame = pd.DataFrame({"lon": [21.0, 21.01], "lat": [52.23, 52.24]})
    with patch(
        "happy_housing.gis.routing.otp_plan_duration_seconds",
        return_value=None,
    ):
        out = attach_transit_commute_minutes(
            frame,
            dest_lon=20.983,
            dest_lat=52.23,
            otp_url="http://127.0.0.1:9",
        )
    assert "transit_commute_min" in out.columns
    assert out["transit_commute_min"].isna().all()


def test_attach_transit_commute_minutes_uses_otp() -> None:
    """Successful OTP fills transit minutes from shortest itinerary."""

    def _fake(
        origin: tuple[float, float],
        destination: tuple[float, float],
        **kwargs: Any,
    ) -> float:
        del origin, destination, kwargs
        return 600.0

    frame = pd.DataFrame({"lon": [21.0], "lat": [52.23]})
    with patch(
        "happy_housing.gis.routing.otp_plan_duration_seconds",
        side_effect=_fake,
    ):
        out = attach_transit_commute_minutes(
            frame,
            dest_lon=20.983,
            dest_lat=52.23,
        )
    assert float(out["transit_commute_min"].iloc[0]) == 10.0


def test_attach_transit_skips_rows_already_filled() -> None:
    frame = pd.DataFrame({
        "lon": [21.0, 21.01],
        "lat": [52.23, 52.24],
        "transit_commute_min": [12.5, float("nan")],
    })
    calls: list[tuple[float, float]] = []

    def _fake(
        origin: tuple[float, float],
        destination: tuple[float, float],
        **kwargs: Any,
    ) -> float:
        del destination, kwargs
        calls.append(origin)
        return 480.0

    with patch(
        "happy_housing.gis.routing.otp_plan_duration_seconds",
        side_effect=_fake,
    ):
        out = attach_transit_commute_minutes(
            frame,
            dest_lon=20.983,
            dest_lat=52.23,
        )
    assert calls == [(21.01, 52.24)]
    assert float(out["transit_commute_min"].iloc[0]) == 12.5
    assert float(out["transit_commute_min"].iloc[1]) == 8.0
