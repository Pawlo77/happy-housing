"""Tests for Explore Offer display helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.offer_display import (
    delta_band,
    delta_hex,
    format_meters,
    livability_rows,
    round_meters,
)


def test_round_and_format_meters() -> None:
    assert round_meters(1234.6) == 1235
    assert format_meters(1234.6) == "1,235 m"
    assert format_meters(None) == "—"
    assert format_meters(float("nan")) == "—"


def test_delta_band_colours() -> None:
    assert delta_band(-800.0) == "green"
    assert delta_band(100.0) == "yellow"
    assert delta_band(900.0) == "red"
    assert delta_hex(-800.0).startswith("#")


def test_livability_rows_round_distances() -> None:
    row = pd.Series({
        "in_oou": 1,
        "commute_min": 12.4,
        "dist_transit_stop_m": 371.9,
        "dist_hospital_m": 8248.5,
        "dist_fire_station_m": 1881.2,
        "dist_police_m": 2177.9,
        "dist_air_station_m": 1608.4,
        "dist_gunb_m": 9516.5,
        "dist_daszynskiego_m": 9206.0,
    })
    frame = livability_rows(row)
    transit = frame.loc[frame["signal"] == "Nearest transit stop", "value"].iloc[0]
    assert transit == "372 m"
    assert "Yes" in frame["value"].astype(str).tolist()
