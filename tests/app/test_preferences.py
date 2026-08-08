"""Tests for Explore user preferences persistence and hard filters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.app.preferences import (
    UserPreferences,
    apply_preferences,
    load_preferences,
    preferences_path,
    save_preferences,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a", "b", "c", "d"],
        "district": ["wola", "mokotow", "wola", "bemowo"],
        "source": ["otodom", "rcn", "otodom", "otodom"],
        "in_oou": [0, 1, 0, 0],
        "price_pln": [500_000.0, 900_000.0, 700_000.0, 400_000.0],
        "price_per_m2": [12_000.0, 20_000.0, 15_000.0, 10_000.0],
        "area_m2": [40.0, 45.0, 50.0, 38.0],
        "rooms": [2, 3, 3, 2],
        "year_built": [2010, 1990, 2018, 2005],
        "commute_min": [15.0, 40.0, 20.0, 25.0],
        "dist_transit_stop_m": [200.0, 900.0, 300.0, 150.0],
        "dist_hospital_m": [1000.0, 5000.0, 2000.0, 800.0],
        "dist_fire_station_m": [500.0, 2000.0, 700.0, 400.0],
        "dist_police_m": [600.0, 2500.0, 800.0, 450.0],
        "dist_air_station_m": [2000.0, 8000.0, 3000.0, 1500.0],
        "dist_gunb_m": [3000.0, 9000.0, 4000.0, 2000.0],
        "delta_price_per_m2": [-200.0, 500.0, -50.0, 100.0],
    })


def test_preferences_roundtrip(tmp_path: Path) -> None:
    path = preferences_path(tmp_path)
    prefs = UserPreferences(
        enabled=True,
        districts=["wola", "mokotow"],
        max_commute_min=30.0,
        require_underpriced=True,
        nl_text="near metro under 30 min",
    )
    save_preferences(path, prefs)
    loaded = load_preferences(path)
    assert loaded.enabled is True
    assert loaded.districts == ["wola", "mokotow"]
    assert loaded.max_commute_min == 30.0
    assert loaded.require_underpriced is True
    assert loaded.nl_text == "near metro under 30 min"


def test_apply_preferences_disabled_returns_all() -> None:
    frame = _frame()
    out = apply_preferences(frame, UserPreferences(enabled=False, districts=["wola"]))
    assert len(out) == len(frame)


def test_apply_preferences_district_multi() -> None:
    out = apply_preferences(
        _frame(),
        UserPreferences(enabled=True, districts=["wola", "bemowo"]),
    )
    assert set(out["listing_id"]) == {"a", "c", "d"}


def test_apply_preferences_max_transit_and_underpriced() -> None:
    out = apply_preferences(
        _frame(),
        UserPreferences(
            enabled=True,
            max_dist_transit_m=350.0,
            require_underpriced=True,
        ),
    )
    assert set(out["listing_id"]) == {"a", "c"}


def test_apply_preferences_outside_only_oou() -> None:
    out = apply_preferences(
        _frame(),
        UserPreferences(enabled=True, oou="outside_only"),
    )
    assert set(out["listing_id"]) == {"a", "c", "d"}
    assert (out["in_oou"] == 0).all()


def test_apply_preferences_max_total_noise() -> None:
    frame = _frame()
    frame["noise_total_lden_db"] = [50.0, 70.0, 52.0, 80.0]
    out = apply_preferences(
        frame,
        UserPreferences(enabled=True, max_noise_total_lden_db=55.0),
    )
    assert set(out["listing_id"]) == {"a", "c"}
