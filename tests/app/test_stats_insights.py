"""Tests for unconventional Stats-page market insights."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.stats_insights import (
    ask_vs_ink,
    bargain_density,
    chopin_surcharge,
    commute_cost_curve,
    room_ladder,
    size_sweet_spot,
    year_paradox,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "district": ["wola", "wola", "mokotow", "mokotow", "wola"],
        "source": ["otodom", "rcn", "otodom", "rcn", "otodom"],
        "price_per_m2": [18000.0, 14000.0, 22000.0, 16000.0, 15000.0],
        "delta_price_per_m2": [-800.0, 200.0, 1500.0, -100.0, -400.0],
        "in_oou": [0, 0, 1, 1, 0],
        "commute_min": [8.0, 12.0, 25.0, 30.0, 10.0],
        "area_m2": [35.0, 55.0, 90.0, 120.0, 42.0],
        "rooms": [1, 2, 3, 4, 2],
        "year_built": [2018, 1995, 1970, 2005, 2020],
        "dist_transit_stop_m": [120.0, 400.0, 80.0, 900.0, 250.0],
        "dist_fire_station_m": [500.0, 800.0, 300.0, 1200.0, 600.0],
    })


def test_chopin_surcharge_compares_oou_medians() -> None:
    out = chopin_surcharge(_frame())
    assert out["n_inside"] == 2
    assert out["n_outside"] == 3
    assert out["median_inside"] > 0
    assert "surcharge_pct" in out


def test_bargain_density_ranks_districts() -> None:
    out = bargain_density(_frame(), min_offers=1)
    assert not out.empty
    assert {"district", "n", "bargain_share", "median_delta"}.issubset(out.columns)
    wola = out.loc[out["district"] == "wola"].iloc[0]
    assert 0.0 <= float(wola["bargain_share"]) <= 1.0


def test_commute_cost_curve_bins_minutes() -> None:
    out = commute_cost_curve(_frame())
    assert not out.empty
    assert "commute_bin" in out.columns
    assert "median_price_per_m2" in out.columns


def test_ask_vs_ink_compares_sources() -> None:
    out = ask_vs_ink(_frame(), min_offers=1)
    assert not out.empty
    assert "ask_median" in out.columns
    assert "ink_median" in out.columns


def test_size_room_year_helpers_return_frames() -> None:
    assert not size_sweet_spot(_frame()).empty
    assert not room_ladder(_frame()).empty
    assert not year_paradox(_frame()).empty


def test_market_temperature_and_volcano() -> None:
    from happy_housing.app.stats_insights import delta_volcano, market_temperature

    big = pd.concat([_frame(), _frame(), _frame()], ignore_index=True)
    temp = market_temperature(big)
    assert not temp.empty
    assert "polarization" in temp.columns
    volcano = delta_volcano(_frame())
    assert set(volcano["side"].unique()) <= {"overpriced", "underpriced"}
