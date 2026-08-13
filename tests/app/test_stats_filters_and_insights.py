"""Tests for Stats-page filters and business analytics helpers."""

import pandas as pd

from happy_housing.app.stats_filters import StatsFilterState, apply_stats_filters
from happy_housing.app.stats_insights import (
    district_scoreboard,
    monthly_trends,
    normalize_district,
    top_value_offers,
    transit_price_curve,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": [f"L{i}" for i in range(8)],
        "district": [
            "Wola",
            "wola",
            "Mokotów",
            "mokotów",
            "Wola",
            "Ursynów",
            "ursynów",
            "Wola",
        ],
        "source": [
            "otodom",
            "rcn",
            "otodom",
            "rcn",
            "otodom",
            "otodom",
            "rcn",
            "fixture",
        ],
        "price_per_m2": [
            18000.0,
            14000.0,
            22000.0,
            16000.0,
            15000.0,
            17000.0,
            15500.0,
            19000.0,
        ],
        "model_price_per_m2": [
            18800.0,
            14500.0,
            20000.0,
            16500.0,
            16000.0,
            16800.0,
            15800.0,
            17500.0,
        ],
        "delta_price_per_m2": [
            -800.0,
            -500.0,
            2000.0,
            -500.0,
            -1000.0,
            200.0,
            -300.0,
            1500.0,
        ],
        "in_oou": [0, 0, 1, 1, 0, 0, 0, 0],
        "area_m2": [35.0, 55.0, 90.0, 120.0, 42.0, 60.0, 70.0, 48.0],
        "rooms": [1, 2, 3, 4, 2, 2, 3, 2],
        "listed_month": [
            "2026-01",
            "2026-01",
            "2026-02",
            "2026-02",
            "2026-03",
            "2026-03",
            "2026-03",
            "2026-01",
        ],
        "dist_transit_stop_m": [80.0, 400.0, 120.0, 900.0, 250.0, 150.0, 600.0, 100.0],
        "commute_min": [8.0, 12.0, 25.0, 30.0, 10.0, 18.0, 22.0, 9.0],
        "year_built": [2018, 1995, 1970, 2005, 2020, 2010, 1985, 2016],
    })


def test_normalize_district_casefolds() -> None:
    assert normalize_district("Mokotów") == "mokotow"
    assert normalize_district("  WOLA ") == "wola"
    assert normalize_district("Śródmieście") == normalize_district("srodmiescie")


def test_apply_stats_filters_valuation_and_district() -> None:
    filtered = apply_stats_filters(
        _frame(),
        StatsFilterState(
            districts=["wola"],
            valuation="underpriced",
            sources=[],
            months=[],
        ),
    )
    assert not filtered.empty
    assert set(filtered["district"].map(normalize_district)) == {"wola"}
    assert (filtered["delta_price_per_m2"] < 0).all()


def test_apply_stats_filters_overpriced_excludes_zero_delta() -> None:
    frame = _frame()
    frame.loc[0, "delta_price_per_m2"] = 0.0
    filtered = apply_stats_filters(frame, StatsFilterState(valuation="overpriced"))
    assert not filtered.empty
    assert (filtered["delta_price_per_m2"] > 0).all()


def test_apply_stats_filters_price_area_bounds() -> None:
    filtered = apply_stats_filters(
        _frame(),
        StatsFilterState(
            price_min=16000.0,
            price_max=20000.0,
            area_min=40.0,
            area_max=80.0,
        ),
    )
    assert filtered["price_per_m2"].between(16000.0, 20000.0).all()
    assert filtered["area_m2"].between(40.0, 80.0).all()


def test_monthly_trends_orders_months() -> None:
    out = monthly_trends(_frame())
    assert list(out["listed_month"]) == ["2026-01", "2026-02", "2026-03"]
    assert {"n", "median_price_per_m2", "median_delta", "median_model"}.issubset(
        out.columns
    )


def test_district_scoreboard_merges_case_variants() -> None:
    out = district_scoreboard(_frame(), min_offers=2)
    assert "wola" in set(out["district"])
    wola = out.loc[out["district"] == "wola"].iloc[0]
    assert int(wola["n"]) >= 3
    assert "bargain_share" in out.columns
    assert "median_model" in out.columns


def test_top_value_offers_ranks_underpriced() -> None:
    out = top_value_offers(_frame(), n=3, side="underpriced")
    assert len(out) == 3
    assert out["delta_price_per_m2"].is_monotonic_increasing
    assert (out["delta_price_per_m2"] < 0).all()


def test_transit_price_curve_bins() -> None:
    out = transit_price_curve(_frame())
    assert not out.empty
    assert "transit_band" in out.columns
    assert "median_price_per_m2" in out.columns
