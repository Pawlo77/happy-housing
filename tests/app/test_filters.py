"""Tests for Explore-page filter helpers."""

from datetime import date

import pandas as pd

from happy_housing.app.filters import (
    FilterState,
    apply_filters,
    apply_starred_only,
    offer_age_months,
)


def test_offer_age_months_from_listed_at() -> None:
    frame = pd.DataFrame({
        "listed_at": ["2026-05-08", "2025-08-08", "bad", None],
        "listed_month": ["2026-05", "2025-08", "unknown", "2020-01"],
    })
    ages = offer_age_months(frame, as_of=date(2026, 8, 8))
    assert float(ages.iloc[0]) == 3.0
    assert float(ages.iloc[1]) == 12.0
    assert pd.isna(ages.iloc[2])
    assert float(ages.iloc[3]) == 79.0


def test_offer_age_months_utc_listed_at_with_month_fallback() -> None:
    """UTC ``listed_at`` + naive YYYY-MM ``listed_month`` must not break ``.dt``."""
    frame = pd.DataFrame({
        "listed_at": ["2026-07-19T19:36:32Z", None, "bad"],
        "listed_month": ["2026-07", "2020-01", "unknown"],
    })
    ages = offer_age_months(frame, as_of=date(2026, 8, 8))
    assert float(ages.iloc[0]) == 0.0
    assert float(ages.iloc[1]) == 79.0
    assert pd.isna(ages.iloc[2])


def test_apply_filters_max_price_pln() -> None:
    """Cap total asking price (PLN), independent of zł/m²."""
    frame = pd.DataFrame({
        "listing_id": ["cheap", "mid", "dear"],
        "district": ["wola", "wola", "wola"],
        "source": ["otodom", "otodom", "otodom"],
        "in_oou": [0, 0, 0],
        "price_pln": [600_000.0, 900_000.0, 1_400_000.0],
        "price_per_m2": [15000.0, 12000.0, 10000.0],
        "listed_at": ["2026-07-01", "2026-07-01", "2026-07-01"],
    })
    out = apply_filters(frame, FilterState(max_price_pln=1_000_000.0))
    assert list(out["listing_id"]) == ["cheap", "mid"]


def test_apply_filters_district_source_oou() -> None:
    """Sidebar filters narrow the scored frame."""
    frame = pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "district": ["wola", "mokotow", "wola"],
        "source": ["otodom", "rcn", "otodom"],
        "in_oou": [0, 1, 1],
        "price_per_m2": [10000.0, 20000.0, 15000.0],
        "listed_at": ["2026-01-01", "2026-02-01", "2026-01-15"],
    })
    state = FilterState(
        district="wola",
        source="otodom",
        oou="inside",
        max_price_per_m2=16000.0,
        max_age_months=None,
    )
    out = apply_filters(frame, state, as_of=date(2026, 8, 8))
    assert list(out["listing_id"]) == ["c"]


def test_apply_filters_max_total_noise() -> None:
    frame = pd.DataFrame({
        "listing_id": ["quiet", "loud"],
        "district": ["wola", "wola"],
        "source": ["otodom", "otodom"],
        "in_oou": [0, 0],
        "in_high_noise": [0, 1],
        "noise_total_lden_db": [48.0, 68.0],
        "price_per_m2": [10000.0, 10000.0],
        "listed_at": ["2026-07-01", "2026-07-01"],
    })
    out = apply_filters(
        frame,
        FilterState(max_noise_total_lden_db=55.0),
        as_of=date(2026, 8, 8),
    )
    assert list(out["listing_id"]) == ["quiet"]
    out2 = apply_filters(frame, FilterState(high_noise="outside"))
    assert list(out2["listing_id"]) == ["quiet"]


def test_apply_starred_only_keeps_starred_ids() -> None:
    frame = pd.DataFrame({"listing_id": ["a", "b", "c"], "x": [1, 2, 3]})
    out = apply_starred_only(frame, frozenset({"b", "c"}))
    assert list(out["listing_id"]) == ["b", "c"]


def test_apply_starred_only_empty_stars() -> None:
    frame = pd.DataFrame({"listing_id": ["a"], "x": [1]})
    assert apply_starred_only(frame, frozenset()).empty
