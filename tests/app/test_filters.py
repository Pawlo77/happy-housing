"""Tests for Explore-page filter helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.filters import (
    FilterState,
    apply_filters,
    apply_starred_only,
)


def test_apply_filters_district_source_oou() -> None:
    """Sidebar filters narrow the scored frame."""
    frame = pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "district": ["wola", "mokotow", "wola"],
        "source": ["otodom", "rcn", "otodom"],
        "in_oou": [0, 1, 1],
        "price_per_m2": [10000.0, 20000.0, 15000.0],
        "listed_month": ["2026-01", "2026-02", "2026-01"],
    })
    state = FilterState(
        district="wola",
        source="otodom",
        oou="inside",
        max_price_per_m2=16000.0,
        month="2026-01",
    )
    out = apply_filters(frame, state)
    assert list(out["listing_id"]) == ["c"]


def test_apply_starred_only_keeps_starred_ids() -> None:
    frame = pd.DataFrame({"listing_id": ["a", "b", "c"], "x": [1, 2, 3]})
    out = apply_starred_only(frame, frozenset({"b", "c"}))
    assert list(out["listing_id"]) == ["b", "c"]


def test_apply_starred_only_empty_stars() -> None:
    frame = pd.DataFrame({"listing_id": ["a"], "x": [1]})
    assert apply_starred_only(frame, frozenset()).empty
