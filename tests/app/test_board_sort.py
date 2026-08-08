"""Tests for Explore Board recommendation enrichment and sorting."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.board_sort import (
    BOARD_SORT_RECOMMENDATION,
    coerce_board_sort,
    enrich_board_recommendations,
    sort_board,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "district": ["wola", "mokotow", "wola"],
        "price_pln": [900_000.0, 700_000.0, 800_000.0],
        "price_per_m2": [18_000.0, 14_000.0, 16_000.0],
        "delta_price_per_m2": [500.0, -800.0, -200.0],
        "area_m2": [50.0, 50.0, 50.0],
        "rooms": [2, 3, 2],
        "commute_min": [30.0, 10.0, 20.0],
        "dist_transit_stop_m": [800.0, 100.0, 300.0],
        "listed_at": ["2026-01-01", "2026-08-01", "2026-05-01"],
    })


def test_coerce_board_sort_defaults() -> None:
    assert coerce_board_sort(None) == BOARD_SORT_RECOMMENDATION
    assert coerce_board_sort("nope") == BOARD_SORT_RECOMMENDATION
    assert coerce_board_sort("price_pln_asc") == "price_pln_asc"


def test_enrich_and_default_sort_puts_unseen_bargain_first() -> None:
    frame = _sample_frame()
    impressions = {
        "b": {"count": 4, "opened": True},
        "c": {"count": 2, "opened": False},
    }
    enriched = enrich_board_recommendations(
        frame,
        impressions=impressions,
        demotion_alpha=1.0,
    )
    assert set(enriched["seen"]) >= {"new", "opened", "seen x2"}
    ordered = sort_board(enriched, BOARD_SORT_RECOMMENDATION)
    # a is new; c seen but not opened; b opened - new first, then c, then b.
    assert list(ordered["listing_id"]) == ["a", "c", "b"]


def test_sort_by_price_and_district() -> None:
    frame = enrich_board_recommendations(
        _sample_frame(),
        impressions={},
        demotion_alpha=1.0,
    )
    by_price = sort_board(frame, "price_pln_asc")
    assert list(by_price["listing_id"]) == ["b", "c", "a"]
    by_district = sort_board(frame, "district_asc")
    assert list(by_district["district"]) == sorted(by_district["district"].astype(str))


def test_sort_by_posted_newest() -> None:
    frame = enrich_board_recommendations(
        _sample_frame(),
        impressions={},
        demotion_alpha=1.0,
    )
    newest = sort_board(frame, "listed_at_desc")
    assert list(newest["listing_id"]) == ["b", "c", "a"]
