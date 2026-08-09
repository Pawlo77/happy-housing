"""Tests for recommendation scoring and memory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.app.preferences import UserPreferences
from happy_housing.app.recommender import (
    clear_recommendation_memory,
    impressions_path,
    load_impressions,
    metrics_path,
    record_impressions,
    retrain_recommender,
    score_offers,
)


def test_score_offers_ranks_underpriced_first() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a", "b"],
        "delta_price_per_m2": [500.0, -800.0],
        "commute_min": [30.0, 10.0],
        "dist_transit_stop_m": [800.0, 100.0],
    })
    ranked = score_offers(frame, demotion_alpha=1.0, impressions={})
    assert list(ranked["listing_id"]) == ["b", "a"]
    assert float(ranked.iloc[0]["confidence"]) > float(ranked.iloc[1]["confidence"])


def test_impressions_demote_seen_offers() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a", "b"],
        "delta_price_per_m2": [-500.0, -500.0],
        "commute_min": [10.0, 10.0],
        "dist_transit_stop_m": [100.0, 100.0],
    })
    impressions = {"a": {"count": 5, "last_seen_at": "x", "opened": False}}
    ranked = score_offers(frame, demotion_alpha=1.0, impressions=impressions)
    assert list(ranked["listing_id"]) == ["b", "a"]


def test_retrain_and_clear_memory(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "listing_id": ["a"],
        "delta_price_per_m2": [-200.0],
        "commute_min": [12.0],
        "dist_transit_stop_m": [200.0],
    })
    metrics = retrain_recommender(
        frame,
        processed_dir=tmp_path,
        top_n=5,
        demotion_alpha=1.0,
    )
    assert metrics["n_candidates"] == 1.0
    assert metrics_path(tmp_path).exists()
    record_impressions(impressions_path(tmp_path), ["a"])
    assert load_impressions(impressions_path(tmp_path))
    clear_recommendation_memory(tmp_path)
    assert load_impressions(impressions_path(tmp_path)) == {}
    assert not metrics_path(tmp_path).exists()


def test_score_offers_penalizes_overpriced() -> None:
    frame = pd.DataFrame({
        "listing_id": ["cheap", "rich"],
        "source": ["otodom", "otodom"],
        "delta_price_per_m2": [-800.0, 2000.0],
        "commute_min": [10.0, 10.0],
        "dist_transit_stop_m": [100.0, 100.0],
    })
    ranked = score_offers(
        frame,
        demotion_alpha=0.0,
        impressions={},
        w_overprice=0.5,
    )
    assert list(ranked["listing_id"]) == ["cheap", "rich"]
    assert float(ranked.iloc[0]["rec_overprice"]) < float(
        ranked.iloc[1]["rec_overprice"]
    )


def test_score_offers_default_pool_skips_rcn() -> None:
    frame = pd.DataFrame({
        "listing_id": ["oto", "deed"],
        "source": ["otodom", "rcn"],
        "delta_price_per_m2": [-100.0, -900.0],
        "commute_min": [10.0, 5.0],
        "dist_transit_stop_m": [100.0, 50.0],
    })
    ranked = score_offers(
        frame, demotion_alpha=0.0, impressions={}, otodom_pool_default=True
    )
    assert list(ranked["listing_id"]) == ["oto"]


def test_score_offers_respects_enabled_preferences() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a", "b"],
        "district": ["wola", "mokotow"],
        "source": ["otodom", "otodom"],
        "delta_price_per_m2": [-900.0, -900.0],
        "commute_min": [10.0, 10.0],
        "dist_transit_stop_m": [100.0, 100.0],
        "in_oou": [0, 0],
    })
    prefs = UserPreferences(enabled=True, districts=["mokotow"])
    ranked = score_offers(
        frame,
        demotion_alpha=1.0,
        impressions={},
        prefs=prefs,
    )
    assert list(ranked["listing_id"]) == ["b"]


def test_retrain_metrics_include_prefs_enabled(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "listing_id": ["a"],
        "delta_price_per_m2": [-200.0],
        "commute_min": [12.0],
        "dist_transit_stop_m": [200.0],
    })
    prefs = UserPreferences(enabled=True)
    metrics = retrain_recommender(
        frame,
        processed_dir=tmp_path,
        top_n=5,
        demotion_alpha=1.0,
        prefs=prefs,
    )
    assert metrics["prefs_enabled"] == 1.0
