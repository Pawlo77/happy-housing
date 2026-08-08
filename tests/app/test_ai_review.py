"""Tests for offer AI review cache and prompt composition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.app.ai_review import (
    OfferAiReview,
    build_review_prompt,
    load_review,
    review_context_hash,
    save_review,
)
from happy_housing.app.app_settings import AppSettings


def _rich_row() -> pd.Series:
    return pd.Series({
        "listing_id": "OTO-1",
        "title": "Flat",
        "district": "Wola",
        "source": "otodom",
        "price_pln": 900000,
        "area_m2": 50,
        "rooms": 2,
        "year_built": 2010,
        "floor_no": 3,
        "building_floors_num": 8,
        "building_type": "block",
        "heating": "urban",
        "has_balcony": 1,
        "has_garage": 0,
        "has_lift": 1,
        "price_per_m2": 18000,
        "model_price_per_m2": 17000,
        "delta_price_per_m2": 1000,
        "in_oou": 0,
        "commute_min": 15,
        "dist_transit_stop_m": 200,
        "dist_metro_m": 450,
        "dist_hospital_m": 1000,
        "dist_fire_station_m": 800,
        "dist_police_m": 900,
        "dist_air_station_m": 1200,
        "dist_gunb_m": 3000,
        "dist_daszynskiego_m": 4200,
        "nbhd_rcn_median_ppm": 16500,
        "nbhd_rcn_mean_ppm": 16800,
        "nbhd_rcn_n": 12,
        "nbhd_rcn_nearest_m": 80,
        "nbhd_rcn_idw_ppm": 16600,
        "description": "Bright flat near metro.",
        "characteristics_json": '[{"key":"floor_no","value":"3"}]',
        "url": "https://example.test/offer",
        "image_url": "https://example.test/img.jpg",
    })


def test_save_and_load_review_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "reviews.duckdb"
    review = OfferAiReview(
        listing_id="OTO-1",
        model="gemma-4-31b-it-mlx",
        review_text="Lean watch — fair-ish ask.",
        created_at="2026-08-08T12:00:00+00:00",
        context_hash="abc123",
    )
    save_review(path, review)
    loaded = load_review(
        path, "OTO-1", model="gemma-4-31b-it-mlx", context_hash="abc123"
    )
    assert loaded is not None
    assert loaded.review_text == review.review_text
    assert load_review(path, "OTO-1", model="other", context_hash="abc123") is None
    assert (
        load_review(path, "OTO-1", model="gemma-4-31b-it-mlx", context_hash="other")
        is None
    )


def test_build_review_prompt_includes_description() -> None:
    prompt = build_review_prompt(_rich_row())
    assert "Bright flat near metro." in prompt
    assert "OTO-1" in prompt


def test_build_review_prompt_includes_prefs_proximity_and_nbhd() -> None:
    settings = AppSettings(
        buyer_pref_notes="Quiet street; Wola or Żoliborz",
        buyer_max_price_pln=950_000.0,
        buyer_preferred_districts="Wola, Żoliborz",
        buyer_max_commute_min=25.0,
        buyer_avoid_oou=True,
        buyer_must_haves="balcony, lift",
        ai_review_include_preferences=True,
        ai_review_include_proximity=True,
        ai_review_include_neighborhood_prices=True,
        ai_review_include_all_remaining=True,
    )
    prompt = build_review_prompt(_rich_row(), settings=settings)
    assert "Quiet street" in prompt
    assert "950000" in prompt or "950,000" in prompt or "950000.0" in prompt
    assert "Nearest metro" in prompt or "dist_metro_m" in prompt
    assert "16500" in prompt
    assert "nbhd_rcn" in prompt or "Neighborhood" in prompt or "RCN" in prompt


def test_build_review_prompt_respects_section_toggles() -> None:
    settings = AppSettings(
        buyer_pref_notes="Must see",
        ai_review_include_preferences=False,
        ai_review_include_description=False,
        ai_review_include_proximity=False,
        ai_review_include_neighborhood_prices=False,
        ai_review_include_all_remaining=False,
        ai_review_include_raw_characteristics=False,
    )
    prompt = build_review_prompt(_rich_row(), settings=settings)
    assert "Must see" not in prompt
    assert "Bright flat near metro." not in prompt
    assert "dist_metro_m" not in prompt
    assert "16500" not in prompt
    assert "characteristics_json" not in prompt
    assert "Listing zł/m²" in prompt or "price_per_m2" in prompt


def test_review_context_hash_changes_with_prefs() -> None:
    a = AppSettings(buyer_pref_notes="A")
    b = AppSettings(buyer_pref_notes="B")
    assert review_context_hash(a) != review_context_hash(b)
