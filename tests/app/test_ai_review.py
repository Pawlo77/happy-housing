"""Tests for offer AI review DuckDB cache (no live LM Studio call)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.app.ai_review import (
    OfferAiReview,
    build_review_prompt,
    load_review,
    save_review,
)


def test_save_and_load_review_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "reviews.duckdb"
    review = OfferAiReview(
        listing_id="OTO-1",
        model="gemma-4-31b-it-mlx",
        review_text="Lean watch — fair-ish ask.",
        created_at="2026-08-08T12:00:00+00:00",
    )
    save_review(path, review)
    loaded = load_review(path, "OTO-1", model="gemma-4-31b-it-mlx")
    assert loaded is not None
    assert loaded.review_text == review.review_text
    assert load_review(path, "OTO-1", model="other") is None


def test_build_review_prompt_includes_description() -> None:
    row = pd.Series({
        "listing_id": "OTO-1",
        "title": "Flat",
        "district": "Wola",
        "source": "otodom",
        "price_pln": 900000,
        "area_m2": 50,
        "rooms": 2,
        "year_built": 2010,
        "price_per_m2": 18000,
        "model_price_per_m2": 17000,
        "delta_price_per_m2": 1000,
        "in_oou": 0,
        "commute_min": 15,
        "dist_transit_stop_m": 200,
        "dist_hospital_m": 1000,
        "dist_fire_station_m": 800,
        "dist_police_m": 900,
        "dist_air_station_m": 1200,
        "dist_gunb_m": 3000,
        "description": "Bright flat near metro.",
    })
    prompt = build_review_prompt(row)
    assert "Bright flat near metro." in prompt
    assert "OTO-1" in prompt
