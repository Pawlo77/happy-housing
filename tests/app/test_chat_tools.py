"""Tests for Assistant tool helpers (no LM Studio)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.app.chat_tools import (
    ChatToolContext,
    offer_summary_row,
    search_offers,
    slim_offer_dict,
)
from happy_housing.app.preferences import UserPreferences


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "listing_id": "A",
            "district": "Wola",
            "price_pln": 800_000,
            "price_per_m2": 16_000,
            "area_m2": 50,
            "rooms": 2,
            "delta_price_per_m2": -500,
            "commute_min": 12,
            "source": "otodom",
            "title": "Wola flat",
            "lat": 52.23,
            "lon": 20.98,
            "model_price_per_m2": 16_500,
            "in_oou": False,
        },
        {
            "listing_id": "B",
            "district": "Mokotów",
            "price_pln": 1_200_000,
            "price_per_m2": 20_000,
            "area_m2": 60,
            "rooms": 3,
            "delta_price_per_m2": 800,
            "commute_min": 25,
            "source": "otodom",
            "title": "Mokotów flat",
            "lat": 52.20,
            "lon": 21.02,
            "model_price_per_m2": 19_200,
            "in_oou": True,
        },
    ])


def test_slim_offer_dict_keeps_key_fields() -> None:
    row = _sample_frame().iloc[0]
    slim = slim_offer_dict(row)
    assert slim["listing_id"] == "A"
    assert "price_pln" in slim
    assert "delta_price_per_m2" in slim
    assert "description" not in slim


def test_search_offers_filters_district_and_price() -> None:
    ctx = ChatToolContext(
        scored=_sample_frame(),
        processed=Path("/tmp"),
        prefs=UserPreferences(),
        active_listing_id=None,
        settings_top_n=5,
        demotion_alpha=1.0,
        pending_open_listing_id=None,
    )
    hits = search_offers(
        ctx,
        district="Wola",
        max_price_pln=900_000,
        limit=10,
    )
    assert len(hits) == 1
    assert hits[0]["listing_id"] == "A"


def test_offer_summary_row_missing() -> None:
    ctx = ChatToolContext(
        scored=_sample_frame(),
        processed=Path("/tmp"),
        prefs=UserPreferences(),
        active_listing_id="Z",
        settings_top_n=5,
        demotion_alpha=1.0,
        pending_open_listing_id=None,
    )
    assert offer_summary_row(ctx, "Z") is None
    assert offer_summary_row(ctx, "A")["listing_id"] == "A"
