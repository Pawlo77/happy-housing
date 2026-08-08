"""Tests for Assistant tool helpers (no LM Studio)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from happy_housing.app.chat_tools import (
    ChatToolContext,
    get_active_offer,
    offer_summary_row,
    open_offer,
    recommend_offers,
    search_offers,
    slim_offer_dict,
    update_preferences,
)
from happy_housing.app.preferences import UserPreferences, load_preferences


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


def test_update_preferences_merges_without_wiping_unrelated(
    tmp_path: Path,
) -> None:
    prefs_file = tmp_path / "user_preferences.json"
    prefs = UserPreferences(
        enabled=False,
        districts=["wola"],
        max_price_pln=900_000.0,
        nl_text="old note",
    )
    ctx = ChatToolContext(
        scored=_sample_frame(),
        processed=tmp_path,
        prefs=prefs,
        prefs_path=prefs_file,
        active_listing_id=None,
        settings_top_n=5,
        demotion_alpha=1.0,
        pending_open_listing_id=None,
    )
    out = update_preferences(
        ctx,
        enabled=True,
        notes="want quiet Wola",
        min_rooms=3,
    )
    assert out["enabled"] is True
    assert out["nl_text"] == "want quiet Wola"
    assert out["min_rooms"] == 3
    assert out["districts"] == ["wola"]
    assert out["max_price_pln"] == 900_000.0
    loaded = load_preferences(prefs_file)
    assert loaded.enabled is True
    assert loaded.districts == ["wola"]
    assert loaded.min_rooms == 3


def test_get_active_and_open_offer() -> None:
    ctx = ChatToolContext(
        scored=_sample_frame(),
        processed=Path("/tmp"),
        prefs=UserPreferences(),
        active_listing_id="B",
        settings_top_n=5,
        demotion_alpha=1.0,
        pending_open_listing_id=None,
    )
    active = get_active_offer(ctx)
    assert active["active"] is True
    assert active["listing_id"] == "B"
    opened = open_offer(ctx, "A")
    assert opened["ok"] is True
    assert ctx.pending_open_listing_id == "A"


def test_recommend_offers_caps_and_skips_impressions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake_top(*_args: object, **kwargs: object) -> pd.DataFrame:
        calls.clear()
        calls.update(kwargs)
        return _sample_frame().head(1)

    monkeypatch.setattr(
        "happy_housing.app.chat_tools.top_recommendations",
        _fake_top,
    )
    ctx = ChatToolContext(
        scored=_sample_frame(),
        processed=tmp_path,
        prefs=UserPreferences(enabled=True),
        active_listing_id=None,
        settings_top_n=5,
        demotion_alpha=1.0,
        pending_open_listing_id=None,
    )
    hits = recommend_offers(ctx, top_n=99)
    assert len(hits) == 1
    assert calls.get("record") is False
    assert int(calls["top_n"]) == 30  # type: ignore[arg-type]
