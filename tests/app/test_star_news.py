"""Tests for starred-offer change news / snapshot sync."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.app.star_news import (
    format_change_html,
    load_news,
    load_snapshots,
    sync_starred_news,
)


def test_sync_baseline_creates_snapshot_without_news(tmp_path: Path) -> None:
    snaps = tmp_path / "snaps.json"
    news = tmp_path / "news.json"
    frame = pd.DataFrame({
        "listing_id": ["OTO-1"],
        "title": ["Flat A"],
        "price_pln": [900_000.0],
        "price_per_m2": [15_000.0],
        "model_price_per_m2": [14_500.0],
        "delta_price_per_m2": [500.0],
        "area_m2": [60.0],
        "rooms": [3],
        "district": ["wola"],
        "commute_min": [12.0],
        "in_oou": [0],
        "url": ["https://example.com/1"],
    })
    items = sync_starred_news(
        frozenset({"OTO-1"}),
        frame,
        snapshots_path=snaps,
        news_path=news,
    )
    assert items == []
    assert "OTO-1" in load_snapshots(snaps)
    assert load_news(news) == []


def test_sync_emits_price_change_and_updates_snapshot(tmp_path: Path) -> None:
    snaps = tmp_path / "snaps.json"
    news = tmp_path / "news.json"
    base = pd.DataFrame({
        "listing_id": ["OTO-1"],
        "title": ["Flat A"],
        "price_pln": [900_000.0],
        "price_per_m2": [15_000.0],
        "model_price_per_m2": [14_500.0],
        "delta_price_per_m2": [500.0],
        "area_m2": [60.0],
        "rooms": [3],
        "district": ["wola"],
        "commute_min": [12.0],
        "in_oou": [0],
        "url": ["https://example.com/1"],
    })
    sync_starred_news(
        frozenset({"OTO-1"}),
        base,
        snapshots_path=snaps,
        news_path=news,
    )
    updated = base.copy()
    updated.loc[0, "price_pln"] = 850_000.0
    updated.loc[0, "price_per_m2"] = 14_166.0
    items = sync_starred_news(
        frozenset({"OTO-1"}),
        updated,
        snapshots_path=snaps,
        news_path=news,
    )
    assert len(items) == 1
    assert items[0].kind == "changed"
    assert items[0].listing_id == "OTO-1"
    fields = {c.field for c in items[0].changes}
    assert "price_pln" in fields
    assert "price_per_m2" in fields
    assert load_snapshots(snaps)["OTO-1"]["price_pln"] == 850_000.0
    assert len(load_news(news)) == 1


def test_sync_reports_removed_starred_offer(tmp_path: Path) -> None:
    snaps = tmp_path / "snaps.json"
    news = tmp_path / "news.json"
    frame = pd.DataFrame({
        "listing_id": ["OTO-1"],
        "title": ["Flat A"],
        "price_pln": [900_000.0],
        "price_per_m2": [15_000.0],
        "model_price_per_m2": [14_500.0],
        "delta_price_per_m2": [500.0],
        "area_m2": [60.0],
        "rooms": [3],
        "district": ["wola"],
        "commute_min": [12.0],
        "in_oou": [0],
        "url": ["https://example.com/1"],
    })
    sync_starred_news(
        frozenset({"OTO-1"}),
        frame,
        snapshots_path=snaps,
        news_path=news,
    )
    empty = frame.iloc[0:0].copy()
    items = sync_starred_news(
        frozenset({"OTO-1"}),
        empty,
        snapshots_path=snaps,
        news_path=news,
    )
    assert len(items) == 1
    assert items[0].kind == "removed"


def test_format_change_html_strikes_old_value() -> None:
    html = format_change_html("price_pln", 900_000.0, 850_000.0)
    assert "line-through" in html
    assert "900,000" in html
    assert "850,000" in html
    assert "Price (PLN)" in html
