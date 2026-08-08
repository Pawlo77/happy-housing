"""Tests for Home hero display helpers."""

from __future__ import annotations

from happy_housing.app.home_hero import format_hero_stats, hero_html


def test_format_hero_stats_formats_counts_and_r2() -> None:
    stats = format_hero_stats(
        refreshed="2026-08-07T18:12:29+00:00",
        n_offers=1020,
        r2=0.6241,
    )
    assert stats.offers_label == "1,020"
    assert stats.r2_label == "0.624"
    assert stats.refreshed_label == "2026-08-07 18:12:29"


def test_format_hero_stats_handles_missing() -> None:
    stats = format_hero_stats(refreshed=None, n_offers=0, r2=None)
    assert stats.offers_label == "0"
    assert stats.r2_label == "—"
    assert stats.refreshed_label == "—"


def test_hero_html_includes_brand_and_stats() -> None:
    stats = format_hero_stats(refreshed=None, n_offers=12, r2=0.5)
    html = hero_html(stats)
    assert "happy-housing" in html
    assert "12" in html
    assert "0.500" in html
    assert "badge" not in html.lower()
