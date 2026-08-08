"""Tests for attribution footer copy."""

from __future__ import annotations

from happy_housing.app.attribution import ATTRIBUTION_LINE


def test_attribution_mentions_core_sources() -> None:
    assert "Otodom" in ATTRIBUTION_LINE
    assert "RCN" in ATTRIBUTION_LINE
    assert "GTFS" in ATTRIBUTION_LINE
    assert "GIOŚ" in ATTRIBUTION_LINE
