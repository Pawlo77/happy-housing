"""Tests for Explore/Stats hover help coverage."""

from __future__ import annotations

from happy_housing.app.map_layers import MAX_GUNB_MAP_POINTS, LayerFlags
from happy_housing.app.spatial_insights import SPATIAL_METRICS
from happy_housing.app.ui_help import (
    FILTER_HELP,
    LAYER_HELP,
    PREF_HELP,
    SPATIAL_METRIC_HELP,
    STATS_FILTER_HELP,
    layer_help,
)


def test_layer_help_covers_every_layer_flag() -> None:
    """Each Map overlay toggle must have price/livability hover copy."""
    for name in LayerFlags.__dataclass_fields__:
        text = layer_help(name)
        assert name in LAYER_HELP
        assert len(text) >= 40
        assert text == LAYER_HELP[name]


def test_offers_help_mentions_green_red_and_pin_click() -> None:
    text = LAYER_HELP["offers"].lower()
    assert "green" in text and "red" in text
    assert "underpriced" in text
    assert "open in offer" in text


def test_gunb_help_mentions_map_sample_cap() -> None:
    assert "sample" in LAYER_HELP["gunb"].lower()
    assert f"{MAX_GUNB_MAP_POINTS:,}" in LAYER_HELP["gunb"]


def test_spatial_metric_help_covers_catalog() -> None:
    """Spatial metric picker help covers every catalog key."""
    for metric in SPATIAL_METRICS:
        assert metric.key in SPATIAL_METRIC_HELP
        assert len(SPATIAL_METRIC_HELP[metric.key]) >= 40


def test_filter_and_pref_help_nonempty() -> None:
    """Sidebar filter / Prefs dictionaries stay populated for wired keys."""
    for mapping in (FILTER_HELP, STATS_FILTER_HELP, PREF_HELP):
        assert mapping
        for text in mapping.values():
            assert len(text) >= 20
