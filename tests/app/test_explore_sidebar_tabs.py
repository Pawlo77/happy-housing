"""Tests for Explore sidebar Filters / Map layers tab labels."""

from __future__ import annotations

from happy_housing.app.page_explore import EXPLORE_SIDEBAR_TABS


def test_explore_sidebar_tabs_are_filters_then_map_layers() -> None:
    assert EXPLORE_SIDEBAR_TABS == ("Filters", "Map layers")
