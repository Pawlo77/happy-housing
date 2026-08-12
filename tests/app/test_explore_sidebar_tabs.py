"""Tests for Explore sidebar Filters / Map layers tab labels."""

from __future__ import annotations

from happy_housing.app.page_explore import EXPLORE_SIDEBAR_TABS


def test_explore_sidebar_tabs_are_filters_then_map_layers() -> None:
    assert EXPLORE_SIDEBAR_TABS == ("Filters", "Map layers")


def test_map_layer_transit_and_shop_toggles_always_declared() -> None:
    """Sub-layer checkboxes stay visible (not gated behind parent if-blocks)."""
    import inspect

    from happy_housing.app import page_explore

    source = inspect.getsource(page_explore._sidebar_map_controls)
    assert "if transit_network:" not in source
    assert "if shops_on:" not in source
    assert '"Metro lines"' in source
    assert '"Groceries"' in source
    assert '"  Metro lines"' not in source
    assert '"  Groceries"' not in source


def test_explore_filters_grouped_with_section_headers() -> None:
    """Filters tab uses ##### sections like Map layers."""
    import inspect

    from happy_housing.app import page_explore

    source = inspect.getsource(page_explore._sidebar_filters)
    assert "##### Location & source" in source
    assert "##### Price & freshness" in source
    assert "##### Livability" in source
    assert "Max total price (PLN)" in source
    assert "explore_max_price_pln" in source
    assert "max_price_pln=" in source
