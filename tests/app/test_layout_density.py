"""Tests for dense top-layout CSS helper."""

from __future__ import annotations

import re

from happy_housing.app.layout_density import (
    _DENSE_TOP_CSS,
    _DENSE_TOP_PADDING,
    apply_dense_top_layout,
)


def test_dense_top_css_only_sets_padding_top() -> None:
    assert "padding-top" in _DENSE_TOP_CSS
    assert "padding-bottom" not in _DENSE_TOP_CSS
    assert "color:" not in _DENSE_TOP_CSS
    assert "font-" not in _DENSE_TOP_CSS
    assert "background" not in _DENSE_TOP_CSS


def test_dense_top_padding_clears_top_nav() -> None:
    """Padding must clear top nav — 1rem clipped Explore tabs."""
    assert _DENSE_TOP_PADDING.endswith("rem")
    value = float(_DENSE_TOP_PADDING.removesuffix("rem"))
    assert value >= 3.5
    assert value <= 6.0
    match = re.search(r"padding-top:\s*([\d.]+)rem", _DENSE_TOP_CSS)
    assert match is not None
    assert float(match.group(1)) == value


def test_apply_dense_top_layout_is_callable() -> None:
    assert callable(apply_dense_top_layout)
