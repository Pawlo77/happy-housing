"""Tests for dense top-layout CSS helper."""

from __future__ import annotations

from happy_housing.app.layout_density import _DENSE_TOP_CSS, apply_dense_top_layout


def test_dense_top_css_only_sets_padding_top() -> None:
    assert "padding-top" in _DENSE_TOP_CSS
    assert "padding-bottom" not in _DENSE_TOP_CSS
    assert "color:" not in _DENSE_TOP_CSS
    assert "font-" not in _DENSE_TOP_CSS
    assert "background" not in _DENSE_TOP_CSS


def test_apply_dense_top_layout_is_callable() -> None:
    assert callable(apply_dense_top_layout)
