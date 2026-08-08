"""Tests for optional CatBoost compare helper."""

from __future__ import annotations

from happy_housing.models.compare import catboost_available, metrics_payload


def test_metrics_payload_unavailable() -> None:
    assert metrics_payload(None) == {"available": False}


def test_catboost_available_is_bool() -> None:
    assert isinstance(catboost_available(), bool)
