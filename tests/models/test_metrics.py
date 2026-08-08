"""Tests for regression metric helpers used by HPO."""

from __future__ import annotations

import numpy as np

from happy_housing.models.metrics import (
    OPT_METRICS,
    compute_regression_metrics,
    is_better,
    metric_direction,
)


def test_opt_metrics_cover_admin_choices() -> None:
    assert set(OPT_METRICS) >= {"mae", "rmse", "mape", "medae", "r2"}


def test_compute_regression_metrics_basic() -> None:
    y = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 310.0])
    metrics = compute_regression_metrics(y, pred)
    assert metrics["mae"] == 10.0
    assert metrics["medae"] == 10.0
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0
    assert "r2" in metrics


def test_is_better_respects_direction() -> None:
    assert metric_direction("mae") == "minimize"
    assert metric_direction("r2") == "maximize"
    assert is_better("mae", 10.0, 12.0) is True
    assert is_better("r2", 0.8, 0.7) is True
    assert is_better("mae", 12.0, 10.0) is False
