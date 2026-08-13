"""Regression metrics and HPO objective helpers for fair-price modeling."""

from typing import Literal

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

OptMetric = Literal["mae", "rmse", "mape", "medae", "r2"]
"""Admin-selectable optimization metrics for hyperparameter search."""

OPT_METRICS: tuple[OptMetric, ...] = ("mae", "rmse", "mape", "medae", "r2")
"""Supported optimization metric names."""

_DEFAULT_METRIC: OptMetric = "mae"
"""Default objective: MAE in zł/m² — interpretable for fair-price error."""


def coerce_opt_metric(value: object, fallback: str = _DEFAULT_METRIC) -> OptMetric:
    """Normalize a persisted / UI metric name."""
    name = str(value or fallback).strip().lower()
    if name in OPT_METRICS:
        return name  # type: ignore[return-value]  # Membership narrows to metric literal
    fb = str(fallback).strip().lower()
    if fb in OPT_METRICS:
        return fb  # type: ignore[return-value]  # Membership narrows to metric literal
    return _DEFAULT_METRIC


def metric_direction(metric: OptMetric) -> Literal["minimize", "maximize"]:
    """Whether lower or higher scores are better for ``metric``."""
    if metric == "r2":
        return "maximize"
    return "minimize"


def is_better(metric: OptMetric, candidate: float, best: float) -> bool:
    """Return True when ``candidate`` improves on ``best`` for ``metric``."""
    if metric_direction(metric) == "maximize":
        return candidate > best
    return candidate < best


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE / RMSE / MAPE / MedAE / R² for zł/m² predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "medae": float("nan"),
            "r2": float("nan"),
            "n": 0.0,
        }
    # MAPE needs non-zero targets; clip tiny zł/m² to avoid blow-ups.
    mape_truth = np.clip(np.abs(y_true), 1.0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(mape_truth, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": float(len(y_true)),
    }
