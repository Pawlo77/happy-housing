"""Optional CatBoost hold-out compare against LightGBM metrics."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from happy_housing.config import ModelConfig
from happy_housing.models.train import FEATURE_COLUMNS, build_feature_matrix

logger = logging.getLogger(__name__)


def catboost_available() -> bool:
    """Return True when the optional ``catboost`` package is installed."""
    import importlib.util

    return importlib.util.find_spec("catboost") is not None


def compare_catboost_holdout(
    frame: pd.DataFrame,
    model_config: ModelConfig,
) -> dict[str, float] | None:
    """Train a CatBoost regressor on the same split; return metrics or None.

    Skips when CatBoost is not installed (optional compare, not a hard dep).
    """
    if not catboost_available():
        logger.info("CatBoost not installed; skipping optional compare")
        return None
    from catboost import CatBoostRegressor

    if model_config.target not in frame.columns:
        raise ValueError(f"Target column missing: {model_config.target}")
    features = build_feature_matrix(frame, FEATURE_COLUMNS)
    target = pd.to_numeric(frame[model_config.target], errors="coerce")
    mask = target.notna() & features.notna().all(axis=1)
    features = features.loc[mask]
    target = target.loc[mask]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=model_config.test_size,
        random_state=model_config.random_state,
    )
    model = CatBoostRegressor(
        iterations=int(model_config.n_estimators),
        depth=6,
        learning_rate=float(model_config.learning_rate),
        loss_function="RMSE",
        verbose=False,
        random_seed=int(model_config.random_state),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics: dict[str, float] = {
        "catboost_mae": float(mean_absolute_error(y_test, pred)),
        "catboost_r2": float(r2_score(y_test, pred)),
        "catboost_n_test": float(len(y_test)),
    }
    logger.info("CatBoost compare metrics: %s", metrics)
    return metrics


def metrics_payload(metrics: dict[str, float] | None) -> dict[str, Any]:
    """Serialize compare result for tests / UI."""
    if metrics is None:
        return {"available": False}
    return {"available": True, **metrics}
