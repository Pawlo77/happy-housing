"""LightGBM price model training and SHAP explanations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from happy_housing.config import ExperimentConfig, ModelConfig
from happy_housing.orchestration.tracking import (
    _resolve_tracking_uri,
    build_mlflow_tracker,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (
    "area_m2",
    "rooms",
    "year_built",
    "in_oou",
    "dist_hospital_m",
    "dist_fire_station_m",
    "dist_police_m",
    "dist_air_station_m",
    "dist_transit_stop_m",
    "dist_daszynskiego_m",
    "commute_min",
    "dist_gunb_m",
)
"""Numeric feature columns used by the price model."""


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Artifacts from a training run."""

    model: lgb.LGBMRegressor
    """Fitted regressor."""
    metrics: dict[str, float]
    """Hold-out metrics."""
    feature_names: list[str]
    """Feature column order."""
    features_train: pd.DataFrame
    """Training features."""
    features_test: pd.DataFrame
    """Test features."""
    y_test: pd.Series
    """Test targets."""


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Select and impute model features."""
    work = frame.copy()
    for col in feature_columns:
        if col not in work.columns:
            work[col] = float("nan")
    features = work.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    medians = features.median(numeric_only=True)
    return features.fillna(medians).fillna(0.0)


def train_price_model(
    frame: pd.DataFrame,
    model_config: ModelConfig,
) -> TrainResult:
    """Train LightGBM on ``price_per_m2`` and return metrics."""
    if model_config.target not in frame.columns:
        raise ValueError(f"Target column missing: {model_config.target}")
    features = build_feature_matrix(frame)
    target = pd.to_numeric(frame[model_config.target], errors="coerce")
    mask = target.notna() & features.notna().all(axis=1)
    features = features.loc[mask]
    target = target.loc[mask]
    features_train, features_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=model_config.test_size,
        random_state=model_config.random_state,
    )
    model = lgb.LGBMRegressor(
        n_estimators=model_config.n_estimators,
        learning_rate=model_config.learning_rate,
        num_leaves=model_config.num_leaves,
        random_state=model_config.random_state,
        verbosity=-1,
    )
    model.fit(features_train, y_train)
    preds = model.predict(features_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": float(len(features_train)),
        "n_test": float(len(features_test)),
    }
    logger.info("Train metrics: %s", metrics)
    return TrainResult(
        model=model,
        metrics=metrics,
        feature_names=list(features.columns),
        features_train=features_train,
        features_test=features_test,
        y_test=y_test,
    )


def predict_price_per_m2(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict zł/m² for enriched listing rows."""
    features = build_feature_matrix(frame)
    return np.asarray(model.predict(features), dtype=float)


def shap_explanation_for_row(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    row_index: int = 0,
) -> shap.Explanation:
    """Compute a TreeExplainer explanation for one row."""
    features = build_feature_matrix(frame)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(features)
    return explanation[row_index]


def run_training_with_mlflow(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    model_path: Path,
) -> TrainResult:
    """Train, log to MLflow, and persist the model booster."""
    result = train_price_model(frame, config.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.booster_.save_model(str(model_path))

    if not config.mlflow.enabled:
        return result

    tracking_uri = _resolve_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    tracker = build_mlflow_tracker(
        config,
        run_name=config.mlflow.run_name or "lightgbm-price-per-m2",
    )
    tracker.start()
    try:
        if tracker._mlflow is not None:
            if config.mlflow.log_params:
                tracker._mlflow.log_params({
                    "n_estimators": config.model.n_estimators,
                    "learning_rate": config.model.learning_rate,
                    "num_leaves": config.model.num_leaves,
                    "test_size": config.model.test_size,
                    "features": ",".join(FEATURE_COLUMNS),
                })
            if config.mlflow.log_metrics:
                tracker._mlflow.log_metrics(result.metrics)
            if config.mlflow.log_artifacts:
                tracker._mlflow.log_artifact(str(model_path))
    finally:
        tracker.close()
    return result
