"""LightGBM price model training, HPO, SHAP, and MLflow logging."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import ParameterSampler, train_test_split

from happy_housing.config import ExperimentConfig, ModelConfig
from happy_housing.models.metrics import (
    OptMetric,
    coerce_opt_metric,
    compute_regression_metrics,
    is_better,
    metric_direction,
)
from happy_housing.models.neighborhood import (
    NEIGHBORHOOD_FEATURE_COLUMNS,
    idw_neighborhood_baseline,
)
from happy_housing.orchestration.tracking import (
    _resolve_tracking_uri,
    build_mlflow_tracker,
)

logger = logging.getLogger(__name__)

CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_m2",
    "rooms",
    "year_built",
    "floor_no",
    "building_floors_num",
    "market_primary",
    "has_balcony",
    "has_garage",
    "has_lift",
    "has_terrace",
    "has_garden",
    "is_private_owner",
    "rent",
    "building_type",
    "heating",
    "construction_status",
    "building_material",
    "district",
    "aircraft_lden_db",
    "aircraft_lnight_db",
    "in_oou",
    "road_lden_db",
    "rail_lden_db",
    "dist_hospital_m",
    "dist_fire_station_m",
    "dist_police_m",
    "dist_air_station_m",
    "air_aq_index",
    "air_pm10_ug_m3",
    "air_pm25_ug_m3",
    "air_no2_ug_m3",
    "dist_transit_stop_m",
    "dist_metro_m",
    "dist_park_m",
    "dist_school_m",
    "dist_shop_m",
    "dist_daszynskiego_m",
    "commute_min",
    "transit_commute_min",
    "dist_gunb_m",
)
"""Structural + GIS features (always included)."""

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "building_type",
    "heating",
    "construction_status",
    "building_material",
    "district",
)
"""Otodom string attributes encoded as LightGBM categories."""

FEATURE_COLUMNS: tuple[str, ...] = CORE_FEATURE_COLUMNS + NEIGHBORHOOD_FEATURE_COLUMNS
"""Full default feature list including neighborhood RCN comps."""

_HPO_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
"""LightGBM hyperparameter search space for ParameterSampler."""

_LABEL_SOURCES_OTODOM: frozenset[str] = frozenset({"otodom", "fixture"})
"""Sources treated as ask-price labels when ``train_otodom_labels_only``."""


@dataclass(frozen=True, slots=True)
class TrainSettings:
    """Runtime training knobs (Admin / app settings + experiment defaults)."""

    opt_metric: OptMetric = "mae"
    """Metric optimized on the validation fold during HPO."""
    val_size: float = 0.15
    """Fraction of non-test rows held out for HPO validation."""
    test_size: float = 0.2
    """Final hold-out fraction (never used for HPO selection)."""
    hpo_trials: int = 12
    """Number of random hyperparameter samples (1 = no search)."""
    use_neighborhood: bool = True
    """Include RCN neighborhood features when present on the frame."""
    random_state: int = 42
    """RNG seed for splits and HPO sampling."""
    train_otodom_labels_only: bool = True
    """Drop RCN deed rows from training labels (comps still OK as features)."""


@dataclass(frozen=True, slots=True)
class SplitBundle:
    """One fold's features + target."""

    features: pd.DataFrame
    """Feature matrix rows."""
    target: pd.Series
    """Aligned target values."""


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Artifacts from a training run."""

    model: lgb.LGBMRegressor
    """Fitted regressor (retrained on train+val with best params)."""
    metrics: dict[str, float]
    """Flattened metrics for metadata (test_* primary)."""
    feature_names: list[str]
    """Feature column order."""
    features_train: pd.DataFrame
    """Train+val features used for the final fit."""
    features_test: pd.DataFrame
    """Held-out test features."""
    y_test: pd.Series
    """Held-out test targets."""
    best_params: dict[str, Any] = field(default_factory=dict)
    """Winning LightGBM hyperparameters."""
    opt_metric: OptMetric = "mae"
    """Metric used to select the winner."""
    val_metrics: dict[str, float] = field(default_factory=dict)
    """Validation metrics of the winning trial."""
    test_metrics: dict[str, float] = field(default_factory=dict)
    """Final test metrics of the production model."""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    """Neighborhood IDW baseline metrics on the test fold (if available)."""
    train_medians: dict[str, float] = field(default_factory=dict)
    """Numeric impute medians from the train fold."""


def active_feature_columns(*, use_neighborhood: bool = True) -> tuple[str, ...]:
    """Feature list depending on whether neighborhood comps are enabled."""
    if use_neighborhood:
        return FEATURE_COLUMNS
    return CORE_FEATURE_COLUMNS


def model_feature_columns(model: lgb.Booster | lgb.LGBMRegressor) -> tuple[str, ...]:
    """Feature names the fitted model actually expects (order preserved)."""
    if isinstance(model, lgb.Booster):
        return tuple(model.feature_name())
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return tuple(booster.feature_name())
    names = getattr(model, "feature_name_", None)
    if names:
        return tuple(str(n) for n in names)
    return FEATURE_COLUMNS


def check_feature_coverage(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    max_missing_fraction: float = 0.85,
    min_usable: int = 8,
) -> None:
    """Raise when too few feature columns have any non-null signal."""
    del max_missing_fraction  # kept for call-site compat
    missing = [c for c in feature_columns if c not in frame.columns]
    usable = 0
    for col in feature_columns:
        if col not in frame.columns:
            continue
        if col in CATEGORICAL_FEATURE_COLUMNS:
            if frame[col].notna().any():
                usable += 1
            continue
        if pd.to_numeric(frame[col], errors="coerce").notna().any():
            usable += 1
    if usable < min_usable:
        raise ValueError(
            f"Feature coverage too low: only {usable} usable columns "
            f"(need ≥{min_usable}); missing={missing[:12]}"
        )
    if missing:
        logger.warning("Model features absent on frame: %s", missing)


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | None = None,
    *,
    train_medians: pd.Series | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Select features; impute numerics with train medians (or frame medians)."""
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    raw = _raw_feature_frame(frame, cols)
    numeric_cols = [c for c in cols if c not in CATEGORICAL_FEATURE_COLUMNS]
    if train_medians is None:
        medians = (
            raw
            .loc[:, numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .median(numeric_only=True)
        )
    else:
        medians = pd.Series(train_medians)
    return _impute_with_medians(raw, medians, cols)


def split_train_val_test(
    features: pd.DataFrame,
    target: pd.Series,
    frame: pd.DataFrame,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[SplitBundle, SplitBundle, SplitBundle]:
    """Time-aware split when ``listed_at`` present; else random IID."""
    test_size = float(np.clip(test_size, 0.1, 0.4))
    val_size = float(np.clip(val_size, 0.05, 0.35))
    n = len(features)
    if "listed_at" in frame.columns and n >= 20:
        listed = pd.to_datetime(
            frame.loc[features.index, "listed_at"],
            errors="coerce",
            utc=True,
        )
        listed = listed.fillna(pd.Timestamp("1970-01-01", tz="UTC"))
        order = listed.sort_values().index
        features = features.loc[order]
        target = target.loc[order]
        n_test = max(1, round(n * test_size))
        n_val = max(1, round(n * val_size))
        n_train = n - n_test - n_val
        if n_train < 5:
            n_train = max(5, n - n_test - 1)
            n_val = max(1, n - n_train - n_test)
        train_idx = features.index[:n_train]
        val_idx = features.index[n_train : n_train + n_val]
        test_idx = features.index[n_train + n_val :]
        return (
            SplitBundle(features.loc[train_idx], target.loc[train_idx]),
            SplitBundle(features.loc[val_idx], target.loc[val_idx]),
            SplitBundle(features.loc[test_idx], target.loc[test_idx]),
        )

    features_tv, features_test, y_tv, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    val_fraction_of_tv = val_size / max(1e-6, (1.0 - test_size))
    val_fraction_of_tv = float(np.clip(val_fraction_of_tv, 0.05, 0.5))
    features_train, features_val, y_train, y_val = train_test_split(
        features_tv,
        y_tv,
        test_size=val_fraction_of_tv,
        random_state=random_state,
    )
    return (
        SplitBundle(features_train, y_train),
        SplitBundle(features_val, y_val),
        SplitBundle(features_test, y_test),
    )


def _fit_lgbm(
    features_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    random_state: int,
) -> lgb.LGBMRegressor:
    work = features_train.copy()
    cat_in_frame: list[str] = []
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col not in work.columns:
            continue
        work[col] = work[col].astype(str).fillna("unknown").astype("category")
        cat_in_frame.append(col)
    model = lgb.LGBMRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params.get("min_child_samples", 20)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(
        work,
        y_train,
        categorical_feature=cat_in_frame or "auto",
    )
    return model


def _default_params(model_config: ModelConfig) -> dict[str, Any]:
    return {
        "n_estimators": model_config.n_estimators,
        "learning_rate": model_config.learning_rate,
        "num_leaves": model_config.num_leaves,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }


def _hpo_candidates(
    model_config: ModelConfig,
    *,
    trials: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Build a list of hyperparameter dicts (always includes YAML defaults)."""
    base = _default_params(model_config)
    if trials <= 1:
        return [base]
    sampler = ParameterSampler(
        _HPO_SPACE,
        n_iter=max(1, trials - 1),
        random_state=random_state,
    )
    seen: set[tuple[Any, ...]] = set()
    candidates = [base]
    seen.add(tuple(sorted(base.items())))
    for sample in sampler:
        key = tuple(sorted(sample.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dict(sample))
    return candidates


def train_price_model(
    frame: pd.DataFrame,
    model_config: ModelConfig,
    train_settings: TrainSettings | None = None,
    *,
    mlflow_module: Any | None = None,
) -> TrainResult:
    """Train with train/val/test splits and validation-driven HPO.

    Args:
        frame: Enriched listings including the target column.
        model_config: YAML model defaults (seed the HPO space).
        train_settings: Admin / runtime overrides.
        mlflow_module: Optional active ``mlflow`` module for nested trial runs.
    """
    settings = train_settings or TrainSettings(
        test_size=model_config.test_size,
        random_state=model_config.random_state,
    )
    opt_metric = coerce_opt_metric(settings.opt_metric)
    feature_cols = active_feature_columns(use_neighborhood=settings.use_neighborhood)
    if model_config.target not in frame.columns:
        raise ValueError(f"Target column missing: {model_config.target}")

    work = frame
    if settings.train_otodom_labels_only and "source" in frame.columns:
        sources = frame["source"].astype(str).str.lower()
        work = frame.loc[sources.isin(_LABEL_SOURCES_OTODOM)].copy()
        logger.info(
            "Train labels filtered to Otodom/fixture: %s → %s rows",
            len(frame),
            len(work),
        )

    check_feature_coverage(work, feature_cols, max_missing_fraction=0.9)

    # Unfilled matrix — medians from train fold only after split.
    raw_features = _raw_feature_frame(work, feature_cols)
    target = pd.to_numeric(work[model_config.target], errors="coerce")
    mask = target.notna()
    raw_features = raw_features.loc[mask]
    target = target.loc[mask]
    if len(raw_features) < 10:
        raise ValueError(f"Need ≥10 labeled rows to train; got {len(raw_features)}")

    train_b, val_b, test_b = split_train_val_test(
        raw_features,
        target,
        work.loc[raw_features.index],
        test_size=settings.test_size,
        val_size=settings.val_size,
        random_state=settings.random_state,
    )

    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURE_COLUMNS]
    train_medians = (
        train_b.features
        .loc[:, numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .median(numeric_only=True)
    )
    features_train = _impute_with_medians(train_b.features, train_medians, feature_cols)
    features_val = _impute_with_medians(val_b.features, train_medians, feature_cols)
    features_test = _impute_with_medians(test_b.features, train_medians, feature_cols)
    y_train, y_val, y_test = train_b.target, val_b.target, test_b.target
    features_tv = pd.concat([features_train, features_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)

    candidates = _hpo_candidates(
        model_config,
        trials=max(1, int(settings.hpo_trials)),
        random_state=settings.random_state,
    )
    best_params = candidates[0]
    best_score = (
        float("-inf") if metric_direction(opt_metric) == "maximize" else float("inf")
    )
    best_val_metrics: dict[str, float] = {}

    for trial_i, params in enumerate(candidates):
        model = _fit_lgbm(features_train, y_train, params, settings.random_state)
        val_pred = model.predict(features_val)
        val_metrics = compute_regression_metrics(y_val.to_numpy(), np.asarray(val_pred))
        score = float(val_metrics[opt_metric])
        logger.info(
            "HPO trial %s/%s %s=%s params=%s",
            trial_i + 1,
            len(candidates),
            opt_metric,
            score,
            params,
        )
        if mlflow_module is not None:
            with mlflow_module.start_run(run_name=f"hpo-{trial_i}", nested=True):
                mlflow_module.log_params({f"hpo_{k}": v for k, v in params.items()})
                mlflow_module.log_metrics({
                    f"val_{k}": v for k, v in val_metrics.items()
                })
                mlflow_module.set_tag("stage", "hpo")
                mlflow_module.set_tag("opt_metric", opt_metric)
        if is_better(opt_metric, score, best_score):
            best_score = score
            best_params = params
            best_val_metrics = val_metrics

    # Final model: train + val, evaluate on untouched test.
    final_model = _fit_lgbm(features_tv, y_tv, best_params, settings.random_state)
    test_pred = final_model.predict(features_test)
    test_metrics = compute_regression_metrics(y_test.to_numpy(), np.asarray(test_pred))

    baseline_metrics: dict[str, float] = {}
    if settings.use_neighborhood and "nbhd_rcn_idw_ppm" in features_test.columns:
        baseline_pred = features_test["nbhd_rcn_idw_ppm"].to_numpy(dtype=float)
        baseline_metrics = compute_regression_metrics(y_test.to_numpy(), baseline_pred)
        baseline_metrics = {f"baseline_{k}": v for k, v in baseline_metrics.items()}

    metrics = {
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in best_val_metrics.items()},
        "n_train": float(len(features_train)),
        "n_val": float(len(features_val)),
        "n_test": float(len(features_test)),
        "n_train_val": float(len(features_tv)),
        "hpo_trials": float(len(candidates)),
        # Back-compat keys used by Home / metadata.
        "mae": float(test_metrics["mae"]),
        "r2": float(test_metrics["r2"]),
        **baseline_metrics,
    }
    return TrainResult(
        model=final_model,
        metrics=metrics,
        feature_names=list(feature_cols),
        features_train=features_tv,
        features_test=features_test,
        y_test=y_test,
        best_params=dict(best_params),
        opt_metric=opt_metric,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        train_medians={str(k): _safe_float(v) for k, v in train_medians.items()},
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce to float; map NA / NaN / bad values to ``default``."""
    try:
        if value is None or pd.isna(value):
            return default
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _raw_feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Feature frame with NaNs preserved (no impute) for split-then-impute."""
    work = frame.copy()
    for col in feature_columns:
        if col not in work.columns:
            work[col] = float("nan") if col not in CATEGORICAL_FEATURE_COLUMNS else None
    pieces: list[pd.Series] = []
    for col in feature_columns:
        if col in CATEGORICAL_FEATURE_COLUMNS:
            pieces.append(
                work[col].astype(str).fillna("unknown").astype("category").rename(col)
            )
        else:
            pieces.append(pd.to_numeric(work[col], errors="coerce").rename(col))
    return pd.concat(pieces, axis=1)


def _impute_with_medians(
    features: pd.DataFrame,
    medians: pd.Series,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Apply train-fold medians to numeric columns; keep categories."""
    out = features.copy()
    for col in feature_columns:
        if col in CATEGORICAL_FEATURE_COLUMNS:
            if col in out.columns:
                out[col] = out[col].astype(str).fillna("unknown").astype("category")
            continue
        raw_fill = medians[col] if col in medians.index else 0.0
        fill = _safe_float(raw_fill)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill)
    return out.loc[:, list(feature_columns)]


def predict_price_per_m2(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    *,
    use_neighborhood: bool = True,
) -> np.ndarray:
    """Score listings with a fitted model (feature matrix aligned to booster)."""
    cols = model_feature_columns(model)
    if not cols:
        cols = active_feature_columns(use_neighborhood=use_neighborhood)
    features = build_feature_matrix(frame, cols)
    # Align category levels for LightGBM
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in features.columns:
            features[col] = (
                features[col].astype(str).fillna("unknown").astype("category")
            )
    return np.asarray(model.predict(features), dtype=float)


def oof_predict_price_per_m2(
    frame: pd.DataFrame,
    model_config: ModelConfig,
    train_settings: TrainSettings,
    *,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold predictions for train-label rows; final-model style for others.

    Rows excluded from training (e.g. RCN when Otodom-only) are scored with a
    model fit on all train labels so Explore deltas stay honest on asks.
    """
    from sklearn.model_selection import KFold

    settings = train_settings
    feature_cols = active_feature_columns(use_neighborhood=settings.use_neighborhood)
    preds = np.full(len(frame), np.nan, dtype=float)
    work = frame
    label_mask = pd.Series(True, index=frame.index)
    if settings.train_otodom_labels_only and "source" in frame.columns:
        sources = frame["source"].astype(str).str.lower()
        label_mask = sources.isin(_LABEL_SOURCES_OTODOM)

    labeled_idx = frame.index[label_mask].to_list()
    if len(labeled_idx) < max(10, n_splits * 2):
        # Too small for OOF — fall back to in-sample after a quick fit
        result = train_price_model(frame, model_config, settings)
        return predict_price_per_m2(
            result.model, frame, use_neighborhood=settings.use_neighborhood
        )

    raw = _raw_feature_frame(frame, feature_cols)
    target = pd.to_numeric(frame[model_config.target], errors="coerce")
    usable = [i for i in labeled_idx if target.loc[i] == target.loc[i]]
    if len(usable) < max(10, n_splits * 2):
        result = train_price_model(frame, model_config, settings)
        return predict_price_per_m2(
            result.model, frame, use_neighborhood=settings.use_neighborhood
        )

    kf = KFold(
        n_splits=min(n_splits, len(usable) // 2),
        shuffle=True,
        random_state=settings.random_state,
    )
    usable_arr = np.asarray(usable)
    params = _default_params(model_config)
    for train_pos, test_pos in kf.split(usable_arr):
        train_ids = usable_arr[train_pos]
        test_ids = usable_arr[test_pos]
        medians = (
            raw
            .loc[
                train_ids,
                [c for c in feature_cols if c not in CATEGORICAL_FEATURE_COLUMNS],
            ]
            .apply(pd.to_numeric, errors="coerce")
            .median(numeric_only=True)
        )
        x_tr = _impute_with_medians(raw.loc[train_ids], medians, feature_cols)
        x_te = _impute_with_medians(raw.loc[test_ids], medians, feature_cols)
        y_tr = target.loc[train_ids]
        fold_model = _fit_lgbm(x_tr, y_tr, params, settings.random_state)
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in x_te.columns:
                x_te[col] = x_te[col].astype(str).fillna("unknown").astype("category")
        fold_pred = fold_model.predict(x_te)
        preds[frame.index.get_indexer(test_ids)] = np.asarray(fold_pred, dtype=float)

    # Non-label rows: fit on all labels, score once
    other_ids = [i for i in frame.index if i not in set(usable)]
    if other_ids:
        medians = (
            raw
            .loc[
                usable,
                [c for c in feature_cols if c not in CATEGORICAL_FEATURE_COLUMNS],
            ]
            .apply(pd.to_numeric, errors="coerce")
            .median(numeric_only=True)
        )
        x_all = _impute_with_medians(raw.loc[usable], medians, feature_cols)
        y_all = target.loc[usable]
        full_model = _fit_lgbm(x_all, y_all, params, settings.random_state)
        x_other = _impute_with_medians(raw.loc[other_ids], medians, feature_cols)
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in x_other.columns:
                x_other[col] = (
                    x_other[col].astype(str).fillna("unknown").astype("category")
                )
        other_pred = full_model.predict(x_other)
        preds[frame.index.get_indexer(other_ids)] = np.asarray(other_pred, dtype=float)

    # Fill any remaining NaN (failed folds) with full-model on labels
    missing = np.isnan(preds)
    if missing.any():
        result = train_price_model(frame, model_config, settings)
        fill = predict_price_per_m2(
            result.model, frame, use_neighborhood=settings.use_neighborhood
        )
        preds[missing] = fill[missing]
    del work
    return preds


def shap_explanation_for_row(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    row_index: int = 0,
    *,
    use_neighborhood: bool = True,
) -> shap.Explanation:
    """Compute a TreeExplainer explanation for one row."""
    cols = model_feature_columns(model)
    if not cols:
        cols = active_feature_columns(use_neighborhood=use_neighborhood)
    features = build_feature_matrix(frame, cols)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in features.columns:
            features[col] = (
                features[col].astype(str).fillna("unknown").astype("category")
            )
    explainer = shap.TreeExplainer(model)
    explanation = explainer(features)
    return explanation[row_index]


def run_training_with_mlflow(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    model_path: Path,
    train_settings: TrainSettings | None = None,
    rcn: pd.DataFrame | None = None,
) -> TrainResult:
    """Train with HPO, log parent + nested runs to MLflow, persist booster."""
    settings = train_settings or TrainSettings(
        test_size=config.model.test_size,
        random_state=config.model.random_state,
    )

    if not config.mlflow.enabled:
        result = train_price_model(frame, config.model, settings)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        result.model.booster_.save_model(str(model_path))
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
        result = train_price_model(
            frame,
            config.model,
            settings,
            mlflow_module=tracker._mlflow if config.mlflow.log_metrics else None,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        result.model.booster_.save_model(str(model_path))

        if tracker._mlflow is not None:
            if config.mlflow.log_params:
                tracker._mlflow.log_params({
                    **{f"best_{k}": v for k, v in result.best_params.items()},
                    "opt_metric": result.opt_metric,
                    "opt_direction": metric_direction(result.opt_metric),
                    "val_size": settings.val_size,
                    "test_size": settings.test_size,
                    "hpo_trials": settings.hpo_trials,
                    "use_neighborhood": settings.use_neighborhood,
                    "features": ",".join(result.feature_names),
                    "n_features": len(result.feature_names),
                })
            if config.mlflow.log_metrics:
                safe_metrics = {
                    k: v
                    for k, v in result.metrics.items()
                    if isinstance(v, int | float) and np.isfinite(v)
                }
                tracker._mlflow.log_metrics(safe_metrics)
            tracker._mlflow.set_tags({
                "model.family": "lightgbm",
                "model.task": "price_per_m2",
                "split.scheme": "train_val_test",
                "pipeline.multi_model": "lgbm+rcn_neighborhood_baseline",
            })
            if config.mlflow.log_artifacts:
                tracker._mlflow.log_artifact(str(model_path))
                importance = {
                    name: float(score)
                    for name, score in zip(
                        result.feature_names,
                        result.model.feature_importances_,
                        strict=True,
                    )
                }
                tracker._mlflow.log_dict(importance, "feature_importance.json")
                tracker._mlflow.log_dict(
                    {
                        "best_params": result.best_params,
                        "val_metrics": result.val_metrics,
                        "test_metrics": result.test_metrics,
                        "baseline_metrics": result.baseline_metrics,
                        "train_settings": asdict(settings),
                    },
                    "training_summary.json",
                )
            # Explicit baseline comparison run when RCN comps exist.
            if (
                rcn is not None
                and not rcn.empty
                and settings.use_neighborhood
                and tracker._mlflow is not None
            ):
                with tracker._mlflow.start_run(
                    run_name="neighborhood-idw-baseline", nested=True
                ):
                    preds = idw_neighborhood_baseline(frame, rcn)
                    y = pd.to_numeric(frame[config.model.target], errors="coerce")
                    mask = y.notna() & np.isfinite(preds)
                    base = compute_regression_metrics(
                        y.loc[mask].to_numpy(), preds[mask.to_numpy()]
                    )
                    tracker._mlflow.log_metrics(base)
                    tracker._mlflow.set_tag("stage", "baseline")
                    tracker._mlflow.set_tag("model.family", "rcn_idw")
    finally:
        tracker.close()
    return result
