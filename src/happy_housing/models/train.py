"""LightGBM price model training, HPO, SHAP, and MLflow logging."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from happy_housing.gis.amenity_catalog import model_distance_feature_columns
from happy_housing.models.feature_eng import (
    ENGINEERED_FEATURE_COLUMNS,
    ENGINEERED_TARGET_COLUMNS,
    apply_district_ppm_stats,
    attach_static_engineered_features,
    fit_district_ppm_stats,
)
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
from happy_housing.vision.photo_features import PHOTO_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_STRUCTURAL_FEATURE_COLUMNS: tuple[str, ...] = (
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
    "road_lnight_db",
    "rail_lden_db",
    "rail_lnight_db",
    "tram_lden_db",
    "tram_lnight_db",
    "industry_lden_db",
    "industry_lnight_db",
    "noise_total_lden_db",
    "noise_total_lnight_db",
    "in_high_noise",
    "dist_air_station_m",
    "air_aq_index",
    "air_pm10_ug_m3",
    "air_pm25_ug_m3",
    "air_no2_ug_m3",
    "in_poor_air",
    "dist_transit_stop_m",
    "dist_daszynskiego_m",
    "commute_min",
    "transit_commute_min",
    "bike_commute_min",
    "walk_commute_min",
    "dist_gunb_m",
    "dist_bike_path_m",
    "bike_path_coverage_m",
    "dist_rail_area_m",
    "dist_military_m",
    "dist_airport_area_m",
    "dist_traffic_road_m",
    "traffic_weight_nearby",
    "dist_major_road_m",
    "major_traffic_weight_nearby",
    "nearest_highway",
    "street_facing_hint",
    "quiet_side_hint",
    "crime_rate_per_1000",
    "crime_criminal_rate_per_1000",
    "crime_detectability_pct",
)
"""Structural + noise + air + transit + zone + traffic + bike + crime features."""

CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    _STRUCTURAL_FEATURE_COLUMNS + model_distance_feature_columns()
)
"""Structural + GIS amenity distances (always included)."""

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "building_type",
    "heating",
    "construction_status",
    "building_material",
    "district",
    "nearest_highway",
)
"""Otodom string attributes encoded as LightGBM categories."""

FEATURE_COLUMNS: tuple[str, ...] = (
    CORE_FEATURE_COLUMNS
    + NEIGHBORHOOD_FEATURE_COLUMNS
    + PHOTO_FEATURE_COLUMNS
    + ENGINEERED_FEATURE_COLUMNS
)
"""Full default feature list including neighborhood, photo CV, engineered."""

CORE_PLUS_PHOTO: tuple[str, ...] = CORE_FEATURE_COLUMNS + PHOTO_FEATURE_COLUMNS
"""Core + photo without neighborhood / engineered (tests / slim runs)."""

_HPO_SPACE: dict[str, list[Any]] = {
    "n_estimators": [400, 800, 1200, 2000, 3000],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [31, 63, 127, 255, 511],
    "max_depth": [-1, 6, 8, 10, 12],
    "min_child_samples": [5, 10, 20, 40, 80],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0, 5.0],
    "min_split_gain": [0.0, 0.01, 0.05, 0.1],
    "objective": ["regression", "mae", "huber"],
}
"""Expanded LightGBM search space (deep trees + L1/Huber for MAPE-ish fits)."""

_LABEL_SOURCES_OTODOM: frozenset[str] = frozenset({"otodom", "olx", "fixture"})
"""Sources treated as ask-price labels when ``train_otodom_labels_only``."""

_LABEL_SOURCES_BUYER: frozenset[str] = frozenset({"otodom", "olx", "fixture"})
"""Buyer-facing ask sources for recommender default pool."""


@dataclass(frozen=True, slots=True)
class TrainSettings:
    """Runtime training knobs (Admin / app settings + experiment defaults)."""

    opt_metric: OptMetric = "mape"
    """Metric optimized on the validation fold during HPO."""
    val_size: float = 0.15
    """Fraction of non-test rows held out for HPO validation."""
    test_size: float = 0.2
    """Final hold-out fraction (never used for HPO selection)."""
    hpo_trials: int = 80
    """Number of random hyperparameter samples (1 = no search)."""
    use_neighborhood: bool = True
    """Include RCN neighborhood features when present on the frame."""
    use_engineered: bool = True
    """Include interaction / log / district train-fold engineered features."""
    random_state: int = 42
    """RNG seed for splits and HPO sampling."""
    train_otodom_labels_only: bool = True
    """Drop RCN deed rows from training labels (comps still OK as features)."""
    log_target: bool = True
    """Train on ``log1p(price_per_m2)``; report metrics in zł/m²."""
    district_calibration: bool = True
    """Add district median residual offsets after the final fit."""
    early_stopping_rounds: int = 80
    """LightGBM early stopping patience on the HPO validation fold."""


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
    log_target: bool = True
    """Whether the booster predicts log1p space."""
    district_offsets: dict[str, float] = field(default_factory=dict)
    """Additive zł/m² correction by district (post-hoc)."""
    district_ppm_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    """Train-fold district median/mean/n for engineered features."""
    district_global_median: float | None = None
    """Train-fold global median zł/m² fallback for unknown districts."""
    district_global_mean: float | None = None
    """Train-fold global mean zł/m² fallback for unknown districts."""
    use_engineered: bool = True
    """Whether engineered columns were used for this model."""


def _optional_commute_flags(frame: pd.DataFrame) -> tuple[bool, bool, bool]:
    """Whether transit / bike / walk columns have any usable values."""

    def _any(col: str) -> bool:
        if col not in frame.columns:
            return False
        return bool(pd.to_numeric(frame[col], errors="coerce").notna().any())

    return (
        _any("transit_commute_min"),
        _any("bike_commute_min"),
        _any("walk_commute_min"),
    )


def active_feature_columns(
    *,
    use_neighborhood: bool = True,
    include_transit: bool = True,
    include_bike: bool = True,
    include_walk: bool = True,
    use_engineered: bool = True,
) -> tuple[str, ...]:
    """Feature list depending on neighborhood / commute-mode / engineered flags."""
    cols: tuple[str, ...]
    if use_neighborhood:
        cols = (
            CORE_FEATURE_COLUMNS + NEIGHBORHOOD_FEATURE_COLUMNS + PHOTO_FEATURE_COLUMNS
        )
    else:
        cols = CORE_PLUS_PHOTO
    if use_engineered:
        cols = cols + ENGINEERED_FEATURE_COLUMNS
    drop: set[str] = set()
    if not include_transit:
        drop.update({"transit_commute_min", "log_transit_commute_min"})
    if not include_bike:
        drop.update({"bike_commute_min", "log_bike_commute_min"})
    if not include_walk:
        drop.update({"walk_commute_min", "log_walk_commute_min"})
    if not drop:
        return cols
    return tuple(c for c in cols if c not in drop)


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
    """Raise when feature columns are absent or lack usable signal.

    Missing columns are a hard error (no silent invent / zero-impute of
    whole features). Row-level NaNs inside present columns are OK.
    """
    del max_missing_fraction  # kept for call-site compat
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Model features absent on frame: {missing}")
    usable = 0
    for col in feature_columns:
        if col in CATEGORICAL_FEATURE_COLUMNS:
            if frame[col].notna().any():
                usable += 1
            continue
        if pd.to_numeric(frame[col], errors="coerce").notna().any():
            usable += 1
    if usable < min_usable:
        raise ValueError(
            f"Feature coverage too low: only {usable} usable columns "
            f"(need ≥{min_usable})"
        )


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


def _default_params(model_config: ModelConfig) -> dict[str, Any]:
    return {
        "n_estimators": model_config.n_estimators,
        "learning_rate": model_config.learning_rate,
        "num_leaves": model_config.num_leaves,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "reg_alpha": 0.0,
        "min_split_gain": 0.0,
        "objective": "regression",
    }


def prepare_engineered_frame(
    frame: pd.DataFrame,
    *,
    use_engineered: bool,
    district_ppm_stats: dict[str, dict[str, float]] | None = None,
    global_median: float | None = None,
    global_mean: float | None = None,
) -> pd.DataFrame:
    """Attach static + district engineered columns when enabled."""
    if not use_engineered:
        return frame
    out = attach_static_engineered_features(frame)
    return apply_district_ppm_stats(
        out,
        district_ppm_stats or {},
        global_median=global_median,
        global_mean=global_mean,
    )


def _prepare_cats(features: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical columns to stable string codes for LightGBM."""
    work = features.copy()
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col not in work.columns:
            continue
        # Use plain strings — avoids pandas category-level mismatch on predict.
        work[col] = work[col].astype(str).fillna("unknown")
    return work


def _fit_lgbm(
    features_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    random_state: int,
    *,
    n_jobs: int = 1,
    features_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_rounds: int = 0,
) -> lgb.LGBMRegressor:
    work = _prepare_cats(features_train)
    cat_in_frame = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in work.columns]
    for col in cat_in_frame:
        work[col] = work[col].astype("category")
    model = lgb.LGBMRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params.get("max_depth", -1)),
        min_child_samples=int(params.get("min_child_samples", 20)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        reg_lambda=float(params.get("reg_lambda", 0.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        min_split_gain=float(params.get("min_split_gain", 0.0)),
        objective=str(params.get("objective", "regression")),
        random_state=random_state,
        n_jobs=int(n_jobs),
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {
        "categorical_feature": cat_in_frame or "auto",
    }
    if (
        features_val is not None
        and y_val is not None
        and int(early_stopping_rounds) > 0
    ):
        val_work = _prepare_cats(features_val)
        for col in cat_in_frame:
            if col not in val_work.columns:
                continue
            val_work[col] = pd.Categorical(
                val_work[col],
                categories=list(work[col].cat.categories),
            )
        fit_kwargs["eval_X"] = val_work
        fit_kwargs["eval_y"] = y_val
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(period=0),
        ]
    model.fit(work, y_train, **fit_kwargs)
    model._hh_category_levels = {
        col: list(work[col].cat.categories) for col in cat_in_frame
    }
    return model


def _predict_raw(
    model: lgb.LGBMRegressor | lgb.Booster, features: pd.DataFrame
) -> np.ndarray:
    """Predict after aligning categorical columns to the booster's levels."""
    work = _prepare_cats(features)
    cat_in_frame = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in work.columns]
    if isinstance(model, lgb.LGBMRegressor) and hasattr(model, "booster_"):
        # Reuse training category levels when available.
        train_cats = getattr(model, "_hh_category_levels", None)
        if isinstance(train_cats, dict):
            for col, levels in train_cats.items():
                if col in work.columns:
                    work[col] = pd.Categorical(work[col], categories=levels)
        else:
            for col in cat_in_frame:
                work[col] = work[col].astype("category")
    else:
        for col in cat_in_frame:
            work[col] = work[col].astype("category")
    return np.asarray(model.predict(work), dtype=float)


def _to_train_target(y: pd.Series | np.ndarray, log_target: bool) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.log1p(np.clip(arr, 0.0, None)) if log_target else arr


def _from_model_pred(pred: np.ndarray, log_target: bool) -> np.ndarray:
    arr = np.asarray(pred, dtype=float)
    return np.expm1(arr) if log_target else arr


def fit_district_offsets(
    districts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Median residual (true - pred) per district for additive calibration."""
    frame = pd.DataFrame({
        "district": districts.astype(str).fillna("unknown").to_numpy(),
        "resid": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
    })
    grouped = frame.groupby("district", dropna=False)["resid"].median()
    return {str(k): float(v) for k, v in grouped.items() if v == v}


def apply_district_offsets(
    pred: np.ndarray,
    districts: pd.Series | None,
    offsets: dict[str, float] | None,
) -> np.ndarray:
    """Add district median residual offsets to predictions."""
    out = np.asarray(pred, dtype=float).copy()
    if not offsets or districts is None or len(districts) != len(out):
        return out
    keys = districts.astype(str).fillna("unknown").to_numpy()
    for i, key in enumerate(keys):
        out[i] += float(offsets.get(str(key), 0.0))
    return out


def model_meta_path(model_path: Path) -> Path:
    """Sidecar JSON next to the LightGBM text model."""
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def save_model_meta(
    model_path: Path,
    *,
    log_target: bool,
    district_offsets: dict[str, float],
    district_ppm_stats: dict[str, dict[str, float]] | None = None,
    district_global_median: float | None = None,
    district_global_mean: float | None = None,
    use_engineered: bool = True,
) -> None:
    """Persist prediction-space metadata beside the booster."""
    import json

    path = model_meta_path(model_path)
    path.write_text(
        json.dumps(
            {
                "log_target": bool(log_target),
                "district_offsets": {
                    str(k): float(v) for k, v in district_offsets.items()
                },
                "district_ppm_stats": {
                    str(district): {
                        str(stat_k): float(stat_v) for stat_k, stat_v in stats.items()
                    }
                    for district, stats in (district_ppm_stats or {}).items()
                },
                "district_global_median": (
                    None
                    if district_global_median is None
                    else float(district_global_median)
                ),
                "district_global_mean": (
                    None
                    if district_global_mean is None
                    else float(district_global_mean)
                ),
                "use_engineered": bool(use_engineered),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_model_meta(model_path: Path) -> dict[str, Any]:
    """Load prediction-space metadata; missing → linear space (legacy boosters)."""
    import json

    path = model_meta_path(model_path)
    empty: dict[str, Any] = {
        "log_target": False,
        "district_offsets": {},
        "district_ppm_stats": {},
        "district_global_median": None,
        "district_global_mean": None,
        "use_engineered": False,
    }
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    offsets = payload.get("district_offsets") or {}
    if not isinstance(offsets, dict):
        offsets = {}
    raw_stats = payload.get("district_ppm_stats") or {}
    stats: dict[str, dict[str, float]] = {}
    if isinstance(raw_stats, dict):
        for district, row in raw_stats.items():
            if not isinstance(row, dict):
                continue
            stats[str(district)] = {str(k): float(v) for k, v in row.items() if v == v}
    g_med = payload.get("district_global_median")
    g_mean = payload.get("district_global_mean")
    return {
        "log_target": bool(payload.get("log_target", False)),
        "district_offsets": {str(k): float(v) for k, v in offsets.items()},
        "district_ppm_stats": stats,
        "district_global_median": (None if g_med is None else float(g_med)),
        "district_global_mean": (None if g_mean is None else float(g_mean)),
        "use_engineered": bool(payload.get("use_engineered", False)),
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
    work = frame
    if settings.train_otodom_labels_only and "source" in frame.columns:
        sources = frame["source"].astype(str).str.lower()
        work = frame.loc[sources.isin(_LABEL_SOURCES_OTODOM)].copy()
        if "listing_kind" in work.columns:
            kinds = work["listing_kind"].astype(str).str.lower().fillna("sale")
            work = work.loc[kinds != "rent"].copy()
        logger.info(
            "Train labels filtered to Otodom/OLX/fixture (sale): %s → %s rows",
            len(frame),
            len(work),
        )

    include_transit, include_bike, include_walk = _optional_commute_flags(work)
    feature_cols = active_feature_columns(
        use_neighborhood=settings.use_neighborhood,
        include_transit=include_transit,
        include_bike=include_bike,
        include_walk=include_walk,
        use_engineered=settings.use_engineered,
    )
    if model_config.target not in work.columns:
        raise ValueError(f"Target column missing: {model_config.target}")

    district_ppm_stats: dict[str, dict[str, float]] = {}
    district_global_median: float | None = None
    district_global_mean: float | None = None
    if settings.use_engineered:
        work = attach_static_engineered_features(work)
        for col in ENGINEERED_TARGET_COLUMNS:
            if col not in work.columns:
                work[col] = np.nan

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

    if settings.use_engineered:
        train_districts = (
            work.loc[train_b.features.index, "district"]
            if "district" in work.columns
            else pd.Series("unknown", index=train_b.features.index)
        )
        district_ppm_stats = fit_district_ppm_stats(train_districts, train_b.target)
        district_global_median = float(train_b.target.median())
        district_global_mean = float(train_b.target.mean())
        work = apply_district_ppm_stats(
            work,
            district_ppm_stats,
            global_median=district_global_median,
            global_mean=district_global_mean,
        )
        raw_features = _raw_feature_frame(work.loc[raw_features.index], feature_cols)
        train_b = SplitBundle(
            raw_features.loc[train_b.features.index],
            train_b.target,
        )
        val_b = SplitBundle(
            raw_features.loc[val_b.features.index],
            val_b.target,
        )
        test_b = SplitBundle(
            raw_features.loc[test_b.features.index],
            test_b.target,
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
    log_target = bool(settings.log_target)
    y_train_fit = pd.Series(
        _to_train_target(y_train, log_target=log_target), index=y_train.index
    )
    y_val_fit = pd.Series(
        _to_train_target(y_val, log_target=log_target), index=y_val.index
    )
    y_tv_fit = pd.Series(
        _to_train_target(y_tv, log_target=log_target), index=y_tv.index
    )

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
    early_rounds = max(0, int(settings.early_stopping_rounds))

    def _eval_trial(
        trial_i: int, params: dict[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, float], float, int]:
        # Cap OpenMP threads per trial so parallel HPO does not oversubscribe.
        model = _fit_lgbm(
            features_train,
            y_train_fit,
            params,
            settings.random_state,
            n_jobs=1,
            features_val=features_val,
            y_val=y_val_fit,
            early_stopping_rounds=early_rounds,
        )
        val_pred = _from_model_pred(
            _predict_raw(model, features_val), log_target=log_target
        )
        val_metrics = compute_regression_metrics(y_val.to_numpy(), val_pred)
        score = float(val_metrics[opt_metric])
        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
        if best_iter <= 0:
            best_iter = int(params["n_estimators"])
        return trial_i, params, val_metrics, score, best_iter

    workers = min(4, len(candidates), max(1, (os.cpu_count() or 2)))
    trial_results: list[tuple[int, dict[str, Any], dict[str, float], float, int]] = []
    if workers <= 1 or len(candidates) == 1:
        trial_results = [_eval_trial(i, p) for i, p in enumerate(candidates)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_eval_trial, i, p) for i, p in enumerate(candidates)]
            for fut in as_completed(futures):
                trial_results.append(fut.result())
        trial_results.sort(key=lambda row: row[0])

    for trial_i, params, val_metrics, score, best_iter in trial_results:
        logger.info(
            "HPO trial %s/%s %s=%s n_trees=%s params=%s",
            trial_i + 1,
            len(candidates),
            opt_metric,
            score,
            best_iter,
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
            best_params = {**params, "n_estimators": max(50, best_iter)}
            best_val_metrics = val_metrics

    # Final model: train + val, evaluate on untouched test.
    final_model = _fit_lgbm(
        features_tv, y_tv_fit, best_params, settings.random_state, n_jobs=-1
    )
    test_pred = _from_model_pred(
        _predict_raw(final_model, features_test), log_target=log_target
    )
    district_offsets: dict[str, float] = {}
    if settings.district_calibration and "district" in features_tv.columns:
        tv_pred = _from_model_pred(
            _predict_raw(final_model, features_tv), log_target=log_target
        )
        district_offsets = fit_district_offsets(
            features_tv["district"], y_tv.to_numpy(), tv_pred
        )
        test_pred = apply_district_offsets(
            test_pred, features_test.get("district"), district_offsets
        )
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
        "log_target": 1.0 if log_target else 0.0,
        "district_calibration": 1.0 if district_offsets else 0.0,
        "use_engineered": 1.0 if settings.use_engineered else 0.0,
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
        log_target=log_target,
        district_offsets=district_offsets,
        district_ppm_stats=district_ppm_stats,
        district_global_median=district_global_median,
        district_global_mean=district_global_mean,
        use_engineered=bool(settings.use_engineered),
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce to float; map NA / NaN / bad values to ``default``."""
    try:
        if value is None or pd.isna(value):
            return default
        number = float(value)  # type: ignore[arg-type]  # pandas/object scalar
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _raw_feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Feature frame with NaNs preserved (no impute) for split-then-impute.

    Raises:
        ValueError: If any requested column is missing from ``frame``.
    """
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Model features absent on frame: {missing}")
    work = frame
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
    model: lgb.LGBMRegressor | lgb.Booster,
    frame: pd.DataFrame,
    *,
    use_neighborhood: bool = True,
    log_target: bool = True,
    district_offsets: dict[str, float] | None = None,
    use_engineered: bool = True,
    district_ppm_stats: dict[str, dict[str, float]] | None = None,
    district_global_median: float | None = None,
    district_global_mean: float | None = None,
) -> np.ndarray:
    """Score listings with a fitted model (feature matrix aligned to booster)."""
    cols = model_feature_columns(model)
    if not cols:
        cols = active_feature_columns(
            use_neighborhood=use_neighborhood,
            use_engineered=use_engineered,
        )
    need_eng = any(c in cols for c in ENGINEERED_FEATURE_COLUMNS)
    work = prepare_engineered_frame(
        frame,
        use_engineered=need_eng or use_engineered,
        district_ppm_stats=district_ppm_stats,
        global_median=district_global_median,
        global_mean=district_global_mean,
    )
    features = build_feature_matrix(work, cols)
    # Align category levels for LightGBM
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in features.columns:
            features[col] = (
                features[col].astype(str).fillna("unknown").astype("category")
            )
    if isinstance(model, lgb.Booster):
        raw = _predict_raw(model, features)
    else:
        raw = _predict_raw(model, features)
    pred = _from_model_pred(raw, log_target=log_target)
    return apply_district_offsets(pred, features.get("district"), district_offsets)


def oof_predict_price_per_m2(
    frame: pd.DataFrame,
    model_config: ModelConfig,
    train_settings: TrainSettings,
    *,
    n_splits: int = 5,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Out-of-fold predictions for train-label rows; final-model style for others.

    Rows excluded from training (e.g. RCN when Otodom-only) are scored with a
    model fit on all train labels so Explore deltas stay honest on asks.
    """
    from sklearn.model_selection import KFold

    settings = train_settings
    include_transit, include_bike, include_walk = _optional_commute_flags(frame)
    feature_cols = active_feature_columns(
        use_neighborhood=settings.use_neighborhood,
        include_transit=include_transit,
        include_bike=include_bike,
        include_walk=include_walk,
        use_engineered=settings.use_engineered,
    )
    preds = np.full(len(frame), np.nan, dtype=float)
    label_mask = pd.Series(True, index=frame.index)
    if settings.train_otodom_labels_only and "source" in frame.columns:
        sources = frame["source"].astype(str).str.lower()
        label_mask = sources.isin(_LABEL_SOURCES_OTODOM)
        if "listing_kind" in frame.columns:
            kinds = frame["listing_kind"].astype(str).str.lower().fillna("sale")
            label_mask = label_mask & (kinds != "rent")

    labeled_idx = frame.index[label_mask].to_list()

    def _predict_bundle(
        result: TrainResult,
    ) -> np.ndarray:
        return predict_price_per_m2(
            result.model,
            frame,
            use_neighborhood=settings.use_neighborhood,
            log_target=result.log_target,
            district_offsets=result.district_offsets,
            use_engineered=result.use_engineered,
            district_ppm_stats=result.district_ppm_stats,
            district_global_median=result.district_global_median,
            district_global_mean=result.district_global_mean,
        )

    if len(labeled_idx) < max(10, n_splits * 2):
        # Too small for OOF — fall back to in-sample after a quick fit
        result = train_price_model(frame, model_config, settings)
        return _predict_bundle(result)

    work = frame
    if settings.use_engineered:
        work = attach_static_engineered_features(work)
        for col in ENGINEERED_TARGET_COLUMNS:
            if col not in work.columns:
                work[col] = np.nan

    target = pd.to_numeric(work[model_config.target], errors="coerce")
    usable = [i for i in labeled_idx if target.loc[i] == target.loc[i]]
    if len(usable) < max(10, n_splits * 2):
        result = train_price_model(frame, model_config, settings)
        return _predict_bundle(result)

    log_target = bool(settings.log_target)
    kf = KFold(
        n_splits=min(n_splits, len(usable) // 2),
        shuffle=True,
        random_state=settings.random_state,
    )
    usable_arr = np.asarray(usable)
    fold_params = dict(params) if params else _default_params(model_config)

    def _fold_features(
        train_ids: np.ndarray,
        score_ids: np.ndarray | list[Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        fold_work = work
        if settings.use_engineered:
            districts = (
                work.loc[train_ids, "district"]
                if "district" in work.columns
                else pd.Series("unknown", index=train_ids)
            )
            stats = fit_district_ppm_stats(districts, target.loc[train_ids])
            g_med = float(target.loc[train_ids].median())
            g_mean = float(target.loc[train_ids].mean())
            fold_work = apply_district_ppm_stats(
                work,
                stats,
                global_median=g_med,
                global_mean=g_mean,
            )
        raw = _raw_feature_frame(fold_work, feature_cols)
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
        x_te = _impute_with_medians(raw.loc[score_ids], medians, feature_cols)
        return x_tr, x_te, medians

    for train_pos, test_pos in kf.split(usable_arr):
        train_ids = usable_arr[train_pos]
        test_ids = usable_arr[test_pos]
        x_tr, x_te, _medians = _fold_features(train_ids, test_ids)
        y_tr = pd.Series(
            _to_train_target(target.loc[train_ids], log_target=log_target),
            index=train_ids,
        )
        fold_model = _fit_lgbm(x_tr, y_tr, fold_params, settings.random_state, n_jobs=1)
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in x_te.columns:
                x_te[col] = x_te[col].astype(str).fillna("unknown").astype("category")
        fold_pred = _from_model_pred(
            _predict_raw(fold_model, x_te),
            log_target=log_target,
        )
        preds[frame.index.get_indexer(test_ids)] = fold_pred

    # Non-label rows: fit on all labels, score once
    other_ids = [i for i in frame.index if i not in set(usable)]
    full_model = None
    x_all = None
    if other_ids:
        x_all, x_other, _medians_all = _fold_features(np.asarray(usable), other_ids)
        y_all = pd.Series(
            _to_train_target(target.loc[usable], log_target=log_target),
            index=usable,
        )
        full_model = _fit_lgbm(
            x_all, y_all, fold_params, settings.random_state, n_jobs=-1
        )
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in x_other.columns:
                x_other[col] = (
                    x_other[col].astype(str).fillna("unknown").astype("category")
                )
        other_pred = _from_model_pred(
            _predict_raw(full_model, x_other),
            log_target=log_target,
        )
        preds[frame.index.get_indexer(other_ids)] = other_pred

    # Fill any remaining NaN (failed folds) with full-model on labels
    missing = np.isnan(preds)
    if missing.any():
        result = train_price_model(frame, model_config, settings)
        fill = _predict_bundle(result)
        preds[missing] = fill[missing]

    if settings.district_calibration and "district" in work.columns:
        # Global offsets from a full-label fit (reuse or build).
        if full_model is None or x_all is None:
            x_all, _x_dummy, _medians_all = _fold_features(
                np.asarray(usable), usable[:1]
            )
            y_all = pd.Series(
                _to_train_target(target.loc[usable], log_target=log_target),
                index=usable,
            )
            full_model = _fit_lgbm(
                x_all, y_all, fold_params, settings.random_state, n_jobs=-1
            )
        if full_model is None or x_all is None:
            return preds
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in x_all.columns:
                x_all[col] = x_all[col].astype(str).fillna("unknown").astype("category")
        lab_pred = _from_model_pred(
            _predict_raw(full_model, x_all),
            log_target=log_target,
        )
        offsets = fit_district_offsets(
            work.loc[usable, "district"],
            target.loc[usable].to_numpy(),
            lab_pred,
        )
        preds = apply_district_offsets(preds, work["district"], offsets)
    return preds


def shap_explanation_for_row(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    row_index: int = 0,
    *,
    use_neighborhood: bool = True,
    use_engineered: bool = True,
    district_ppm_stats: dict[str, dict[str, float]] | None = None,
    district_global_median: float | None = None,
    district_global_mean: float | None = None,
) -> shap.Explanation:
    """Compute a TreeExplainer explanation for one row."""
    cols = model_feature_columns(model)
    if not cols:
        cols = active_feature_columns(
            use_neighborhood=use_neighborhood,
            use_engineered=use_engineered,
        )
    need_eng = any(c in cols for c in ENGINEERED_FEATURE_COLUMNS)
    work = prepare_engineered_frame(
        frame,
        use_engineered=need_eng or use_engineered,
        district_ppm_stats=district_ppm_stats,
        global_median=district_global_median,
        global_mean=district_global_mean,
    )
    features = build_feature_matrix(work, cols)
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
        save_model_meta(
            model_path,
            log_target=result.log_target,
            district_offsets=result.district_offsets,
            district_ppm_stats=result.district_ppm_stats,
            district_global_median=result.district_global_median,
            district_global_mean=result.district_global_mean,
            use_engineered=result.use_engineered,
        )
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
        save_model_meta(
            model_path,
            log_target=result.log_target,
            district_offsets=result.district_offsets,
            district_ppm_stats=result.district_ppm_stats,
            district_global_median=result.district_global_median,
            district_global_mean=result.district_global_mean,
            use_engineered=result.use_engineered,
        )

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
                    "use_engineered": settings.use_engineered,
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
