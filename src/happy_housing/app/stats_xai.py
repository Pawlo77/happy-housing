"""Model explainability helpers for the Stats → Model tab."""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

from happy_housing.models.train import (
    CATEGORICAL_FEATURE_COLUMNS,
    build_feature_matrix,
    model_feature_columns,
)

logger = logging.getLogger(__name__)


def booster_importance_table(
    booster: lgb.Booster,
    *,
    top_n: int = 25,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """LightGBM feature importance (gain/split) as a sorted table."""
    names = list(booster.feature_name())
    if not names:
        return pd.DataFrame(columns=["feature", "importance", "share"])
    values = np.asarray(
        booster.feature_importance(importance_type=importance_type),
        dtype=float,
    )
    if values.size != len(names):
        return pd.DataFrame(columns=["feature", "importance", "share"])
    out = pd.DataFrame({"feature": names, "importance": values})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    total = float(out["importance"].sum()) or 1.0
    out["share"] = out["importance"] / total
    return out.head(max(1, int(top_n))).reset_index(drop=True)


def sample_frame_for_shap(
    frame: pd.DataFrame,
    *,
    sample_n: int = 300,
    random_state: int = 42,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Cap rows for global SHAP so Stats stays interactive."""
    if frame.empty or sample_n <= 0:
        return frame.copy()
    work = frame
    if feature_columns is not None:
        keep = [c for c in feature_columns if c in frame.columns]
        if keep:
            work = frame.loc[:, keep]
    if len(work) <= sample_n:
        return work.reset_index(drop=True)
    return work.sample(n=int(sample_n), random_state=int(random_state)).reset_index(
        drop=True
    )


def rank_mean_abs_shap(
    *,
    feature_names: list[str] | tuple[str, ...],
    shap_values: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP contribution."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])
    mean_abs = np.nanmean(np.abs(values), axis=0)
    mean_signed = np.nanmean(values, axis=0)
    out = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    })
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out.head(max(1, int(top_n))).reset_index(drop=True)


def mean_abs_shap_table(
    booster: lgb.Booster,
    explainer: object,
    frame: pd.DataFrame,
    *,
    sample_n: int = 300,
    top_n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Global mean |SHAP| on a sampled lens for Stats Model tab."""
    empty = pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])
    if frame.empty:
        return empty
    cols = model_feature_columns(booster)
    if not cols:
        return empty
    listing_fp = ""
    if "listing_id" in frame.columns:
        listing_fp = str(int(pd.util.hash_pandas_object(frame["listing_id"]).sum()))
    else:
        listing_fp = f"n={len(frame)}"
    return _cached_mean_abs_shap_table(
        listing_fp,
        int(sample_n),
        int(top_n),
        int(random_state),
        booster,
        explainer,
        frame,
        tuple(cols),
    )


def _cached_mean_abs_shap_table(
    listing_fp: str,
    sample_n: int,
    top_n: int,
    random_state: int,
    booster: lgb.Booster,
    explainer: object,
    frame: pd.DataFrame,
    cols: tuple[str, ...],
) -> pd.DataFrame:
    """Session-memoized SHAP summary keyed by lens fingerprint + sample size."""
    cache_key = ("stats_mean_abs_shap", listing_fp, sample_n, top_n, random_state, cols)
    cached = st.session_state.get("_stats_shap_summary_key")
    if cached == cache_key:
        payload = st.session_state.get("_stats_shap_summary_val")
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
    empty = pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])
    sample = sample_frame_for_shap(
        frame,
        sample_n=sample_n,
        random_state=random_state,
        feature_columns=cols,
    )
    try:
        features = build_feature_matrix(sample, cols)
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in features.columns:
                features[col] = (
                    features[col].astype(str).fillna("unknown").astype("category")
                )
        explanation = explainer(features)  # type: ignore[operator]  # SHAP union is callable
        values = np.asarray(explanation.values, dtype=float)
        names = list(features.columns)
        out = rank_mean_abs_shap(feature_names=names, shap_values=values, top_n=top_n)
    except Exception as exc:
        logger.warning("Global SHAP summary failed: %s", exc)
        out = empty
    st.session_state["_stats_shap_summary_key"] = cache_key
    st.session_state["_stats_shap_summary_val"] = out
    return out.copy()
