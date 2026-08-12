"""Derived listing features for the price model (static + train-fold stats)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENGINEERED_STATIC_COLUMNS: tuple[str, ...] = (
    "area_per_room",
    "floor_ratio",
    "building_age",
    "is_ground_floor",
    "is_top_floor",
    "log_area_m2",
    "log_rooms",
    "log_commute_min",
    "log_transit_commute_min",
    "log_bike_commute_min",
    "log_walk_commute_min",
    "log_dist_metro_m",
    "log_dist_transit_stop_m",
    "log_dist_park_m",
    "log_dist_shop_m",
    "log_dist_daszynskiego_m",
    "rooms_x_area",
    "floor_x_area",
    "listed_month",
    "listed_year",
    "listed_dow",
    "commute_vs_straight_ratio",
)
"""Target-free engineered columns (safe to compute on the full frame)."""

ENGINEERED_TARGET_COLUMNS: tuple[str, ...] = (
    "district_train_median_ppm",
    "district_train_mean_ppm",
    "district_train_n",
    "nbhd_minus_district_ppm",
)
"""Train-fold district price stats (must be fit on train labels only)."""

ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    ENGINEERED_STATIC_COLUMNS + ENGINEERED_TARGET_COLUMNS
)
"""All engineered columns included when feature engineering is enabled."""

_CURRENT_YEAR: int = 2026
"""Reference year for ``building_age`` (pipeline year)."""


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def attach_static_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interaction / log / calendar features (no label leakage)."""
    out = frame.copy()
    area = _num(out, "area_m2")
    rooms = _num(out, "rooms")
    floor = _num(out, "floor_no")
    floors = _num(out, "building_floors_num")
    year = _num(out, "year_built")
    out["area_per_room"] = area / rooms.replace(0.0, np.nan)
    out["floor_ratio"] = floor / floors.replace(0.0, np.nan)
    out["building_age"] = float(_CURRENT_YEAR) - year
    out["is_ground_floor"] = (floor.fillna(-1) <= 0).astype(float)
    out["is_top_floor"] = (
        floors.notna() & floor.notna() & (floor >= floors - 0.5)
    ).astype(float)
    out["log_area_m2"] = np.log1p(area.clip(lower=0.0))
    out["log_rooms"] = np.log1p(rooms.clip(lower=0.0))
    out["log_commute_min"] = np.log1p(_num(out, "commute_min").clip(lower=0.0))
    out["log_transit_commute_min"] = np.log1p(
        _num(out, "transit_commute_min").clip(lower=0.0)
    )
    out["log_bike_commute_min"] = np.log1p(
        _num(out, "bike_commute_min").clip(lower=0.0)
    )
    out["log_walk_commute_min"] = np.log1p(
        _num(out, "walk_commute_min").clip(lower=0.0)
    )
    out["log_dist_metro_m"] = np.log1p(_num(out, "dist_metro_m").clip(lower=0.0))
    out["log_dist_transit_stop_m"] = np.log1p(
        _num(out, "dist_transit_stop_m").clip(lower=0.0)
    )
    out["log_dist_park_m"] = np.log1p(_num(out, "dist_park_m").clip(lower=0.0))
    out["log_dist_shop_m"] = np.log1p(_num(out, "dist_shop_m").clip(lower=0.0))
    out["log_dist_daszynskiego_m"] = np.log1p(
        _num(out, "dist_daszynskiego_m").clip(lower=0.0)
    )
    out["rooms_x_area"] = rooms * area
    out["floor_x_area"] = floor * area
    if "listed_at" in out.columns:
        listed = pd.to_datetime(out["listed_at"], errors="coerce", utc=True)
        out["listed_month"] = listed.dt.month.astype(float)
        out["listed_year"] = listed.dt.year.astype(float)
        out["listed_dow"] = listed.dt.dayofweek.astype(float)
    else:
        out["listed_month"] = np.nan
        out["listed_year"] = np.nan
        out["listed_dow"] = np.nan
    straight_min = _num(out, "dist_daszynskiego_m") / 1000.0 / 30.0 * 60.0
    commute = _num(out, "commute_min")
    out["commute_vs_straight_ratio"] = commute / straight_min.replace(0.0, np.nan)
    return out


def fit_district_ppm_stats(
    districts: pd.Series,
    price_per_m2: pd.Series,
) -> dict[str, dict[str, float]]:
    """Per-district median/mean/count from train-fold labels only."""
    work = pd.DataFrame({
        "district": districts.astype(str).fillna("unknown"),
        "ppm": pd.to_numeric(price_per_m2, errors="coerce"),
    }).dropna(subset=["ppm"])
    if work.empty:
        return {}
    grouped = work.groupby("district", sort=False)["ppm"]
    stats: dict[str, dict[str, float]] = {}
    for name, series in grouped:
        stats[str(name)] = {
            "median": float(series.median()),
            "mean": float(series.mean()),
            "n": float(len(series)),
        }
    return stats


def apply_district_ppm_stats(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    *,
    global_median: float | None = None,
    global_mean: float | None = None,
) -> pd.DataFrame:
    """Map train-fold district price stats onto rows (unknown → global)."""
    out = frame.copy()
    districts = (
        out["district"].astype(str).fillna("unknown")
        if "district" in out.columns
        else pd.Series("unknown", index=out.index)
    )
    medians: list[float] = []
    means: list[float] = []
    counts: list[float] = []
    fallback_med = (
        float(global_median)
        if global_median is not None and global_median == global_median
        else float("nan")
    )
    fallback_mean = (
        float(global_mean)
        if global_mean is not None and global_mean == global_mean
        else fallback_med
    )
    for name in districts:
        row = stats.get(str(name))
        if row is None:
            medians.append(fallback_med)
            means.append(fallback_mean)
            counts.append(0.0)
        else:
            medians.append(float(row["median"]))
            means.append(float(row["mean"]))
            counts.append(float(row["n"]))
    out["district_train_median_ppm"] = medians
    out["district_train_mean_ppm"] = means
    out["district_train_n"] = counts
    nbhd = _num(out, "nbhd_rcn_idw_ppm")
    out["nbhd_minus_district_ppm"] = nbhd - out["district_train_median_ppm"]
    return out
