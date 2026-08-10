"""Tests for price-model engineered features."""

from __future__ import annotations

import pandas as pd

from happy_housing.models.feature_eng import (
    ENGINEERED_STATIC_COLUMNS,
    ENGINEERED_TARGET_COLUMNS,
    apply_district_ppm_stats,
    attach_static_engineered_features,
    fit_district_ppm_stats,
)


def test_attach_static_engineered_features_adds_all_columns() -> None:
    frame = pd.DataFrame({
        "area_m2": [50.0, 100.0],
        "rooms": [2.0, 4.0],
        "floor_no": [0.0, 5.0],
        "building_floors_num": [4.0, 5.0],
        "year_built": [2000.0, 2015.0],
        "commute_min": [10.0, 20.0],
        "dist_metro_m": [500.0, 1500.0],
        "dist_transit_stop_m": [100.0, 200.0],
        "dist_park_m": [300.0, 400.0],
        "dist_shop_m": [50.0, 80.0],
        "dist_daszynskiego_m": [3000.0, 6000.0],
        "listed_at": ["2026-01-15", "2026-06-01"],
    })
    out = attach_static_engineered_features(frame)
    for col in ENGINEERED_STATIC_COLUMNS:
        assert col in out.columns
    assert float(out["area_per_room"].iloc[0]) == 25.0
    assert float(out["is_ground_floor"].iloc[0]) == 1.0
    assert float(out["is_top_floor"].iloc[1]) == 1.0
    assert float(out["listed_month"].iloc[0]) == 1.0


def test_district_ppm_stats_fit_and_apply_no_leakage_fill() -> None:
    districts = pd.Series(["Wola", "Wola", "Mokotów"])
    ppm = pd.Series([10_000.0, 12_000.0, 20_000.0])
    stats = fit_district_ppm_stats(districts, ppm)
    assert stats["Wola"]["median"] == 11_000.0
    assert stats["Wola"]["n"] == 2.0
    frame = pd.DataFrame({
        "district": ["Wola", "Ursynów", "Mokotów"],
        "nbhd_rcn_idw_ppm": [11_500.0, 9_000.0, 19_000.0],
    })
    out = apply_district_ppm_stats(
        frame,
        stats,
        global_median=15_000.0,
        global_mean=14_000.0,
    )
    for col in ENGINEERED_TARGET_COLUMNS:
        assert col in out.columns
    assert float(out["district_train_median_ppm"].iloc[0]) == 11_000.0
    assert float(out["district_train_median_ppm"].iloc[1]) == 15_000.0  # unseen
    assert float(out["nbhd_minus_district_ppm"].iloc[0]) == 500.0
