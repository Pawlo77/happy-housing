"""Tests for Stats livability ladders and model XAI helpers."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from happy_housing.app.stats_filters import StatsFilterState, apply_stats_filters
from happy_housing.app.stats_insights import (
    distance_price_curve,
    high_noise_surcharge,
    noise_source_premiums,
    noise_total_ladder,
)
from happy_housing.app.stats_xai import (
    booster_importance_table,
    sample_frame_for_shap,
)


def _livability_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a", "b", "c", "d", "e", "f"],
        "price_per_m2": [12000.0, 14000.0, 16000.0, 18000.0, 20000.0, 22000.0],
        "noise_total_lden_db": [0.0, 48.0, 55.0, 60.0, 65.0, 70.0],
        "in_high_noise": [0, 0, 1, 1, 1, 1],
        "road_lden_db": [0.0, 50.0, 55.0, 60.0, 0.0, 70.0],
        "rail_lden_db": [0.0, 0.0, 0.0, 55.0, 60.0, 0.0],
        "tram_lden_db": [0.0, 0.0, 45.0, 0.0, 55.0, 0.0],
        "aircraft_lden_db": [0.0, 0.0, 0.0, 0.0, 0.0, 60.0],
        "industry_lden_db": [0.0, 0.0, 0.0, 0.0, 55.0, 0.0],
        "dist_bike_path_m": [50.0, 200.0, 400.0, 800.0, 1500.0, 3000.0],
        "traffic_weight_nearby": [0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
        "dist_park_m": [100.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0],
        "air_aq_index": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_noise_total_ladder_bands() -> None:
    out = noise_total_ladder(_livability_frame())
    assert {"lden_band", "n", "median_price_per_m2"}.issubset(out.columns)
    assert int(out["n"].sum()) == 6


def test_high_noise_surcharge() -> None:
    stats = high_noise_surcharge(_livability_frame())
    assert stats["n_inside"] == 4.0
    assert stats["n_outside"] == 2.0
    assert stats["surcharge_pct"] == stats["surcharge_pct"]


def test_noise_source_premiums_lists_sources() -> None:
    out = noise_source_premiums(_livability_frame())
    assert {"source", "n_exposed", "median_ppm_exposed", "premium_pct"}.issubset(
        out.columns
    )
    assert set(out["source"]) >= {"road", "rail", "tram", "aircraft", "industry"}


def test_distance_price_curve_generic() -> None:
    out = distance_price_curve(
        _livability_frame(),
        dist_col="dist_bike_path_m",
        bins=[0, 250, 500, 1000, 50_000],
        labels=["<250m", "250-500m", "500m-1km", "1km+"],
        band_col="bike_band",
    )
    assert "bike_band" in out.columns
    assert int(out["n"].sum()) == 6


def test_apply_stats_filters_high_noise() -> None:
    frame = _livability_frame()
    frame["district"] = "wola"
    frame["source"] = "otodom"
    out = apply_stats_filters(frame, StatsFilterState(high_noise="outside"))
    assert list(out["listing_id"]) == ["a", "b"]


def test_booster_importance_table_sorted() -> None:
    booster = MagicMock()
    booster.feature_name.return_value = ["area_m2", "noise_total_lden_db", "rooms"]
    booster.feature_importance.return_value = np.array([10.0, 30.0, 5.0])
    out = booster_importance_table(booster, top_n=2)
    assert list(out["feature"]) == ["noise_total_lden_db", "area_m2"]
    assert float(out["importance"].iloc[0]) == 30.0


def test_sample_frame_for_shap_caps_rows() -> None:
    frame = pd.DataFrame({
        "listing_id": [f"L{i}" for i in range(50)],
        "price_per_m2": np.linspace(10_000, 20_000, 50),
    })
    sample = sample_frame_for_shap(frame, sample_n=10, random_state=0)
    assert len(sample) == 10


def test_mean_abs_shap_table_from_arrays() -> None:
    """Mean |SHAP| ranks features without needing a real LightGBM model."""
    from happy_housing.app.stats_xai import rank_mean_abs_shap

    out = rank_mean_abs_shap(
        feature_names=["a", "b", "c"],
        shap_values=np.array([
            [1.0, -2.0, 0.5],
            [0.0, 4.0, -0.5],
        ]),
        top_n=2,
    )
    assert list(out["feature"]) == ["b", "a"]
    assert float(out.loc[out["feature"] == "b", "mean_abs_shap"].iloc[0]) == 3.0
