"""SHAP / feature-matrix alignment with a persisted LightGBM booster."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd
import shap

from happy_housing.models.train import (
    build_feature_matrix,
    model_feature_columns,
)


def test_model_feature_columns_match_booster(tmp_path: Path) -> None:
    """SHAP matrix uses booster feature names, not the latest catalog."""
    cols = ("area_m2", "rooms", "dist_metro_m")
    train = pd.DataFrame({
        "area_m2": [40.0, 50.0, 60.0, 70.0],
        "rooms": [1.0, 2.0, 2.0, 3.0],
        "dist_metro_m": [200.0, 400.0, 100.0, 800.0],
        "y": [12000.0, 14000.0, 13000.0, 15000.0],
    })
    x = train.loc[:, list(cols)]
    model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
    model.fit(x, train["y"])
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    booster = lgb.Booster(model_file=str(path))
    assert model_feature_columns(booster) == cols

    # Row has extra modern columns; matrix for SHAP must stay at 3.
    row = pd.DataFrame({
        "area_m2": [55.0],
        "rooms": [2.0],
        "dist_metro_m": [300.0],
        "dist_park_m": [50.0],
        "commute_min": [12.0],
    })
    features = build_feature_matrix(row, model_feature_columns(booster))
    assert list(features.columns) == list(cols)
    explainer = shap.TreeExplainer(booster)
    explanation = explainer(features)
    assert explanation.values.shape[1] == 3
