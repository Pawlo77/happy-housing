"""Tests for Admin commute-target resolve / stale / recompute."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.commute_target import (
    DEFAULT_COMMUTE_LABEL,
    commute_target_stale,
    recompute_commute_features,
    resolve_commute_target,
)
from happy_housing.config import load_experiment_config


def test_resolve_commute_target_prefers_app_settings() -> None:
    config = load_experiment_config(Path("configs/experiment.yaml"))
    settings = AppSettings(
        commute_target_lon=21.01,
        commute_target_lat=52.25,
        commute_target_label="Plac Bankowy",
    )
    target = resolve_commute_target(settings, config)
    assert target.lon == 21.01
    assert target.lat == 52.25
    assert target.label == "Plac Bankowy"


def test_resolve_falls_back_to_yaml_when_unset() -> None:
    config = load_experiment_config(Path("configs/experiment.yaml"))
    settings = AppSettings(
        commute_target_lon=None,
        commute_target_lat=None,
        commute_target_label="",
    )
    target = resolve_commute_target(settings, config)
    assert target.lon == float(config.commute_target_lon)
    assert target.lat == float(config.commute_target_lat)
    assert target.label == DEFAULT_COMMUTE_LABEL


def test_commute_target_stale_when_features_lag() -> None:
    settings = AppSettings(
        commute_target_lon=21.0,
        commute_target_lat=52.2,
        commute_features_lon=20.983,
        commute_features_lat=52.23,
    )
    assert commute_target_stale(settings) is True
    settings.commute_features_lon = 21.0
    settings.commute_features_lat = 52.2
    assert commute_target_stale(settings) is False


def test_recompute_updates_distance_commute_and_fingerprint(tmp_path: Path) -> None:
    """Changing target must refresh dist_daszynskiego_m + commute_min + scores."""
    scored = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lat": [52.23, 52.25],
        "lon": [21.0, 21.05],
        "price_per_m2": [15000.0, 16000.0],
        "area_m2": [50.0, 60.0],
        "rooms": [2.0, 3.0],
        "year_built": [2000.0, 2010.0],
        "floor_no": [3.0, 4.0],
        "has_balcony": [1.0, 0.0],
        "has_garage": [0.0, 1.0],
        "has_lift": [1.0, 1.0],
        "in_oou": [0.0, 0.0],
        "dist_hospital_m": [500.0, 600.0],
        "dist_fire_station_m": [700.0, 800.0],
        "dist_police_m": [400.0, 450.0],
        "dist_park_m": [200.0, 250.0],
        "dist_air_station_m": [1000.0, 1100.0],
        "dist_transit_stop_m": [150.0, 180.0],
        "dist_gunb_m": [300.0, 320.0],
        "dist_daszynskiego_m": [9999.0, 9999.0],
        "commute_min": [99.0, 99.0],
        "source": ["oto", "oto"],
    })
    # Tiny booster on one feature so predict works after recompute.
    train = pd.DataFrame({
        "area_m2": [50.0, 60.0, 55.0],
        "y": [15000.0, 16000.0, 15500.0],
    })
    model = lgb.LGBMRegressor(
        n_estimators=5, learning_rate=0.5, verbosity=-1, random_state=0
    )
    model.fit(train[["area_m2"]], train["y"])
    model_path = tmp_path / "lightgbm_price_per_m2.txt"
    model.booster_.save_model(str(model_path))

    scored_path = tmp_path / "listings_scored.parquet"
    scored.to_parquet(scored_path, index=False)
    settings_file = settings_path(tmp_path)
    settings = AppSettings(
        commute_target_lon=20.983,
        commute_target_lat=52.23,
        commute_target_label="Rondo Daszyńskiego",
        commute_features_lon=0.0,
        commute_features_lat=0.0,
        model_use_neighborhood=False,
    )
    save_app_settings(settings_file, settings)

    config = load_experiment_config(Path("configs/experiment.yaml"))
    config = replace(
        config,
        routing=replace(config.routing, require_osrm=False),
    )
    result = recompute_commute_features(
        tmp_path,
        dest_lon=20.983,
        dest_lat=52.23,
        label="Rondo Daszyńskiego",
        config=config,
        projected_crs=config.bbox.projected_crs,
        osrm_url="http://127.0.0.1:1",  # force distance fallback
        use_otp_transit=False,
    )
    assert result["n_offers"] == 2
    out = pd.read_parquet(scored_path)
    assert float(out["dist_daszynskiego_m"].iloc[0]) < 5000.0
    assert float(out["commute_min"].iloc[0]) < 50.0
    assert "model_price_per_m2" in out.columns
    assert "delta_price_per_m2" in out.columns
    loaded = load_app_settings(settings_file)
    assert abs(float(loaded.commute_features_lon or 0) - 20.983) < 1e-6
    assert abs(float(loaded.commute_features_lat or 0) - 52.23) < 1e-6
    assert commute_target_stale(loaded) is False
