"""Tests for train feature matrix hygiene and splits."""

import pandas as pd
import pytest

from happy_housing.config import ModelConfig
from happy_housing.models.train import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TrainSettings,
    build_feature_matrix,
    check_feature_coverage,
    split_train_val_test,
    train_price_model,
)


def _with_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pad missing model columns so unit tests exercise train without full GIS."""
    extras: dict[str, object] = {}
    for col in FEATURE_COLUMNS:
        if col in frame.columns:
            continue
        if col in CATEGORICAL_FEATURE_COLUMNS:
            extras[col] = "unknown"
        else:
            extras[col] = 0.0
    if not extras:
        return frame
    return pd.concat(
        [frame, pd.DataFrame(extras, index=frame.index)],
        axis=1,
    )


def test_build_feature_matrix_uses_provided_medians() -> None:
    frame = pd.DataFrame({
        "area_m2": [50.0, None, 70.0],
        "rooms": [2.0, 3.0, None],
    })
    medians = pd.Series({"area_m2": 99.0, "rooms": 4.0})
    out = build_feature_matrix(
        frame,
        ("area_m2", "rooms"),
        train_medians=medians,
    )
    assert float(out.iloc[1]["area_m2"]) == 99.0
    assert float(out.iloc[2]["rooms"]) == 4.0


def test_check_feature_coverage_raises_when_core_missing() -> None:
    frame = pd.DataFrame({"area_m2": [50.0, 60.0]})
    with pytest.raises(ValueError, match="absent on frame"):
        check_feature_coverage(
            frame,
            ("area_m2", "rooms", "year_built"),
            min_usable=8,
        )


def test_split_train_val_test_time_aware() -> None:
    n = 100
    frame = pd.DataFrame({
        "listed_at": pd.date_range("2024-01-01", periods=n, freq="D"),
        "x": range(n),
        "y": range(n),
    })
    features = frame[["x"]]
    target = frame["y"].astype(float)
    train_b, _val_b, test_b = split_train_val_test(
        features,
        target,
        frame,
        test_size=0.2,
        val_size=0.15,
        random_state=0,
    )
    # Time-aware: test indices should be the latest rows
    assert test_b.features.index.max() == n - 1
    assert test_b.features.index.min() > train_b.features.index.max()


def test_impute_with_medians_handles_na_median() -> None:
    from happy_housing.models.train import _impute_with_medians

    features = pd.DataFrame({"area_m2": [50.0, None], "rooms": [2.0, 3.0]})
    medians = pd.Series({"area_m2": pd.NA, "rooms": 2.5})
    out = _impute_with_medians(features, medians, ("area_m2", "rooms"))
    assert float(out.iloc[1]["area_m2"]) == 0.0
    assert float(out.iloc[0]["rooms"]) == 2.0


def test_split_train_val_test_mixed_timezone_listed_at() -> None:
    """Otodom UTC strings + naive fixture dates must not crash the split."""
    n = 40
    listed = pd.concat([
        pd.Series(pd.date_range("2024-01-01", periods=n // 2, freq="D", tz="UTC")),
        pd.Series(pd.date_range("2024-06-01", periods=n // 2, freq="D")),
    ]).reset_index(drop=True)
    frame = pd.DataFrame({"listed_at": listed, "x": range(n), "y": range(n)})
    train_b, val_b, test_b = split_train_val_test(
        frame[["x"]],
        frame["y"].astype(float),
        frame,
        test_size=0.2,
        val_size=0.15,
        random_state=0,
    )
    assert len(train_b.features) + len(val_b.features) + len(test_b.features) == n


def test_train_otodom_only_excludes_rcn_labels() -> None:
    frame = pd.DataFrame({
        "source": ["otodom"] * 20 + ["rcn"] * 20,
        "area_m2": [50.0] * 40,
        "rooms": [2.0] * 40,
        "year_built": [2000.0] * 40,
        "floor_no": [3.0] * 40,
        "building_floors_num": [8.0] * 40,
        "market_primary": [0.0] * 40,
        "has_balcony": [1.0] * 40,
        "has_garage": [0.0] * 40,
        "has_lift": [1.0] * 40,
        "is_private_owner": [1.0] * 40,
        "rent": [0.0] * 40,
        "aircraft_lden_db": [0.0] * 40,
        "aircraft_lnight_db": [0.0] * 40,
        "in_oou": [0.0] * 40,
        "dist_hospital_m": [500.0] * 40,
        "dist_fire_station_m": [400.0] * 40,
        "dist_police_m": [300.0] * 40,
        "dist_air_station_m": [1000.0] * 40,
        "air_aq_index": [2.0] * 40,
        "air_pm10_ug_m3": [20.0] * 40,
        "air_pm25_ug_m3": [10.0] * 40,
        "air_no2_ug_m3": [15.0] * 40,
        "dist_transit_stop_m": [200.0] * 40,
        "dist_metro_m": [800.0] * 40,
        "dist_park_m": [150.0] * 40,
        "dist_daszynskiego_m": [3000.0] * 40,
        "commute_min": [15.0] * 40,
        "dist_gunb_m": [600.0] * 40,
        "price_per_m2": [12_000.0 + i * 10 for i in range(40)],
    })
    result = train_price_model(
        _with_all_features(frame),
        ModelConfig(n_estimators=20, num_leaves=7, test_size=0.25),
        TrainSettings(
            hpo_trials=1,
            use_neighborhood=False,
            train_buyer_asks_only=True,
            test_size=0.25,
            val_size=0.2,
            random_state=0,
        ),
    )
    # Only 20 Otodom rows → train+val+test from those
    assert result.metrics["n_train_val"] + result.metrics["n_test"] == 20.0
