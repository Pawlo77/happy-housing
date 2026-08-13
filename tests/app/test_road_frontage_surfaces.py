"""Tests for road-density / frontage filters and preference hard cuts."""

import pandas as pd

from happy_housing.app.filters import FilterState, apply_filters
from happy_housing.app.preferences import (
    FRONTAGE_STREET,
    UserPreferences,
    apply_preferences,
)
from happy_housing.app.stats_insights import (
    frontage_hint_price_table,
    highway_class_price_table,
)
from happy_housing.models.train import (
    CATEGORICAL_FEATURE_COLUMNS,
    CORE_FEATURE_COLUMNS,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a", "b", "c", "d"],
        "district": ["wola"] * 4,
        "source": ["otodom"] * 4,
        "price_per_m2": [15000.0, 16000.0, 14000.0, 15500.0],
        "delta_price_per_m2": [-100.0, 50.0, -200.0, 0.0],
        "traffic_weight_nearby": [10.0, 80.0, 20.0, 5.0],
        "major_traffic_weight_nearby": [0.0, 40.0, 5.0, 0.0],
        "street_facing_hint": [1, 0, 0, 0],
        "quiet_side_hint": [0, 0, 1, 0],
        "nearest_highway": ["primary", "residential", "tertiary", "residential"],
        "in_oou": [0, 0, 0, 0],
        "in_high_noise": [0, 0, 0, 0],
        "in_poor_air": [0, 0, 0, 0],
    })


def test_explore_filters_traffic_and_frontage() -> None:
    frame = _frame()
    filtered = apply_filters(
        frame,
        FilterState(max_traffic_weight_nearby=25.0, frontage="quiet"),
    )
    assert list(filtered["listing_id"]) == ["c"]


def test_preference_filters_major_traffic_and_street() -> None:
    frame = _frame()
    prefs = UserPreferences(
        enabled=True,
        max_major_traffic_weight_nearby=10.0,
        frontage=FRONTAGE_STREET,
    )
    out = apply_preferences(frame, prefs)
    assert list(out["listing_id"]) == ["a"]


def test_model_features_include_road_density_and_frontage() -> None:
    for col in (
        "dist_major_road_m",
        "major_traffic_weight_nearby",
        "nearest_highway",
        "street_facing_hint",
        "quiet_side_hint",
    ):
        assert col in CORE_FEATURE_COLUMNS
    assert "nearest_highway" in CATEGORICAL_FEATURE_COLUMNS


def test_stats_highway_and_frontage_tables() -> None:
    frame = _frame()
    hwy = highway_class_price_table(frame)
    assert not hwy.empty
    assert "primary" in set(hwy["nearest_highway"])
    frontage = frontage_hint_price_table(frame)
    assert set(frontage["frontage"]) == {"street_facing", "quiet_side"}
