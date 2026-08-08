"""Model feature columns include catalog amenity distances."""

from __future__ import annotations

from happy_housing.gis.amenity_catalog import (
    dist_column,
    model_distance_feature_columns,
)
from happy_housing.models.train import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS


def test_model_distance_features_include_gym_and_ev() -> None:
    cols = model_distance_feature_columns()
    assert dist_column("gym") in cols
    assert dist_column("indoor_pool") in cols
    assert dist_column("charging_station") in cols
    assert dist_column("library") in cols


def test_core_features_include_city_poi_distances() -> None:
    assert "dist_gym_m" in CORE_FEATURE_COLUMNS
    assert "dist_indoor_pool_m" in CORE_FEATURE_COLUMNS
    assert "dist_charging_station_m" in CORE_FEATURE_COLUMNS
    assert "dist_park_m" in CORE_FEATURE_COLUMNS
    assert FEATURE_COLUMNS[0] == CORE_FEATURE_COLUMNS[0]
