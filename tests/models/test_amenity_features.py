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
    assert "dist_bike_path_m" in CORE_FEATURE_COLUMNS
    assert "bike_path_coverage_m" in CORE_FEATURE_COLUMNS
    assert "dist_rail_area_m" in CORE_FEATURE_COLUMNS
    assert "dist_military_m" in CORE_FEATURE_COLUMNS
    assert "dist_airport_area_m" in CORE_FEATURE_COLUMNS
    assert "dist_traffic_road_m" in CORE_FEATURE_COLUMNS
    assert "traffic_weight_nearby" in CORE_FEATURE_COLUMNS
    assert "noise_total_lden_db" in CORE_FEATURE_COLUMNS
    assert "tram_lden_db" in CORE_FEATURE_COLUMNS
    assert "industry_lden_db" in CORE_FEATURE_COLUMNS
    assert "in_high_noise" in CORE_FEATURE_COLUMNS
    assert "crime_rate_per_1000" in CORE_FEATURE_COLUMNS
    assert "crime_criminal_rate_per_1000" in CORE_FEATURE_COLUMNS
    assert "crime_detectability_pct" in CORE_FEATURE_COLUMNS
    assert FEATURE_COLUMNS[0] == CORE_FEATURE_COLUMNS[0]
