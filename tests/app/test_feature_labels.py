"""Tests for model feature hover descriptions."""

from __future__ import annotations

from happy_housing.app.feature_labels import feature_description


def test_feature_description_covers_core_and_amenity_distances() -> None:
    assert feature_description("district") == "Warsaw district"
    assert "floor" in feature_description("floor_no").casefold()
    assert "commute" in feature_description("commute_min").casefold()
    assert "crime" in feature_description("crime_rate_per_1000").casefold()
    assert feature_description("dist_hospital_m") == "Nearest hospital (m)"
    assert "tennis wall" in feature_description("dist_tennis_wall_m").casefold()
    assert "metro entrance" in feature_description("dist_metro_entrance_m").casefold()
    assert "squash" in feature_description("dist_squash_m").casefold()


def test_feature_description_falls_back_for_unknown() -> None:
    assert feature_description("weird_new_signal") == "weird new signal"
    assert feature_description("") == ""
