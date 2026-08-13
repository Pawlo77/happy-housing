"""Tests for model feature hover descriptions."""

from happy_housing.app.feature_labels import feature_catalog_frame, feature_description


def test_feature_description_covers_core_and_amenity_distances() -> None:
    assert feature_description("district") == "Warsaw district"
    assert "floor" in feature_description("floor_no").casefold()
    assert "commute" in feature_description("commute_min").casefold()
    assert "crime" in feature_description("crime_rate_per_1000").casefold()
    assert feature_description("dist_hospital_m") == "Nearest hospital (m)"
    assert "tennis wall" in feature_description("dist_tennis_wall_m").casefold()
    assert "metro entrance" in feature_description("dist_metro_entrance_m").casefold()
    assert "squash" in feature_description("dist_squash_m").casefold()


def test_feature_description_expands_lden_and_lnight() -> None:
    """Explainability text spells out day-evening-night / night indicators."""
    aircraft_day = feature_description("aircraft_lden_db")
    aircraft_night = feature_description("aircraft_lnight_db")
    total_day = feature_description("noise_total_lden_db")
    high = feature_description("in_high_noise")
    assert "lden" not in aircraft_day.casefold()
    assert "lnight" not in aircraft_night.casefold()
    assert "day-evening-night" in aircraft_day.casefold()
    assert "night" in aircraft_night.casefold()
    assert "day-evening-night" in total_day.casefold()
    assert "lden" not in high.casefold()
    assert "day-evening-night" in high.casefold()


def test_feature_description_expands_rcn() -> None:
    """Neighborhood comps spell out Rejestr Cen, not bare RCN."""
    for key in (
        "nbhd_rcn_median_ppm",
        "nbhd_rcn_mean_ppm",
        "nbhd_rcn_idw_ppm",
        "nbhd_rcn_n",
        "nbhd_rcn_nearest_m",
    ):
        text = feature_description(key)
        assert "rcn" not in text.casefold()
        assert "rejestr cen" in text.casefold()


def test_feature_description_falls_back_for_unknown() -> None:
    assert feature_description("weird_new_signal") == "weird new signal"
    assert feature_description("") == ""


def test_feature_catalog_frame_pairs_names_with_descriptions() -> None:
    frame = feature_catalog_frame(["district", "dist_gym_m", ""])
    assert list(frame.columns) == ["feature", "description"]
    assert list(frame["feature"]) == ["district", "dist_gym_m"]
    assert frame.iloc[0]["description"] == "Warsaw district"
    assert "gym" in str(frame.iloc[1]["description"]).casefold()
