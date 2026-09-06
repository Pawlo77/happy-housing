"""Home Model feature-column resolution."""

from happy_housing.app.page_home import resolve_home_feature_columns


def test_resolve_home_feature_columns_prefers_metadata() -> None:
    assert resolve_home_feature_columns(
        ["district", "area_m2"],
        booster_columns=["other"],
    ) == ["district", "area_m2"]


def test_resolve_home_feature_columns_falls_back_to_booster() -> None:
    assert resolve_home_feature_columns(
        [],
        booster_columns=["district", "area_m2", "dist_metro_m"],
    ) == ["district", "area_m2", "dist_metro_m"]


def test_resolve_home_feature_columns_skips_blank_metadata() -> None:
    assert resolve_home_feature_columns(
        ["", "  "],
        booster_columns=["district"],
    ) == ["district"]


def test_resolve_home_feature_columns_empty_when_neither() -> None:
    assert resolve_home_feature_columns(None, booster_columns=None) == []
