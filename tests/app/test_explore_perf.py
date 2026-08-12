"""Lazy Explore map-dataset selection (skip heavy loads when unused)."""

from __future__ import annotations

from happy_housing.app.map_layers import LayerFlags, explore_datasets_needed


def test_explore_datasets_needed_empty_on_board_chat_prefs() -> None:
    flags = LayerFlags()
    assert explore_datasets_needed(section="Board", flags=flags) == frozenset()
    assert explore_datasets_needed(section="Chat", flags=flags) == frozenset()
    assert explore_datasets_needed(section="Prefs", flags=flags) == frozenset()


def test_explore_datasets_needed_skips_gunb_when_off() -> None:
    flags = LayerFlags(gunb=False, traffic=False, hospitals=False)
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "gunb" not in needed
    assert "traffic" not in needed
    assert "amenities" not in needed


def test_explore_datasets_needed_includes_bike_paths() -> None:
    flags = LayerFlags(
        bike_paths=True,
        city_boundary=False,
        military=False,
        rail_areas=False,
        airport=False,
        noise_overlay="off",
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "bike_paths" in needed


def test_explore_datasets_needed_includes_warsaw_noise() -> None:
    flags = LayerFlags(
        noise_overlay="total_lden",
        city_boundary=False,
        military=False,
        rail_areas=False,
        airport=False,
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "warsaw_noise" in needed


def test_explore_datasets_needed_includes_gas_and_charging() -> None:
    flags = LayerFlags(
        gas_stations=True,
        charging_stations=True,
        city_boundary=False,
        military=False,
        rail_areas=False,
        airport=False,
        noise_overlay="off",
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "amenities" in needed


def test_explore_datasets_needed_includes_gios_aq() -> None:
    flags = LayerFlags(
        aq_overlay="aq_index",
        city_boundary=False,
        military=False,
        rail_areas=False,
        airport=False,
        noise_overlay="off",
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "gios_aq" in needed


def test_explore_datasets_needed_includes_defaults_for_map() -> None:
    flags = LayerFlags(
        noise_overlay="total_lden",
        city_boundary=True,
        district_boundaries=True,
        military=True,
        rail_areas=True,
        airport=True,
        city_poi_types=("gym", "indoor_pool"),
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "oou" not in needed
    assert "warsaw_noise" in needed
    assert "city_boundary" in needed
    assert "district_boundaries" in needed
    assert "military" in needed
    assert "city_pois" in needed


def test_explore_datasets_needed_news_only_stars_sync() -> None:
    needed = explore_datasets_needed(section="News", flags=LayerFlags())
    assert needed == frozenset({"star_news"})


def test_explore_datasets_needed_offer_skips_gis_on_overview() -> None:
    flags = LayerFlags(gunb=True, city_poi_types=("gym",))
    assert (
        explore_datasets_needed(section="Offer", flags=flags, offer_subtab="Overview")
        == frozenset()
    )
    needed = explore_datasets_needed(section="Offer", flags=flags, offer_subtab="Map")
    assert "gunb" in needed
    assert "city_pois" in needed


def test_amenity_layers_skips_copy_when_all_toggles_off() -> None:
    """No work when hospitals / fire / police are all disabled."""
    import geopandas as gpd
    from shapely.geometry import Point

    from happy_housing.app.maps import amenity_layers

    amenities = gpd.GeoDataFrame(
        {"amenity": ["hospital"], "name": ["X"]},
        geometry=[Point(21.0, 52.2)],
        crs="EPSG:4326",
    )
    assert (
        amenity_layers(
            amenities,
            show_hospital=False,
            show_fire=False,
            show_police=False,
        )
        == []
    )
