"""Lazy Explore map-dataset selection (skip heavy loads when unused)."""

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


def test_explore_map_not_mounted_on_offer_for_speed() -> None:
    """Offer interactions must not remount Explore Map / warm Map GIS."""
    from happy_housing.app.map_layers import (
        explore_datasets_needed_preserving_map,
        explore_needs_recommended_pins,
        keep_explore_map_mounted,
        recommended_pins_memo_key,
    )

    flags = LayerFlags(gunb=True, city_poi_types=("gym",), military=True)
    needed = explore_datasets_needed_preserving_map(
        section="Offer",
        flags=flags,
        offer_subtab="Overview",
    )
    assert needed == frozenset()
    assert keep_explore_map_mounted("Offer") is False
    assert keep_explore_map_mounted("Map") is True
    assert keep_explore_map_mounted("Board") is False
    assert explore_needs_recommended_pins("Map") is True
    assert explore_needs_recommended_pins("Offer", "Overview") is False
    assert explore_needs_recommended_pins("Offer", "Map") is True
    assert explore_needs_recommended_pins("Board") is False
    key_a = recommended_pins_memo_key(
        scored_mtime=1.0,
        scored_rows=10,
        prefs_payload={"enabled": False},
        starred=frozenset({"a"}),
        top_n=15,
        demotion_alpha=1.0,
        w_bargain=0.5,
        w_commute=0.25,
        w_transit=0.2,
        w_overprice=0.15,
        otodom_pool_default=True,
        amenity_weights=None,
        commute_mode="transit",
    )
    key_b = recommended_pins_memo_key(
        scored_mtime=1.0,
        scored_rows=10,
        prefs_payload={"enabled": False},
        starred=frozenset({"a"}),
        top_n=15,
        demotion_alpha=1.0,
        w_bargain=0.5,
        w_commute=0.25,
        w_transit=0.2,
        w_overprice=0.15,
        otodom_pool_default=True,
        amenity_weights=None,
        commute_mode="transit",
    )
    key_c = recommended_pins_memo_key(
        scored_mtime=1.0,
        scored_rows=10,
        prefs_payload={"enabled": True},
        starred=frozenset({"a"}),
        top_n=15,
        demotion_alpha=1.0,
        w_bargain=0.5,
        w_commute=0.25,
        w_transit=0.2,
        w_overprice=0.15,
        otodom_pool_default=True,
        amenity_weights=None,
        commute_mode="transit",
    )
    assert key_a == key_b
    assert key_a != key_c


def test_layer_flags_default_zones_off_for_fast_map() -> None:
    """Heavy zone overlays stay opt-in so first Map paint stays light."""
    flags = LayerFlags()
    assert flags.military is False
    assert flags.rail_areas is False
    assert flags.airport is False
    assert flags.city_boundary is False
    assert flags.district_boundaries is False


def test_explore_datasets_needed_spatial_skips_zones_when_off() -> None:
    flags = LayerFlags(
        military=False,
        rail_areas=False,
        airport=False,
        city_boundary=False,
        district_boundaries=False,
        noise_overlay="off",
    )
    needed = explore_datasets_needed(section="Spatial", flags=flags)
    assert "traffic" in needed
    assert "military" not in needed
    assert "airport" not in needed
    assert "city_boundary" not in needed


def test_explore_datasets_needed_spatial_skips_map_only_overlays() -> None:
    """Spatial uses traffic + zones only — not noise/AQ/shops/transit GIS."""
    flags = LayerFlags(
        noise_overlay="total_lden",
        aq_overlay="pm25",
        shop_show_grocery=True,
        transit_intensity=True,
        gunb=True,
        parks=True,
        military=True,
        traffic=True,
    )
    needed = explore_datasets_needed(section="Spatial", flags=flags)
    assert needed == frozenset({"traffic", "military"})
    assert "warsaw_noise" not in needed
    assert "gios_aq" not in needed
    assert "shops" not in needed
    assert "gunb" not in needed
    assert "parks" not in needed
    assert "stops" not in needed


def test_prepare_listings_for_map_ignores_unused_columns() -> None:
    """Map prep must not require copying description / media columns."""
    import pandas as pd

    from happy_housing.app.map_layers import prepare_listings_for_map

    frame = pd.DataFrame({
        "listing_id": ["a"],
        "lat": [52.2],
        "lon": [21.0],
        "delta_price_per_m2": [-100.0],
        "price_per_m2": [15000.0],
        "area_m2": [50.0],
        "rooms": [2.0],
        "district": ["Wola"],
        "description": ["x" * 50_000],
        "image_urls_json": ["[]"],
    })
    out = prepare_listings_for_map(frame)
    assert "description" not in out.columns
    assert "tooltip_detail" in out.columns
    assert len(out) == 1


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
