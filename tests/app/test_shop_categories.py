"""Tests for OSM shop subcategory filters on the Explore map."""

import pandas as pd

from happy_housing.app.map_layers import (
    SHOP_CATEGORY_LABELS,
    LayerFlags,
    explore_datasets_needed,
    filter_shops_by_categories,
    shop_category_for_tag,
)
from happy_housing.app.maps import shops_layers


def test_shop_category_groups_common_tags() -> None:
    assert shop_category_for_tag("supermarket") == "grocery"
    assert shop_category_for_tag("convenience") == "grocery"
    assert shop_category_for_tag("greengrocer") == "grocery"
    assert shop_category_for_tag("bakery") == "grocery"
    assert shop_category_for_tag("chemist") == "health_beauty"
    assert shop_category_for_tag("hairdresser") == "health_beauty"
    assert shop_category_for_tag("clothes") == "fashion"
    assert shop_category_for_tag("shoes") == "fashion"
    assert shop_category_for_tag("furniture") == "home"
    assert shop_category_for_tag("electronics") == "home"
    assert shop_category_for_tag("car_repair") == "mobility"
    assert shop_category_for_tag("bicycle") == "mobility"
    assert shop_category_for_tag("vacant") == "other"
    assert shop_category_for_tag("weird_new_tag") == "other"
    assert set(SHOP_CATEGORY_LABELS) == {
        "grocery",
        "health_beauty",
        "fashion",
        "home",
        "mobility",
        "other",
    }


def test_filter_shops_by_categories_keeps_selected_only() -> None:
    frame = pd.DataFrame({
        "name": ["Biedronka", "Zara", "Orlen-ish"],
        "shop": ["supermarket", "clothes", "car"],
        "lat": [52.23, 52.24, 52.25],
        "lon": [21.0, 21.01, 21.02],
    })
    grocery = filter_shops_by_categories(frame, categories={"grocery"})
    assert list(grocery["shop"]) == ["supermarket"]
    fashion_mobility = filter_shops_by_categories(
        frame, categories={"fashion", "mobility"}
    )
    assert set(fashion_mobility["shop"]) == {"clothes", "car"}
    empty = filter_shops_by_categories(frame, categories=set())
    assert empty.empty


def test_shops_layers_emits_one_layer_per_enabled_category() -> None:
    frame = pd.DataFrame({
        "name": ["A", "B", "C"],
        "shop": ["supermarket", "clothes", "florist"],
        "lat": [52.23, 52.24, 52.25],
        "lon": [21.0, 21.01, 21.02],
    })
    layers = shops_layers(
        frame,
        show_grocery=True,
        show_health_beauty=False,
        show_fashion=True,
        show_home=False,
        show_mobility=False,
        show_other=True,
    )
    ids = {getattr(layer, "id", None) for layer in layers}
    assert "shops-grocery" in ids
    assert "shops-fashion" in ids
    assert "shops-other" in ids
    assert "shops-health_beauty" not in ids


def test_explore_datasets_needed_shops_from_subcategories() -> None:
    flags = LayerFlags(
        shop_show_grocery=True,
        shop_show_fashion=False,
        city_boundary=False,
        military=False,
        rail_areas=False,
        airport=False,
        noise_overlay="off",
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "shops" in needed
    assert flags.any_shops() is True
    assert LayerFlags().any_shops() is False


def test_explore_datasets_needed_transit_from_modes() -> None:
    flags = LayerFlags(
        transit_show_metro=True,
        city_boundary=False,
        military=False,
        rail_areas=False,
        airport=False,
        noise_overlay="off",
    )
    needed = explore_datasets_needed(section="Map", flags=flags)
    assert "stops" in needed
    assert "route_shapes" in needed
    assert flags.any_transit_network() is True
    assert LayerFlags().any_transit_network() is False
