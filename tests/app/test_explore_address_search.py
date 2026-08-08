"""Tests for Explore Map address search camera + pin."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.maps import (
    ADDRESS_FOCUS_ZOOM,
    MapAddressFocus,
    address_search_layer,
    explore_map_camera,
    explore_map_selection_key,
)


def test_explore_map_camera_defaults_to_listing_mean() -> None:
    frame = pd.DataFrame({"lat": [52.2, 52.4], "lon": [21.0, 21.2]})
    lat, lon, zoom = explore_map_camera(
        frame,
        focus=None,
        fallback_lat=52.23,
        fallback_lon=21.0,
    )
    assert abs(lat - 52.3) < 1e-9
    assert abs(lon - 21.1) < 1e-9
    assert zoom == 10.5


def test_explore_map_camera_focus_centers_and_zooms() -> None:
    focus = MapAddressFocus(lon=21.01234, lat=52.23456, label="ul. Krucza 1")
    lat, lon, zoom = explore_map_camera(
        pd.DataFrame({"lat": [52.0], "lon": [21.0]}),
        focus=focus,
        fallback_lat=52.23,
        fallback_lon=21.0,
    )
    assert lat == focus.lat
    assert lon == focus.lon
    assert zoom == ADDRESS_FOCUS_ZOOM
    assert zoom >= 15.0


def test_explore_map_selection_key_changes_with_focus() -> None:
    assert explore_map_selection_key(None) == "explore_map_deck"
    focus = MapAddressFocus(lon=21.0, lat=52.2, label="x")
    key = explore_map_selection_key(focus)
    assert "21.00000" in key
    assert "52.20000" in key
    assert key != "explore_map_deck"


def test_address_search_layer_builds_icon() -> None:
    layer = address_search_layer(21.0, 52.2, label="Test address")
    assert layer is not None
    assert "address-search" in str(layer.id)
