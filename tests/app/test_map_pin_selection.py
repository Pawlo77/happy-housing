"""Tests for Explore map pin selection helpers (fast fragment reruns)."""

from __future__ import annotations

from unittest.mock import MagicMock

from happy_housing.app.explore_nav import (
    MAP_PIN_SELECTION_KEY,
    resolve_map_pin_selection,
)
from happy_housing.app.map_layers import LayerFlags
from happy_housing.app.maps import (
    MapAddressFocus,
    explore_map_layer_cache_key,
    listing_ids_fingerprint,
)


def test_listing_ids_fingerprint_stable_and_order_sensitive() -> None:
    a = listing_ids_fingerprint(["OTO-1", "OTO-2"])
    b = listing_ids_fingerprint(["OTO-1", "OTO-2"])
    c = listing_ids_fingerprint(["OTO-2", "OTO-1"])
    assert a == b
    assert a != c
    assert a != listing_ids_fingerprint(["OTO-1"])


def test_explore_map_layer_cache_key_changes_with_flags_and_listings() -> None:
    base = explore_map_layer_cache_key(
        listing_fingerprint="abc",
        flags=LayerFlags(parks=False),
        hour=8,
        basemap="Streets",
        commute_lon=21.0,
        commute_lat=52.2,
        recommended_ids=frozenset(),
        focus=None,
        overlay_counts={"parks": 0, "shops": 0},
    )
    parks_on = explore_map_layer_cache_key(
        listing_fingerprint="abc",
        flags=LayerFlags(parks=True),
        hour=8,
        basemap="Streets",
        commute_lon=21.0,
        commute_lat=52.2,
        recommended_ids=frozenset(),
        focus=None,
        overlay_counts={"parks": 10, "shops": 0},
    )
    other_listings = explore_map_layer_cache_key(
        listing_fingerprint="xyz",
        flags=LayerFlags(parks=False),
        hour=8,
        basemap="Streets",
        commute_lon=21.0,
        commute_lat=52.2,
        recommended_ids=frozenset(),
        focus=None,
        overlay_counts={"parks": 0, "shops": 0},
    )
    with_focus = explore_map_layer_cache_key(
        listing_fingerprint="abc",
        flags=LayerFlags(parks=False),
        hour=8,
        basemap="Streets",
        commute_lon=21.0,
        commute_lat=52.2,
        recommended_ids=frozenset(),
        focus=MapAddressFocus(lon=21.01, lat=52.23, label="x"),
        overlay_counts={"parks": 0, "shops": 0},
    )
    assert base != parks_on
    assert base != other_listings
    assert base != with_focus


def test_resolve_map_pin_selection_stores_and_clears() -> None:
    session: dict[str, object] = {}
    event = {
        "selection": {
            "objects": {"listings": [{"listing_id": "OTO-9", "tooltip_name": "OTO-9"}]}
        }
    }
    assert resolve_map_pin_selection(event, session=session) == "OTO-9"
    assert session[MAP_PIN_SELECTION_KEY] == "OTO-9"

    assert resolve_map_pin_selection(None, session=session) is None
    assert MAP_PIN_SELECTION_KEY not in session


def test_resolve_map_pin_selection_uses_streamlit_session_by_default() -> None:
    from unittest.mock import patch

    event = {"selection": {"objects": {"listings": [{"listing_id": "OTO-1"}]}}}
    fake_st = MagicMock()
    fake_st.session_state = {}
    with patch("happy_housing.app.explore_nav.st", fake_st):
        assert resolve_map_pin_selection(event) == "OTO-1"
        assert fake_st.session_state[MAP_PIN_SELECTION_KEY] == "OTO-1"


def test_cached_explore_map_layers_reuses_until_key_changes() -> None:
    from unittest.mock import patch

    from happy_housing.app.maps import cached_explore_map_layers

    session: dict[str, object] = {}
    calls = {"n": 0}

    def builder() -> list[str]:
        calls["n"] += 1
        return [f"layer-{calls['n']}"]

    fake_st = MagicMock()
    fake_st.session_state = session
    with patch("happy_housing.app.maps.st", fake_st):
        first = cached_explore_map_layers(("k1",), builder)
        second = cached_explore_map_layers(("k1",), builder)
        third = cached_explore_map_layers(("k2",), builder)
    assert first == ["layer-1"]
    assert second == ["layer-1"]
    assert third == ["layer-2"]
    assert calls["n"] == 2
    # Shallow copy: mutating a returned list must not poison the cache.
    first.append("x")
    with patch("happy_housing.app.maps.st", fake_st):
        # k2 is current cache key — rebuild for k1.
        again_k1 = cached_explore_map_layers(("k1",), builder)
        again_k1_b = cached_explore_map_layers(("k1",), builder)
    assert again_k1 == ["layer-3"]
    assert again_k1_b == ["layer-3"]
    again_k1.append("y")
    with patch("happy_housing.app.maps.st", fake_st):
        assert cached_explore_map_layers(("k1",), builder) == ["layer-3"]
