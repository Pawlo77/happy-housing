"""Tests for Explore map styling helpers."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk

from happy_housing.app.map_layers import prepare_listings_for_map
from happy_housing.app.maps import (
    basemap_style,
    deck_map_provider,
    listings_scatter_layer,
)


def test_listing_colors_vectorized_over_under() -> None:
    """Overpriced pins go red-heavy; underpriced go green-heavy."""
    frame = pd.DataFrame({
        "listing_id": ["a", "b"],
        "district": ["wola", "wola"],
        "lat": [52.2, 52.3],
        "lon": [21.0, 21.1],
        "delta_price_per_m2": [500.0, -200.0],
        "price_per_m2": [18000.0, 14000.0],
        "area_m2": [50.0, 80.0],
    })
    out = prepare_listings_for_map(frame)
    assert int(out.loc[0, "color_r"]) == 220
    assert int(out.loc[1, "color_g"]) == 170
    assert out["radius"].between(22, 55).all()
    assert listings_scatter_layer(frame) is not None


def test_satellite_basemap_requires_mapbox_provider() -> None:
    """pydeck asserts map_provider=mapbox whenever map_style is a dict."""
    style = basemap_style("Satellite")
    assert isinstance(style, dict)
    assert deck_map_provider(style) == "mapbox"
    view = pdk.ViewState(latitude=52.23, longitude=21.0, zoom=11)
    deck = pdk.Deck(
        map_style=style,
        map_provider=deck_map_provider(style),
        layers=[],
        initial_view_state=view,
    )
    assert deck.map_provider == "mapbox"


def test_streets_basemap_keeps_default_provider() -> None:
    """URL styles use pydeck's default carto provider (no mapbox token)."""
    style = basemap_style("Streets")
    assert isinstance(style, str)
    assert deck_map_provider(style) is None
