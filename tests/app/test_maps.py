"""Tests for Explore map styling helpers."""

import pandas as pd
import pydeck as pdk

from happy_housing.app.map_layers import (
    listing_delta_rgba,
    prepare_listings_for_map,
)
from happy_housing.app.maps import (
    basemap_style,
    deck_map_provider,
    listings_scatter_layer,
)


def test_listing_delta_rgba_is_continuous_not_binary() -> None:
    """Δ zł/m² maps to a smooth red↔green ramp, not a 0/1 split."""
    deep_green = listing_delta_rgba(-2_000.0)
    mild_green = listing_delta_rgba(-200.0)
    fair = listing_delta_rgba(0.0)
    mild_red = listing_delta_rgba(200.0)
    deep_red = listing_delta_rgba(2_000.0)

    assert deep_green[1] > mild_green[1] > fair[1]
    assert deep_red[0] > mild_red[0] > fair[0]
    assert deep_green[0] < deep_red[0]
    assert deep_green[1] > deep_red[1]
    # Fair is neither saturated green nor saturated red.
    assert fair[0] != deep_green[0]
    assert fair[1] != deep_red[1]


def test_listing_colors_vectorized_over_under() -> None:
    """Overpriced pins go red-heavy; underpriced go green-heavy."""
    frame = pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "district": ["wola", "wola", "wola"],
        "lat": [52.2, 52.3, 52.25],
        "lon": [21.0, 21.1, 21.05],
        "delta_price_per_m2": [1_500.0, -1_500.0, 0.0],
        "price_per_m2": [18000.0, 14000.0, 16000.0],
        "area_m2": [50.0, 80.0, 60.0],
    })
    out = prepare_listings_for_map(frame)
    assert int(out.loc[0, "color_r"]) > int(out.loc[2, "color_r"])
    assert int(out.loc[1, "color_g"]) > int(out.loc[2, "color_g"])
    assert int(out.loc[0, "color_r"]) > int(out.loc[1, "color_r"])
    assert out["radius"].between(22, 55).all()
    layers = listings_scatter_layer(frame)
    assert isinstance(layers, list)
    assert layers
    rec_layers = listings_scatter_layer(frame, recommended_ids={"b"})
    assert len(rec_layers) == 2


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
