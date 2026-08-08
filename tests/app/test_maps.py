"""Tests for Explore map styling helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.map_layers import prepare_listings_for_map
from happy_housing.app.maps import listings_scatter_layer


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
    assert out["radius"].between(90, 320).all()
    assert listings_scatter_layer(frame) is not None
