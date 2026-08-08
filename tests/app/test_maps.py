"""Tests for Explore map styling helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.maps import _listing_colors


def test_listing_colors_vectorized_over_under() -> None:
    """Overpriced pins go red-heavy; underpriced go green-heavy."""
    frame = pd.DataFrame({
        "lat": [52.2, 52.3],
        "lon": [21.0, 21.1],
        "delta_price_per_m2": [500.0, -200.0],
        "area_m2": [50.0, 80.0],
    })
    out = _listing_colors(frame)
    assert int(out.loc[0, "color_r"]) == 220
    assert int(out.loc[1, "color_g"]) == 170
    assert out["radius"].between(60, 280).all()
