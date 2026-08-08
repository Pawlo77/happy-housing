"""Tests for commute minutes with OSRM unavailable (distance proxy)."""

from __future__ import annotations

import pandas as pd

from happy_housing.gis.routing import attach_commute_minutes


def test_attach_commute_minutes_distance_fallback() -> None:
    """Without OSRM, commute_min derives from dist_daszynskiego_m at 30 km/h."""
    frame = pd.DataFrame({
        "lon": [21.0],
        "lat": [52.23],
        "dist_daszynskiego_m": [3000.0],
    })
    out = attach_commute_minutes(
        frame,
        dest_lon=20.983,
        dest_lat=52.23,
        osrm_url="http://127.0.0.1:9",
    )
    # 3 km at 30 km/h → 6 minutes.
    assert float(out["commute_min"].iloc[0]) == 6.0
