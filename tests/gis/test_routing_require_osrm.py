"""OSRM hard-require vs distance-proxy fallback."""

import pandas as pd
import pytest

from happy_housing.gis.routing import attach_commute_minutes


def test_attach_commute_minutes_require_osrm_raises() -> None:
    frame = pd.DataFrame({
        "lon": [21.0],
        "lat": [52.23],
        "dist_daszynskiego_m": [3000.0],
    })
    with pytest.raises(RuntimeError, match="OSRM required"):
        attach_commute_minutes(
            frame,
            dest_lon=20.983,
            dest_lat=52.23,
            osrm_url="http://127.0.0.1:9",
            require_osrm=True,
        )


def test_attach_commute_minutes_proxy_when_not_required() -> None:
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
        require_osrm=False,
    )
    assert float(out["commute_min"].iloc[0]) == 6.0
