"""Tests for scored-listings listed_month derivation."""

import pandas as pd

from happy_housing.app.data import _with_listed_month


def test_with_listed_month_parses_iso_and_rcn_gml() -> None:
    frame = pd.DataFrame({
        "listed_at": [
            "2026-07-14T17:55:55Z",
            "2026-02-15",
            "PL.PZGIK.1465_84a10d85-81c4-4299-8a13-45779244040e_2021-07-13T09-16-23",
            None,
        ]
    })
    out = _with_listed_month(frame)
    assert list(out["listed_month"]) == ["2026-07", "2026-02", "2021-07", "unknown"]
