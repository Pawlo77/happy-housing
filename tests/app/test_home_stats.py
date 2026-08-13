"""Tests for Home data-source catalog and market summary helpers."""

import pandas as pd

from happy_housing.app.home_stats import market_summary
from happy_housing.app.metadata import DEFAULT_SOURCE_CATALOG, DEFAULT_SOURCES


def test_source_catalog_has_names_and_https_urls() -> None:
    """Every catalog entry exposes a display name and https URL."""
    assert len(DEFAULT_SOURCE_CATALOG) >= 5
    names = [s.name for s in DEFAULT_SOURCE_CATALOG]
    assert names == DEFAULT_SOURCES
    for src in DEFAULT_SOURCE_CATALOG:
        assert src.name
        assert src.url.startswith("https://")
        assert src.role
        assert src.key
        assert src.unit


def test_market_summary_aggregates_key_stats() -> None:
    """Summary exposes counts and price/delta aggregates for charts."""
    frame = pd.DataFrame({
        "source": ["otodom", "otodom", "rcn"],
        "district": ["wola", "mokotow", "wola"],
        "price_per_m2": [10000.0, 20000.0, 8000.0],
        "delta_price_per_m2": [500.0, -200.0, 100.0],
        "in_oou": [0, 1, 0],
    })
    summary = market_summary(frame)
    assert summary["n_offers"] == 3
    assert summary["by_source"]["otodom"] == 2
    assert summary["median_price_per_m2"] == 10000.0
    assert summary["share_overpriced"] == 2 / 3
