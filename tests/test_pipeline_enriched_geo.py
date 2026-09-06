"""Tests for pipeline enrich GeoParquet helpers."""

import pandas as pd

from happy_housing.pipeline import enriched_listings_geodataframe


def test_enriched_listings_geodataframe_matches_frame_length() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "lon": [21.0, 21.1, None],
        "lat": [52.2, 52.3, 52.4],
        "price_per_m2": [10_000.0, 11_000.0, 12_000.0],
    })
    geo = enriched_listings_geodataframe(frame)
    assert len(geo) == 3
    assert list(geo["listing_id"]) == ["a", "b", "c"]
    assert geo.crs is not None
    assert geo.crs.to_epsg() == 4326
