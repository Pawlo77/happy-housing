"""Incremental enrich: reuse prior GIS rows; recompute new or moved listings."""

import pandas as pd

from happy_housing.pipeline import (
    overlay_listing_columns,
    split_incremental_enrich_rows,
)


def test_split_incremental_enrich_reuses_unchanged_coords() -> None:
    current = pd.DataFrame({
        "listing_id": ["old", "new"],
        "lat": [52.23, 52.24],
        "lon": [21.00, 21.01],
        "price_pln": [900_000.0, 800_000.0],
    })
    prior = pd.DataFrame({
        "listing_id": ["old"],
        "lat": [52.23],
        "lon": [21.00],
        "price_pln": [850_000.0],
        "dist_daszynskiego_m": [1200.0],
        "commute_min": [8.5],
    })
    need, reuse = split_incremental_enrich_rows(current, prior)
    assert list(need["listing_id"]) == ["new"]
    assert list(reuse["listing_id"]) == ["old"]
    assert float(reuse["price_pln"].iloc[0]) == 900_000.0
    assert float(reuse["dist_daszynskiego_m"].iloc[0]) == 1200.0
    assert float(reuse["commute_min"].iloc[0]) == 8.5


def test_split_incremental_enrich_recomputes_moved_listing() -> None:
    current = pd.DataFrame({
        "listing_id": ["a"],
        "lat": [52.30],
        "lon": [21.10],
        "price_pln": [1.0],
    })
    prior = pd.DataFrame({
        "listing_id": ["a"],
        "lat": [52.23],
        "lon": [21.00],
        "price_pln": [1.0],
        "dist_daszynskiego_m": [1200.0],
    })
    need, reuse = split_incremental_enrich_rows(current, prior)
    assert list(need["listing_id"]) == ["a"]
    assert reuse.empty


def test_split_incremental_enrich_recomputes_when_gis_incomplete() -> None:
    current = pd.DataFrame({
        "listing_id": ["a"],
        "lat": [52.23],
        "lon": [21.00],
        "price_pln": [1.0],
    })
    prior = pd.DataFrame({
        "listing_id": ["a"],
        "lat": [52.23],
        "lon": [21.00],
        "price_pln": [1.0],
        "dist_daszynskiego_m": [float("nan")],
    })
    need, reuse = split_incremental_enrich_rows(current, prior)
    assert list(need["listing_id"]) == ["a"]
    assert reuse.empty


def test_split_incremental_enrich_without_prior_enriches_all() -> None:
    current = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lat": [52.23, 52.24],
        "lon": [21.00, 21.01],
    })
    need, reuse = split_incremental_enrich_rows(current, None)
    assert len(need) == 2
    assert reuse.empty


def test_overlay_listing_columns_keeps_enrich_features() -> None:
    prior = pd.DataFrame({
        "listing_id": ["a"],
        "price_pln": [1.0],
        "dist_daszynskiego_m": [99.0],
    })
    current = pd.DataFrame({
        "listing_id": ["a"],
        "price_pln": [2.0],
        "title": ["fresh"],
    })
    out = overlay_listing_columns(prior, current)
    assert float(out["price_pln"].iloc[0]) == 2.0
    assert str(out["title"].iloc[0]) == "fresh"
    assert float(out["dist_daszynskiego_m"].iloc[0]) == 99.0
