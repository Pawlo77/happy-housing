"""Tests for RCN neighborhood spatial features."""

import pandas as pd

from happy_housing.models.neighborhood import (
    attach_neighborhood_rcn_features,
    idw_neighborhood_baseline,
)


def _rcn_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "lon": [21.00, 21.001, 21.002, 21.05],
        "lat": [52.23, 52.231, 52.229, 52.30],
        "price_per_m2": [15_000.0, 16_000.0, 14_000.0, 30_000.0],
        "price_pln": [750_000.0, 800_000.0, 700_000.0, 1_500_000.0],
        "area_m2": [50.0, 50.0, 50.0, 50.0],
    })


def test_attach_neighborhood_rcn_features_nearby_median() -> None:
    listings = pd.DataFrame({
        "listing_id": ["A"],
        "lon": [21.0005],
        "lat": [52.2305],
        "price_per_m2": [15_500.0],
    })
    out = attach_neighborhood_rcn_features(
        listings,
        _rcn_frame(),
        radius_m=500.0,
        min_neighbors=1,
    )
    assert "nbhd_rcn_median_ppm" in out.columns
    assert "nbhd_rcn_n" in out.columns
    assert out.iloc[0]["nbhd_rcn_n"] >= 1
    assert 13_000 < float(out.iloc[0]["nbhd_rcn_median_ppm"]) < 17_000


def test_idw_baseline_predicts_finite() -> None:
    listings = pd.DataFrame({
        "lon": [21.0005],
        "lat": [52.2305],
    })
    preds = idw_neighborhood_baseline(listings, _rcn_frame(), radius_m=800.0)
    assert len(preds) == 1
    assert preds[0] == preds[0]  # not NaN


def test_neighborhood_excludes_self_sale_at_same_coords() -> None:
    """RCN row at identical lat/lon must not enter its own nbhd aggregates."""
    rcn = pd.DataFrame({
        "listing_id": ["RCN-1", "RCN-2"],
        "lon": [21.0, 21.001],
        "lat": [52.23, 52.231],
        "price_per_m2": [10_000.0, 20_000.0],
    })
    listings = pd.DataFrame({
        "listing_id": ["RCN-1"],
        "lon": [21.0],
        "lat": [52.23],
        "price_per_m2": [10_000.0],
    })
    out = attach_neighborhood_rcn_features(listings, rcn, radius_m=500.0)
    # Only RCN-2 should remain as a comp (~111m away)
    assert float(out.iloc[0]["nbhd_rcn_n"]) == 1.0
    assert float(out.iloc[0]["nbhd_rcn_median_ppm"]) == 20_000.0
    assert float(out.iloc[0]["nbhd_rcn_nearest_m"]) > 5.0
