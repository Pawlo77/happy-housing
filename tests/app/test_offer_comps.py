"""Tests for Offer Comps membership, size band, loyalty, and metro flags."""

import pandas as pd
import pytest

from happy_housing.app.offer_comps import (
    drop_centroid_fallbacks,
    flag_metro_rows,
    local_loyalty,
    local_monthly_medians,
    membership_stats,
    prepare_offer_comps,
    similar_size_band,
)
from happy_housing.models.neighborhood import query_rcn_neighbors


def _deeds() -> pd.DataFrame:
    return pd.DataFrame({
        "lon": [21.00, 21.001, 21.002, 20.91],
        "lat": [52.23, 52.231, 52.229, 52.239],
        "price_per_m2": [15_000.0, 16_000.0, 14_000.0, 18_000.0],
        "price_pln": [750_000.0, 800_000.0, 700_000.0, 900_000.0],
        "area_m2": [50.0, 52.0, 48.0, 80.0],
        "listed_at": [
            "2026-06-01",
            "2025-06-01",
            "2024-01-01",
            "2026-01-01",
        ],
        "district": ["Wola", "Wola", "Wola", "Bemowo"],
        "coord_source": ["parcel", "parcel", "parcel", "centroid"],
        "source": ["rcn"] * 4,
    })


def test_membership_percentile_and_gap() -> None:
    neighbors = query_rcn_neighbors(52.23, 21.00, _deeds(), radius_m=400.0)
    stats = membership_stats(15_500.0, neighbors)
    assert stats["n"] >= 2
    assert 0.0 <= float(stats["percentile"]) <= 100.0
    assert stats["median_ppm"] is not None
    assert stats["gap_ppm"] == pytest.approx(15_500.0 - float(stats["median_ppm"]))


def test_membership_empty_neighbors() -> None:
    stats = membership_stats(15_000.0, pd.DataFrame())
    assert stats["n"] == 0
    assert stats["percentile"] is None


def test_similar_size_band_keeps_close_area() -> None:
    band = similar_size_band(_deeds(), area_m2=50.0)
    assert set(band["area_m2"]) <= {50.0, 52.0, 48.0}


def test_drop_centroid_fallbacks() -> None:
    kept = drop_centroid_fallbacks(_deeds())
    assert (kept["coord_source"] == "parcel").all()
    assert len(kept) == 3


def test_flag_metro_unlabelled_and_outside_city() -> None:
    frame = pd.DataFrame({
        "lon": [21.01, 21.35],
        "lat": [52.23, 52.20],
        "district": ["Wola", "rcn"],
    })
    out = flag_metro_rows(frame)
    assert bool(out.iloc[0]["is_metro"]) is False
    assert bool(out.iloc[1]["is_metro"]) is True


def test_prepare_offer_comps_empty_radius_caption_fields() -> None:
    far = pd.DataFrame({
        "lon": [21.20],
        "lat": [52.10],
        "price_per_m2": [15_000.0],
        "area_m2": [50.0],
        "listed_at": ["2026-01-01"],
        "district": ["Wola"],
        "coord_source": ["parcel"],
        "source": ["rcn"],
    })
    bundle = prepare_offer_comps(
        lat=52.23,
        lon=21.00,
        ask_ppm=15_500.0,
        area_m2=50.0,
        listing_id="ASK-1",
        rcn=far,
        asks=pd.DataFrame(),
        radius_m=50.0,
        show_nearby_asks=False,
    )
    assert bundle["stats"]["n"] == 0
    assert bundle["neighbors"].empty
    assert bundle["map_rcn"].empty


def test_local_loyalty_and_monthly() -> None:
    neighbors = query_rcn_neighbors(52.23, 21.00, _deeds(), radius_m=800.0)
    loyalty = local_loyalty(neighbors, as_of="2026-08-01")
    assert loyalty["n"] >= 1
    assert "iqr_ppm" in loyalty
    months = local_monthly_medians(neighbors)
    assert not months.empty
    assert {"listed_month", "n", "median_ppm"}.issubset(months.columns)
