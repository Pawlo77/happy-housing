"""Tests for Explore Spatial metric heatmaps and district rollups."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.spatial_insights import (
    SPATIAL_FIELD_MODE_HISTORICAL,
    SPATIAL_FIELD_MODE_MODEL,
    SPATIAL_METRICS,
    district_spatial_summary,
    prepare_spatial_field_points,
    prepare_spatial_marker_points,
    prepare_spatial_points,
    spatial_kpis,
)
from happy_housing.app.stats_insights import normalize_district
from happy_housing.gis.intensity_field import prepare_weighted_points_overlay


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a", "b", "c", "d"],
        "district": ["wola", "wola", "mokotow", "mokotow"],
        "source": ["otodom", "rcn", "otodom", "rcn"],
        "lat": [52.23, 52.24, 52.20, 52.21],
        "lon": [21.00, 21.01, 21.02, 21.03],
        "price_per_m2": [15_000.0, 12_000.0, 20_000.0, 18_000.0],
        "model_price_per_m2": [16_000.0, 11_500.0, 21_000.0, 17_000.0],
        "area_m2": [40.0, 50.0, 60.0, 55.0],
        "commute_min": [10.0, 20.0, 35.0, 40.0],
        "delta_price_per_m2": [-200.0, 100.0, -800.0, 50.0],
        "listed_month": ["2026-08", "2026-07", "2026-01", "2026-08"],
    })


def test_spatial_metrics_catalog_nonempty() -> None:
    assert "price_ppm" in {m.key for m in SPATIAL_METRICS}
    assert "sold_density" in {m.key for m in SPATIAL_METRICS}


def test_prepare_spatial_points_price_weights() -> None:
    points = prepare_spatial_points(_frame(), "price_ppm")
    assert not points.empty
    assert set(points.columns) >= {"lon", "lat", "weight"}
    assert points["weight"].min() > 0


def test_prepare_spatial_points_sold_keeps_rcn_only() -> None:
    points = prepare_spatial_points(_frame(), "sold_density")
    assert len(points) == 2
    assert (points["weight"] > 0).all()


def test_district_spatial_summary_avg_area() -> None:
    summary = district_spatial_summary(_frame(), "avg_area", min_offers=2)
    wola = summary.loc[summary["district"] == "wola"].iloc[0]
    assert float(wola["value"]) == 45.0
    assert int(wola["n"]) == 2


def test_spatial_kpis_commute() -> None:
    kpis = spatial_kpis(_frame(), "commute")
    assert kpis["n_points"] >= 1
    assert "headline" in kpis
    assert kpis["headline_value"] == kpis["headline_value"]


def test_spatial_field_model_vs_historical_price() -> None:
    frame = _frame()
    model = prepare_spatial_field_points(
        frame, "price_ppm", field_mode=SPATIAL_FIELD_MODE_MODEL
    )
    hist = prepare_spatial_field_points(
        frame, "price_ppm", field_mode=SPATIAL_FIELD_MODE_HISTORICAL
    )
    assert list(model["weight"]) != list(hist["weight"])
    assert float(hist.loc[hist["lon"] == 21.00, "weight"].iloc[0]) == 15_000.0
    assert float(model.loc[model["lon"] == 21.00, "weight"].iloc[0]) == 16_000.0


def test_spatial_markers_are_historical_asks() -> None:
    markers = prepare_spatial_marker_points(_frame(), "price_ppm")
    assert len(markers) == 4
    assert markers["tooltip_detail"].str.contains("Historical ask").all()


def test_spatial_weighted_overlay_is_bitmap_payload() -> None:
    pts = prepare_spatial_field_points(_frame(), "price_ppm")
    overlay = prepare_weighted_points_overlay(
        pts["lon"].to_numpy(),
        pts["lat"].to_numpy(),
        pts["weight"].to_numpy(),
        bbox_wgs84=(20.85, 52.10, 21.28, 52.37),
        layer_key="spatial-test",
    )
    assert overlay is not None
    assert overlay.image_data_uri.startswith("data:image/png;base64,")


def test_normalize_district_strips_diacritics() -> None:
    assert normalize_district("Śródmieście") == normalize_district("srodmiescie")
    assert normalize_district("Mokotów") == "mokotow"
