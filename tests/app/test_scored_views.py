"""Tests for slim scored-frame projections used by Board / Home / Spatial."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.scored_views import (
    BOARD_VIEW_COLUMNS,
    HOME_VIEW_COLUMNS,
    drop_heavy_listing_columns,
    project_frame,
)


def _fat_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": ["a"],
        "lat": [52.2],
        "lon": [21.0],
        "district": ["Wola"],
        "source": ["otodom"],
        "price_per_m2": [15_000.0],
        "delta_price_per_m2": [-100.0],
        "in_oou": [0],
        "price_pln": [750_000.0],
        "area_m2": [50.0],
        "rooms": [2],
        "description": ["x" * 20_000],
        "image_urls_json": ['["http://example/a.jpg"]'],
        "floor_plan_urls_json": ["[]"],
        "characteristics_json": ["[]"],
        "additional_information_json": ["[]"],
    })


def test_drop_heavy_listing_columns_removes_media_and_json() -> None:
    out = drop_heavy_listing_columns(_fat_frame())
    assert "description" not in out.columns
    assert "image_urls_json" not in out.columns
    assert "listing_id" in out.columns
    assert "price_per_m2" in out.columns


def test_project_frame_keeps_requested_only() -> None:
    out = project_frame(_fat_frame(), HOME_VIEW_COLUMNS)
    assert set(out.columns) <= set(HOME_VIEW_COLUMNS)
    assert "description" not in out.columns
    assert len(out) == 1


def test_board_view_columns_cover_sort_and_score_inputs() -> None:
    assert "listing_id" in BOARD_VIEW_COLUMNS
    assert "delta_price_per_m2" in BOARD_VIEW_COLUMNS
    assert "commute_min" in BOARD_VIEW_COLUMNS
    assert "description" not in BOARD_VIEW_COLUMNS
