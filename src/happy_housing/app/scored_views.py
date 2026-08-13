"""Column projections that keep Explore / Stats / Home off fat scored frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from happy_housing.app.commute_modes import COMMUTE_MODE_COLUMN
from happy_housing.gis.amenity_catalog import amenity_keys, dist_column

HEAVY_LISTING_COLUMNS: frozenset[str] = frozenset({
    "description",
    "image_url",
    "image_urls_json",
    "floor_plan_urls_json",
    "characteristics_json",
    "additional_information_json",
})
"""Wide Otodom text/media columns rarely needed outside Offer."""

HOME_VIEW_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "source",
    "district",
    "price_per_m2",
    "delta_price_per_m2",
    "in_oou",
)
"""Columns consumed by Home market summary / charts."""

BOARD_VIEW_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((
        "listing_id",
        "title",
        "district",
        "source",
        "market",
        "area_m2",
        "rooms",
        "floor_no",
        "year_built",
        "price_pln",
        "price_per_m2",
        "model_price_per_m2",
        "delta_price_per_m2",
        "has_balcony",
        "has_garage",
        "has_lift",
        "commute_min",
        *COMMUTE_MODE_COLUMN.values(),
        "dist_transit_stop_m",
        "in_oou",
        "listed_at",
        "url",
        *(dist_column(key) for key in amenity_keys()),
    ))
)
"""Board display + recommender score inputs (no description/media)."""

SPATIAL_VIEW_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "lat",
    "lon",
    "district",
    "source",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "area_m2",
    "listed_month",
    "commute_min",
)
"""Columns used by Spatial field / marker / district rollups."""


def project_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a shallow column subset (no copy when already projected)."""
    if frame.empty:
        keep = [c for c in columns if c in frame.columns]
        return frame.loc[:, keep] if keep else frame.iloc[:, 0:0]
    keep = [c for c in columns if c in frame.columns]
    if not keep:
        return frame.iloc[:, 0:0]
    if list(frame.columns) == keep:
        return frame
    return frame.loc[:, keep]


def drop_heavy_listing_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop Otodom text/media JSON columns when present."""
    drop = [c for c in HEAVY_LISTING_COLUMNS if c in frame.columns]
    if not drop:
        return frame
    return frame.drop(columns=drop)
