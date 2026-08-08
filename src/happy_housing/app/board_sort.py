"""Explore Board enrichment and sort helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from happy_housing.app.preferences import UserPreferences
from happy_housing.app.recommender import score_offers

BOARD_SORT_RECOMMENDATION = "recommendation"
"""Default sort: recommender confidence with unseen offers first."""

BOARD_SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    (BOARD_SORT_RECOMMENDATION, "Recommendation (best · unseen first)"),
    ("price_pln_asc", "Price PLN ↑"),
    ("price_pln_desc", "Price PLN ↓"),
    ("price_per_m2_asc", "zł/m² ↑"),
    ("price_per_m2_desc", "zł/m² ↓"),
    ("delta_asc", "Delta (most underpriced)"),
    ("delta_desc", "Delta (most overpriced)"),
    ("district_asc", "District A-Z"),
    ("district_desc", "District Z-A"),
    ("listed_at_desc", "Posted (newest)"),
    ("listed_at_asc", "Posted (oldest)"),
    ("area_m2_asc", "Area m² ↑"),
    ("area_m2_desc", "Area m² ↓"),
    ("rooms_asc", "Rooms ↑"),
    ("rooms_desc", "Rooms ↓"),
    ("commute_asc", "Commute min ↑"),
    ("commute_desc", "Commute min ↓"),
)
"""Selectable board sort keys and UI labels."""

_SORT_SPECS: dict[str, tuple[str, bool]] = {
    "price_pln_asc": ("price_pln", True),
    "price_pln_desc": ("price_pln", False),
    "price_per_m2_asc": ("price_per_m2", True),
    "price_per_m2_desc": ("price_per_m2", False),
    "delta_asc": ("delta_price_per_m2", True),
    "delta_desc": ("delta_price_per_m2", False),
    "district_asc": ("district", True),
    "district_desc": ("district", False),
    "listed_at_desc": ("listed_at", False),
    "listed_at_asc": ("listed_at", True),
    "area_m2_asc": ("area_m2", True),
    "area_m2_desc": ("area_m2", False),
    "rooms_asc": ("rooms", True),
    "rooms_desc": ("rooms", False),
    "commute_asc": ("commute_min", True),
    "commute_desc": ("commute_min", False),
}
"""Non-recommendation sorts: column name + ascending flag."""


def coerce_board_sort(value: object) -> str:
    """Return a known board sort key, defaulting to recommendation."""
    key = str(value or BOARD_SORT_RECOMMENDATION).strip()
    allowed = {opt for opt, _ in BOARD_SORT_OPTIONS}
    return key if key in allowed else BOARD_SORT_RECOMMENDATION


def enrich_board_recommendations(
    frame: pd.DataFrame,
    *,
    impressions: dict[str, dict[str, Any]],
    demotion_alpha: float,
    prefs: UserPreferences | None = None,
    amenity_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach recommender confidence and seen flags for board display/sort."""
    if frame.empty or "listing_id" not in frame.columns:
        out = frame.copy()
        out["confidence"] = pd.Series(dtype=float)
        out["seen"] = pd.Series(dtype=str)
        out["opened"] = pd.Series(dtype=bool)
        out["impression_count"] = pd.Series(dtype=int)
        return out
    work = frame.copy()
    work["listing_id"] = work["listing_id"].astype(str)
    ranked = score_offers(
        work,
        demotion_alpha=demotion_alpha,
        impressions=impressions,
        prefs=prefs,
        amenity_weights=amenity_weights,
    )
    conf = ranked.set_index("listing_id")["confidence"]
    work["confidence"] = work["listing_id"].map(conf).astype(float)
    counts: list[int] = []
    opened_flags: list[bool] = []
    seen_labels: list[str] = []
    for lid in work["listing_id"]:
        entry = impressions.get(str(lid)) or {}
        count = int(entry.get("count") or 0)
        opened = bool(entry.get("opened"))
        counts.append(count)
        opened_flags.append(opened)
        if opened:
            seen_labels.append("opened")
        elif count > 0:
            seen_labels.append(f"seen x{count}")
        else:
            seen_labels.append("new")
    work["impression_count"] = counts
    work["opened"] = opened_flags
    work["seen"] = seen_labels
    return work


def sort_board(frame: pd.DataFrame, sort_key: str) -> pd.DataFrame:
    """Sort board rows; recommendation puts new/unopened + high confidence first."""
    if frame.empty:
        return frame.reset_index(drop=True)
    key = coerce_board_sort(sort_key)
    work = frame.copy()
    if key == BOARD_SORT_RECOMMENDATION:
        if "opened" not in work.columns:
            work["opened"] = False
        if "impression_count" not in work.columns:
            work["impression_count"] = 0
        if "confidence" not in work.columns:
            work["confidence"] = 0.0
        return work.sort_values(
            by=["opened", "impression_count", "confidence"],
            ascending=[True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)

    column, ascending = _SORT_SPECS[key]
    if column not in work.columns:
        return work.reset_index(drop=True)
    series = work[column]
    if column in {
        "price_pln",
        "price_per_m2",
        "delta_price_per_m2",
        "area_m2",
        "rooms",
        "commute_min",
    }:
        work = work.assign(_sort_key=pd.to_numeric(series, errors="coerce"))
        return (
            work
            .sort_values(
                by="_sort_key",
                ascending=ascending,
                kind="mergesort",
                na_position="last",
            )
            .drop(columns=["_sort_key"])
            .reset_index(drop=True)
        )
    if column == "listed_at":
        work = work.assign(
            _sort_key=pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        )
        return (
            work
            .sort_values(
                by="_sort_key",
                ascending=ascending,
                kind="mergesort",
                na_position="last",
            )
            .drop(columns=["_sort_key"])
            .reset_index(drop=True)
        )
    work = work.assign(_sort_key=series.astype(str))
    return (
        work
        .sort_values(
            by="_sort_key",
            ascending=ascending,
            kind="mergesort",
            na_position="last",
        )
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )
