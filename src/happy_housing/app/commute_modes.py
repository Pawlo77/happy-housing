"""Commute mode labels, columns, and display-mode helpers."""

from typing import Literal

import numpy as np
import pandas as pd

CommuteDisplayMode = Literal["transit", "car", "bike", "walk"]
"""Modes available for offer hover / filters / recommender commute signal."""

COMMUTE_DISPLAY_MODES: tuple[CommuteDisplayMode, ...] = (
    "transit",
    "car",
    "bike",
    "walk",
)
"""Stable order for Admin selectboxes."""

COMMUTE_DISPLAY_DEFAULT: CommuteDisplayMode = "transit"
"""Default offer-hover commute mode (public transport)."""

COMMUTE_MODE_COLUMN: dict[CommuteDisplayMode, str] = {
    "car": "commute_min",
    "transit": "transit_commute_min",
    "bike": "bike_commute_min",
    "walk": "walk_commute_min",
}
"""Display mode → listing column (``commute_min`` remains car for back-compat)."""

COMMUTE_MODE_LABELS: dict[CommuteDisplayMode, str] = {
    "transit": "Public transport",
    "car": "Car",
    "bike": "Bike",
    "walk": "Walk",
}
"""Admin / UI labels."""

COMMUTE_TOOLTIP_LABELS: dict[CommuteDisplayMode, str] = {
    "transit": "Transit",
    "car": "Car",
    "bike": "Bike",
    "walk": "Walk",
}
"""Short labels for map hover lines."""


def coerce_commute_display_mode(raw: object | None) -> CommuteDisplayMode:
    """Normalize Admin / settings value to a known commute display mode."""
    if raw is None:
        return COMMUTE_DISPLAY_DEFAULT
    key = str(raw).strip().casefold()
    if key in COMMUTE_MODE_COLUMN:
        return key  # type: ignore[return-value]  # Validated literal mode key
    aliases = {
        "public": "transit",
        "public_transport": "transit",
        "pt": "transit",
        "driving": "car",
        "bicycle": "bike",
        "cycling": "bike",
        "foot": "walk",
        "walking": "walk",
    }
    mapped = aliases.get(key)
    if mapped in COMMUTE_MODE_COLUMN:
        return mapped  # type: ignore[return-value]  # Mapping values are mode literals
    return COMMUTE_DISPLAY_DEFAULT


def commute_column_for_mode(mode: object | None) -> str:
    """Listing column for the given display mode."""
    return COMMUTE_MODE_COLUMN[coerce_commute_display_mode(mode)]


def commute_tooltip_label(mode: object | None) -> str:
    """Short mode word for map tooltips (e.g. ``Transit``)."""
    return COMMUTE_TOOLTIP_LABELS[coerce_commute_display_mode(mode)]


def commute_series(frame: pd.DataFrame, mode: object | None) -> pd.Series:
    """Numeric commute minutes for ``mode``; all-NaN when column missing."""
    col = commute_column_for_mode(mode)
    if frame.empty:
        return pd.Series(dtype=float)
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")
