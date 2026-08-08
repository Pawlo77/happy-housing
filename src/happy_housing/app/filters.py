"""Sidebar filter state and pure filter application for Explore."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class FilterState:
    """User-selected Explore filters from the sidebar."""

    district: str = "(all)"
    """District name or ``(all)``."""
    source: str = "(all)"
    """Listing source or ``(all)``."""
    oou: str = "(all)"
    """OOU membership: ``(all)``, ``inside``, or ``outside``."""
    max_price_per_m2: float | None = None
    """Upper bound on listing zł/m²."""
    month: str = "(all)"
    """Listed month ``YYYY-MM`` or ``(all)``."""


def apply_filters(frame: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Return rows matching ``state`` without mutating the input."""
    out = frame
    if state.district != "(all)" and "district" in out.columns:
        out = out[out["district"].astype(str) == state.district]
    if state.source != "(all)" and "source" in out.columns:
        out = out[out["source"].astype(str) == state.source]
    if state.oou == "inside" and "in_oou" in out.columns:
        out = out[out["in_oou"] == 1]
    elif state.oou == "outside" and "in_oou" in out.columns:
        out = out[out["in_oou"] == 0]
    if state.max_price_per_m2 is not None and "price_per_m2" in out.columns:
        out = out[out["price_per_m2"] <= float(state.max_price_per_m2)]
    if state.month != "(all)" and "listed_month" in out.columns:
        out = out[out["listed_month"].astype(str) == state.month]
    return out.reset_index(drop=True)
