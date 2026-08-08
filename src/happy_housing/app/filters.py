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
    """Aircraft Lden ≥55 dB: ``(all)``, ``inside``, or ``outside``."""
    max_price_per_m2: float | None = None
    """Upper bound on listing zł/m²."""
    month: str = "(all)"
    """Listed month ``YYYY-MM`` or ``(all)``."""


def apply_filters(frame: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Return rows matching ``state`` without mutating the input."""
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if state.district != "(all)" and "district" in frame.columns:
        mask &= frame["district"].astype(str).eq(state.district)
    if state.source != "(all)" and "source" in frame.columns:
        mask &= frame["source"].astype(str).eq(state.source)
    if state.oou == "inside" and "in_oou" in frame.columns:
        mask &= frame["in_oou"].eq(1)
    elif state.oou == "outside" and "in_oou" in frame.columns:
        mask &= frame["in_oou"].eq(0)
    if state.max_price_per_m2 is not None and "price_per_m2" in frame.columns:
        mask &= frame["price_per_m2"].le(float(state.max_price_per_m2))
    if state.month != "(all)" and "listed_month" in frame.columns:
        mask &= frame["listed_month"].astype(str).eq(state.month)
    return frame.loc[mask].reset_index(drop=True)


def apply_starred_only(
    frame: pd.DataFrame,
    starred_ids: frozenset[str] | set[str],
) -> pd.DataFrame:
    """Keep only rows whose ``listing_id`` is starred."""
    if frame.empty or "listing_id" not in frame.columns:
        return frame.copy()
    ids = {str(x) for x in starred_ids}
    mask = frame["listing_id"].astype(str).isin(ids)
    return frame.loc[mask].reset_index(drop=True)
