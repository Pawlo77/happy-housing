"""Sidebar filter state and pure filter application for Explore."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
    max_age_months: int | None = None
    """Keep offers with age ≤ this many months; ``None`` = no age filter."""


def offer_age_months(
    frame: pd.DataFrame,
    *,
    as_of: date | None = None,
) -> pd.Series:
    """Whole months since listing date (``listed_at``, else ``listed_month``)."""
    ref = as_of or date.today()
    n = len(frame)
    ages = pd.Series([pd.NA] * n, index=frame.index, dtype="Float64")
    if n == 0:
        return ages

    if "listed_at" in frame.columns:
        listed = pd.to_datetime(frame["listed_at"], errors="coerce", utc=False)
    else:
        listed = pd.Series(pd.NaT, index=frame.index)

    if "listed_month" in frame.columns:
        month_raw = frame["listed_month"].astype(str)
        month_dates = pd.to_datetime(
            month_raw.where(~month_raw.isin({"unknown", "NaT", "nan", "None"})),
            format="%Y-%m",
            errors="coerce",
        )
        listed = listed.fillna(month_dates)

    valid = listed.notna()
    if not valid.any():
        return ages

    years = listed.dt.year.astype("Int64")
    months = listed.dt.month.astype("Int64")
    days = listed.dt.day.astype("Int64")
    age = (ref.year - years) * 12 + (ref.month - months)
    age = age.where(days.isna() | (ref.day >= days), age - 1)
    age = age.clip(lower=0)
    return ages.where(~valid, age.astype("Float64"))


def apply_filters(
    frame: pd.DataFrame,
    state: FilterState,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Return rows matching ``state`` without mutating the input."""
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if state.district != "(all)" and "district" in frame.columns:
        mask &= frame["district"].astype(str).eq(state.district)
    if state.source != "(all)" and "source" in frame.columns:
        mask &= frame["source"].astype(str).eq(state.source)
    if state.oou == "inside" and "in_oou" in frame.columns:
        mask &= pd.to_numeric(frame["in_oou"], errors="coerce").eq(1)
    elif state.oou == "outside" and "in_oou" in frame.columns:
        mask &= pd.to_numeric(frame["in_oou"], errors="coerce").eq(0)
    if state.max_price_per_m2 is not None and "price_per_m2" in frame.columns:
        mask &= pd.to_numeric(frame["price_per_m2"], errors="coerce").le(
            float(state.max_price_per_m2)
        )
    if state.max_age_months is not None:
        ages = offer_age_months(frame, as_of=as_of)
        mask &= ages.notna() & ages.le(float(state.max_age_months))
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
