"""Sidebar filter state and pure filter application for Explore."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from happy_housing.app.listing_attrs import apply_property_attr_filters


@dataclass(slots=True)
class FilterState:
    """User-selected Explore filters from the sidebar."""

    district: str = "(all)"
    """District name or ``(all)``."""
    source: str = "(all)"
    """Listing source or ``(all)``."""
    oou: str = "(all)"
    """Aircraft Lden ≥55 dB: ``(all)``, ``inside``, or ``outside``."""
    high_noise: str = "(all)"
    """Total Lden ≥55 dB: ``(all)``, ``inside``, or ``outside``."""
    poor_air: str = "(all)"
    """IDW AQI ≥3: ``(all)``, ``inside``, or ``outside``."""
    max_noise_total_lden_db: float | None = None
    """Upper bound on energy-sum total Lden (dB); ``None`` = no cap."""
    max_traffic_weight_nearby: float | None = None
    """Upper bound on OSM traffic-proxy weight sum; ``None`` = no cap."""
    max_major_traffic_weight_nearby: float | None = None
    """Upper bound on major-road traffic weight sum; ``None`` = no cap."""
    frontage: str = "(all)"
    """``(all)``, ``street`` (street_facing_hint), or ``quiet`` (quiet_side_hint)."""
    max_price_per_m2: float | None = None
    """Upper bound on listing zł/m²."""
    max_age_months: int | None = None
    """Keep offers with age ≤ this many months; ``None`` = no age filter."""
    object_type: str = "(all)"
    """``flat`` / ``house`` / ``(all)``."""
    ownership: str = "(all)"
    """``full`` / ``usufruct`` / ``other`` / ``(all)``."""
    has_garage: str = "(all)"
    has_outdoor_parking: str = "(all)"
    has_basement: str = "(all)"
    has_balcony: str = "(all)"
    has_garden: str = "(all)"
    has_terrace: str = "(all)"
    has_lift: str = "(all)"
    is_attic: str = "(all)"
    needs_renovation: str = "(all)"
    plumbing_replaced: str = "(all)"
    wiring_replaced: str = "(all)"
    floor_min: int | None = None
    floor_max: int | None = None
    year_built_min: int | None = None
    year_renovated_min: int | None = None


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
    if state.high_noise == "inside" and "in_high_noise" in frame.columns:
        mask &= pd.to_numeric(frame["in_high_noise"], errors="coerce").eq(1)
    elif state.high_noise == "outside" and "in_high_noise" in frame.columns:
        mask &= pd.to_numeric(frame["in_high_noise"], errors="coerce").eq(0)
    if state.poor_air == "inside" and "in_poor_air" in frame.columns:
        mask &= pd.to_numeric(frame["in_poor_air"], errors="coerce").eq(1)
    elif state.poor_air == "outside" and "in_poor_air" in frame.columns:
        mask &= pd.to_numeric(frame["in_poor_air"], errors="coerce").eq(0)
    if (
        state.max_noise_total_lden_db is not None
        and "noise_total_lden_db" in frame.columns
    ):
        mask &= pd.to_numeric(frame["noise_total_lden_db"], errors="coerce").le(
            float(state.max_noise_total_lden_db)
        )
    if (
        state.max_traffic_weight_nearby is not None
        and "traffic_weight_nearby" in frame.columns
    ):
        mask &= pd.to_numeric(frame["traffic_weight_nearby"], errors="coerce").le(
            float(state.max_traffic_weight_nearby)
        )
    if (
        state.max_major_traffic_weight_nearby is not None
        and "major_traffic_weight_nearby" in frame.columns
    ):
        mask &= pd.to_numeric(frame["major_traffic_weight_nearby"], errors="coerce").le(
            float(state.max_major_traffic_weight_nearby)
        )
    if state.frontage == "street" and "street_facing_hint" in frame.columns:
        mask &= pd.to_numeric(frame["street_facing_hint"], errors="coerce").eq(1)
    elif state.frontage == "quiet" and "quiet_side_hint" in frame.columns:
        mask &= pd.to_numeric(frame["quiet_side_hint"], errors="coerce").eq(1)
    if state.max_price_per_m2 is not None and "price_per_m2" in frame.columns:
        mask &= pd.to_numeric(frame["price_per_m2"], errors="coerce").le(
            float(state.max_price_per_m2)
        )
    if state.max_age_months is not None:
        ages = offer_age_months(frame, as_of=as_of)
        mask &= ages.notna() & ages.le(float(state.max_age_months))
    filtered = frame.loc[mask].reset_index(drop=True)
    return apply_property_attr_filters(
        filtered,
        object_type=state.object_type,
        ownership=state.ownership,
        has_garage=state.has_garage,
        has_outdoor_parking=state.has_outdoor_parking,
        has_basement=state.has_basement,
        has_balcony=state.has_balcony,
        has_garden=state.has_garden,
        has_terrace=state.has_terrace,
        has_lift=state.has_lift,
        is_attic=state.is_attic,
        needs_renovation=state.needs_renovation,
        plumbing_replaced=state.plumbing_replaced,
        wiring_replaced=state.wiring_replaced,
        floor_min=state.floor_min,
        floor_max=state.floor_max,
        year_built_min=state.year_built_min,
        year_renovated_min=state.year_renovated_min,
    )


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
