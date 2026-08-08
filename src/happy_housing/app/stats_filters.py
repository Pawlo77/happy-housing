"""Filter state for the Stats page (business analytics lens)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from happy_housing.app.listing_attrs import apply_property_attr_filters
from happy_housing.app.stats_insights import normalize_district


@dataclass(slots=True)
class StatsFilterState:
    """User lens applied before Stats aggregations and charts."""

    districts: list[str] = field(default_factory=list)
    """Normalized district names; empty means all."""
    sources: list[str] = field(default_factory=list)
    """Source labels as stored; empty means all."""
    months: list[str] = field(default_factory=list)
    """Listed months ``YYYY-MM``; empty means all."""
    oou: str = "(all)"
    """Aircraft Lden ≥55 dB: ``(all)``, ``inside``, or ``outside``."""
    valuation: str = "(all)"
    """Residual side: ``(all)``, ``underpriced``, or ``overpriced``."""
    price_min: float | None = None
    """Lower bound on listing zł/m²."""
    price_max: float | None = None
    """Upper bound on listing zł/m²."""
    area_min: float | None = None
    """Lower bound on area m²."""
    area_max: float | None = None
    """Upper bound on area m²."""
    rooms_min: int | None = None
    """Minimum room count (inclusive)."""
    rooms_max: int | None = None
    """Maximum room count (inclusive)."""
    object_type: str = "(all)"
    ownership: str = "(all)"
    has_garage: str = "(all)"
    has_outdoor_parking: str = "(all)"
    has_basement: str = "(all)"
    has_balcony: str = "(all)"
    has_garden: str = "(all)"
    has_terrace: str = "(all)"
    is_attic: str = "(all)"
    needs_renovation: str = "(all)"
    plumbing_replaced: str = "(all)"
    wiring_replaced: str = "(all)"
    floor_min: int | None = None
    floor_max: int | None = None
    year_built_min: int | None = None
    year_renovated_min: int | None = None


def apply_stats_filters(frame: pd.DataFrame, state: StatsFilterState) -> pd.DataFrame:
    """Return rows matching ``state`` without mutating the input."""
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if state.districts and "district" in frame.columns:
        wanted = {normalize_district(d) for d in state.districts}
        mask &= frame["district"].map(normalize_district).isin(wanted)
    if state.sources and "source" in frame.columns:
        wanted_src = {str(s) for s in state.sources}
        mask &= frame["source"].astype(str).isin(wanted_src)
    if state.months and "listed_month" in frame.columns:
        wanted_m = {str(m) for m in state.months}
        mask &= frame["listed_month"].astype(str).isin(wanted_m)
    if state.oou == "inside" and "in_oou" in frame.columns:
        mask &= pd.to_numeric(frame["in_oou"], errors="coerce").eq(1)
    elif state.oou == "outside" and "in_oou" in frame.columns:
        mask &= pd.to_numeric(frame["in_oou"], errors="coerce").eq(0)
    if state.valuation == "underpriced" and "delta_price_per_m2" in frame.columns:
        mask &= pd.to_numeric(frame["delta_price_per_m2"], errors="coerce").lt(0)
    elif state.valuation == "overpriced" and "delta_price_per_m2" in frame.columns:
        mask &= pd.to_numeric(frame["delta_price_per_m2"], errors="coerce").gt(0)
    if state.price_min is not None and "price_per_m2" in frame.columns:
        mask &= pd.to_numeric(frame["price_per_m2"], errors="coerce").ge(
            float(state.price_min)
        )
    if state.price_max is not None and "price_per_m2" in frame.columns:
        mask &= pd.to_numeric(frame["price_per_m2"], errors="coerce").le(
            float(state.price_max)
        )
    if state.area_min is not None and "area_m2" in frame.columns:
        mask &= pd.to_numeric(frame["area_m2"], errors="coerce").ge(
            float(state.area_min)
        )
    if state.area_max is not None and "area_m2" in frame.columns:
        mask &= pd.to_numeric(frame["area_m2"], errors="coerce").le(
            float(state.area_max)
        )
    if state.rooms_min is not None and "rooms" in frame.columns:
        mask &= pd.to_numeric(frame["rooms"], errors="coerce").ge(int(state.rooms_min))
    if state.rooms_max is not None and "rooms" in frame.columns:
        mask &= pd.to_numeric(frame["rooms"], errors="coerce").le(int(state.rooms_max))
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
        is_attic=state.is_attic,
        needs_renovation=state.needs_renovation,
        plumbing_replaced=state.plumbing_replaced,
        wiring_replaced=state.wiring_replaced,
        floor_min=state.floor_min,
        floor_max=state.floor_max,
        year_built_min=state.year_built_min,
        year_renovated_min=state.year_renovated_min,
    )
