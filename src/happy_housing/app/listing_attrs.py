"""Promote / normalize listing amenity columns for filters (no LLM)."""

from __future__ import annotations

import json
import logging

import pandas as pd

from happy_housing.ingest.otodom import flags_from_extras_and_tags

logger = logging.getLogger(__name__)

ATTR_FILTER_COLUMNS: tuple[str, ...] = (
    "object_type",
    "floor_no",
    "is_attic",
    "has_garden",
    "has_garage",
    "has_outdoor_parking",
    "has_basement",
    "has_balcony",
    "has_terrace",
    "has_lift",
    "ownership",
    "needs_renovation",
    "year_built",
    "year_renovated",
    "plumbing_replaced",
    "wiring_replaced",
)
"""Columns used by Stats / Explore property filters."""


def _tags_set(raw: object) -> set[str]:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {str(x).upper() for x in parsed}
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return {t.strip().upper() for t in text.split(",") if t.strip()}


def _coalesce_flag(existing: object, derived: int) -> int:
    cur: int | None = None
    if existing is not None and not pd.isna(existing):
        try:
            cur = int(str(existing))
        except (TypeError, ValueError):
            cur = None
    if cur is None:
        return int(derived)
    return int(max(cur, derived))


def promote_listing_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill basement / outdoor parking / attic / object_type from structured cols.

    Safe on scored frames that predate new Otodom columns — derives from
    ``tags_json`` / ``floor_label`` / ``estate`` when present.
    """
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for col in (
        "has_basement",
        "has_outdoor_parking",
        "has_garage",
        "has_balcony",
        "has_terrace",
        "has_garden",
        "has_lift",
        "is_attic",
    ):
        if col not in out.columns:
            out[col] = pd.NA

    if "tags_json" in out.columns or "floor_label" in out.columns:
        tags_series = (
            out["tags_json"]
            if "tags_json" in out.columns
            else pd.Series([None] * len(out))
        )
        floor_labels = (
            out["floor_label"]
            if "floor_label" in out.columns
            else pd.Series([None] * len(out))
        )
        floor_nos = (
            out["floor_no"]
            if "floor_no" in out.columns
            else pd.Series([None] * len(out))
        )
        derived_rows: list[dict[str, int]] = []
        for tags_raw, flab, fno in zip(
            tags_series.to_numpy(),
            floor_labels.to_numpy(),
            floor_nos.to_numpy(),
            strict=True,
        ):
            try:
                floor_i = int(float(fno)) if fno is not None and fno == fno else None
            except (TypeError, ValueError):
                floor_i = None
            derived_rows.append(
                flags_from_extras_and_tags(
                    extras=set(),
                    tags=_tags_set(tags_raw),
                    floor_label=str(flab)
                    if flab is not None and flab == flab
                    else None,
                    floor_no=floor_i,
                )
            )
        derived = pd.DataFrame(derived_rows, index=out.index)
        for col in derived.columns:
            out[col] = [
                _coalesce_flag(a, int(b))
                for a, b in zip(
                    out[col].to_numpy(), derived[col].to_numpy(), strict=True
                )
            ]

    if "object_type" not in out.columns:
        out["object_type"] = pd.NA
    estate = (
        out["estate"].astype(str).str.upper()
        if "estate" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    btype = (
        out["building_type"].astype(str).str.lower()
        if "building_type" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    missing_ot = out["object_type"].isna() | (out["object_type"].astype(str) == "")
    house = estate.eq("HOUSE") | btype.str.contains("house|dom", regex=True, na=False)
    flat = estate.eq("FLAT") | (~house & btype.ne("") & btype.ne("nan"))
    out.loc[missing_ot & house, "object_type"] = "house"
    out.loc[missing_ot & flat & ~house, "object_type"] = "flat"

    if "ownership" not in out.columns:
        out["ownership"] = pd.NA
    if "building_ownership" in out.columns:
        map_own = {
            "full_ownership": "full",
            "usufruct": "usufruct",
            "limited_ownership": "other",
            "share": "other",
        }
        mapped = out["building_ownership"].astype(str).str.lower().map(map_own)
        out["ownership"] = out["ownership"].where(out["ownership"].notna(), mapped)

    if "needs_renovation" not in out.columns:
        out["needs_renovation"] = pd.NA
    if "construction_status" in out.columns:
        status = out["construction_status"].astype(str).str.lower()
        needs = pd.Series(pd.NA, index=out.index, dtype="Float64")
        needs = needs.mask(status.str.contains("renovation", na=False), 1)
        needs = needs.mask(status.isin({"ready_to_use", "to_completion"}), 0)
        out["needs_renovation"] = out["needs_renovation"].where(
            out["needs_renovation"].notna(), needs
        )

    logger.debug("Promoted listing flags on %s rows", len(out))
    return out


def apply_property_attr_filters(
    frame: pd.DataFrame,
    *,
    object_type: str = "(all)",
    ownership: str = "(all)",
    has_garage: str = "(all)",
    has_outdoor_parking: str = "(all)",
    has_basement: str = "(all)",
    has_balcony: str = "(all)",
    has_garden: str = "(all)",
    has_terrace: str = "(all)",
    is_attic: str = "(all)",
    needs_renovation: str = "(all)",
    plumbing_replaced: str = "(all)",
    wiring_replaced: str = "(all)",
    floor_min: int | None = None,
    floor_max: int | None = None,
    year_built_min: int | None = None,
    year_renovated_min: int | None = None,
) -> pd.DataFrame:
    """Apply optional property-attribute filters (``(all)`` / ``yes`` / ``no``)."""
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index, dtype=bool)

    def _yes_no(col: str, choice: str) -> None:
        nonlocal mask
        if choice == "(all)" or col not in frame.columns:
            return
        values = pd.to_numeric(frame[col], errors="coerce")
        if choice == "yes":
            mask &= values.eq(1)
        elif choice == "no":
            mask &= values.eq(0)

    if object_type != "(all)" and "object_type" in frame.columns:
        mask &= frame["object_type"].astype(str).str.lower().eq(object_type.lower())
    if ownership != "(all)" and "ownership" in frame.columns:
        mask &= frame["ownership"].astype(str).str.lower().eq(ownership.lower())

    for col, choice in (
        ("has_garage", has_garage),
        ("has_outdoor_parking", has_outdoor_parking),
        ("has_basement", has_basement),
        ("has_balcony", has_balcony),
        ("has_garden", has_garden),
        ("has_terrace", has_terrace),
        ("is_attic", is_attic),
        ("needs_renovation", needs_renovation),
        ("plumbing_replaced", plumbing_replaced),
        ("wiring_replaced", wiring_replaced),
    ):
        _yes_no(col, choice)

    if floor_min is not None and "floor_no" in frame.columns:
        mask &= pd.to_numeric(frame["floor_no"], errors="coerce").ge(int(floor_min))
    if floor_max is not None and "floor_no" in frame.columns:
        mask &= pd.to_numeric(frame["floor_no"], errors="coerce").le(int(floor_max))
    if year_built_min is not None and "year_built" in frame.columns:
        mask &= pd.to_numeric(frame["year_built"], errors="coerce").ge(
            int(year_built_min)
        )
    if year_renovated_min is not None and "year_renovated" in frame.columns:
        mask &= pd.to_numeric(frame["year_renovated"], errors="coerce").ge(
            int(year_renovated_min)
        )
    return frame.loc[mask].reset_index(drop=True)
