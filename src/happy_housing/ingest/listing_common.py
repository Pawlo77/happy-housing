"""Shared listing-ingest constants and frame helpers."""

import re
from pathlib import Path
from typing import Any

import pandas as pd

BUYER_LISTING_SOURCES: frozenset[str] = frozenset({"otodom", "nol", "olx", "fixture"})
"""Buyer-facing ask sources (Explore / train labels / cross-portal dedup)."""

BUYER_SOURCE_KEEP_PRIORITY: dict[str, int] = {
    "otodom": 0,
    "nol": 1,
    "olx": 2,
    "fixture": 3,
}
"""Lower rank wins when collapsing cross-portal duplicate groups."""

_INACTIVE_ASK_PARQUETS: tuple[tuple[str, str], ...] = (
    ("otodom", "otodom_listings.parquet"),
    ("nol", "nol_listings.parquet"),
)
"""Closed search-card artifacts unioned into Stats historical ask context."""

LISTING_ACTIVE_COLUMN: str = "listing_active"
"""1 = live ask; 0 = withdrawn/closed (kept for history / Explore filter)."""

BUYER_LISTING_SCHEMA_COLUMNS: tuple[str, ...] = (
    "year_built",
    "description",
    "building_type",
    "estate",
    "building_floors_num",
    "construction_status",
    "building_material",
    "windows_type",
    "building_ownership",
    "heating",
    "rent",
    "is_private_owner",
    "has_balcony",
    "has_garage",
    "has_outdoor_parking",
    "has_lift",
    "has_terrace",
    "has_garden",
    "has_basement",
    "is_attic",
    "image_urls_json",
    "floor_plan_urls_json",
    "characteristics_json",
    "tags_json",
    LISTING_ACTIVE_COLUMN,
)
"""Detail / model columns every buyer ask row should expose (nullable when unknown)."""

_SCHEMA_DEFAULTS: dict[str, Any] = {
    LISTING_ACTIVE_COLUMN: 1,
    "has_balcony": 0,
    "has_garage": 0,
    "has_outdoor_parking": 0,
    "has_lift": 0,
    "has_terrace": 0,
    "has_garden": 0,
    "has_basement": 0,
    "is_attic": 0,
    "floor_plan_urls_json": "[]",
}
"""Default values for missing schema keys on newly normalized rows."""


def flags_from_extras_and_tags(
    *,
    extras: set[str],
    tags: set[str],
    floor_label: str | None,
    floor_no: int | float | None,
) -> dict[str, int]:
    """Derive boolean amenity / attic flags from extras and search tags."""
    tag_u = {str(t).upper() for t in tags}
    extras_l = {str(e).lower() for e in extras}
    has_garage = int("garage" in extras_l or bool({"GARAGE"} & tag_u))
    has_outdoor_parking = int(
        "parking" in extras_l or bool({"PARKING_SPOT", "PARKING"} & tag_u)
    )
    has_basement = int("basement" in extras_l or "BASEMENT" in tag_u)
    label = str(floor_label or "").upper()
    is_attic = int(
        label in {"GARRET", "ATTIC"} or (floor_no is not None and int(floor_no) >= 99)
    )
    return {
        "has_balcony": int("balcony" in extras_l or "BALCONY" in tag_u),
        "has_garage": has_garage,
        "has_outdoor_parking": has_outdoor_parking,
        "has_lift": int("lift" in extras_l or bool({"LIFT", "ELEVATOR"} & tag_u)),
        "has_terrace": int("terrace" in extras_l or "TERRACE" in tag_u),
        "has_garden": int("garden" in extras_l or "GARDEN" in tag_u),
        "has_basement": has_basement,
        "is_attic": is_attic,
    }


def extras_from_free_text(text: str) -> set[str]:
    """Map Polish/English listing text tokens to Otodom-style extra keys."""
    lower = text.lower()
    extras: set[str] = set()
    if re.search(r"\bbalkon|\bbalcony", lower):
        extras.add("balcony")
    if re.search(r"garaż|garaz|\bgarage", lower):
        extras.add("garage")
    if re.search(r"\bparking|miejsce postojowe|postoj", lower):
        extras.add("parking")
    if re.search(r"\bwind[aą]|\blift|\belevator", lower):
        extras.add("lift")
    if re.search(r"\btaras|\bterrace", lower):
        extras.add("terrace")
    if re.search(r"\bogród|\bogrod|\bgarden", lower):
        extras.add("garden")
    if re.search(r"\bpiwnic|\bbasement", lower):
        extras.add("basement")
    return extras


def align_buyer_listing_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure a normalized buyer row exposes the shared schema columns."""
    out = dict(row)
    for col in BUYER_LISTING_SCHEMA_COLUMNS:
        if col not in out or out[col] is None:
            out[col] = _SCHEMA_DEFAULTS.get(col)
    return out


def align_buyer_listing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure every buyer listing row exposes the shared schema columns."""
    if frame.empty:
        return frame
    out = frame.copy()
    for col in BUYER_LISTING_SCHEMA_COLUMNS:
        if col not in out.columns:
            default = _SCHEMA_DEFAULTS.get(col)
            out[col] = default
        elif col in _SCHEMA_DEFAULTS:
            out[col] = out[col].fillna(_SCHEMA_DEFAULTS[col])
    return out


def listing_description(row: dict[str, Any] | pd.Series) -> str:
    """Best-effort free text for shared NLP / flag extraction."""
    data = row.to_dict() if isinstance(row, pd.Series) else row
    for key in ("description", "short_description", "title"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def coalesce_listing_descriptions(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill empty ``description`` from ``short_description`` / ``title``."""
    if frame.empty:
        return frame
    out = frame.copy()
    if "description" not in out.columns:
        out["description"] = pd.NA
    desc = out["description"].astype(object)
    for col in ("short_description", "title"):
        if col not in out.columns:
            continue
        fallback = out[col].astype(object)
        missing = desc.isna() | (desc.astype(str).str.strip() == "")
        desc = desc.where(~missing, fallback)
    out["description"] = desc
    return out


def prepare_listings_for_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Align schema and description text before shared flag/NLP enrich."""
    return coalesce_listing_descriptions(align_buyer_listing_frame(frame))


def listing_is_active(value: object) -> bool:
    """Return False when a row is marked closed/inactive."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return True
    return int(number) != 0


def filter_active_listings(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop inactive/closed rows for recommender and live-only lists."""
    if frame.empty or LISTING_ACTIVE_COLUMN not in frame.columns:
        return frame
    mask = frame[LISTING_ACTIVE_COLUMN].map(listing_is_active)
    return frame.loc[mask].copy()


def append_inactive_otodom_asks(
    frame: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """Union closed Otodom/NOL search rows for Stats historical ask-price context."""
    extras: list[pd.DataFrame] = []
    known: set[str] = (
        set(frame["listing_id"].astype(str)) if "listing_id" in frame.columns else set()
    )
    for source, filename in _INACTIVE_ASK_PARQUETS:
        path = processed_dir / filename
        if not path.exists():
            continue
        stored = pd.read_parquet(path)
        if stored.empty or LISTING_ACTIVE_COLUMN not in stored.columns:
            continue
        inactive = stored.loc[
            ~stored[LISTING_ACTIVE_COLUMN].map(listing_is_active)
        ].copy()
        if inactive.empty:
            continue
        inactive["source"] = source
        if known:
            inactive = inactive.loc[~inactive["listing_id"].astype(str).isin(known)]
        if inactive.empty:
            continue
        extras.append(inactive)
        known.update(inactive["listing_id"].astype(str))
    if not extras:
        return frame
    return pd.concat([frame, *extras], ignore_index=True, sort=False)


def merge_listing_frames(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union existing + fresh rows; fresh wins on duplicate ``listing_id``."""
    if existing.empty:
        return fresh.copy()
    if fresh.empty:
        return existing.copy()
    combined = pd.concat([existing, fresh], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=["listing_id"], keep="last").reset_index(
        drop=True
    )
