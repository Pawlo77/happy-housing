"""Shared listing-ingest constants and frame helpers."""

from pathlib import Path

import pandas as pd

BUYER_LISTING_SOURCES: frozenset[str] = frozenset({"otodom", "olx", "fixture"})
"""Buyer-facing ask sources (Explore / train labels / cross-portal dedup)."""

BUYER_SOURCE_KEEP_PRIORITY: dict[str, int] = {"otodom": 0, "olx": 1, "fixture": 2}
"""Lower rank wins when collapsing cross-portal duplicate groups."""

LISTING_ACTIVE_COLUMN: str = "listing_active"
"""1 = live ask shown on map; 0 = withdrawn/closed but kept for history."""


def listing_is_active(value: object) -> bool:
    """Return False when a row is marked closed/inactive."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return True
    return int(number) != 0


def filter_active_listings(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop inactive/closed rows for Explore map and live offer lists."""
    if frame.empty or LISTING_ACTIVE_COLUMN not in frame.columns:
        return frame
    mask = frame[LISTING_ACTIVE_COLUMN].map(listing_is_active)
    return frame.loc[mask].copy()


def append_inactive_otodom_asks(
    frame: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """Union closed Otodom search rows for Stats historical ask-price context."""
    oto_path = processed_dir / "otodom_listings.parquet"
    if not oto_path.exists():
        return frame
    oto = pd.read_parquet(oto_path)
    if oto.empty or LISTING_ACTIVE_COLUMN not in oto.columns:
        return frame
    inactive = oto.loc[~oto[LISTING_ACTIVE_COLUMN].map(listing_is_active)].copy()
    if inactive.empty:
        return frame
    inactive["source"] = "otodom"
    if "listing_id" not in frame.columns:
        return pd.concat([frame, inactive], ignore_index=True, sort=False)
    known = set(frame["listing_id"].astype(str))
    extra = inactive.loc[~inactive["listing_id"].astype(str).isin(known)]
    if extra.empty:
        return frame
    return pd.concat([frame, extra], ignore_index=True, sort=False)


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
