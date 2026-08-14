"""Cross-portal deduplication for buyer-facing listing sources."""

import logging
from collections import defaultdict

import pandas as pd

from happy_housing.ingest.listing_common import (
    BUYER_LISTING_SOURCES,
    BUYER_SOURCE_KEEP_PRIORITY,
)

logger = logging.getLogger(__name__)


def normalize_listing_image_url(url: object) -> str | None:
    """Normalize a cover photo URL for cross-portal matching."""
    if not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw.startswith("http"):
        return None
    base = raw.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    return base.lower()


def _numeric(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _rooms_token(value: object) -> int:
    rooms = _numeric(value)
    if rooms is None:
        return -1
    return round(rooms)


def _geo_dedup_key(row: pd.Series) -> str | None:
    lat = _numeric(row.get("lat"))
    lon = _numeric(row.get("lon"))
    price = _numeric(row.get("price_pln"))
    area = _numeric(row.get("area_m2"))
    if lat is None or lon is None or price is None or area is None:
        return None
    if price <= 0 or area <= 0:
        return None
    price_bucket = round(price / 500)
    area_bucket = round(area * 2)
    rooms = _rooms_token(row.get("rooms"))
    return f"geo:{round(lat, 4)}:{round(lon, 4)}:{price_bucket}:{area_bucket}:{rooms}"


def _image_dedup_key(row: pd.Series) -> str | None:
    image = normalize_listing_image_url(row.get("image_url"))
    if image is None:
        return None
    return f"img:{image}"


def _dedup_keys_for_row(row: pd.Series) -> tuple[str, ...]:
    keys: list[str] = []
    geo = _geo_dedup_key(row)
    if geo is not None:
        keys.append(geo)
    image = _image_dedup_key(row)
    if image is not None:
        keys.append(image)
    return tuple(keys)


def _row_completeness(row: pd.Series) -> int:
    score = 0
    for col in ("lat", "lon", "description", "image_urls_json", "floor_no"):
        value = row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        score += 1
    return score


def _keep_rank(row: pd.Series) -> tuple[int, int, float]:
    source = str(row.get("source") or "").casefold()
    src_prio = BUYER_SOURCE_KEEP_PRIORITY.get(source, 99)
    completeness = -_row_completeness(row)
    listed = pd.to_datetime(row.get("listed_at"), errors="coerce", utc=True)
    ts = -listed.timestamp() if not pd.isna(listed) else 0.0
    return src_prio, completeness, ts


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def dedupe_cross_source_listings(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop cross-portal duplicates among buyer sources; keep RCN / other rows.

    Rows match when they share a geo fingerprint (coords + economics) or the
    same normalized cover photo URL. Otodom is preferred over OLX over fixtures.
    """
    if frame.empty or "listing_id" not in frame.columns:
        return frame
    if "source" not in frame.columns:
        return frame

    out = frame.copy()
    sources = out["source"].astype(str).str.casefold()
    buyer_mask = sources.isin(BUYER_LISTING_SOURCES)
    if not buyer_mask.any():
        return out

    buyer = out.loc[buyer_mask].copy()
    other = out.loc[~buyer_mask]
    if len(buyer) < 2:
        return out

    key_to_positions: dict[str, list[int]] = defaultdict(list)
    for pos, (_, row) in enumerate(buyer.iterrows()):
        for key in _dedup_keys_for_row(row):
            key_to_positions[key].append(pos)

    uf = _UnionFind(len(buyer))
    buyer_sources = buyer["source"].astype(str).str.casefold().tolist()
    for positions in key_to_positions.values():
        if len(positions) < 2:
            continue
        for left_i in range(len(positions)):
            for right_i in range(left_i + 1, len(positions)):
                left = positions[left_i]
                right = positions[right_i]
                if buyer_sources[left] == buyer_sources[right]:
                    continue
                uf.union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for pos in range(len(buyer)):
        groups[uf.find(pos)].append(pos)

    keep_positions: set[int] = set()
    dropped = 0
    buyer_records = buyer.to_dict(orient="records")
    for members in groups.values():
        if len(members) == 1:
            keep_positions.add(members[0])
            continue
        ranked = sorted(
            members,
            key=lambda idx: _keep_rank(pd.Series(buyer_records[idx])),
        )
        keep_positions.add(ranked[0])
        dropped += len(members) - 1

    if dropped:
        logger.info("Cross-source dedup removed %s duplicate buyer listings", dropped)

    kept_buyer = buyer.iloc[sorted(keep_positions)].reset_index(drop=True)
    return pd.concat([kept_buyer, other], ignore_index=True, sort=False)
