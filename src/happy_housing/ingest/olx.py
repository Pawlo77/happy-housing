"""OLX.pl Warsaw flat listings via public offers API (``olx-api-wrapper``).

City-wide offset pagination is unreliable on OLX (pages overlap). We tile by
Warsaw ``district_id`` from the geo-encoder API, walk ``created_at:desc``, and
merge incrementally into ``olx_listings.parquet`` like Otodom.
"""

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from happy_housing.ingest.fx import apply_fx_to_listings_frame, fx_cache_path
from happy_housing.ingest.http_client import resilient_get
from happy_housing.ingest.limits import reached_cap, resolve_ingest_cap
from happy_housing.ingest.listed_at import coalesce_listed_at
from happy_housing.ingest.listing_common import (
    align_buyer_listing_row,
    extras_from_free_text,
    flags_from_extras_and_tags,
    merge_listing_frames,
)
from happy_housing.ingest.proxy_pool import bind_http_proxy

logger = logging.getLogger(__name__)

OLX_OFFERS_URL: str = "https://www.olx.pl/api/v1/offers/"
"""Public OLX offers search endpoint."""

OLX_DISTRICTS_URL: str = "https://www.olx.pl/api/v1/geo-encoder/cities/17871/districts"
"""Warsaw district list for tiling past the city-wide 1000-hit window."""

WARSAW_CITY_ID: int = 17871
"""OLX city id for Warszawa."""

WARSAW_REGION_ID: int = 2
"""OLX region id for Mazowieckie."""

CATEGORY_SALE: int = 14
"""``mieszkania / sprzedaż`` category id."""

CATEGORY_RENT: int = 15
"""``mieszkania / wynajem`` category id."""

DEFAULT_PAGE_SIZE: Final[int] = 40
"""Offers requested per district page."""

_FALLBACK_DISTRICT_IDS: tuple[int, ...] = (
    367,
    365,
    369,
    353,
    355,
    381,
    379,
    361,
    351,
    377,
    371,
    373,
    383,
    533,
    375,
    357,
    359,
    363,
)
"""Hardcoded Warsaw district ids when geo-encoder fetch fails."""

_ROOMS_KEY: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}
"""OLX ``rooms`` param keys → int."""

FetchPageFn = Callable[..., Any]
"""Injectable page fetcher for tests (``OlxPublic.get_offers``-compatible)."""

FetchDistrictIdsFn = Callable[[], Sequence[int]]
"""Injectable Warsaw district id loader."""


def _param_map(params: object) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(params, list):
        return out
    for item in params:
        if isinstance(item, Mapping):
            key = item.get("key")
            value = item.get("value")
        else:
            key = getattr(item, "key", None)
            value = getattr(item, "value", None)
        if not isinstance(key, str) or not key:
            continue
        out[key] = value
    return out


def _param_key(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw = value.get("key")
        return str(raw) if raw is not None else None
    raw = getattr(value, "key", None)
    return str(raw) if raw is not None else None


def _param_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if value.get("value") is not None:
            try:
                return float(value["value"])
            except (TypeError, ValueError):
                pass
        key = value.get("key")
        label = value.get("label")
    else:
        raw_val = getattr(value, "value", None)
        if raw_val is not None:
            try:
                return float(raw_val)
            except (TypeError, ValueError):
                pass
        key = getattr(value, "key", None)
        label = getattr(value, "label", None)
    for candidate in (key, label):
        if candidate is None:
            continue
        text = str(candidate).replace(" ", "").replace(",", ".")
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _parse_floor(value: object) -> int | None:
    key = _param_key(value)
    if key is None:
        return None
    lower = key.lower()
    if lower in {"ground_floor", "floor_0", "parter"}:
        return 0
    match = re.search(r"(\d+)", lower)
    if match:
        return int(match.group(1))
    return None


def _parse_rooms(value: object) -> int | None:
    key = _param_key(value)
    if key is None:
        return None
    mapped = _ROOMS_KEY.get(key.lower())
    if mapped is not None:
        return mapped
    match = re.search(r"(\d+)", key)
    return int(match.group(1)) if match else None


def _money_amount(price: object) -> tuple[float | None, str]:
    if price is None:
        return None, "PLN"
    if isinstance(price, Mapping):
        amount = price.get("amount", price.get("value"))
        currency = str(price.get("currency") or "PLN").upper()
        try:
            return (float(amount) if amount is not None else None), currency
        except (TypeError, ValueError):
            return None, currency
    amount = getattr(price, "amount", None)
    currency = str(getattr(price, "currency", "PLN") or "PLN").upper()
    try:
        return (float(amount) if amount is not None else None), currency
    except (TypeError, ValueError):
        return None, currency


def _param_label(value: object) -> str | None:
    """Human-readable OLX param label when present."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        label = value.get("label")
    else:
        label = getattr(value, "label", None)
    if label is None:
        return None
    text = str(label).strip()
    return text or None


def _param_text(value: object) -> str:
    """Concat param key, value key, and label for token matching."""
    parts = [_param_key(value), _param_label(value)]
    if isinstance(value, Mapping):
        raw = value.get("value")
        if raw is not None:
            parts.append(str(raw))
    return " ".join(p for p in parts if p)


def _olx_extras_from_params(params: dict[str, Any]) -> set[str]:
    """Map OLX param keys/labels to shared amenity extra tokens."""
    extras: set[str] = set()
    for key, val in params.items():
        blob = f"{key} {_param_text(val)}".lower()
        extras |= extras_from_free_text(blob)
        if key in {"winda", "elevator", "lift"}:
            extras.add("lift")
        if key in {"garage", "garaz", "garaż"}:
            extras.add("garage")
        if key in {"parking", "parking_place"}:
            extras.add("parking")
        if key in {"balkon", "balcony"}:
            extras.add("balcony")
    return extras


def _olx_categorical(params: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty OLX param label for ``keys``."""
    for key in keys:
        label = _param_label(params.get(key))
        if label:
            return label
        param_key = _param_key(params.get(key))
        if param_key:
            return param_key.replace("_", " ")
    return None


def _photo_urls_json(photos: object) -> str:
    """Serialize OLX photo links for gallery parity with Otodom."""
    if not isinstance(photos, list):
        return "[]"
    urls: list[str] = []
    for photo in photos:
        if isinstance(photo, Mapping):
            link = photo.get("link")
        else:
            link = getattr(photo, "link", None)
        if isinstance(link, str) and link.strip():
            urls.append(link.replace("{width}", "800").replace("{height}", "600"))
    return json.dumps(urls, ensure_ascii=False)


def _photo_url(photos: object) -> str | None:
    if not isinstance(photos, list) or not photos:
        return None
    first = photos[0]
    if isinstance(first, Mapping):
        link = first.get("link")
    else:
        link = getattr(first, "link", None)
    if not isinstance(link, str) or not link.strip():
        return None
    return link.replace("{width}", "800").replace("{height}", "600")


def _nested_name(node: object, *attrs: str) -> str | None:
    cur: object = node
    for attr in attrs:
        if cur is None:
            return None
        cur = cur.get(attr) if isinstance(cur, Mapping) else getattr(cur, attr, None)
    if cur is None:
        return None
    text = str(cur).strip()
    return text or None


def offer_to_mapping(offer: Any) -> dict[str, Any]:
    """Convert an ``olx`` Offer / dict into a plain mapping for normalize."""
    if isinstance(offer, Mapping):
        return dict(offer)
    params_raw = getattr(offer, "params", None) or []
    params: list[dict[str, Any]] = []
    for item in params_raw:
        if isinstance(item, Mapping):
            params.append(dict(item))
            continue
        value = getattr(item, "value", None)
        value_map: dict[str, Any] | None
        if value is None:
            value_map = None
        elif isinstance(value, Mapping):
            value_map = dict(value)
        else:
            value_map = {
                "key": getattr(value, "key", None),
                "value": getattr(value, "value", None),
                "label": getattr(value, "label", None),
                "currency": getattr(value, "currency", None),
            }
        params.append({"key": getattr(item, "key", None), "value": value_map})
    location = getattr(offer, "location", None)
    loc_map: dict[str, Any] = {}
    if location is not None:
        city = getattr(location, "city", None)
        district = getattr(location, "district", None)
        loc_map = {
            "city": {
                "name": getattr(city, "name", None) if city is not None else None,
            },
            "district": {
                "id": getattr(district, "id", None) if district is not None else None,
                "name": (
                    getattr(district, "name", None) if district is not None else None
                ),
            },
        }
    map_obj = getattr(offer, "map", None)
    map_map = {
        "lat": getattr(map_obj, "lat", None) if map_obj is not None else None,
        "lon": getattr(map_obj, "lon", None) if map_obj is not None else None,
    }
    photos_raw = getattr(offer, "photos", None) or []
    photos = []
    for photo in photos_raw:
        if isinstance(photo, Mapping):
            photos.append(dict(photo))
        else:
            photos.append({"link": getattr(photo, "link", None)})
    category = getattr(offer, "category", None)
    price = getattr(offer, "price", None)
    price_map: dict[str, Any] | None
    if price is None:
        price_map = None
    elif isinstance(price, Mapping):
        price_map = dict(price)
    else:
        price_map = {
            "amount": getattr(price, "amount", None),
            "currency": str(getattr(price, "currency", "PLN")),
        }
    return {
        "id": getattr(offer, "id", None),
        "title": getattr(offer, "title", None),
        "url": getattr(offer, "url", None),
        "description": getattr(offer, "description", None),
        "created_time": getattr(offer, "created_time", None),
        "last_refresh_time": getattr(offer, "last_refresh_time", None),
        "category": {
            "id": getattr(category, "id", None) if category is not None else None,
        },
        "map": map_map,
        "location": loc_map,
        "price": price_map,
        "photos": photos,
        "params": params,
        "business": bool(getattr(offer, "business", False)),
    }


def normalize_olx_offer(
    offer: Mapping[str, Any] | Any,
    listing_kind: str = "sale",
) -> dict[str, Any] | None:
    """Map one OLX offer to the happy-housing listing row schema.

    Returns:
        Normalized row, or ``None`` when coords / price / area are unusable.
    """
    payload = offer_to_mapping(offer)
    offer_id = payload.get("id")
    try:
        olx_id = int(offer_id)  # type: ignore[arg-type]  # JSON/object id
    except (TypeError, ValueError):
        return None
    raw_map = payload.get("map")
    map_info: Mapping[str, Any] = raw_map if isinstance(raw_map, Mapping) else {}
    try:
        lat = float(map_info.get("lat"))  # type: ignore[arg-type]  # JSON scalar
        lon = float(map_info.get("lon"))  # type: ignore[arg-type]  # JSON scalar
    except (TypeError, ValueError):
        return None
    params = _param_map(payload.get("params"))
    amount, currency = _money_amount(payload.get("price"))
    if amount is None:
        amount = _param_number(params.get("price"))
        currency_raw = params.get("price")
        if isinstance(currency_raw, Mapping) and currency_raw.get("currency"):
            currency = str(currency_raw["currency"]).upper()
    area = _param_number(params.get("m"))
    if amount is None or amount <= 0 or area is None or area <= 0:
        return None
    rooms = _parse_rooms(params.get("rooms"))
    floor_no = _parse_floor(params.get("floor_select"))
    market_key = (_param_key(params.get("market")) or "").lower()
    market_primary = market_key in {"primary", "pierwotny"}
    ppm = _param_number(params.get("price_per_m"))
    if ppm is None or ppm <= 0:
        ppm = float(amount) / float(area)
    district = _nested_name(payload.get("location"), "district", "name") or "Warszawa"
    listed_at = coalesce_listed_at(
        payload.get("last_refresh_time"),
        payload.get("created_time"),
    )
    image = _photo_url(payload.get("photos"))
    kind = "rent" if str(listing_kind).lower() == "rent" else "sale"
    description = str(payload.get("description") or "").strip() or None
    title = str(payload.get("title") or "").strip() or None
    extras = _olx_extras_from_params(params) | extras_from_free_text(
        f"{title or ''} {description or ''}"
    )
    flags = flags_from_extras_and_tags(
        extras=extras,
        tags=set(),
        floor_label=_param_label(params.get("floor_select")),
        floor_no=floor_no,
    )
    year_built_raw = _param_number(params.get("build_year"))
    if year_built_raw is None:
        year_built_raw = _param_number(params.get("rok_budowy"))
    year_built = int(year_built_raw) if year_built_raw is not None else None
    rent_raw = _param_number(params.get("rent"))
    if rent_raw is None:
        rent_raw = _param_number(params.get("czynsz"))
    rent = float(rent_raw) if rent_raw is not None else None
    is_business = bool(payload.get("business"))
    row = {
        "listing_id": f"OLX-{olx_id}",
        "olx_id": olx_id,
        "title": title,
        "url": str(payload.get("url") or "").strip() or None,
        "description": description,
        "lat": lat,
        "lon": lon,
        "price_original": float(amount),
        "price_currency": currency,
        "price_pln": float(amount) if currency == "PLN" else None,
        "area_m2": float(area),
        "rooms": rooms,
        "year_built": year_built,
        "district": district,
        "listed_at": listed_at,
        "floor_no": floor_no,
        "market": "primary" if market_primary else "secondary",
        "market_primary": market_primary,
        "price_per_m2": float(ppm),
        "image_url": image,
        "image_urls_json": _photo_urls_json(payload.get("photos")),
        "listing_kind": kind,
        "is_business": is_business,
        "source": "olx",
        "building_type": _olx_categorical(params, "building_type", "typ_nieruchomosci"),
        "construction_status": _olx_categorical(
            params, "construction_status", "stan_wykonczenia"
        ),
        "building_material": _olx_categorical(params, "building_material"),
        "heating": _olx_categorical(params, "heating", "ogrzewanie"),
        "rent": rent,
        "is_private_owner": 0 if is_business else 1,
        **flags,
    }
    return align_buyer_listing_row(row)


def fetch_warsaw_district_ids(
    session: requests.Session | None = None,
    timeout_s: float = 30.0,
) -> list[int]:
    """Load Warsaw district ids from OLX geo-encoder (fallback hardcoded)."""
    http = session or requests.Session()
    bind_http_proxy(http)
    response = resilient_get(
        http,
        OLX_DISTRICTS_URL,
        timeout=timeout_s,
        retries=3,
        label="OLX Warsaw districts",
        headers={"Accept": "application/json", "User-Agent": "happy-housing/0.1"},
    )
    if response is None:
        logger.warning("OLX districts fetch failed — using fallback ids")
        return list(_FALLBACK_DISTRICT_IDS)
    try:
        payload = response.json()
    except ValueError:
        return list(_FALLBACK_DISTRICT_IDS)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return list(_FALLBACK_DISTRICT_IDS)
    ids: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            ids.append(int(row["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids or list(_FALLBACK_DISTRICT_IDS)


def _default_fetch_page(
    *,
    category_id: int,
    district_id: int,
    offset: int,
    limit: int,
    listing_kind: str,
) -> Any:
    del listing_kind
    from olx.public import OlxPublic

    client = OlxPublic(full_response=True)
    return client.get_offers(
        category_id=category_id,
        offset=offset,
        limit=limit,
        sort_by="created_at:desc",
        extra_params={
            "city_id": WARSAW_CITY_ID,
            "region_id": WARSAW_REGION_ID,
            "district_id": district_id,
        },
    )


def _page_offers(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    data = getattr(raw, "data", None)
    if isinstance(data, list):
        return data
    if isinstance(raw, Mapping) and isinstance(raw.get("data"), list):
        return list(raw["data"])
    return []


def _page_total(raw: Any) -> int | None:
    meta = getattr(raw, "metadata", None)
    if meta is not None:
        total = getattr(meta, "total_elements", None)
        if total is not None:
            try:
                return int(total)
            except (TypeError, ValueError):
                return None
    if isinstance(raw, Mapping):
        metadata = raw.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("total_elements") is not None:
            try:
                return int(metadata["total_elements"])
            except (TypeError, ValueError):
                return None
    return None


def scrape_olx_warsaw(
    output_parquet: Path,
    *,
    max_listings: int | None = None,
    incremental: bool = True,
    include_rent: bool = False,
    district_ids: Sequence[int] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sleep_s: float = 0.15,
    stop_after_known_pages: int = 2,
    fetch_page: FetchPageFn | None = None,
    fetch_district_ids: FetchDistrictIdsFn | None = None,
) -> pd.DataFrame:
    """Download Warsaw flat offers from OLX into a parquet file.

    Tiles by district, sorts newest-first, and merges into an existing parquet
    when ``incremental`` is true (early-stop after known pages).
    """
    listing_cap = resolve_ingest_cap(max_listings)
    existing = (
        pd.read_parquet(output_parquet)
        if incremental and output_parquet.exists()
        else pd.DataFrame()
    )
    known_ids: set[str] = set()
    if not existing.empty and "listing_id" in existing.columns:
        known_ids = {str(x) for x in existing["listing_id"].tolist()}
    use_incremental = bool(incremental and known_ids)
    fetcher = fetch_page or _default_fetch_page
    districts = list(district_ids) if district_ids is not None else None
    if districts is None:
        loader = fetch_district_ids or fetch_warsaw_district_ids
        districts = list(loader())
    segments: list[tuple[str, int]] = [("sale", CATEGORY_SALE)]
    if include_rent:
        segments.append(("rent", CATEGORY_RENT))

    rows: list[dict[str, Any]] = []
    for kind, category_id in segments:
        for district_id in districts:
            if reached_cap(len(rows), listing_cap):
                break
            offset = 0
            known_pages = 0
            while True:
                if reached_cap(len(rows), listing_cap):
                    break
                try:
                    raw = fetcher(
                        category_id=category_id,
                        district_id=int(district_id),
                        offset=offset,
                        limit=page_size,
                        listing_kind=kind,
                    )
                except Exception as exc:
                    logger.warning(
                        "OLX page fail district=%s offset=%s: %s",
                        district_id,
                        offset,
                        exc,
                    )
                    break
                offers = _page_offers(raw)
                if not offers:
                    break
                page_all_known = True
                page_new = 0
                for offer in offers:
                    if reached_cap(len(rows), listing_cap):
                        break
                    normalized = normalize_olx_offer(offer, listing_kind=kind)
                    if normalized is None:
                        continue
                    listing_id = str(normalized["listing_id"])
                    if listing_id in known_ids:
                        continue
                    page_all_known = False
                    page_new += 1
                    known_ids.add(listing_id)
                    rows.append(normalized)
                total = _page_total(raw)
                logger.info(
                    "OLX %s district=%s offset=%s got=%s new=%s total_el=%s",
                    kind,
                    district_id,
                    offset,
                    len(offers),
                    page_new,
                    total,
                )
                if use_incremental and page_all_known:
                    known_pages += 1
                    if known_pages >= max(1, int(stop_after_known_pages)):
                        break
                else:
                    known_pages = 0
                if total is not None and offset + page_size >= total:
                    break
                if page_new == 0 and not use_incremental and total is None:
                    break
                offset += page_size
                if sleep_s > 0:
                    time.sleep(sleep_s)

    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        fx_path = fx_cache_path(output_parquet.parent)
        fresh = apply_fx_to_listings_frame(fresh, fx_path)
        if "price_pln" in fresh.columns and "area_m2" in fresh.columns:
            mask = fresh["price_pln"].notna() & fresh["area_m2"].gt(0)
            fresh.loc[mask, "price_per_m2"] = (
                fresh.loc[mask, "price_pln"] / fresh.loc[mask, "area_m2"]
            )
    merged = merge_listing_frames(existing, fresh)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    logger.info(
        "OLX scrape wrote %s rows (%s fresh) → %s",
        len(merged),
        len(fresh),
        output_parquet,
    )
    return merged
