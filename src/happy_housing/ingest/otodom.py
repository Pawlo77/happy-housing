"""Otodom.pl listing ingest via public HTML ``__NEXT_DATA__`` payloads."""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

import pandas as pd
import requests

from happy_housing.ingest.fx import (
    apply_fx_to_listings_frame,
    currency_from_characteristics,
    fx_cache_path,
    price_value_from_characteristics,
)
from happy_housing.ingest.http_client import (
    resilient_get,
    retry_sleep_s,
)
from happy_housing.ingest.limits import reached_cap, resolve_ingest_cap
from happy_housing.ingest.listed_at import coalesce_listed_at
from happy_housing.ingest.proxy_pool import bind_http_proxy

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
"""Browser-like User-Agent for Otodom HTML requests."""

_SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/warszawa"
"""Warsaw flat-sale search base URL."""

DEFAULT_CONNECT_TIMEOUT_S: Final[float] = 10.0
"""TCP connect timeout for Otodom HTTP calls."""

DEFAULT_READ_TIMEOUT_S: Final[float] = 60.0
"""Read timeout for Otodom HTTP calls."""

DEFAULT_TIMEOUT: Final[tuple[float, float]] = (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_READ_TIMEOUT_S,
)
"""``(connect, read)`` timeout pair passed to ``requests``."""

_MAX_CONSECUTIVE_PAGE_FAILURES: Final[int] = 5
"""Abort search walk after this many consecutive failed pages."""

_DEFAULT_STOP_AFTER_KNOWN_PAGES: Final[int] = 2
"""Incremental mode: stop after this many fully-known newest pages."""

_ROOMS_MAP: dict[str, int] = {
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN_OR_MORE": 10,
}
"""Otodom roomsNumber enum → int."""

_FLOOR_ENUM: dict[str, int] = {
    "GROUND": 0,
    "FLOOR_0": 0,
    "FIRST": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "SIXTH": 6,
    "SEVENTH": 7,
    "EIGHTH": 8,
    "NINTH": 9,
    "TENTH": 10,
    "CELLAR": -1,
    "GARRET": 99,
    "ABOVE_TENTH": 11,
}
"""Search-card floorNumber enums → int."""


def parse_floor_no(value: object) -> int | None:
    """Normalize Otodom floor labels (``floor_5``, ``FIFTH``, ``3``) to int."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper().replace(" ", "_")
    if upper in _FLOOR_ENUM:
        return _FLOOR_ENUM[upper]
    match = re.search(r"(?:floor[_-]?)?(\d+)", text, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def _parse_rooms(rooms_raw: object) -> int | None:
    rooms = _ROOMS_MAP.get(str(rooms_raw))
    if rooms is not None:
        return rooms
    if rooms_raw is None:
        return None
    try:
        return int(str(rooms_raw))
    except (TypeError, ValueError):
        return None


def _tags_from_search(item: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    raw = item.get("tags") or []
    if isinstance(raw, list):
        for tag in raw:
            if isinstance(tag, dict) and tag.get("value"):
                tags.add(str(tag["value"]).upper())
            elif isinstance(tag, str):
                tags.add(tag.upper())
    return tags


def _extras_from_additional(additional: object) -> set[str]:
    found: set[str] = set()
    if not isinstance(additional, list):
        return found
    for block in additional:
        if not isinstance(block, dict):
            continue
        label = str(block.get("label") or "").lower()
        values = block.get("values") or []
        if not isinstance(values, list):
            continue
        joined = " ".join(str(v).lower() for v in values)
        if label in {"extras_types", "security_types", "lift", "advertiser_type"}:
            for token in (
                "balcony",
                "garage",
                "lift",
                "parking",
                "terrace",
                "garden",
                "basement",
                "air_conditioning",
            ):
                if token in joined:
                    found.add(token)
        if "agency" in joined:
            found.add("agency")
        if "private" in joined:
            found.add("private")
    return found


def _char_map(characteristics: object) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(characteristics, list):
        return out
    for row in characteristics:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or row.get("label") or "").strip().lower()
        if not key:
            continue
        value = row.get("value")
        if value is None:
            continue
        out[key] = str(value).strip()
    return out


def flags_from_extras_and_tags(
    *,
    extras: set[str],
    tags: set[str],
    floor_label: str | None,
    floor_no: int | float | None,
) -> dict[str, int]:
    """Derive boolean amenity / attic flags from Otodom extras and search tags."""
    tag_u = {str(t).upper() for t in tags}
    extras_l = {str(e).lower() for e in extras}
    has_garage = int("garage" in extras_l or bool({"GARAGE"} & tag_u))
    has_outdoor_parking = int(
        "parking" in extras_l or bool({"PARKING_SPOT", "PARKING"} & tag_u)
    )
    # If both garage and parking tokens exist, keep both flags.
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


def extract_image_urls(ad: dict[str, Any]) -> list[str]:
    """Prefer medium (else large/small) photo URLs from an Otodom ad."""
    images = ad.get("images")
    if not isinstance(images, list):
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for item in images:
        if not isinstance(item, dict):
            continue
        url = ""
        for key in ("medium", "large", "small", "thumbnail"):
            raw = item.get(key)
            if isinstance(raw, str) and raw.strip():
                url = raw.strip()
                break
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def extract_floor_plan_urls(ad: dict[str, Any]) -> list[str]:
    """Floor-plan / layout image URLs from Otodom ``floorPlans``."""
    plans = ad.get("floorPlans")
    if not isinstance(plans, list):
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for item in plans:
        url = ""
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            for key in ("medium", "large", "small", "url", "thumbnail"):
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    url = raw.strip()
                    break
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def flatten_otodom_ad(ad: dict[str, Any]) -> dict[str, Any]:
    """Flatten an Otodom detail ``ad`` payload into listing columns."""
    coords = (ad.get("location") or {}).get("coordinates") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    chars = _char_map(ad.get("characteristics") or ad.get("attributes"))
    extras = _extras_from_additional(ad.get("additionalInformation"))
    year_built = None
    for key in ("build_year", "buildyear", "rok_budowy", "year"):
        if key in chars:
            try:
                year_built = int(float(chars[key]))
            except (TypeError, ValueError):
                year_built = None
            if year_built is not None:
                break
    if year_built is None:
        blob = json.dumps(ad, ensure_ascii=False)
        match = re.search(
            r'"(?:buildYear|build_year|rok_budowy)"\s*:\s*"?(\d{4})"?',
            blob,
            flags=re.I,
        )
        if match:
            year_built = int(match.group(1))

    market = (chars.get("market") or str(ad.get("market") or "")).strip().lower()
    floor_no = parse_floor_no(chars.get("floor_no"))
    floors_raw = chars.get("building_floors_num")
    try:
        building_floors_num = int(float(floors_raw)) if floors_raw else None
    except (TypeError, ValueError):
        building_floors_num = None
    rent_raw = chars.get("rent")
    try:
        rent = float(str(rent_raw)) if rent_raw not in {None, ""} else None
    except (TypeError, ValueError):
        rent = None
    rooms_raw = chars.get("rooms_num")
    rooms = _parse_rooms(rooms_raw) if rooms_raw else None

    advertiser = (
        str(ad.get("advertiserType") or ad.get("advertType") or "").strip().lower()
    )
    is_private = 1 if "private" in advertiser or "private" in extras else 0
    if "agency" in advertiser or "agency" in extras:
        is_private = 0

    flags = flags_from_extras_and_tags(
        extras=extras,
        tags=set(),
        floor_label=chars.get("floor_no"),
        floor_no=floor_no,
    )
    estate = str(ad.get("estate") or chars.get("estate") or "").strip().upper() or None
    image_urls = extract_image_urls(ad)
    floor_plan_urls = extract_floor_plan_urls(ad)

    return {
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "year_built": year_built,
        "listed_at": coalesce_listed_at(
            ad.get("dateCreated"), ad.get("createdAt"), ad.get("modifiedAt")
        ),
        "description": _ad_description(ad),
        "image_url": image_urls[0] if image_urls else None,
        "image_urls_json": json.dumps(image_urls, ensure_ascii=False),
        "floor_plan_urls_json": json.dumps(floor_plan_urls, ensure_ascii=False),
        "market": market or None,
        "market_primary": 1 if market == "primary" else (0 if market else None),
        "building_type": chars.get("building_type"),
        "estate": estate,
        "floor_no": floor_no,
        "building_floors_num": building_floors_num,
        "construction_status": chars.get("construction_status"),
        "building_material": chars.get("building_material"),
        "windows_type": chars.get("windows_type"),
        "building_ownership": chars.get("building_ownership"),
        "heating": chars.get("heating"),
        "rent": rent,
        "rooms": rooms,
        "has_balcony": flags["has_balcony"],
        "has_garage": flags["has_garage"],
        "has_outdoor_parking": flags["has_outdoor_parking"],
        "has_lift": flags["has_lift"],
        "has_terrace": flags["has_terrace"],
        "has_garden": flags["has_garden"],
        "has_basement": flags["has_basement"],
        "is_attic": flags["is_attic"],
        "is_private_owner": is_private,
        "advertiser_type": advertiser or None,
        "characteristics_json": json.dumps(
            ad.get("characteristics") or [],
            ensure_ascii=False,
        ),
        "additional_information_json": json.dumps(
            ad.get("additionalInformation") or [],
            ensure_ascii=False,
        ),
    }


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    })
    bind_http_proxy(session)
    return session


def otodom_session() -> requests.Session:
    """Public browser-like session for Otodom HTML fetches."""
    return _session()


def _retry_sleep_s(attempt: int, status_code: int | None = None) -> float:
    """Backoff delay; 403/429 get a longer cool-down."""
    return retry_sleep_s(attempt, status_code=status_code)


def _http_get(
    session: requests.Session,
    url: str,
    *,
    timeout: float | tuple[float, float] = DEFAULT_TIMEOUT,
    retries: int = 5,
    label: str,
) -> requests.Response | None:
    """GET with retries; rotate free proxies on Otodom 403/429 blocks."""
    return resilient_get(
        session,
        url,
        timeout=timeout,
        retries=retries,
        label=label,
    )


def _extract_next_data(html: str) -> dict[str, Any]:
    """Parse the ``__NEXT_DATA__`` JSON blob from an Otodom HTML page."""
    marker = 'id="__NEXT_DATA__"'
    start = html.find(marker)
    if start < 0:
        raise ValueError("Otodom page missing __NEXT_DATA__")
    start = html.find(">", start) + 1
    end = html.find("</script>", start)
    if end < 0:
        raise ValueError("Otodom __NEXT_DATA__ script not closed")
    payload: dict[str, Any] = json.loads(html[start:end])
    return payload


def fetch_search_page(
    session: requests.Session,
    page: int,
    limit: int = 36,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    retries: int = 5,
    sort_by: str | None = None,
    sort_direction: str | None = None,
) -> dict[str, Any] | None:
    """Fetch one Otodom search results page.

    Returns:
        ``searchAds`` dict, or ``None`` if the page could not be fetched after
        retries (caller should skip / continue rather than abort the run).
    """
    params: dict[str, Any] = {"limit": limit, "page": page}
    if sort_by:
        params["by"] = sort_by
    if sort_direction:
        params["direction"] = sort_direction
    url = f"{_SEARCH_URL}?{urlencode(params)}"
    try:
        response = _http_get(
            session,
            url,
            timeout=timeout_s,
            retries=retries,
            label=f"Otodom search page {page}",
        )
    except requests.RequestException as exc:
        logger.warning("Otodom search page %s fetch failed: %s", page, exc)
        return None
    if response is None:
        return None
    try:
        data = _extract_next_data(response.text)
        ads = data["props"]["pageProps"]["data"]["searchAds"]
        if not isinstance(ads, dict):
            raise TypeError("Otodom searchAds payload must be a dict")
        return ads
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Otodom search page %s parse failed: %s", page, exc)
        return None


def fetch_listing_detail(
    session: requests.Session,
    slug: str,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    retries: int = 4,
) -> dict[str, Any]:
    """Fetch detail page and flatten all Otodom ad fields we care about.

    Raises:
        requests.RequestException: When the detail page cannot be fetched.
        ValueError: When the HTML payload cannot be parsed.
    """
    url = f"https://www.otodom.pl/pl/oferta/{slug}"
    response = _http_get(
        session,
        url,
        timeout=timeout_s,
        retries=retries,
        label=f"Otodom detail {slug}",
    )
    if response is None:
        raise requests.RequestException(f"Otodom detail {slug} failed after retries")
    data = _extract_next_data(response.text)
    ad = data["props"]["pageProps"].get("ad") or {}
    if not isinstance(ad, dict):
        return flatten_otodom_ad({})
    return flatten_otodom_ad(ad)


def _district_from_item(item: dict[str, Any]) -> str:
    locations = (
        item.get("location", {}).get("reverseGeocoding", {}).get("locations", [])
    )
    for loc in locations:
        if loc.get("locationLevel") == "district":
            return str(loc.get("name") or "")
    return ""


def _normalize_search_item(item: dict[str, Any]) -> dict[str, Any]:
    """Map a search-card item to a flat listing row (coords filled later)."""
    money = item.get("totalPrice") or {}
    price = money.get("value") if isinstance(money, dict) else None
    currency = (
        str(money.get("currency") or "PLN").strip().upper()
        if isinstance(money, dict)
        else "PLN"
    ) or "PLN"
    ppm_money = item.get("pricePerSquareMeter") or {}
    ppm = ppm_money.get("value") if isinstance(ppm_money, dict) else None
    rooms = _parse_rooms(item.get("roomsNumber"))
    tags = _tags_from_search(item)
    floor_no = parse_floor_no(item.get("floorNumber"))
    is_private = item.get("isPrivateOwner")
    flags = flags_from_extras_and_tags(
        extras=set(),
        tags=tags,
        floor_label=str(item.get("floorNumber") or "") or None,
        floor_no=floor_no,
    )
    search_images = extract_image_urls({"images": item.get("images") or []})
    area = (
        float(item["areaInSquareMeters"])
        if item.get("areaInSquareMeters") is not None
        else None
    )
    # Defer FX to frame-level apply (needs cache path); stash original + currency.
    price_f = float(price) if price is not None else None
    ppm_f = float(ppm) if ppm is not None else None
    return {
        "listing_id": f"OTO-{item.get('id')}",
        "otodom_id": item.get("id"),
        "slug": item.get("slug"),
        "title": item.get("title"),
        "district": _district_from_item(item),
        "price_original": price_f,
        "price_currency": currency,
        "price_pln": price_f,
        "price_per_m2": ppm_f if currency == "PLN" else None,
        "area_m2": area,
        "rooms": rooms,
        "floor_no": floor_no,
        "floor_label": item.get("floorNumber"),
        "estate": item.get("estate"),
        "image_url": search_images[0] if search_images else None,
        "image_urls_json": json.dumps(search_images, ensure_ascii=False),
        "floor_plan_urls_json": "[]",
        "url": f"https://www.otodom.pl/pl/oferta/{item.get('slug')}",
        "short_description": item.get("shortDescription"),
        "listed_at": coalesce_listed_at(
            item.get("dateCreated"),
            item.get("createdAtFirst"),
            item.get("createdAt"),
        ),
        "is_private_owner": (
            1 if is_private is True else (0 if is_private is False else None)
        ),
        "has_balcony": flags["has_balcony"],
        "has_garage": flags["has_garage"],
        "has_outdoor_parking": flags["has_outdoor_parking"],
        "has_lift": flags["has_lift"],
        "has_terrace": flags["has_terrace"],
        "has_garden": flags["has_garden"],
        "has_basement": flags["has_basement"],
        "is_attic": flags["is_attic"],
        "tags_json": json.dumps(sorted(tags), ensure_ascii=False),
        "lat": None,
        "lon": None,
        "year_built": None,
    }


def _ad_description(ad: dict[str, Any]) -> str | None:
    """Extract a free-text listing description from an Otodom ad payload."""
    for key in (
        "description",
        "adDescription",
        "description_text",
        "longDescription",
    ):
        value = ad.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Nested localizations sometimes carry translated bodies.
    for key in ("description", "adDescription"):
        nested = ad.get(key)
        if isinstance(nested, dict):
            for lang in ("pl", "en", "default"):
                text = nested.get(lang)
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return None


def _existing_by_id(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "listing_id" not in frame.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        lid = str(row.get("listing_id") or "")
        if lid:
            out[lid] = row
    return out


_DETAIL_FILL_COLUMNS: tuple[str, ...] = (
    "year_built",
    "listed_at",
    "description",
    "market",
    "market_primary",
    "building_type",
    "estate",
    "floor_no",
    "building_floors_num",
    "construction_status",
    "building_material",
    "windows_type",
    "building_ownership",
    "heating",
    "rent",
    "rooms",
    "has_balcony",
    "has_garage",
    "has_outdoor_parking",
    "has_lift",
    "has_terrace",
    "has_garden",
    "has_basement",
    "is_attic",
    "is_private_owner",
    "advertiser_type",
    "characteristics_json",
    "additional_information_json",
    "lat",
    "lon",
)
"""Detail fields merged into sparse Otodom rows during media backfill."""


def _cell_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    return text.lower() in {"nan", "none", "unknown", "[]"}


def merge_otodom_detail_into_row(
    row: pd.Series,
    detail: dict[str, Any],
) -> pd.Series:
    """Fill empty Otodom columns from a detail-page payload (no overwrite)."""
    out = row.copy()
    for col in _DETAIL_FILL_COLUMNS:
        if col not in detail:
            continue
        new_val = detail.get(col)
        if _cell_empty(new_val):
            continue
        if col in out.index and not _cell_empty(out.get(col)):
            continue
        out[col] = new_val
    return out


def listing_photo_count(row: dict[str, Any] | pd.Series) -> int:
    """Count listing photo URLs (``image_urls_json`` preferred, else cover)."""
    raw = row.get("image_urls_json")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            urls = [str(u) for u in payload if str(u).startswith("http")]
            if urls:
                return len(urls)
    cover = row.get("image_url")
    if isinstance(cover, str) and cover.startswith("http"):
        return 1
    return 0


def listing_needs_attr_backfill(row: dict[str, Any] | pd.Series) -> bool:
    """True once for gallery-complete rows that never received a detail merge.

    Sparse seller fields (heating/material) often stay empty forever — do not
    re-fetch those listings on every backfill batch.
    """
    if listing_needs_media_backfill(row):
        return False
    # ``characteristics_json`` is written on every successful detail flatten.
    raw = row.get("characteristics_json")
    if isinstance(raw, str) and raw.strip() and raw.strip() not in {"null", "None"}:
        return False
    for col in (
        "building_material",
        "heating",
        "building_type",
        "construction_status",
        "floor_no",
        "building_floors_num",
        "year_built",
    ):
        if _cell_empty(row.get(col)):
            return True
    return False


def listing_needs_media_backfill(row: dict[str, Any] | pd.Series) -> bool:
    """True when gallery JSON was never persisted (cover-only legacy rows)."""
    raw = row.get("image_urls_json")
    if raw is None:
        return True
    if isinstance(raw, float) and raw != raw:  # NaN
        return True
    if not isinstance(raw, str) or not raw.strip():
        return True
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return True
    return not isinstance(payload, list)


def card_needs_detail_refresh(
    card: dict[str, Any],
    existing_by_id: dict[str, dict[str, Any]],
) -> bool:
    """True when listing is new, missing coords/media, or search price changed."""
    lid = str(card.get("listing_id") or "")
    prior = existing_by_id.get(lid)
    if prior is None:
        return True
    if listing_needs_media_backfill(prior):
        return True
    lat = prior.get("lat")
    lon = prior.get("lon")
    if lat is None or lon is None or lat != lat or lon != lon:
        return True
    new_price = card.get("price_pln")
    old_price = prior.get("price_pln")
    if new_price is None or old_price is None:
        return new_price != old_price
    try:
        return abs(float(new_price) - float(old_price)) > 0.5
    except (TypeError, ValueError):
        return True


def backfill_otodom_media(
    frame: pd.DataFrame,
    *,
    detail_workers: int = 4,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    max_listings: int | None = None,
    sleep_s: float = 0.15,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
    include_sparse_attrs: bool = True,
) -> pd.DataFrame:
    """Fetch Otodom detail galleries for rows missing ``image_urls_json``.

    Updates ``image_url`` / ``image_urls_json`` / ``floor_plan_urls_json`` and
    fills empty structural attrs from the detail payload. Rows without a usable
    slug are skipped.

    Transient failures (403/429/network) leave the row pending so a later run
    can retry. Permanent gone (404/410) writes an empty gallery JSON.
    """
    if frame.empty:
        out = frame.copy()
        if "image_urls_json" not in out.columns:
            out["image_urls_json"] = "[]"
        if "floor_plan_urls_json" not in out.columns:
            out["floor_plan_urls_json"] = "[]"
        return out

    out = frame.copy()
    if "image_urls_json" not in out.columns:
        out["image_urls_json"] = pd.NA
    if "floor_plan_urls_json" not in out.columns:
        out["floor_plan_urls_json"] = pd.NA

    pending_idx: list[Any] = []
    seen: set[Any] = set()
    for idx, row in out.iterrows():
        payload = row.to_dict()
        slug_ok = isinstance(row.get("slug"), str) and bool(
            str(row.get("slug") or "").strip()
        )
        if not slug_ok:
            continue
        need = listing_needs_media_backfill(payload)
        if include_sparse_attrs and listing_needs_attr_backfill(payload):
            need = True
        if need and idx not in seen:
            pending_idx.append(idx)
            seen.add(idx)
    if max_listings is not None:
        pending_idx = pending_idx[: max(0, int(max_listings))]
    if not pending_idx:
        logger.info("Otodom media backfill: nothing pending")
        return out

    logger.info(
        "Otodom media backfill: fetching galleries/attrs for %s / %s listings",
        len(pending_idx),
        len(out),
    )

    def _fetch(idx: Any) -> tuple[Any, dict[str, Any] | None, int | None]:
        slug = str(out.at[idx, "slug"]).strip()
        if sleep_s > 0:
            time.sleep(float(sleep_s))
        session = _session()
        try:
            detail = fetch_listing_detail(session, slug, timeout_s=timeout_s)
        except requests.HTTPError as exc:
            status = int(exc.response.status_code) if exc.response is not None else None
            logger.warning("Media backfill failed for %s: %s", slug, exc)
            return idx, None, status
        except Exception as exc:
            logger.warning("Media backfill failed for %s: %s", slug, exc)
            return idx, None, None
        return idx, detail, None

    def _checkpoint() -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(checkpoint_path, index=False)
        logger.info("Otodom media backfill checkpoint → %s", checkpoint_path)

    updates = 0
    closed_gone = 0
    transient_fail = 0
    since_checkpoint = 0
    workers = max(1, int(detail_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, idx) for idx in pending_idx]
        for fut in as_completed(futures):
            idx, detail, status = fut.result()
            if detail is None:
                if status in {404, 410} and listing_needs_media_backfill(
                    out.loc[idx].to_dict()
                ):
                    out.at[idx, "image_urls_json"] = "[]"
                    if not isinstance(out.at[idx, "floor_plan_urls_json"], str):
                        out.at[idx, "floor_plan_urls_json"] = "[]"
                    closed_gone += 1
                    since_checkpoint += 1
                else:
                    transient_fail += 1
                if checkpoint_path is not None and since_checkpoint >= max(
                    1, int(checkpoint_every)
                ):
                    _checkpoint()
                    since_checkpoint = 0
                continue
            urls_raw = detail.get("image_urls_json")
            if isinstance(urls_raw, str) and urls_raw.strip():
                out.at[idx, "image_urls_json"] = urls_raw
            else:
                out.at[idx, "image_urls_json"] = "[]"
            cover = detail.get("image_url")
            if isinstance(cover, str) and cover.startswith("http"):
                out.at[idx, "image_url"] = cover
            plans = detail.get("floor_plan_urls_json")
            if isinstance(plans, str) and plans.strip():
                out.at[idx, "floor_plan_urls_json"] = plans
            elif not isinstance(out.at[idx, "floor_plan_urls_json"], str):
                out.at[idx, "floor_plan_urls_json"] = "[]"
            merged = merge_otodom_detail_into_row(out.loc[idx], detail)
            for col in _DETAIL_FILL_COLUMNS:
                if col not in merged.index:
                    continue
                if col not in out.columns:
                    out[col] = pd.NA
                out.at[idx, col] = merged[col]
            updates += 1
            since_checkpoint += 1
            if checkpoint_path is not None and since_checkpoint >= max(
                1, int(checkpoint_every)
            ):
                _checkpoint()
                since_checkpoint = 0

    if checkpoint_path is not None and since_checkpoint > 0:
        _checkpoint()

    logger.info(
        "Otodom media backfill: updated=%s gone_closed=%s transient_fail=%s",
        updates,
        closed_gone,
        transient_fail,
    )
    return out


def merge_otodom_frames(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
) -> pd.DataFrame:
    """Union existing + fresh rows; fresh wins on duplicate ``listing_id``."""
    if existing.empty:
        return fresh.copy()
    if fresh.empty:
        return existing.copy()
    combined = pd.concat([existing, fresh], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=["listing_id"], keep="last").reset_index(
        drop=True
    )


def max_existing_listed_at(existing: pd.DataFrame) -> pd.Timestamp | None:
    """Newest parseable ``listed_at`` in a prior Otodom frame, else ``None``."""
    if existing.empty or "listed_at" not in existing.columns:
        return None
    series = pd.to_datetime(existing["listed_at"], errors="coerce", utc=True)
    if series.isna().all():
        return None
    return pd.Timestamp(series.max())


def cards_at_or_before_watermark(
    cards: list[dict[str, Any]],
    watermark: pd.Timestamp | None,
) -> bool:
    """True when every card with a date is at/before ``watermark``."""
    if watermark is None or not cards:
        return False
    saw = False
    for card in cards:
        ts = pd.to_datetime(card.get("listed_at"), errors="coerce", utc=True)
        if pd.isna(ts):
            return False
        saw = True
        if pd.Timestamp(ts) > watermark:
            return False
    return saw


def scrape_otodom_warsaw(
    output_parquet: Path,
    *,
    max_pages: int | None = None,
    max_listings: int | None = None,
    detail_workers: int = 8,
    sleep_s: float = 0.25,
    page_size: int = 72,
    incremental: bool = True,
    stop_after_known_pages: int = _DEFAULT_STOP_AFTER_KNOWN_PAGES,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Scrape Warsaw flat-sale listings from Otodom into a parquet file.

    Search-page HTTP failures are skipped after retries (run continues). Detail
    failures drop that listing only.     When ``incremental`` and an existing
    parquet is present, search is sorted newest-first (``by=LATEST`` +
    ``direction=DESC``) and stops after ``stop_after_known_pages`` fully-known
    pages; only new/price-changed cards hit the detail endpoint; results merge
    into the prior file.

    Args:
        output_parquet: Destination path.
        max_pages: Max search pages; ``None`` walks every page Otodom reports.
        max_listings: Hard cap on geocoded listings; ``None`` keeps all.
        detail_workers: Parallel detail-page workers.
        sleep_s: Pause between search page requests.
        page_size: Search page size requested from Otodom.
        incremental: Prefer newest-first + merge when prior parquet exists.
        stop_after_known_pages: Incremental early-stop after N known pages.
        timeout_s: Connect/read timeout(s) for HTTP calls.

    Returns:
        Normalized listings dataframe.
    """
    page_cap = resolve_ingest_cap(max_pages)
    listing_cap = resolve_ingest_cap(max_listings)
    existing = (
        pd.read_parquet(output_parquet)
        if incremental and output_parquet.exists()
        else pd.DataFrame()
    )
    existing_by_id = _existing_by_id(existing)
    use_incremental = bool(incremental and existing_by_id)
    sort_by = "LATEST" if use_incremental else None
    sort_direction = "DESC" if use_incremental else None
    listed_watermark = max_existing_listed_at(existing) if use_incremental else None
    if use_incremental:
        logger.info(
            "Otodom incremental mode: %s prior listings; sort=%s direction=%s "
            "stop_after_known=%s watermark=%s",
            len(existing_by_id),
            sort_by,
            sort_direction,
            stop_after_known_pages,
            listed_watermark,
        )

    session = _session()
    cards: list[dict[str, Any]] = []
    page = 1
    total_items: int | None = None
    consecutive_failures = 0
    known_page_streak = 0
    while True:
        if page_cap is not None and page > page_cap:
            break
        logger.info("Otodom search page %s", page)
        try:
            ads = fetch_search_page(
                session,
                page=page,
                limit=page_size,
                timeout_s=timeout_s,
                sort_by=sort_by,
                sort_direction=sort_direction,
            )
        except Exception as exc:
            logger.warning("Otodom search page %s unexpected error: %s", page, exc)
            ads = None
        if ads is None:
            consecutive_failures += 1
            logger.warning(
                "Skipping Otodom search page %s (%s consecutive failures)",
                page,
                consecutive_failures,
            )
            if consecutive_failures >= _MAX_CONSECUTIVE_PAGE_FAILURES:
                logger.error(
                    "Aborting Otodom search after %s consecutive page failures",
                    consecutive_failures,
                )
                break
            page += 1
            time.sleep(sleep_s)
            continue
        consecutive_failures = 0
        items = ads.get("items") or []
        if not items:
            break
        page_cards = [_normalize_search_item(item) for item in items]
        page_all_known = True
        page_needs_refresh = False
        for card in page_cards:
            cards.append(card)
            lid = str(card.get("listing_id") or "")
            if lid not in existing_by_id:
                page_all_known = False
            elif card_needs_detail_refresh(card, existing_by_id):
                page_needs_refresh = True
        if use_incremental and page_all_known and not page_needs_refresh:
            known_page_streak += 1
            stop_need = max(1, int(stop_after_known_pages))
            if cards_at_or_before_watermark(page_cards, listed_watermark):
                # Newest-first + already at/before prior max listed_at → 1 page.
                stop_need = 1
            logger.info(
                "Otodom page %s fully known (%s/%s); early-stop soon",
                page,
                known_page_streak,
                stop_need,
            )
            if known_page_streak >= stop_need:
                logger.info("Otodom incremental early-stop at page %s", page)
                break
        else:
            known_page_streak = 0
        pagination = ads.get("pagination") or {}
        total_pages = int(pagination.get("totalPages") or page)
        if total_items is None and pagination.get("totalItems") is not None:
            try:
                total_items = int(pagination["totalItems"])
                logger.info(
                    "Otodom reports totalItems=%s totalPages=%s",
                    total_items,
                    total_pages,
                )
            except (TypeError, ValueError):
                total_items = None
        time.sleep(sleep_s)
        if page >= total_pages or reached_cap(len(cards), listing_cap):
            break
        page += 1

    # Deduplicate by listing id.
    by_id: dict[str, dict[str, Any]] = {}
    for row in cards:
        lid = str(row["listing_id"])
        by_id[lid] = row
    cards = list(by_id.values())
    if listing_cap is not None:
        cards = cards[:listing_cap]
    logger.info("Otodom unique search cards: %s", len(cards))

    to_enrich = [
        card
        for card in cards
        if (not use_incremental) or card_needs_detail_refresh(card, existing_by_id)
    ]
    skipped = len(cards) - len(to_enrich)
    if skipped:
        logger.info("Otodom incremental skip detail for %s unchanged listings", skipped)

    enriched: list[dict[str, Any]] = []

    def _enrich(row: dict[str, Any]) -> dict[str, Any] | None:
        slug = row.get("slug")
        if not slug:
            return None
        detail_session = _session()
        try:
            detail = fetch_listing_detail(
                detail_session, str(slug), timeout_s=timeout_s
            )
        except Exception as exc:
            logger.warning("Detail failed for %s: %s", slug, exc)
            return None
        if detail["lat"] is None or detail["lon"] is None:
            return None
        out = dict(row)
        for key, value in detail.items():
            if value is not None:
                out[key] = value
        for flag in (
            "has_balcony",
            "has_garage",
            "has_outdoor_parking",
            "has_lift",
            "has_terrace",
            "has_garden",
            "has_basement",
            "is_attic",
        ):
            out[flag] = max(int(row.get(flag) or 0), int(detail.get(flag) or 0))
        chars_raw = out.get("characteristics_json")
        chars: object = None
        if isinstance(chars_raw, str) and chars_raw.strip():
            try:
                chars = json.loads(chars_raw)
            except json.JSONDecodeError:
                chars = None
        char_currency = currency_from_characteristics(chars)
        char_price = price_value_from_characteristics(chars)
        if char_currency:
            out["price_currency"] = char_currency
        if char_price is not None:
            out["price_original"] = char_price
            if char_currency in {None, "PLN"}:
                out["price_pln"] = char_price
        if (
            out.get("price_per_m2") is None
            and out.get("price_pln")
            and out.get("area_m2")
            and str(out.get("price_currency") or "PLN").upper() == "PLN"
        ):
            out["price_per_m2"] = out["price_pln"] / out["area_m2"]
        if out.get("year_built") is None:
            out["year_built"] = 1990
        if out.get("listed_at") is None:
            out["listed_at"] = coalesce_listed_at(None)
        if out.get("rooms") is None:
            out["rooms"] = 2
        return out

    with ThreadPoolExecutor(max_workers=detail_workers) as pool:
        futures = [pool.submit(_enrich, card) for card in to_enrich]
        for fut in as_completed(futures):
            try:
                detail_row = fut.result()
            except Exception as exc:
                logger.warning("Detail worker crashed: %s", exc)
                continue
            if detail_row is None:
                continue
            enriched.append(detail_row)
            if reached_cap(len(enriched), listing_cap):
                for pending in futures:
                    pending.cancel()
                break

    fresh = pd.DataFrame(enriched)
    frame = merge_otodom_frames(existing, fresh) if use_incremental else fresh
    required = [
        "listing_id",
        "lat",
        "lon",
        "price_pln",
        "area_m2",
        "rooms",
        "year_built",
        "district",
        "listed_at",
    ]
    if frame.empty:
        raise ValueError("Otodom scrape produced zero geocoded listings.")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Otodom frame missing columns: {missing}")
    frame = frame.dropna(subset=["lat", "lon", "price_pln", "area_m2"])
    cache = fx_cache_path(output_parquet.parent)
    frame = apply_fx_to_listings_frame(frame, cache)
    frame = frame.dropna(subset=["price_pln", "area_m2"])
    frame["price_per_m2"] = frame["price_pln"] / frame["area_m2"]
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)
    logger.info(
        "Wrote %s Otodom listings to %s (fresh=%s prior=%s)",
        len(frame),
        output_parquet,
        len(fresh),
        len(existing),
    )
    return frame
