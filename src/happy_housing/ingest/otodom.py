"""Otodom.pl listing ingest via public HTML ``__NEXT_DATA__`` payloads."""

from __future__ import annotations

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

from happy_housing.ingest.limits import reached_cap, resolve_ingest_cap

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

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({
    408,
    425,
    429,
    500,
    502,
    503,
    504,
})
"""HTTP statuses that warrant a retry."""

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

    return {
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "year_built": year_built,
        "listed_at": ad.get("dateCreated") or ad.get("createdAt"),
        "description": _ad_description(ad),
        "market": market or None,
        "market_primary": 1 if market == "primary" else (0 if market else None),
        "building_type": chars.get("building_type"),
        "floor_no": floor_no,
        "building_floors_num": building_floors_num,
        "construction_status": chars.get("construction_status"),
        "building_material": chars.get("building_material"),
        "windows_type": chars.get("windows_type"),
        "building_ownership": chars.get("building_ownership"),
        "heating": chars.get("heating"),
        "rent": rent,
        "rooms": rooms,
        "has_balcony": 1 if "balcony" in extras else 0,
        "has_garage": 1 if "garage" in extras or "parking" in extras else 0,
        "has_lift": 1 if "lift" in extras else 0,
        "has_terrace": 1 if "terrace" in extras else 0,
        "has_garden": 1 if "garden" in extras else 0,
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
    return session


def otodom_session() -> requests.Session:
    """Public browser-like session for Otodom HTML fetches."""
    return _session()


def is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for timeouts, connection errors, and retryable HTTP codes."""
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return int(exc.response.status_code) in _RETRYABLE_STATUS
    return isinstance(exc, requests.RequestException)


def _retry_sleep_s(attempt: int, *, status_code: int | None = None) -> float:
    """Backoff delay; 403/429 get a longer cool-down."""
    base = min(2.0 * attempt, 20.0)
    if status_code in {403, 429}:
        return min(base * 2.5, 45.0)
    return base


def _http_get(
    session: requests.Session,
    url: str,
    *,
    timeout: float | tuple[float, float] = DEFAULT_TIMEOUT,
    retries: int = 5,
    label: str,
) -> requests.Response | None:
    """GET with retries; return ``None`` after exhausting retryable failures."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        status_code: int | None = None
        try:
            response = session.get(url, timeout=timeout)
            status_code = int(response.status_code)
            response.raise_for_status()
            return response
        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            last_exc = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status_code = int(exc.response.status_code)
            retryable = is_retryable_http_error(exc) or status_code in {
                403,
                *_RETRYABLE_STATUS,
            }
            if not retryable or attempt >= retries:
                logger.warning(
                    "%s failed permanently (attempt %s/%s): %s",
                    label,
                    attempt,
                    retries,
                    exc,
                )
                break
            sleep_for = _retry_sleep_s(attempt, status_code=status_code)
            logger.warning(
                "%s failed (attempt %s/%s): %s; retry in %.1fs",
                label,
                attempt,
                retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    if last_exc is not None:
        logger.debug("Last error for %s: %s", label, last_exc)
    return None


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
) -> dict[str, Any] | None:
    """Fetch one Otodom search results page.

    Returns:
        ``searchAds`` dict, or ``None`` if the page could not be fetched after
        retries (caller should skip / continue rather than abort the run).
    """
    params: dict[str, Any] = {"limit": limit, "page": page}
    if sort_by:
        params["by"] = sort_by
    url = f"{_SEARCH_URL}?{urlencode(params)}"
    response = _http_get(
        session,
        url,
        timeout=timeout_s,
        retries=retries,
        label=f"Otodom search page {page}",
    )
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
    price = (item.get("totalPrice") or {}).get("value")
    ppm = (item.get("pricePerSquareMeter") or {}).get("value")
    rooms = _parse_rooms(item.get("roomsNumber"))
    tags = _tags_from_search(item)
    floor_no = parse_floor_no(item.get("floorNumber"))
    is_private = item.get("isPrivateOwner")
    return {
        "listing_id": f"OTO-{item.get('id')}",
        "otodom_id": item.get("id"),
        "slug": item.get("slug"),
        "title": item.get("title"),
        "district": _district_from_item(item),
        "price_pln": float(price) if price is not None else None,
        "price_per_m2": float(ppm) if ppm is not None else None,
        "area_m2": float(item["areaInSquareMeters"])
        if item.get("areaInSquareMeters") is not None
        else None,
        "rooms": rooms,
        "floor_no": floor_no,
        "floor_label": item.get("floorNumber"),
        "estate": item.get("estate"),
        "image_url": (item.get("images") or [{}])[0].get("medium"),
        "url": f"https://www.otodom.pl/pl/oferta/{item.get('slug')}",
        "short_description": item.get("shortDescription"),
        "listed_at": item.get("dateCreated") or item.get("createdAtFirst"),
        "is_private_owner": (
            1 if is_private is True else (0 if is_private is False else None)
        ),
        "has_balcony": 1 if "BALCONY" in tags else 0,
        "has_garage": 1 if {"GARAGE", "PARKING_SPOT", "PARKING"} & tags else 0,
        "has_lift": 1 if "LIFT" in tags or "ELEVATOR" in tags else 0,
        "has_terrace": 1 if "TERRACE" in tags else 0,
        "has_garden": 1 if "GARDEN" in tags else 0,
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


def card_needs_detail_refresh(
    card: dict[str, Any],
    existing_by_id: dict[str, dict[str, Any]],
) -> bool:
    """True when listing is new, missing coords, or search price changed."""
    lid = str(card.get("listing_id") or "")
    prior = existing_by_id.get(lid)
    if prior is None:
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
    failures drop that listing only. When ``incremental`` and an existing
    parquet is present, search is sorted newest-first and stops after
    ``stop_after_known_pages`` fully-known pages; only new/price-changed cards
    hit the detail endpoint; results merge into the prior file.

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
    if use_incremental:
        logger.info(
            "Otodom incremental mode: %s prior listings; sort=%s stop_after_known=%s",
            len(existing_by_id),
            sort_by,
            stop_after_known_pages,
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
            logger.info(
                "Otodom page %s fully known (%s/%s); early-stop soon",
                page,
                known_page_streak,
                stop_after_known_pages,
            )
            if known_page_streak >= max(1, stop_after_known_pages):
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
            "has_lift",
            "has_terrace",
            "has_garden",
        ):
            out[flag] = max(int(row.get(flag) or 0), int(detail.get(flag) or 0))
        if (
            out.get("price_per_m2") is None
            and out.get("price_pln")
            and out.get("area_m2")
        ):
            out["price_per_m2"] = out["price_pln"] / out["area_m2"]
        if out.get("year_built") is None:
            out["year_built"] = 1990
        if out.get("listed_at") is None:
            out["listed_at"] = "2026-01-01"
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
