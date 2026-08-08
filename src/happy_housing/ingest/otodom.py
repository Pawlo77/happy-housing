"""Otodom.pl listing ingest via public HTML ``__NEXT_DATA__`` payloads."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
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
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Fetch one Otodom search results page and return pageProps data."""
    params = {"limit": limit, "page": page}
    url = f"{_SEARCH_URL}?{urlencode(params)}"
    response = session.get(url, timeout=timeout_s)
    response.raise_for_status()
    data = _extract_next_data(response.text)
    ads = data["props"]["pageProps"]["data"]["searchAds"]
    if not isinstance(ads, dict):
        raise TypeError("Otodom searchAds payload must be a dict")
    return ads


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
    rooms_raw = item.get("roomsNumber")
    rooms_map = {
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
    rooms = rooms_map.get(str(rooms_raw))
    if rooms is None and rooms_raw is not None:
        try:
            rooms = int(rooms_raw)
        except (TypeError, ValueError):
            rooms = None
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
        "image_url": (item.get("images") or [{}])[0].get("medium"),
        "url": f"https://www.otodom.pl/pl/oferta/{item.get('slug')}",
        "lat": None,
        "lon": None,
        "year_built": None,
        "listed_at": None,
    }


def fetch_listing_detail(
    session: requests.Session,
    slug: str,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Fetch detail page fields: coordinates, build year, created date."""
    url = f"https://www.otodom.pl/pl/oferta/{slug}"
    response = session.get(url, timeout=timeout_s)
    response.raise_for_status()
    data = _extract_next_data(response.text)
    ad = data["props"]["pageProps"].get("ad") or {}
    coords = (ad.get("location") or {}).get("coordinates") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    year_built = None
    listed_at = ad.get("dateCreated") or ad.get("createdAt")
    characteristics = ad.get("characteristics") or []
    if isinstance(characteristics, list):
        for row in characteristics:
            key = str(row.get("key") or row.get("label") or "").lower()
            if key in {"buildyear", "build_year", "rok_budowy", "year"}:
                try:
                    year_built = int(float(str(row.get("value"))))
                except (TypeError, ValueError):
                    year_built = None
    # Fallback: scan JSON text for build year-like fields.
    if year_built is None:
        blob = json.dumps(ad, ensure_ascii=False)
        match = re.search(
            r'"(?:buildYear|build_year|rok_budowy)"\s*:\s*"?(\d{4})"?',
            blob,
            flags=re.I,
        )
        if match:
            year_built = int(match.group(1))
    return {
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "year_built": year_built,
        "listed_at": listed_at,
        "description": _ad_description(ad),
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


def scrape_otodom_warsaw(
    output_parquet: Path,
    *,
    max_pages: int | None = None,
    max_listings: int | None = None,
    detail_workers: int = 6,
    sleep_s: float = 0.35,
    page_size: int = 72,
) -> pd.DataFrame:
    """Scrape Warsaw flat-sale listings from Otodom into a parquet file.

    Args:
        output_parquet: Destination path.
        max_pages: Max search pages; ``None`` walks every page Otodom reports.
        max_listings: Hard cap on geocoded listings; ``None`` keeps all.
        detail_workers: Parallel detail-page workers.
        sleep_s: Pause between search page requests.
        page_size: Search page size requested from Otodom.

    Returns:
        Normalized listings dataframe.
    """
    page_cap = resolve_ingest_cap(max_pages)
    listing_cap = resolve_ingest_cap(max_listings)
    session = _session()
    cards: list[dict[str, Any]] = []
    page = 1
    while True:
        if page_cap is not None and page > page_cap:
            break
        logger.info("Otodom search page %s", page)
        ads = fetch_search_page(session, page=page, limit=page_size)
        items = ads.get("items") or []
        if not items:
            break
        for item in items:
            cards.append(_normalize_search_item(item))
        pagination = ads.get("pagination") or {}
        total_pages = int(pagination.get("totalPages") or page)
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

    enriched: list[dict[str, Any]] = []

    def _enrich(row: dict[str, Any]) -> dict[str, Any] | None:
        slug = row.get("slug")
        if not slug:
            return None
        detail_session = _session()
        try:
            detail = fetch_listing_detail(detail_session, str(slug))
        except Exception as exc:
            logger.warning("Detail failed for %s: %s", slug, exc)
            return None
        if detail["lat"] is None or detail["lon"] is None:
            return None
        out = {**row, **detail}
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
        futures = [pool.submit(_enrich, card) for card in cards]
        for fut in as_completed(futures):
            detail_row = fut.result()
            if detail_row is None:
                continue
            enriched.append(detail_row)
            if reached_cap(len(enriched), listing_cap):
                for pending in futures:
                    pending.cancel()
                break

    frame = pd.DataFrame(enriched)
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
    logger.info("Wrote %s Otodom listings to %s", len(frame), output_parquet)
    return frame
