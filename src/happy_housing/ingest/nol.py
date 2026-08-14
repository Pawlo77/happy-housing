"""Nieruchomości Online Warsaw flat listings via JSON-LD (``resilient_get``)."""

import html as html_lib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

import pandas as pd
import requests

from happy_housing.ingest.fx import apply_fx_to_listings_frame, fx_cache_path
from happy_housing.ingest.http_client import resilient_get
from happy_housing.ingest.limits import reached_cap, resolve_ingest_cap
from happy_housing.ingest.listed_at import coalesce_listed_at
from happy_housing.ingest.listing_common import (
    LISTING_ACTIVE_COLUMN,
    align_buyer_listing_row,
    extras_from_free_text,
    flags_from_extras_and_tags,
    listing_is_active,
)
from happy_housing.ingest.proxy_pool import bind_http_proxy

logger = logging.getLogger(__name__)

SEARCH_URL: str = "https://www.nieruchomosci-online.pl/szukaj.html"
"""Warsaw flat search base URL."""

DEFAULT_CONNECT_TIMEOUT_S: Final[float] = 10.0
"""TCP connect timeout for NOL HTTP calls."""

DEFAULT_READ_TIMEOUT_S: Final[float] = 60.0
"""Read timeout for NOL HTTP calls."""

DEFAULT_TIMEOUT: Final[tuple[float, float]] = (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_READ_TIMEOUT_S,
)
"""``(connect, read)`` timeout pair."""

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
"""Browser-like User-Agent (same family as Otodom)."""

_LD_JSON_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
"""JSON-LD script tags."""

_ID_IN_URL_RE = re.compile(r"/(\d{5,})\.html(?:[?#]|$)", re.IGNORECASE)
"""Native offer id in listing URL."""

_TILES_PREFIX: Final[str] = "tilesData:"
"""Embedded search-page map/attribute JSON."""

_DETAIL_PRESERVE_KEYS: frozenset[str] = frozenset({
    "lat",
    "lon",
    "characteristics_json",
    "image_urls_json",
    "floor_plan_urls_json",
    "building_type",
    "building_floors_num",
    "heating",
    "construction_status",
    "building_material",
    "building_ownership",
    "year_built",
    "description",
})
"""Detail fields kept when merging a newer search card."""

FetchSearchPageFn = Callable[..., str | None]
"""Injectable search-page fetcher ``(page, **kwargs) -> html``."""

_AMENITY_EXTRAS: dict[str, str] = {
    "balcony": "balcony",
    "balkon": "balcony",
    "garage": "garage",
    "garaż": "garage",
    "garaz": "garage",
    "parking": "parking",
    "elevator": "lift",
    "lift": "lift",
    "winda": "lift",
    "garden": "garden",
    "ogród": "garden",
    "ogrod": "garden",
    "terrace": "terrace",
    "taras": "terrace",
    "basement": "basement",
    "piwnica": "basement",
}
"""JSON-LD amenity names → shared extra tokens."""


def listing_id_from_url(url: object) -> str | None:
    """Return ``NOL-{id}`` from a listing URL, or ``None``."""
    if not isinstance(url, str):
        return None
    match = _ID_IN_URL_RE.search(url)
    if match is None:
        return None
    return f"NOL-{match.group(1)}"


def native_id_from_listing_id(listing_id: object) -> str | None:
    """Strip the ``NOL-`` prefix."""
    text = str(listing_id or "")
    if text.startswith("NOL-"):
        return text[4:]
    match = _ID_IN_URL_RE.search(text)
    return match.group(1) if match else None


def extract_json_ld_blocks(html: str) -> list[dict[str, Any]]:
    """Parse ``application/ld+json`` script payloads from HTML."""
    blocks: list[dict[str, Any]] = []
    for raw in _LD_JSON_RE.findall(html):
        text = html_lib.unescape(raw.strip())
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            blocks.append(data)
        elif isinstance(data, list):
            blocks.extend(item for item in data if isinstance(item, dict))
    return blocks


def extract_tiles_data(html: str) -> dict[str, dict[str, Any]]:
    """Parse search-page ``tilesData`` map keyed by native offer id."""
    idx = html.find(_TILES_PREFIX)
    if idx < 0:
        return {}
    start = html.find("{", idx)
    if start < 0:
        return {}
    try:
        payload, _end = json.JSONDecoder().raw_decode(html[start:])
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        native = str(key).lstrip("a")
        if native.isdigit():
            out[native] = value
    return out


def _float(value: object) -> float | None:
    if value is None or value is False:
        return None
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number:
        return None
    return number


def _int(value: object) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _cell_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if not isinstance(value, str):
        return False
    return not value.strip() or value.strip().lower() in {"nan", "none", "[]"}


def _nol_listed_at(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return coalesce_listed_at(None) or "2024-01-01T00:00:00+00:00"
    text = raw.replace("CEST", "+02:00").replace("CET", "+01:00")
    return coalesce_listed_at(text) or coalesce_listed_at(None) or text


def _extras_from_amenities(items: object) -> set[str]:
    extras: set[str] = set()
    if not isinstance(items, list):
        return extras
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip().lower()
        value = item.get("value")
        if value in {False, "false", "Nie", "nie", "No", "no"}:
            continue
        for token, extra in _AMENITY_EXTRAS.items():
            if token in name:
                extras.add(extra)
    return extras


def _yes(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("tak") or text in {"yes", "true", "1"}:
        return True
    if text.startswith("nie") or text in {"no", "false", "0"}:
        return False
    return None


def _attr_values(tile: Mapping[str, Any], alias: str) -> list[str]:
    raw = tile.get("attributes")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("alias") or "") != alias:
            continue
        values = item.get("values")
        if not isinstance(values, list):
            continue
        for entry in values:
            if isinstance(entry, Mapping) and entry.get("value") is not None:
                text = str(entry["value"]).strip()
                if text and text != "/":
                    out.append(text)
    return out


def _apply_tile(row: dict[str, Any], tile: Mapping[str, Any] | None) -> None:
    if tile is None:
        return
    geo = tile.get("map")
    if isinstance(geo, Mapping):
        lat = _float(geo.get("latitude"))
        lon = _float(geo.get("longitude"))
        if lat is not None and lon is not None:
            row["lat"] = lat
            row["lon"] = lon
    floors = _attr_values(tile, "floor")
    if floors:
        row["floor_no"] = _int(floors[0])
        if len(floors) > 1:
            row["building_floors_num"] = _int(floors[-1])
    years = _attr_values(tile, "year-built")
    if years:
        row["year_built"] = _int(years[0])
    rooms = _attr_values(tile, "rooms")
    if rooms and row.get("rooms") is None:
        row["rooms"] = _int(rooms[0])
    parking = " ".join(_attr_values(tile, "parking-type-0")).lower()
    if "garaż" in parking or "garaz" in parking:
        row["has_garage"] = 1
    elif "parking" in parking:
        row["has_outdoor_parking"] = 1
    condition = " ".join(_attr_values(tile, "condition-type")).lower()
    if "remont" in condition:
        row["construction_status"] = "to_renovation"


def search_offers_from_json_ld(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect nested ``Offer`` objects from CollectionPage JSON-LD."""
    offers: list[dict[str, Any]] = []
    for block in blocks:
        entity = (
            block.get("mainEntity") if block.get("@type") == "CollectionPage" else block
        )
        if not isinstance(entity, Mapping):
            continue
        top = entity.get("offers")
        bundles = top if isinstance(top, list) else [top]
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                continue
            nested = bundle.get("offers")
            items = nested if isinstance(nested, list) else [bundle]
            for item in items:
                if isinstance(item, Mapping) and item.get("url"):
                    offers.append(dict(item))
    return offers


def apartment_from_json_ld(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the Apartment / Accommodation JSON-LD block when present."""
    for block in blocks:
        kind = str(block.get("@type") or "")
        if kind in {"Apartment", "Accommodation", "House", "Residence"}:
            return block
    return None


def normalize_nol_search_offer(
    offer: Mapping[str, Any],
    tile: Mapping[str, Any] | None = None,
    listing_kind: str = "sale",
) -> dict[str, Any] | None:
    """Map a search JSON-LD Offer (+ optional tilesData) to the buyer schema."""
    url = str(offer.get("url") or "").strip()
    listing_id = listing_id_from_url(url)
    if listing_id is None:
        return None
    price = _float(offer.get("price"))
    spec = offer.get("priceSpecification")
    ppm = None
    currency = str(offer.get("priceCurrency") or "PLN").upper() or "PLN"
    if isinstance(spec, Mapping):
        ppm = _float(spec.get("price"))
        if spec.get("priceCurrency"):
            currency = str(spec["priceCurrency"]).upper()
    item = offer.get("itemOffered")
    item_map: Mapping[str, Any] = item if isinstance(item, Mapping) else {}
    floor_size = item_map.get("floorSize")
    area = None
    if isinstance(floor_size, Mapping):
        area = _float(floor_size.get("value"))
    if price is None or price <= 0 or area is None or area <= 0:
        return None
    rooms = _int(item_map.get("numberOfRooms"))
    address = item_map.get("address")
    district = "Warszawa"
    if isinstance(address, Mapping):
        district = str(
            address.get("addressLocality") or address.get("streetAddress") or "Warszawa"
        )
    description = str(item_map.get("description") or "").strip() or None
    title = str(offer.get("name") or "").strip() or None
    extras = extras_from_free_text(f"{title or ''} {description or ''}")
    flags = flags_from_extras_and_tags(
        extras=extras,
        tags=set(),
        floor_label=None,
        floor_no=None,
    )
    image = offer.get("image")
    image_url = image if isinstance(image, str) else None
    if isinstance(image, list) and image:
        image_url = str(image[0]) if image[0] else None
    ppm_f = ppm if ppm is not None and ppm > 0 else (price / area)
    row: dict[str, Any] = {
        "listing_id": listing_id,
        "nol_id": native_id_from_listing_id(listing_id),
        "title": title,
        "url": url,
        "description": description,
        "lat": None,
        "lon": None,
        "price_original": price,
        "price_currency": currency,
        "price_pln": price if currency == "PLN" else None,
        "area_m2": area,
        "rooms": rooms,
        "district": district,
        "listed_at": _nol_listed_at(offer.get("datePosted")),
        "market": "secondary",
        "market_primary": False,
        "price_per_m2": float(ppm_f) if currency == "PLN" else None,
        "image_url": image_url,
        "image_urls_json": json.dumps(
            [image_url] if image_url else [],
            ensure_ascii=False,
        ),
        "listing_kind": "rent" if listing_kind == "rent" else "sale",
        "source": "nol",
        "estate": "FLAT",
        **flags,
    }
    _apply_tile(row, tile)
    return align_buyer_listing_row(row)


def normalize_nol_detail(apartment: Mapping[str, Any]) -> dict[str, Any] | None:
    """Map a detail-page Apartment JSON-LD block to the buyer schema."""
    url = str(apartment.get("url") or "").strip()
    listing_id = listing_id_from_url(url)
    if listing_id is None:
        return None
    offers = apartment.get("offers")
    offer: Mapping[str, Any] = {}
    if isinstance(offers, list) and offers and isinstance(offers[0], Mapping):
        offer = offers[0]
    elif isinstance(offers, Mapping):
        offer = offers
    price = _float(offer.get("price") or apartment.get("price"))
    spec = offer.get("priceSpecification")
    ppm = _float(spec.get("price")) if isinstance(spec, Mapping) else None
    currency = str(offer.get("priceCurrency") or "PLN").upper() or "PLN"
    floor_size = apartment.get("floorSize")
    area = _float(floor_size.get("value")) if isinstance(floor_size, Mapping) else None
    if price is None or price <= 0 or area is None or area <= 0:
        return None
    geo = apartment.get("geo")
    lat = lon = None
    if isinstance(geo, Mapping):
        lat = _float(geo.get("latitude"))
        lon = _float(geo.get("longitude"))
    extras = _extras_from_amenities(apartment.get("amenityFeature"))
    extras |= extras_from_free_text(str(apartment.get("description") or ""))
    props: dict[str, str] = {}
    raw_props = apartment.get("additionalProperty")
    if isinstance(raw_props, list):
        for item in raw_props:
            if isinstance(item, Mapping) and item.get("name"):
                props[str(item["name"]).strip().lower()] = str(item.get("value") or "")
    floor_no = _int(apartment.get("floorLevel")) or _int(props.get("floor level"))
    flags = flags_from_extras_and_tags(
        extras=extras,
        tags=set(),
        floor_label=None,
        floor_no=floor_no,
    )
    if _yes(props.get("basement access")):
        flags["has_basement"] = 1
    if _yes(props.get("garage")):
        flags["has_garage"] = 1
    if _yes(props.get("terrace")):
        flags["has_terrace"] = 1
    images = apartment.get("image")
    urls: list[str] = []
    if isinstance(images, str) and images.startswith("http"):
        urls = [images]
    elif isinstance(images, list):
        urls = [str(u) for u in images if isinstance(u, str) and u.startswith("http")]
    agent = apartment.get("agent")
    is_private = 0 if isinstance(agent, Mapping) and agent.get("name") else 1
    heating = None
    for item in apartment.get("amenityFeature") or []:
        if isinstance(item, Mapping) and "heat" in str(item.get("name") or "").lower():
            heating = str(item.get("name"))
    construction = None
    if _yes(props.get("renovation needed")):
        construction = "to_renovation"
    ppm_f = ppm if ppm is not None and ppm > 0 else price / area
    row: dict[str, Any] = {
        "listing_id": listing_id,
        "nol_id": native_id_from_listing_id(listing_id),
        "title": str(apartment.get("name") or "").strip() or None,
        "url": url,
        "description": str(apartment.get("description") or "").strip() or None,
        "lat": lat,
        "lon": lon,
        "price_original": price,
        "price_currency": currency,
        "price_pln": price if currency == "PLN" else None,
        "area_m2": area,
        "rooms": _int(apartment.get("numberOfRooms"))
        or _int(props.get("number of rooms")),
        "year_built": _int(apartment.get("yearBuilt")) or _int(props.get("year built")),
        "district": "Warszawa",
        "listed_at": _nol_listed_at(apartment.get("datePosted")),
        "floor_no": floor_no,
        "building_floors_num": _int(props.get("total floors in building")),
        "building_type": props.get("building type") or None,
        "construction_status": construction,
        "heating": heating,
        "market": "secondary",
        "market_primary": False,
        "price_per_m2": float(ppm_f) if currency == "PLN" else None,
        "image_url": urls[0] if urls else None,
        "image_urls_json": json.dumps(urls, ensure_ascii=False),
        "listing_kind": "sale",
        "source": "nol",
        "estate": "FLAT",
        "is_private_owner": is_private,
        "characteristics_json": json.dumps(dict(props), ensure_ascii=False),
        "detail_price_pln": price if currency == "PLN" else None,
        **flags,
    }
    address = apartment.get("address")
    if isinstance(address, Mapping) and address.get("addressLocality"):
        row["district"] = str(address["addressLocality"])
    return align_buyer_listing_row(row)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    })
    bind_http_proxy(session)
    return session


def fetch_nol_search_html(
    page: int,
    *,
    listing_kind: str = "sale",
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
) -> str | None:
    """GET one Warsaw search page (newest-first is the site default)."""
    kind = "wynajem" if listing_kind == "rent" else "sprzedaz"
    query = f"3,mieszkanie,{kind},,Warszawa"
    url = f"{SEARCH_URL}?{query}"
    if page > 1:
        url = f"{url}&{urlencode({'p': page})}"
    http = session or _session()
    response = resilient_get(
        http,
        url,
        timeout=timeout_s,
        retries=5,
        label=f"NOL search p={page}",
    )
    if response is None:
        return None
    return str(response.text)


def fetch_nol_detail_html(
    url: str,
    *,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
) -> str | None:
    """GET one listing detail HTML page."""
    http = session or _session()
    response = resilient_get(
        http,
        url,
        timeout=timeout_s,
        retries=4,
        label="NOL detail",
    )
    if response is None:
        return None
    return str(response.text)


def cards_from_search_html(
    html: str,
    *,
    listing_kind: str = "sale",
) -> list[dict[str, Any]]:
    """Normalize search JSON-LD offers, attaching tilesData coords when present."""
    tiles = extract_tiles_data(html)
    cards: list[dict[str, Any]] = []
    for offer in search_offers_from_json_ld(extract_json_ld_blocks(html)):
        native = native_id_from_listing_id(listing_id_from_url(offer.get("url")))
        tile = tiles.get(native) if native else None
        row = normalize_nol_search_offer(offer, tile, listing_kind=listing_kind)
        if row is not None:
            cards.append(row)
    return cards


def merge_nol_search_cards(
    existing: pd.DataFrame,
    cards: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge search cards into existing parquet without wiping detail fields."""
    existing_by_id: dict[str, dict[str, Any]] = {}
    if not existing.empty and "listing_id" in existing.columns:
        for row in existing.to_dict(orient="records"):
            lid = str(row.get("listing_id") or "")
            if lid:
                existing_by_id[lid] = row
    merged_by_id: dict[str, dict[str, Any]] = {}
    for card in cards:
        lid = str(card.get("listing_id") or "")
        if not lid:
            continue
        prior = existing_by_id.get(lid)
        if prior is None:
            merged_by_id[lid] = dict(card)
            continue
        out = dict(prior)
        for key, value in card.items():
            if key in _DETAIL_PRESERVE_KEYS and not _cell_empty(prior.get(key)):
                continue
            out[key] = value
        out[LISTING_ACTIVE_COLUMN] = 1
        merged_by_id[lid] = out
    for lid, prior in existing_by_id.items():
        if lid not in merged_by_id:
            merged_by_id[lid] = dict(prior)
    if not merged_by_id:
        return pd.DataFrame()
    return pd.DataFrame(list(merged_by_id.values()))


def _coords_missing(row: Mapping[str, Any]) -> bool:
    lat = row.get("lat")
    lon = row.get("lon")
    if lat is None or lon is None:
        return True
    if isinstance(lat, float) and lat != lat:
        return True
    return bool(isinstance(lon, float) and lon != lon)


def row_needs_nol_detail_backfill(row: dict[str, Any] | pd.Series) -> bool:
    """True when a stored NOL row still needs a detail JSON-LD fetch."""
    if not listing_is_active(row.get(LISTING_ACTIVE_COLUMN)):
        return False
    if _coords_missing(row):
        return True
    raw = row.get("characteristics_json")
    has_chars = (
        isinstance(raw, str)
        and bool(raw.strip())
        and raw.strip() not in {"null", "None"}
    )
    return not has_chars


def mark_nol_listing_closed(row: dict[str, Any]) -> dict[str, Any]:
    """Mark a search row as withdrawn while keeping last-known economics."""
    out = dict(row)
    out[LISTING_ACTIVE_COLUMN] = 0
    price = out.get("price_pln")
    if price is not None:
        out["detail_price_pln"] = price
    if not isinstance(out.get("characteristics_json"), str):
        out["characteristics_json"] = "{}"
    return out


def enrich_nol_listing_row(
    row: dict[str, Any],
    *,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    fetch_html: Callable[..., str | None] | None = None,
) -> dict[str, Any] | None:
    """Fetch detail JSON-LD and merge into ``row``; mark 404/410 closed."""
    url = str(row.get("url") or "")
    if not url:
        return None
    fetcher = fetch_html or fetch_nol_detail_html
    try:
        html = fetcher(url, timeout_s=timeout_s, session=session)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {404, 410}:
            return mark_nol_listing_closed(row)
        logger.warning("NOL detail failed for %s: %s", url, exc)
        return None
    except Exception as exc:
        logger.warning("NOL detail failed for %s: %s", url, exc)
        return None
    if html is None:
        return None
    apartment = apartment_from_json_ld(extract_json_ld_blocks(html))
    if apartment is None:
        return None
    detail = normalize_nol_detail(apartment)
    if detail is None:
        return None
    out = dict(row)
    for key, value in detail.items():
        if value is not None and not _cell_empty(value):
            out[key] = value
    out[LISTING_ACTIVE_COLUMN] = 1
    return align_buyer_listing_row(out)


def backfill_nol_details(
    output_parquet: Path,
    *,
    max_listings: int | None = 100,
    detail_workers: int = 1,
    sleep_s: float = 1.0,
    timeout_s: float | tuple[float, float] = DEFAULT_TIMEOUT,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """Fetch NOL detail JSON-LD for pending rows in ``output_parquet``."""
    if not output_parquet.exists():
        return pd.DataFrame()
    out = pd.read_parquet(output_parquet)
    if out.empty:
        return out
    pending_idx: list[Any] = []
    for idx, row in out.iterrows():
        if row_needs_nol_detail_backfill(row.to_dict()):
            pending_idx.append(idx)
    cap = resolve_ingest_cap(max_listings)
    if cap is not None:
        pending_idx = pending_idx[:cap]
    if not pending_idx:
        logger.info("NOL detail backfill: nothing pending")
        return out
    logger.info(
        "NOL detail backfill: %s pending (workers=%s)",
        len(pending_idx),
        detail_workers,
    )
    updated = 0
    since_checkpoint = 0
    workers = max(1, int(detail_workers))

    def _apply(idx: Any) -> tuple[Any, dict[str, Any] | None]:
        payload = out.loc[idx].to_dict()
        merged = enrich_nol_listing_row(payload, timeout_s=timeout_s)
        if (
            sleep_s > 0
            and merged is not None
            and listing_is_active(merged.get(LISTING_ACTIVE_COLUMN))
        ):
            time.sleep(float(sleep_s))
        return idx, merged

    if workers == 1:
        results = [_apply(idx) for idx in pending_idx]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [
                fut.result()
                for fut in as_completed([
                    pool.submit(_apply, idx) for idx in pending_idx
                ])
            ]
    for idx, merged in results:
        if merged is None:
            continue
        for key, value in merged.items():
            out.at[idx, key] = value
        updated += 1
        since_checkpoint += 1
        if since_checkpoint >= max(1, int(checkpoint_every)):
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_parquet, index=False)
            since_checkpoint = 0
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    logger.info("NOL detail backfill updated %s rows", updated)
    return out


def scrape_nol_warsaw(
    output_parquet: Path,
    *,
    max_pages: int | None = None,
    max_listings: int | None = None,
    incremental: bool = True,
    include_rent: bool = False,
    fetch_details: bool = False,
    stop_after_known_pages: int = 2,
    sleep_s: float = 0.2,
    fetch_search_page: FetchSearchPageFn | None = None,
) -> pd.DataFrame:
    """Download Warsaw NOL flats into parquet (JSON-LD search, incremental)."""
    page_cap = resolve_ingest_cap(max_pages)
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
    fetcher = fetch_search_page or (
        lambda page, listing_kind="sale", **_: fetch_nol_search_html(
            page, listing_kind=listing_kind
        )
    )
    segments: list[str] = ["sale"]
    if include_rent:
        segments.append("rent")
    cards: list[dict[str, Any]] = []
    for kind in segments:
        known_page_streak = 0
        prev_ids: set[str] | None = None
        page = 1
        while True:
            if page_cap is not None and page > page_cap:
                break
            if reached_cap(len(cards), listing_cap):
                break
            html = fetcher(page, listing_kind=kind)
            if not html:
                break
            page_cards = cards_from_search_html(html, listing_kind=kind)
            if not page_cards:
                break
            ids = {str(c["listing_id"]) for c in page_cards}
            if prev_ids is not None and ids == prev_ids:
                logger.info("NOL %s stop: repeated page id set at p=%s", kind, page)
                break
            prev_ids = ids
            if use_incremental and ids and ids <= known_ids:
                known_page_streak += 1
                if known_page_streak >= max(1, int(stop_after_known_pages)):
                    logger.info(
                        "NOL %s incremental stop after %s known pages",
                        kind,
                        known_page_streak,
                    )
                    break
            else:
                known_page_streak = 0
            cards.extend(page_cards)
            known_ids.update(ids)
            page += 1
            if sleep_s > 0 and fetch_search_page is None:
                time.sleep(float(sleep_s))
    merged = merge_nol_search_cards(existing, cards)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    if not merged.empty:
        merged = apply_fx_to_listings_frame(
            merged,
            fx_cache_path(output_parquet.parent),
        )
        merged.to_parquet(output_parquet, index=False)
    else:
        merged.to_parquet(output_parquet, index=False)
    logger.info("NOL search wrote %s rows → %s", len(merged), output_parquet)
    if not fetch_details:
        return merged
    return backfill_nol_details(
        output_parquet,
        max_listings=listing_cap,
        detail_workers=1,
        sleep_s=sleep_s,
    )
