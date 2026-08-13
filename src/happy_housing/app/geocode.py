"""OSM Nominatim place search for Admin commute-target picker."""

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
"""Public Nominatim search endpoint (OpenStreetMap)."""

_USER_AGENT = "HappyHousing/1.0 (local Warsaw housing app; contact: local-dev)"
"""Nominatim requires an identifying User-Agent."""

# Warsaw-ish viewbox: left, top, right, bottom (Nominatim order).
_WARSAW_VIEWBOX = "20.85,52.40,21.30,52.10"
"""Bias suggestions toward Warsaw without hard-bounding."""

MIN_LIVE_QUERY_CHARS = 2
"""Minimum trimmed query length before Nominatim live search."""

_MIN_REQUEST_INTERVAL_S = 1.0
"""Nominatim usage policy: max ~1 request/second."""

_last_request_monotonic: float = 0.0
"""Monotonic timestamp of the last Nominatim HTTP call."""


def live_search_ready(query: str, min_chars: int = MIN_LIVE_QUERY_CHARS) -> bool:
    """True when ``query`` is long enough for a live place search."""
    return len(str(query or "").strip()) >= int(min_chars)


def _throttle_nominatim() -> None:
    """Sleep so consecutive Nominatim calls respect ``_MIN_REQUEST_INTERVAL_S``."""
    global _last_request_monotonic
    now = time.monotonic()
    wait = _MIN_REQUEST_INTERVAL_S - (now - _last_request_monotonic)
    if wait > 0:
        time.sleep(wait)
    _last_request_monotonic = time.monotonic()


@dataclass(frozen=True, slots=True)
class PlaceSuggestion:
    """One geocoded place from an open map geocoder."""

    label: str
    """Human-readable display name."""
    lon: float
    """WGS84 longitude."""
    lat: float
    """WGS84 latitude."""
    osm_type: str
    """OSM element type (node/way/relation)."""
    osm_id: int
    """OSM element id."""


def format_place_option_labels(
    hits: list[PlaceSuggestion] | tuple[object, ...],
) -> list[str]:
    """Stable selectbox labels ``1. …`` so duplicate display names stay distinct."""
    labels: list[str] = []
    for i, row in enumerate(hits):
        if isinstance(row, PlaceSuggestion):
            name = row.label
        elif isinstance(row, dict):
            name = str(row.get("label") or "")
        else:
            name = str(getattr(row, "label", row))
        labels.append(f"{i + 1}. {name}")
    return labels


def resolve_place_pick(
    pick: str,
    hits: list[PlaceSuggestion] | tuple[object, ...],
    labels: list[str] | None = None,
) -> PlaceSuggestion | dict[str, object] | None:
    """Map a selectbox pick to the corresponding hit by index (not by name)."""
    options = labels if labels is not None else format_place_option_labels(hits)
    if pick not in options:
        return None
    idx = options.index(pick)
    if idx < 0 or idx >= len(hits):
        return None
    return hits[idx]  # type: ignore[return-value]  # Geocoder hit shape validated


def search_places(
    query: str,
    *,
    limit: int = 8,
    timeout_s: float = 12.0,
) -> list[PlaceSuggestion]:
    """Search Nominatim for places (Poland-biased, Warsaw-preferred)."""
    needle = str(query or "").strip()
    if not needle:
        return []
    _throttle_nominatim()
    response = requests.get(
        _NOMINATIM_SEARCH,
        params={
            "q": needle,
            "format": "json",
            "limit": int(max(1, min(limit, 15))),
            "addressdetails": 0,
            "countrycodes": "pl",
            "viewbox": _WARSAW_VIEWBOX,
            "bounded": 0,
        },
        headers={"User-Agent": _USER_AGENT, "Accept-Language": "pl,en"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        logger.warning("Unexpected Nominatim payload type: %s", type(payload))
        return []
    hits: list[PlaceSuggestion] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            lon = float(row["lon"])
            lat = float(row["lat"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(row.get("display_name") or "").strip()
        if not label:
            continue
        hits.append(
            PlaceSuggestion(
                label=label,
                lon=lon,
                lat=lat,
                osm_type=str(row.get("osm_type") or ""),
                osm_id=int(row.get("osm_id") or 0),
            )
        )
    logger.info("Nominatim %r → %s hits", needle, len(hits))
    return hits
