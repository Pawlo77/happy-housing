"""OSM Nominatim place search for Admin commute-target picker."""

from __future__ import annotations

import logging
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
