"""Unit tests for Otodom HTML/__NEXT_DATA__ parsers (no live network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from happy_housing.ingest.otodom import (
    _extract_next_data,
    _normalize_search_item,
    fetch_listing_detail,
)


def _wrap_next_data(payload: dict[str, Any]) -> str:
    return (
        "<html><head></head><body>"
        f'<script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}</script></body></html>"
    )


def test_extract_next_data_roundtrip() -> None:
    """Parser returns the embedded JSON object."""
    payload = {"props": {"pageProps": {"ok": True}}}
    assert _extract_next_data(_wrap_next_data(payload)) == payload


def test_extract_next_data_missing_raises() -> None:
    """Missing marker raises ValueError."""
    with pytest.raises(ValueError, match="missing __NEXT_DATA__"):
        _extract_next_data("<html></html>")


def test_normalize_search_item_maps_price_and_rooms() -> None:
    """Search card maps total price, zł/m², and rooms enums."""
    item = {
        "id": 123,
        "slug": "test-oferta",
        "title": "Mieszkanie",
        "totalPrice": {"value": 900_000},
        "pricePerSquareMeter": {"value": 18_000},
        "areaInSquareMeters": 50,
        "roomsNumber": "TWO",
        "images": [{"medium": "https://example.com/a.jpg"}],
        "location": {
            "reverseGeocoding": {
                "locations": [
                    {"locationLevel": "district", "name": "Mokotów"},
                ]
            }
        },
    }
    row = _normalize_search_item(item)
    assert row["listing_id"] == "OTO-123"
    assert row["price_pln"] == 900_000.0
    assert row["price_per_m2"] == 18_000.0
    assert row["rooms"] == 2
    assert row["district"] == "Mokotów"
    assert row["image_url"] == "https://example.com/a.jpg"


class _FakeResponse:
    def __init__(self, html: str) -> None:
        self.text = html

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, html: str) -> None:
        self._html = html

    def get(self, url: str, timeout: float = 0.0) -> _FakeResponse:
        _ = url, timeout
        return _FakeResponse(self._html)


def test_fetch_listing_detail_reads_coordinates() -> None:
    """Detail page exposes lat/lon and optional build year."""
    payload = {
        "props": {
            "pageProps": {
                "ad": {
                    "location": {
                        "coordinates": {
                            "latitude": 52.23,
                            "longitude": 21.01,
                        }
                    },
                    "dateCreated": "2026-03-01T10:00:00Z",
                    "characteristics": [
                        {"key": "buildYear", "value": "2015"},
                    ],
                }
            }
        }
    }
    session = _FakeSession(_wrap_next_data(payload))
    detail = fetch_listing_detail(session, "test-slug")  # type: ignore[arg-type]
    assert detail["lat"] == pytest.approx(52.23)
    assert detail["lon"] == pytest.approx(21.01)
    assert detail["year_built"] == 2015
    assert detail["listed_at"] == "2026-03-01T10:00:00Z"
