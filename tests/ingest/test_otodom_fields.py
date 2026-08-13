"""Unit tests for Otodom full-field detail / search normalization."""

import json
from typing import Any

from happy_housing.ingest.otodom import (
    _extract_next_data,
    _normalize_search_item,
    flatten_otodom_ad,
    parse_floor_no,
)


def _wrap_next_data(payload: dict[str, Any]) -> str:
    return (
        "<html><head></head><body>"
        f'<script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}</script></body></html>"
    )


def test_parse_floor_no_from_enums_and_strings() -> None:
    assert parse_floor_no("floor_5") == 5
    assert parse_floor_no("FIFTH") == 5
    assert parse_floor_no("GROUND") == 0
    assert parse_floor_no("3") == 3
    assert parse_floor_no(None) is None


def test_normalize_search_item_keeps_extra_search_fields() -> None:
    item = {
        "id": 9,
        "slug": "abc",
        "title": "Flat",
        "totalPrice": {"value": 100},
        "pricePerSquareMeter": {"value": 10},
        "areaInSquareMeters": 10,
        "roomsNumber": "TWO",
        "floorNumber": "THIRD",
        "dateCreated": "2026-01-02",
        "createdAtFirst": "2026-01-01",
        "shortDescription": "Nice",
        "isPrivateOwner": False,
        "estate": "FLAT",
        "images": [{"medium": "https://x"}],
        "tags": [{"value": "BALCONY"}, {"value": "GARAGE"}],
        "location": {
            "reverseGeocoding": {
                "locations": [{"locationLevel": "district", "name": "Wola"}]
            }
        },
    }
    row = _normalize_search_item(item)
    assert row["floor_no"] == 3
    assert str(row["listed_at"]).startswith("2026-01-02")
    assert row["short_description"] == "Nice"
    assert row["is_private_owner"] == 0
    assert row["has_balcony"] == 1
    assert row["has_garage"] == 1
    assert row["estate"] == "FLAT"


def test_flatten_otodom_ad_maps_characteristics_and_extras() -> None:
    ad = {
        "location": {"coordinates": {"latitude": 52.2, "longitude": 21.0}},
        "dateCreated": "2026-03-01",
        "description": "Sunny flat",
        "market": "PRIMARY",
        "advertiserType": "agency",
        "characteristics": [
            {"key": "market", "value": "primary"},
            {"key": "building_type", "value": "apartment"},
            {"key": "floor_no", "value": "floor_5"},
            {"key": "building_floors_num", "value": "6"},
            {"key": "heating", "value": "urban"},
            {"key": "build_year", "value": "2018"},
            {"key": "rent", "value": "550"},
            {"key": "rooms_num", "value": "2"},
        ],
        "additionalInformation": [
            {
                "label": "extras_types",
                "values": [
                    "extras_types::balcony",
                    "extras_types::garage",
                    "extras_types::lift",
                ],
            }
        ],
    }
    flat = flatten_otodom_ad(ad)
    assert flat["lat"] == 52.2
    assert flat["lon"] == 21.0
    assert flat["year_built"] == 2018
    assert flat["floor_no"] == 5
    assert flat["building_floors_num"] == 6
    assert flat["market"] == "primary"
    assert flat["market_primary"] == 1
    assert flat["building_type"] == "apartment"
    assert flat["heating"] == "urban"
    assert flat["rent"] == 550.0
    assert flat["has_balcony"] == 1
    assert flat["has_garage"] == 1
    assert flat["has_lift"] == 1
    assert flat["description"] == "Sunny flat"
    assert "characteristics_json" in flat
    assert json.loads(flat["characteristics_json"])[0]["key"] == "market"


def test_extract_next_data_still_works() -> None:
    payload = {"props": {"pageProps": {"ok": True}}}
    assert _extract_next_data(_wrap_next_data(payload)) == payload
