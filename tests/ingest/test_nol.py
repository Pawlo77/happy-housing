"""Tests for Nieruchomości Online JSON-LD ingest (no live network)."""

from typing import Any

import pandas as pd
import pytest

from happy_housing.ingest.listing_common import BUYER_LISTING_SCHEMA_COLUMNS
from happy_housing.ingest.nol import (
    backfill_nol_details,
    extract_json_ld_blocks,
    extract_tiles_data,
    listing_id_from_url,
    merge_nol_search_cards,
    normalize_nol_detail,
    normalize_nol_search_offer,
    row_needs_nol_detail_backfill,
    scrape_nol_warsaw,
)


def _search_offer(
    offer_id: str = "26815983",
    *,
    price: str = "860000",
    ppm: str = "17972.16",
    area: str = "48.13",
    rooms: int = 2,
) -> dict[str, Any]:
    return {
        "@type": "Offer",
        "price": price,
        "priceCurrency": "PLN",
        "priceSpecification": {"price": ppm, "priceCurrency": "PLN"},
        "url": (
            f"https://warszawa.nieruchomosci-online.pl/mieszkanie,x/{offer_id}.html"
        ),
        "image": "https://i.example/a.jpg",
        "name": "Sprzedam Mieszkanie Warszawa",
        "itemOffered": {
            "@type": "Accommodation",
            "description": "Mieszkanie z balkonem.",
            "address": {
                "streetAddress": "Wołoska",
                "addressLocality": "Warszawa",
            },
            "floorSize": {"value": area, "unitCode": "MTR"},
            "numberOfRooms": rooms,
        },
    }


def _search_html(
    offers: list[dict[str, Any]], tiles: dict[str, Any] | None = None
) -> str:
    payload = {
        "@type": "CollectionPage",
        "mainEntity": {
            "@type": "Product",
            "offers": [{"@type": "AggregateOffer", "offers": offers}],
        },
    }
    import json

    ld = json.dumps(payload, ensure_ascii=False)
    tiles_js = ""
    if tiles is not None:
        tiles_js = f"tilesData: {json.dumps(tiles, ensure_ascii=False)},\n"
    return (
        f'<script type="application/ld+json">{ld}</script>'
        f"<script>var cfg = {{ {tiles_js} other: 1 }};</script>"
        '<a href="szukaj.html?3,mieszkanie,sprzedaz,,Warszawa&amp;p=2">2</a>'
    )


def _detail_apartment(*, lat: str = "52.2", lon: str = "21.0") -> dict[str, Any]:
    return {
        "@type": "Apartment",
        "name": "Mieszkanie",
        "description": "2 pokoje z windą.",
        "url": "https://warszawa.nieruchomosci-online.pl/mieszkanie,x/26815983.html",
        "image": ["https://i.example/a.jpg", "https://i.example/b.jpg"],
        "amenityFeature": [
            {"@type": "LocationFeatureSpecification", "name": "Balcony", "value": True},
            {
                "@type": "LocationFeatureSpecification",
                "name": "Elevator",
                "value": True,
            },
        ],
        "additionalProperty": [
            {"name": "Floor level", "value": "1"},
            {"name": "Total floors in building", "value": "4"},
            {"name": "Year built", "value": "1959"},
            {"name": "Building type", "value": "kamienica"},
            {"name": "Basement access", "value": "Tak, 0.00 m²"},
        ],
        "geo": {"@type": "GeoCoordinates", "latitude": lat, "longitude": lon},
        "offers": [{"price": "860000", "priceCurrency": "PLN"}],
        "agent": {"@type": "RealEstateAgent", "name": "Anna"},
        "datePosted": "2026-07-09CEST12:04:05Z",
        "numberOfRooms": 2,
        "floorLevel": "1",
        "yearBuilt": 1959,
        "floorSize": {"value": "48.13", "unitCode": "MTR"},
    }


def test_listing_id_from_url() -> None:
    assert (
        listing_id_from_url(
            "https://warszawa.nieruchomosci-online.pl/foo/26815983.html"
        )
        == "NOL-26815983"
    )
    assert listing_id_from_url("https://example.com/x") is None


def test_normalize_search_offer_maps_schema() -> None:
    row = normalize_nol_search_offer(_search_offer())
    assert row is not None
    assert row["listing_id"] == "NOL-26815983"
    assert row["source"] == "nol"
    assert row["price_pln"] == pytest.approx(860_000.0)
    assert row["area_m2"] == pytest.approx(48.13)
    assert row["rooms"] == 2
    assert row["has_balcony"] == 1
    assert row["lat"] is None
    for col in BUYER_LISTING_SCHEMA_COLUMNS:
        assert col in row


def test_normalize_search_skips_missing_price() -> None:
    offer = _search_offer(price="")
    offer["priceSpecification"]["price"] = ""
    assert normalize_nol_search_offer(offer) is None


def test_tiles_data_fills_coords_and_floors() -> None:
    tiles = {
        "a26815983": {
            "map": {"latitude": "52.1995815", "longitude": "21.0011704"},
            "attributes": [
                {
                    "alias": "floor",
                    "values": [
                        {"value": "1", "type": "big-number"},
                        {"value": "/", "type": "number"},
                        {"value": "4", "type": "number"},
                    ],
                },
                {
                    "alias": "year-built",
                    "values": [{"value": "1959", "type": "big-number"}],
                },
            ],
        }
    }
    html = _search_html([_search_offer()], tiles=tiles)
    parsed = extract_tiles_data(html)
    assert "26815983" in parsed
    row = normalize_nol_search_offer(_search_offer(), tile=parsed["26815983"])
    assert row is not None
    assert row["lat"] == pytest.approx(52.1995815)
    assert row["lon"] == pytest.approx(21.0011704)
    assert row["floor_no"] == 1
    assert row["building_floors_num"] == 4
    assert row["year_built"] == 1959


def test_normalize_detail_jsonld() -> None:
    row = normalize_nol_detail(_detail_apartment())
    assert row is not None
    assert row["lat"] == pytest.approx(52.2)
    assert row["year_built"] == 1959
    assert row["building_type"] == "kamienica"
    assert row["has_lift"] == 1
    assert row["has_balcony"] == 1
    assert row["has_basement"] == 1
    assert row["is_private_owner"] == 0
    assert row["building_floors_num"] == 4
    assert "1959" in (row.get("characteristics_json") or "")


def test_extract_json_ld_blocks() -> None:
    html = '<script type="application/ld+json">{"@type": "Apartment"}</script>'
    blocks = extract_json_ld_blocks(html)
    assert blocks[0]["@type"] == "Apartment"


def test_merge_preserves_detail_coords() -> None:
    existing = pd.DataFrame([
        {
            **(normalize_nol_search_offer(_search_offer()) or {}),
            "lat": 52.1,
            "lon": 21.0,
            "characteristics_json": '{"ok": true}',
        }
    ])
    fresh = [normalize_nol_search_offer(_search_offer(price="870000"))]
    merged = merge_nol_search_cards(existing, [r for r in fresh if r])
    row = merged.iloc[0]
    assert float(row["lat"]) == 52.1
    assert float(row["price_pln"]) == 870_000.0
    assert row["characteristics_json"] == '{"ok": true}'


def test_scrape_incremental_early_stop(tmp_path: Any) -> None:
    out = tmp_path / "nol.parquet"
    first = scrape_nol_warsaw(
        out,
        incremental=False,
        fetch_details=False,
        max_pages=1,
        fetch_search_page=lambda page, **_: _search_html([_search_offer("10001")]),
    )
    assert len(first) == 1
    pages: list[int] = []

    def _fetch(page: int, **_: object) -> str:
        pages.append(page)
        return _search_html([_search_offer("10001")])

    scrape_nol_warsaw(
        out,
        incremental=True,
        fetch_details=False,
        stop_after_known_pages=2,
        fetch_search_page=_fetch,
    )
    assert pages == [1, 2]


def test_backfill_merges_detail(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "nol.parquet"
    card = normalize_nol_search_offer(_search_offer())
    assert card is not None
    pd.DataFrame([card]).to_parquet(out, index=False)
    import json

    html = (
        f'<script type="application/ld+json">{json.dumps(_detail_apartment())}</script>'
    )
    monkeypatch.setattr(
        "happy_housing.ingest.nol.fetch_nol_detail_html",
        lambda url, **_: html,
    )
    updated = backfill_nol_details(out, max_listings=5, detail_workers=1, sleep_s=0.0)
    assert float(updated.iloc[0]["lat"]) == pytest.approx(52.2)
    assert row_needs_nol_detail_backfill(updated.iloc[0].to_dict()) is False
