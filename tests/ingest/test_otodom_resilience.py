"""Tests for resilient Otodom HTTP fetch and incremental merge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from happy_housing.ingest.http_client import is_retryable_http_error
from happy_housing.ingest.otodom import (
    card_needs_detail_refresh,
    fetch_search_page,
    merge_otodom_frames,
    scrape_otodom_warsaw,
)


def _wrap_next_data(payload: dict[str, Any]) -> str:
    return (
        "<html><body>"
        f'<script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}</script></body></html>"
    )


def _search_html(items: list[dict[str, Any]], total_pages: int = 1) -> str:
    return _wrap_next_data({
        "props": {
            "pageProps": {
                "data": {
                    "searchAds": {
                        "items": items,
                        "pagination": {
                            "totalPages": total_pages,
                            "totalItems": len(items),
                        },
                    }
                }
            }
        }
    })


def test_is_retryable_http_error_for_5xx_and_429() -> None:
    response = MagicMock()
    response.status_code = 500
    err = requests.HTTPError(response=response)
    assert is_retryable_http_error(err) is True
    response.status_code = 429
    assert is_retryable_http_error(requests.HTTPError(response=response)) is True
    response.status_code = 404
    assert is_retryable_http_error(requests.HTTPError(response=response)) is False
    assert is_retryable_http_error(requests.Timeout()) is True


def test_fetch_search_page_returns_none_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard HTTP failures must not raise after exhausting retries."""
    sleeps: list[float] = []
    monkeypatch.setattr(
        "happy_housing.ingest.otodom.time.sleep",
        lambda s: sleeps.append(s),
    )

    class BoomSession:
        def get(self, url: str, timeout: object = None) -> MagicMock:
            _ = url, timeout
            response = MagicMock()
            response.status_code = 500
            response.raise_for_status.side_effect = requests.HTTPError(
                "500", response=response
            )
            return response

    ads = fetch_search_page(BoomSession(), page=201, retries=3)  # type: ignore[arg-type]
    assert ads is None
    assert len(sleeps) == 2


def test_merge_otodom_frames_keeps_fresh_and_prior() -> None:
    existing = pd.DataFrame([
        {
            "listing_id": "OTO-1",
            "price_pln": 100_000.0,
            "lat": 52.1,
            "lon": 21.0,
            "area_m2": 40.0,
            "rooms": 2,
            "year_built": 2000,
            "district": "wola",
            "listed_at": "2026-01-01",
        },
        {
            "listing_id": "OTO-2",
            "price_pln": 200_000.0,
            "lat": 52.2,
            "lon": 21.1,
            "area_m2": 50.0,
            "rooms": 3,
            "year_built": 2010,
            "district": "mokotow",
            "listed_at": "2026-01-02",
        },
    ])
    fresh = pd.DataFrame([
        {
            "listing_id": "OTO-1",
            "price_pln": 110_000.0,
            "lat": 52.1,
            "lon": 21.0,
            "area_m2": 40.0,
            "rooms": 2,
            "year_built": 2000,
            "district": "wola",
            "listed_at": "2026-02-01",
        },
        {
            "listing_id": "OTO-3",
            "price_pln": 300_000.0,
            "lat": 52.3,
            "lon": 21.2,
            "area_m2": 60.0,
            "rooms": 3,
            "year_built": 2015,
            "district": "bemowo",
            "listed_at": "2026-03-01",
        },
    ])
    merged = merge_otodom_frames(existing, fresh)
    assert set(merged["listing_id"]) == {"OTO-1", "OTO-2", "OTO-3"}
    assert (
        float(merged.loc[merged["listing_id"] == "OTO-1", "price_pln"].iloc[0])
        == 110_000.0
    )


def test_card_needs_detail_refresh_for_new_or_price_change() -> None:
    existing = {
        "OTO-1": {
            "listing_id": "OTO-1",
            "price_pln": 100.0,
            "lat": 52.0,
            "lon": 21.0,
            "image_urls_json": '["https://cdn.example/1.jpg"]',
        }
    }
    assert card_needs_detail_refresh(
        {"listing_id": "OTO-2", "price_pln": 50.0}, existing
    )
    assert card_needs_detail_refresh(
        {"listing_id": "OTO-1", "price_pln": 120.0}, existing
    )
    assert not card_needs_detail_refresh(
        {"listing_id": "OTO-1", "price_pln": 100.0}, existing
    )


def test_scrape_skips_failed_search_page_and_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One dead search page must not abort the run when later pages work."""
    monkeypatch.setattr("happy_housing.ingest.otodom.time.sleep", lambda _s: None)

    pages = {
        1: {
            "items": [
                {
                    "id": 1,
                    "slug": "a",
                    "title": "A",
                    "totalPrice": {"value": 100},
                    "pricePerSquareMeter": {"value": 10},
                    "areaInSquareMeters": 10,
                    "roomsNumber": "TWO",
                    "images": [{}],
                    "location": {"reverseGeocoding": {"locations": []}},
                }
            ],
            "pagination": {"totalPages": 3, "totalItems": 3},
        },
        2: None,
        3: {
            "items": [
                {
                    "id": 3,
                    "slug": "c",
                    "title": "C",
                    "totalPrice": {"value": 300},
                    "pricePerSquareMeter": {"value": 30},
                    "areaInSquareMeters": 10,
                    "roomsNumber": "THREE",
                    "images": [{}],
                    "location": {"reverseGeocoding": {"locations": []}},
                }
            ],
            "pagination": {"totalPages": 3, "totalItems": 3},
        },
    }

    def fake_search(
        session: object,
        page: int,
        limit: int = 36,
        timeout_s: object = None,
        retries: int = 5,
        sort_by: str | None = None,
        sort_direction: str | None = None,
    ) -> dict[str, Any] | None:
        _ = session, limit, timeout_s, retries, sort_by, sort_direction
        return pages[page]

    def fake_detail(session: object, slug: str, **kwargs: object) -> dict[str, Any]:
        _ = session, kwargs
        return {
            "lat": 52.2,
            "lon": 21.0,
            "year_built": 2001,
            "listed_at": "2026-01-01",
            "description": None,
            "market": None,
            "market_primary": None,
            "building_type": None,
            "floor_no": None,
            "building_floors_num": None,
            "construction_status": None,
            "building_material": None,
            "windows_type": None,
            "building_ownership": None,
            "heating": None,
            "rent": None,
            "rooms": 2,
            "has_balcony": 0,
            "has_garage": 0,
            "has_lift": 0,
            "has_terrace": 0,
            "has_garden": 0,
            "is_private_owner": 0,
            "advertiser_type": None,
            "characteristics_json": "[]",
            "additional_information_json": "[]",
        }

    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_search_page", fake_search)
    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_listing_detail", fake_detail)

    out = tmp_path / "otodom.parquet"
    frame = scrape_otodom_warsaw(
        out,
        max_pages=3,
        detail_workers=2,
        sleep_s=0.0,
        incremental=False,
    )
    assert set(frame["listing_id"]) == {"OTO-1", "OTO-3"}
    assert out.exists()


def test_scrape_incremental_skips_unchanged_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("happy_housing.ingest.otodom.time.sleep", lambda _s: None)
    existing = pd.DataFrame([
        {
            "listing_id": "OTO-1",
            "slug": "a",
            "price_pln": 100.0,
            "price_per_m2": 10.0,
            "area_m2": 10.0,
            "rooms": 2,
            "year_built": 2000,
            "district": "wola",
            "listed_at": "2026-01-01",
            "lat": 52.1,
            "lon": 21.0,
            "has_balcony": 0,
            "has_garage": 0,
            "has_lift": 0,
            "has_terrace": 0,
            "has_garden": 0,
            "image_urls_json": '["https://cdn.example/a.jpg"]',
            "floor_plan_urls_json": "[]",
        }
    ])
    out = tmp_path / "otodom.parquet"
    existing.to_parquet(out, index=False)

    detail_calls: list[str] = []
    search_sorts: list[str | None] = []
    search_directions: list[str | None] = []

    def fake_search(
        session: object,
        page: int,
        limit: int = 36,
        timeout_s: object = None,
        retries: int = 5,
        sort_by: str | None = None,
        sort_direction: str | None = None,
    ) -> dict[str, Any] | None:
        _ = session, limit, timeout_s, retries
        search_sorts.append(sort_by)
        search_directions.append(sort_direction)
        if page > 1:
            return {"items": [], "pagination": {"totalPages": 1}}
        return {
            "items": [
                {
                    "id": 1,
                    "slug": "a",
                    "title": "A",
                    "totalPrice": {"value": 100},
                    "pricePerSquareMeter": {"value": 10},
                    "areaInSquareMeters": 10,
                    "roomsNumber": "TWO",
                    "images": [{}],
                    "location": {"reverseGeocoding": {"locations": []}},
                },
                {
                    "id": 2,
                    "slug": "b",
                    "title": "B",
                    "totalPrice": {"value": 200},
                    "pricePerSquareMeter": {"value": 20},
                    "areaInSquareMeters": 10,
                    "roomsNumber": "TWO",
                    "images": [{}],
                    "location": {"reverseGeocoding": {"locations": []}},
                },
            ],
            "pagination": {"totalPages": 1, "totalItems": 2},
        }

    def fake_detail(session: object, slug: str, **kwargs: object) -> dict[str, Any]:
        _ = session, kwargs
        detail_calls.append(str(slug))
        return {
            "lat": 52.2,
            "lon": 21.1,
            "year_built": 2011,
            "listed_at": "2026-02-01",
            "description": None,
            "market": None,
            "market_primary": None,
            "building_type": None,
            "floor_no": None,
            "building_floors_num": None,
            "construction_status": None,
            "building_material": None,
            "windows_type": None,
            "building_ownership": None,
            "heating": None,
            "rent": None,
            "rooms": 2,
            "has_balcony": 0,
            "has_garage": 0,
            "has_lift": 0,
            "has_terrace": 0,
            "has_garden": 0,
            "is_private_owner": 0,
            "advertiser_type": None,
            "characteristics_json": "[]",
            "additional_information_json": "[]",
        }

    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_search_page", fake_search)
    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_listing_detail", fake_detail)

    frame = scrape_otodom_warsaw(
        out,
        max_pages=1,
        detail_workers=2,
        sleep_s=0.0,
        incremental=True,
    )
    assert search_sorts == ["LATEST"]
    assert search_directions == ["DESC"]
    assert detail_calls == ["b"]
    assert set(frame["listing_id"]) == {"OTO-1", "OTO-2"}


def test_fetch_search_page_passes_latest_desc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_get(
        session: object,
        url: str,
        *,
        timeout: object = None,
        retries: int = 5,
        label: str = "",
    ) -> MagicMock:
        _ = session, timeout, retries, label
        captured["url"] = url
        response = MagicMock()
        response.text = _search_html([
            {
                "id": 9,
                "slug": "x",
                "title": "X",
                "totalPrice": {"value": 1},
                "areaInSquareMeters": 10,
                "roomsNumber": "TWO",
                "images": [{}],
                "location": {"reverseGeocoding": {"locations": []}},
            }
        ])
        return response

    monkeypatch.setattr("happy_housing.ingest.otodom._http_get", fake_get)
    ads = fetch_search_page(
        MagicMock(),
        page=1,
        sort_by="LATEST",
        sort_direction="DESC",
    )
    assert ads is not None
    assert "by=LATEST" in captured["url"]
    assert "direction=DESC" in captured["url"]


def test_scrape_incremental_early_stops_after_known_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newest-first pages: stop once two fully-known pages appear."""
    monkeypatch.setattr("happy_housing.ingest.otodom.time.sleep", lambda _s: None)
    existing = pd.DataFrame([
        {
            "listing_id": lid,
            "slug": slug,
            "price_pln": 100.0,
            "price_per_m2": 10.0,
            "area_m2": 10.0,
            "rooms": 2,
            "year_built": 2000,
            "district": "wola",
            "listed_at": "2026-01-01",
            "lat": 52.1,
            "lon": 21.0,
            "has_balcony": 0,
            "has_garage": 0,
            "has_lift": 0,
            "has_terrace": 0,
            "has_garden": 0,
            "image_urls_json": '["https://cdn.example/a.jpg"]',
            "floor_plan_urls_json": "[]",
        }
        for lid, slug in (
            ("OTO-10", "old-a"),
            ("OTO-11", "old-b"),
            ("OTO-12", "old-c"),
            ("OTO-13", "old-d"),
        )
    ])
    out = tmp_path / "otodom.parquet"
    existing.to_parquet(out, index=False)
    pages_hit: list[int] = []

    def _item(listing_id: int, slug: str) -> dict[str, Any]:
        return {
            "id": listing_id,
            "slug": slug,
            "title": slug,
            "totalPrice": {"value": 100},
            "pricePerSquareMeter": {"value": 10},
            "areaInSquareMeters": 10,
            "roomsNumber": "TWO",
            "images": [{}],
            "location": {"reverseGeocoding": {"locations": []}},
            "dateCreated": "2026-01-01T00:00:00Z",
        }

    def fake_search(
        session: object,
        page: int,
        limit: int = 36,
        timeout_s: object = None,
        retries: int = 5,
        sort_by: str | None = None,
        sort_direction: str | None = None,
    ) -> dict[str, Any] | None:
        _ = session, limit, timeout_s, retries, sort_by, sort_direction
        pages_hit.append(page)
        if page == 1:
            items = [_item(99, "brand-new"), _item(10, "old-a")]
        elif page == 2:
            items = [_item(11, "old-b"), _item(12, "old-c")]
        elif page == 3:
            items = [_item(13, "old-d"), _item(12, "old-c")]
        else:
            items = [_item(11, "old-b"), _item(10, "old-a")]
        return {
            "items": items,
            "pagination": {"totalPages": 20, "totalItems": 500},
        }

    def fake_detail(session: object, slug: str, **kwargs: object) -> dict[str, Any]:
        _ = session, kwargs
        return {
            "lat": 52.2,
            "lon": 21.1,
            "year_built": 2011,
            "listed_at": "2026-03-01",
            "description": None,
            "market": None,
            "market_primary": None,
            "building_type": None,
            "floor_no": None,
            "building_floors_num": None,
            "construction_status": None,
            "building_material": None,
            "windows_type": None,
            "building_ownership": None,
            "heating": None,
            "rent": None,
            "rooms": 2,
            "has_balcony": 0,
            "has_garage": 0,
            "has_lift": 0,
            "has_terrace": 0,
            "has_garden": 0,
            "is_private_owner": 0,
            "advertiser_type": None,
            "characteristics_json": "[]",
            "additional_information_json": "[]",
            "image_urls_json": '["https://cdn.example/n.jpg"]',
        }

    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_search_page", fake_search)
    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_listing_detail", fake_detail)

    frame = scrape_otodom_warsaw(
        out,
        max_pages=None,
        detail_workers=2,
        sleep_s=0.0,
        incremental=True,
        stop_after_known_pages=2,
    )
    assert pages_hit == [1, 2]
    assert "OTO-99" in set(frame["listing_id"])


def test_cards_at_or_before_watermark() -> None:
    from happy_housing.ingest.otodom import (
        cards_at_or_before_watermark,
        max_existing_listed_at,
    )

    existing = pd.DataFrame({"listed_at": ["2026-02-01", "2026-03-10"]})
    watermark = max_existing_listed_at(existing)
    assert watermark is not None
    assert cards_at_or_before_watermark(
        [{"listed_at": "2026-03-10"}, {"listed_at": "2026-01-01"}],
        watermark,
    )
    assert not cards_at_or_before_watermark(
        [{"listed_at": "2026-03-11"}],
        watermark,
    )
