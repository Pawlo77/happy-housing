"""Tests for search-only Otodom scrape and background detail backfill."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from happy_housing.ingest.detail_backfill import (
    count_pending_detail_backfill,
    pending_detail_backfill_rows,
)
from happy_housing.ingest.otodom import (
    backfill_otodom_details,
    merge_otodom_search_cards,
    row_needs_detail_backfill,
    scrape_otodom_warsaw,
)


def _search_card(
    listing_id: str = "1",
    *,
    price_pln: float = 500_000.0,
    slug: str = "offer-a",
) -> dict[str, Any]:
    return {
        "listing_id": listing_id,
        "otodom_id": listing_id,
        "slug": slug,
        "title": "Flat",
        "url": f"https://www.otodom.pl/pl/oferta/{slug}",
        "district": "Mokotów",
        "price_pln": price_pln,
        "price_original": price_pln,
        "price_currency": "PLN",
        "price_per_m2": 10_000.0,
        "area_m2": 50.0,
        "rooms": 2,
        "floor_no": 1,
        "floor_label": "1",
        "estate": "block",
        "listed_at": "2026-01-01T00:00:00Z",
        "short_description": "Nice",
        "image_url": "https://example.com/a.jpg",
        "image_urls_json": '["https://example.com/a.jpg"]',
        "floor_plan_urls_json": "[]",
        "is_private_owner": 0,
        "has_balcony": 0,
        "has_garage": 0,
        "has_outdoor_parking": 0,
        "has_lift": 0,
        "has_terrace": 0,
        "has_garden": 0,
        "has_basement": 0,
        "is_attic": 0,
        "tags_json": "[]",
        "lat": None,
        "lon": None,
        "year_built": None,
    }


def test_merge_otodom_search_cards_preserves_detail_coords() -> None:
    existing = pd.DataFrame([
        {
            **_search_card("1", price_pln=500_000.0),
            "lat": 52.1,
            "lon": 21.0,
            "characteristics_json": '{"x": 1}',
            "detail_price_pln": 500_000.0,
        }
    ])
    cards = [_search_card("1", price_pln=510_000.0)]
    merged = merge_otodom_search_cards(existing, cards)
    row = merged.iloc[0]
    assert float(row["lat"]) == 52.1
    assert float(row["lon"]) == 21.0
    assert row["characteristics_json"] == '{"x": 1}'
    assert float(row["price_pln"]) == 510_000.0


def test_merge_otodom_search_cards_keeps_existing_not_in_search_pass() -> None:
    existing = pd.DataFrame([_search_card("old", slug="old-offer")])
    merged = merge_otodom_search_cards(
        existing, [_search_card("new", slug="new-offer")]
    )
    ids = set(merged["listing_id"].astype(str))
    assert ids == {"old", "new"}


def test_row_needs_detail_backfill_missing_coords() -> None:
    assert row_needs_detail_backfill(_search_card()) is True


def test_row_needs_detail_backfill_price_change() -> None:
    row = _search_card(price_pln=520_000.0)
    row["lat"] = 52.0
    row["lon"] = 21.0
    row["characteristics_json"] = '{"ok": true}'
    row["detail_price_pln"] = 500_000.0
    assert row_needs_detail_backfill(row) is True


def test_row_needs_detail_backfill_complete_row() -> None:
    row = _search_card()
    row["lat"] = 52.0
    row["lon"] = 21.0
    row["characteristics_json"] = '{"ok": true}'
    row["detail_price_pln"] = row["price_pln"]
    assert row_needs_detail_backfill(row) is False


def test_count_pending_detail_backfill(tmp_path: Path) -> None:
    out = tmp_path / "otodom_listings.parquet"
    pd.DataFrame([
        _search_card("a"),
        {
            **_search_card("b"),
            "lat": 52.0,
            "lon": 21.0,
            "characteristics_json": "{}",
            "detail_price_pln": 500_000.0,
        },
    ]).to_parquet(out, index=False)
    counts = count_pending_detail_backfill(tmp_path)
    assert counts["otodom"] == 1
    assert counts["nol"] == 0
    pending = pending_detail_backfill_rows(out, source="otodom", limit=10)
    assert len(pending) == 1
    assert pending[0]["listing_id"] == "a"


def test_scrape_otodom_search_only_writes_rows_without_coords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "otodom.parquet"

    def _fake_fetch(
        session: object,
        page: int,
        *,
        page_size: int = 72,
        sort_by: str | None = None,
        sort_direction: str | None = None,
        limit: int | None = None,
        timeout_s: object = None,
        retries: int = 5,
    ) -> dict[str, Any] | None:
        _ = session, page_size, sort_by, sort_direction, limit, timeout_s, retries
        if page > 1:
            return None
        return {
            "items": [{"id": 99, "slug": "x", "title": "T"}],
            "pagination": {"totalPages": 1, "totalItems": 1},
        }

    monkeypatch.setattr(
        "happy_housing.ingest.otodom.fetch_search_page",
        _fake_fetch,
    )
    monkeypatch.setattr(
        "happy_housing.ingest.otodom._normalize_search_item",
        lambda item: _search_card(str(item["id"]), slug=str(item["slug"])),
    )
    frame = scrape_otodom_warsaw(
        out,
        incremental=False,
        fetch_details=False,
    )
    assert len(frame) == 1
    assert pd.isna(frame.iloc[0]["lat"])
    assert out.exists()


def test_backfill_otodom_details_updates_coords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "otodom.parquet"
    pd.DataFrame([_search_card("1", slug="offer-a")]).to_parquet(out, index=False)

    def _fake_detail(
        session: object, slug: str, *, timeout_s: object = None
    ) -> dict[str, Any]:
        _ = session, slug, timeout_s
        return {
            "lat": 52.2,
            "lon": 21.1,
            "year_built": 2010,
            "characteristics_json": '{"rooms": 2}',
            "image_urls_json": '["https://example.com/z.jpg"]',
            "floor_plan_urls_json": "[]",
            "description": "Full",
            "price_pln": 500_000.0,
            "price_currency": "PLN",
            "has_balcony": 1,
            "has_garage": 0,
            "has_outdoor_parking": 0,
            "has_lift": 0,
            "has_terrace": 0,
            "has_garden": 0,
            "has_basement": 0,
            "is_attic": 0,
        }

    monkeypatch.setattr(
        "happy_housing.ingest.otodom.fetch_listing_detail",
        _fake_detail,
    )
    updated = backfill_otodom_details(
        out,
        max_listings=5,
        detail_workers=1,
        sleep_s=0.0,
    )
    row = updated.iloc[0]
    assert float(row["lat"]) == 52.2
    assert float(row["detail_price_pln"]) == 500_000.0
    assert count_pending_detail_backfill(tmp_path)["otodom"] == 0


def test_count_pending_detail_backfill_nol(tmp_path: Path) -> None:
    out = tmp_path / "nol_listings.parquet"
    pd.DataFrame([
        {
            "listing_id": "NOL-1",
            "url": "https://warszawa.nieruchomosci-online.pl/x/1.html",
            "price_pln": 500_000.0,
            "area_m2": 50.0,
            "lat": None,
            "lon": None,
            "listing_active": 1,
        }
    ]).to_parquet(out, index=False)
    counts = count_pending_detail_backfill(tmp_path)
    assert counts["nol"] == 1
    assert counts["otodom"] == 0
    pending = pending_detail_backfill_rows(out, source="nol", limit=10)
    assert len(pending) == 1
    assert pending[0]["listing_id"] == "NOL-1"
