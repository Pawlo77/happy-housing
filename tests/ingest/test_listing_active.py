"""Tests for inactive/closed listing handling."""

from pathlib import Path

import pandas as pd
import pytest
import requests

from happy_housing.ingest.listing_common import (
    LISTING_ACTIVE_COLUMN,
    append_inactive_otodom_asks,
    filter_active_listings,
)
from happy_housing.ingest.otodom import (
    backfill_otodom_details,
    enrich_otodom_listing_row,
    mark_otodom_listing_closed,
    row_needs_detail_backfill,
)


def _search_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "listing_id": "OTO-1",
        "slug": "gone-offer",
        "price_pln": 600_000.0,
        "area_m2": 55.0,
        "rooms": 2,
        "district": "Mokotów",
        "lat": None,
        "lon": None,
    }
    row.update(overrides)
    return row


def test_mark_otodom_listing_closed_sets_inactive_flag() -> None:
    closed = mark_otodom_listing_closed(_search_row())
    assert closed[LISTING_ACTIVE_COLUMN] == 0
    assert closed["detail_price_pln"] == 600_000.0


def test_row_needs_detail_backfill_false_when_inactive() -> None:
    row = mark_otodom_listing_closed(_search_row())
    assert row_needs_detail_backfill(row) is False


def test_enrich_otodom_listing_row_marks_410_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 410

    def _raise_410(*_args: object, **_kwargs: object) -> None:
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(
        "happy_housing.ingest.otodom.fetch_listing_detail",
        _raise_410,
    )
    merged = enrich_otodom_listing_row(_search_row(), session=object())  # type: ignore[arg-type]
    assert merged is not None
    assert merged[LISTING_ACTIVE_COLUMN] == 0


def test_backfill_otodom_details_persists_closed_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "otodom.parquet"
    pd.DataFrame([_search_row()]).to_parquet(out, index=False)

    response = requests.Response()
    response.status_code = 404

    def _raise_404(*_args: object, **_kwargs: object) -> None:
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(
        "happy_housing.ingest.otodom.fetch_listing_detail",
        _raise_404,
    )
    updated = backfill_otodom_details(
        out, max_listings=5, detail_workers=1, sleep_s=0.0
    )
    assert int(updated.iloc[0][LISTING_ACTIVE_COLUMN]) == 0
    assert row_needs_detail_backfill(updated.iloc[0].to_dict()) is False


def test_filter_active_listings_drops_closed() -> None:
    frame = pd.DataFrame([
        {"listing_id": "A", LISTING_ACTIVE_COLUMN: 1},
        {"listing_id": "B", LISTING_ACTIVE_COLUMN: 0},
    ])
    out = filter_active_listings(frame)
    assert set(out["listing_id"]) == {"A"}


def test_append_inactive_otodom_asks_for_stats(tmp_path: Path) -> None:
    scored = pd.DataFrame([
        {"listing_id": "OLX-1", "price_pln": 700_000.0, LISTING_ACTIVE_COLUMN: 1},
    ])
    oto = tmp_path / "otodom_listings.parquet"
    pd.DataFrame([
        {
            **_search_row(listing_id="OTO-9"),
            LISTING_ACTIVE_COLUMN: 0,
            "source": "otodom",
        }
    ]).to_parquet(oto, index=False)
    out = append_inactive_otodom_asks(scored, tmp_path)
    assert len(out) == 2
    assert "OTO-9" in set(out["listing_id"].astype(str))


def test_append_inactive_asks_includes_closed_nol(tmp_path: Path) -> None:
    scored = pd.DataFrame([
        {"listing_id": "OTO-1", "price_pln": 700_000.0, LISTING_ACTIVE_COLUMN: 1},
    ])
    pd.DataFrame([
        {
            "listing_id": "NOL-9",
            "price_pln": 610_000.0,
            LISTING_ACTIVE_COLUMN: 0,
            "source": "nol",
        }
    ]).to_parquet(tmp_path / "nol_listings.parquet", index=False)
    out = append_inactive_otodom_asks(scored, tmp_path)
    assert "NOL-9" in set(out["listing_id"].astype(str))
