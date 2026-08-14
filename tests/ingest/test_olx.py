"""Unit tests for OLX Warsaw public-API ingest (no live network)."""

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from happy_housing.ingest.olx import (
    normalize_olx_offer,
    scrape_olx_warsaw,
)


def _param(
    key: str, *, value_key: str | None = None, value: Any = None, label: str = ""
) -> dict[str, Any]:
    return {
        "key": key,
        "value": {
            "key": value_key,
            "value": value,
            "label": label,
            "currency": "PLN" if key == "price" else None,
        },
    }


def _offer_payload(
    offer_id: int = 1001,
    *,
    segment: str = "sale",
    created: str = "2026-08-13T10:00:00+02:00",
    lat: float = 52.23,
    lon: float = 21.01,
    price: float = 500_000.0,
    area: str = "42.5",
    rooms: str = "two",
    floor: str = "floor_3",
    district: str = "Wola",
) -> dict[str, Any]:
    return {
        "id": offer_id,
        "title": f"Flat {offer_id}",
        "url": f"https://www.olx.pl/d/oferta/flat-{offer_id}.html",
        "description": "Nice flat",
        "created_time": created,
        "last_refresh_time": created,
        "category": {"id": 14 if segment == "sale" else 15},
        "map": {"lat": lat, "lon": lon},
        "location": {
            "city": {"name": "Warszawa"},
            "district": {"id": 359, "name": district},
        },
        "price": {"amount": price, "currency": "PLN"},
        "photos": [
            {"link": "https://cdn.example/{width}x{height}", "filename": "a"},
        ],
        "params": [
            _param("m", value_key=area, label=f"{area} m²"),
            _param("rooms", value_key=rooms, label="2 pokoje"),
            _param("floor_select", value_key=floor, label="3"),
            _param("market", value_key="secondary", label="Wtórny"),
            _param("price", value=price, label=f"{price} zł"),
            _param("price_per_m", value_key="11764.71", label="11764.71 zł/m²"),
        ],
        "business": False,
    }


def test_normalize_olx_offer_sale_fields() -> None:
    row = normalize_olx_offer(_offer_payload(), listing_kind="sale")
    assert row is not None
    assert row["listing_id"] == "OLX-1001"
    assert row["olx_id"] == 1001
    assert row["lat"] == pytest.approx(52.23)
    assert row["lon"] == pytest.approx(21.01)
    assert row["price_pln"] == pytest.approx(500_000.0)
    assert row["area_m2"] == pytest.approx(42.5)
    assert row["rooms"] == 2
    assert row["floor_no"] == 3
    assert row["district"] == "Wola"
    assert row["listing_kind"] == "sale"
    assert row["market_primary"] is False
    assert "2026-08-13" in str(row["listed_at"])
    assert "cdn.example" in str(row["image_url"])


def test_normalize_skips_missing_coords() -> None:
    payload = _offer_payload()
    payload["map"] = {"lat": None, "lon": None}
    assert normalize_olx_offer(payload, listing_kind="sale") is None


def test_scrape_olx_warsaw_incremental_with_fake_client(
    tmp_path: Any,
) -> None:
    out = tmp_path / "olx_listings.parquet"
    existing = pd.DataFrame([
        {
            "listing_id": "OLX-1",
            "olx_id": 1,
            "lat": 52.2,
            "lon": 21.0,
            "price_pln": 400_000.0,
            "area_m2": 40.0,
            "rooms": 2,
            "year_built": None,
            "district": "Wola",
            "listed_at": "2026-08-10T08:00:00+00:00",
            "listing_kind": "sale",
            "price_per_m2": 10_000.0,
        }
    ])
    existing.to_parquet(out, index=False)

    pages: dict[tuple[int, int, str], list[dict[str, Any]]] = {
        (359, 0, "sale"): [
            _offer_payload(2, created="2026-08-13T12:00:00+02:00"),
            _offer_payload(1, created="2026-08-10T10:00:00+02:00"),
        ],
        (359, 40, "sale"): [],
    }

    def fake_fetch(
        *,
        category_id: int,
        district_id: int,
        offset: int,
        limit: int,
        listing_kind: str,
    ) -> SimpleNamespace:
        del category_id, limit
        data = pages.get((district_id, offset, listing_kind), [])
        return SimpleNamespace(
            data=data,
            metadata=SimpleNamespace(total_elements=len(data) if offset == 0 else 0),
        )

    frame = scrape_olx_warsaw(
        out,
        incremental=True,
        max_listings=50,
        include_rent=False,
        district_ids=(359,),
        page_size=40,
        sleep_s=0.0,
        fetch_page=fake_fetch,
        fetch_district_ids=lambda: [359],
    )
    assert set(frame["listing_id"]) == {"OLX-1", "OLX-2"}
    assert out.exists()
    assert len(pd.read_parquet(out)) == 2
