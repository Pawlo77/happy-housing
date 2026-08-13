"""Tests for NBP FX conversion helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from happy_housing.ingest.fx import (
    apply_fx_to_listings_frame,
    currency_from_characteristics,
    format_original_price,
    normalize_price_fields,
)


def test_normalize_pln_passthrough(tmp_path: Path) -> None:
    out = normalize_price_fields(
        amount=500_000,
        currency="PLN",
        area_m2=50.0,
        price_per_m2=10_000.0,
        cache_path=tmp_path / "fx.json",
    )
    assert out["price_pln"] == 500_000
    assert out["price_currency"] == "PLN"
    assert out["price_original"] == 500_000
    assert out["price_per_m2"] == 10_000.0
    assert out["fx_rate_to_pln"] == 1.0


def test_normalize_eur_converts_and_recomputes_ppm(tmp_path: Path) -> None:
    cache = tmp_path / "fx.json"
    with patch(
        "happy_housing.ingest.fx.fetch_nbp_mid_rate",
        return_value=(4.25, "2026-08-08"),
    ):
        out = normalize_price_fields(
            amount=1_000_000,
            currency="EUR",
            area_m2=100.0,
            price_per_m2=5_409.0,  # misleading PLN ppm from Otodom
            cache_path=cache,
        )
    assert out["price_pln"] == 4_250_000.0
    assert out["price_currency"] == "EUR"
    assert out["price_original"] == 1_000_000
    assert out["price_per_m2"] == 42_500.0
    assert out["fx_rate_to_pln"] == 4.25


def test_currency_from_characteristics() -> None:
    chars = [
        {"key": "price", "value": "99000", "currency": "EUR"},
        {"key": "price_per_m", "value": "5409", "currency": "PLN"},
    ]
    assert currency_from_characteristics(chars) == "EUR"
    assert currency_from_characteristics([]) is None


def test_apply_fx_retrofits_from_characteristics(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        {
            "listing_id": "OTO-1",
            "price_pln": 99_000.0,  # wrongly stored EUR as PLN
            "price_per_m2": 5_409.0,
            "area_m2": 18.3,
            "characteristics_json": json.dumps([
                {"key": "price", "value": "99000", "currency": "EUR"}
            ]),
        }
    ])
    with patch(
        "happy_housing.ingest.fx.fetch_nbp_mid_rate",
        return_value=(4.0, "2026-08-08"),
    ):
        out = apply_fx_to_listings_frame(frame, tmp_path / "fx.json")
    assert out.iloc[0]["price_currency"] == "EUR"
    assert out.iloc[0]["price_original"] == 99_000.0
    assert out.iloc[0]["price_pln"] == 396_000.0
    assert abs(out.iloc[0]["price_per_m2"] - (396_000.0 / 18.3)) < 1e-6


def test_format_original_price() -> None:
    assert format_original_price(1_000_000, "EUR") == "1 000 000 €"
    assert format_original_price(500_000, "PLN") is None


def test_fetch_uses_cache(tmp_path: Path) -> None:
    cache = tmp_path / "fx.json"
    mock_fetch = MagicMock(return_value=(4.2, "2026-08-08"))
    with patch("happy_housing.ingest.fx.fetch_nbp_mid_rate", mock_fetch):
        from happy_housing.ingest.fx import get_pln_rate

        get_pln_rate("EUR", cache)
        get_pln_rate("EUR", cache)
    assert mock_fetch.call_count == 1
