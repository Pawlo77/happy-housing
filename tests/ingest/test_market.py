"""Tests for market_primary derivation (Otodom market + RCN rodzaj_rynku)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.ingest.market import (
    backfill_market_primary,
    ensure_market_primary,
    market_primary_from_rodzaj_rynku,
)


def test_market_primary_from_rodzaj_rynku() -> None:
    assert market_primary_from_rodzaj_rynku(1) == 1.0
    assert market_primary_from_rodzaj_rynku("2") == 0.0
    assert pd.isna(market_primary_from_rodzaj_rynku(None))
    assert pd.isna(market_primary_from_rodzaj_rynku("x"))


def test_ensure_market_primary_from_market_string() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "market": ["primary", "secondary", None],
    })
    out = ensure_market_primary(frame)
    assert float(out.loc[0, "market_primary"]) == 1.0
    assert float(out.loc[1, "market_primary"]) == 0.0
    assert pd.isna(out.loc[2, "market_primary"])


def test_ensure_market_primary_from_rodzaj_rynku() -> None:
    frame = pd.DataFrame({
        "listing_id": ["r1", "r2"],
        "rodzaj_rynku": [1, 2],
    })
    out = ensure_market_primary(frame)
    assert float(out.loc[0, "market_primary"]) == 1.0
    assert float(out.loc[1, "market_primary"]) == 0.0


def test_backfill_market_primary_from_otodom_parquet(tmp_path: Path) -> None:
    oto = pd.DataFrame({
        "listing_id": ["OTO-1", "OTO-2"],
        "market": ["primary", "secondary"],
        "market_primary": [1, 0],
    })
    path = tmp_path / "otodom.parquet"
    oto.to_parquet(path, index=False)
    scored = pd.DataFrame({
        "listing_id": ["OTO-1", "OTO-2", "RCN-0"],
        "source": ["otodom", "otodom", "rcn"],
        "price_per_m2": [1.0, 2.0, 3.0],
    })
    out = backfill_market_primary(scored, otodom_path=path)
    assert "market_primary" in out.columns
    assert float(out.loc[out["listing_id"] == "OTO-1", "market_primary"].iloc[0]) == 1.0
    assert float(out.loc[out["listing_id"] == "OTO-2", "market_primary"].iloc[0]) == 0.0
    assert pd.isna(out.loc[out["listing_id"] == "RCN-0", "market_primary"].iloc[0])
