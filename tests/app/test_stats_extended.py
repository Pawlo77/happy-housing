"""Tests for extended Stats inventory / affordability / livability helpers."""

import pandas as pd

from happy_housing.app.stats_insights import (
    affordability_bands,
    aircraft_noise_ladder,
    market_segment_mix,
    metro_price_curve,
    price_band_inventory,
    rooms_inventory,
    source_mix_summary,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "district": ["wola", "wola", "mokotow", "mokotow", "praga"],
        "source": ["otodom", "otodom", "rcn", "otodom", "rcn"],
        "price_pln": [600_000.0, 900_000.0, 1_200_000.0, 1_800_000.0, 450_000.0],
        "price_per_m2": [15000.0, 18000.0, 20000.0, 24000.0, 12000.0],
        "delta_price_per_m2": [-500.0, 200.0, 800.0, -100.0, -300.0],
        "area_m2": [40.0, 50.0, 60.0, 75.0, 37.5],
        "rooms": [1, 2, 2, 3, 1],
        "market_primary": [1, 0, 0, 1, 0],
        "dist_metro_m": [200.0, 600.0, 1200.0, 300.0, 2500.0],
        "aircraft_lden_db": [0.0, 55.0, 60.0, 0.0, 70.0],
    })


def test_price_band_inventory_has_shares() -> None:
    out = price_band_inventory(_frame())
    assert not out.empty
    assert {"price_band", "n", "share", "median_delta"}.issubset(out.columns)
    assert abs(float(out["share"].sum()) - 1.0) < 1e-6


def test_source_and_rooms_inventory() -> None:
    src = source_mix_summary(_frame())
    assert set(src["source"]) >= {"otodom", "rcn"}
    rooms = rooms_inventory(_frame())
    assert int(rooms["n"].sum()) == 5


def test_metro_and_noise_ladders() -> None:
    metro = metro_price_curve(_frame())
    assert "metro_band" in metro.columns
    noise = aircraft_noise_ladder(_frame())
    assert "lden_band" in noise.columns
    assert float(noise["n"].sum()) == 5


def test_affordability_and_segment_mix() -> None:
    afford = affordability_bands(_frame())
    assert {"price_band", "n", "median_area_m2"}.issubset(afford.columns)
    seg = market_segment_mix(_frame())
    assert set(seg["segment"]) <= {"primary", "secondary"}
    assert int(seg["n"].sum()) == 5


def test_market_segment_mix_derives_from_market_and_skips_unknown() -> None:
    frame = pd.DataFrame({
        "market": ["primary", "secondary", None],
        "price_per_m2": [10_000.0, 12_000.0, 9_000.0],
    })
    seg = market_segment_mix(frame)
    assert set(seg["segment"]) == {"primary", "secondary"}
    assert int(seg["n"].sum()) == 2
