"""Tests for listing ingest and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from happy_housing.ingest import LISTING_REQUIRED_COLUMNS, load_listings_json
from happy_housing.ingest.listing_common import merge_listing_frames


def test_load_listings_fixture() -> None:
    """Committed listings fixture loads with required columns and price_per_m2."""
    path = Path("data/fixtures/listings.json")
    frame = load_listings_json(path)
    assert len(frame) >= 10
    for col in LISTING_REQUIRED_COLUMNS:
        assert col in frame.columns
    assert (frame["price_per_m2"] > 0).all()


def test_merge_listing_frames_fresh_wins() -> None:
    """Fresh rows replace existing rows with the same listing_id."""
    old = pd.DataFrame([
        {"listing_id": "A", "price_pln": 100.0},
        {"listing_id": "B", "price_pln": 200.0},
    ])
    new = pd.DataFrame([
        {"listing_id": "A", "price_pln": 150.0},
        {"listing_id": "C", "price_pln": 300.0},
    ])
    merged = merge_listing_frames(old, new)
    assert set(merged["listing_id"]) == {"A", "B", "C"}
    assert float(merged.loc[merged["listing_id"] == "A", "price_pln"].iloc[0]) == 150.0


def test_load_listings_rejects_empty(tmp_path: Path) -> None:
    """Empty JSON array is rejected."""
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_listings_json(path)
