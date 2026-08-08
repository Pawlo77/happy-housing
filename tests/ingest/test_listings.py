"""Tests for listing ingest and schema validation."""

from pathlib import Path

import pytest

from happy_housing.ingest import LISTING_REQUIRED_COLUMNS, load_listings_json


def test_load_listings_fixture() -> None:
    """Committed listings fixture loads with required columns and price_per_m2."""
    path = Path("data/fixtures/listings.json")
    frame = load_listings_json(path)
    assert len(frame) >= 10
    for col in LISTING_REQUIRED_COLUMNS:
        assert col in frame.columns
    assert (frame["price_per_m2"] > 0).all()


def test_load_listings_rejects_empty(tmp_path: Path) -> None:
    """Empty JSON array is rejected."""
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_listings_json(path)
