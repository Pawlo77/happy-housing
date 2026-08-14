"""Tests for shared buyer-listing schema alignment."""

import pandas as pd

from happy_housing.ingest.listing_common import (
    BUYER_LISTING_SCHEMA_COLUMNS,
    align_buyer_listing_row,
    coalesce_listing_descriptions,
    extras_from_free_text,
    flags_from_extras_and_tags,
    prepare_listings_for_model,
)


def test_align_buyer_listing_row_adds_missing_schema_columns() -> None:
    row = align_buyer_listing_row({"listing_id": "OLX-1", "price_pln": 100.0})
    for col in BUYER_LISTING_SCHEMA_COLUMNS:
        assert col in row


def test_flags_from_description_tokens() -> None:
    extras = extras_from_free_text("Mieszkanie z balkonem i windą")
    flags = flags_from_extras_and_tags(
        extras=extras,
        tags=set(),
        floor_label=None,
        floor_no=2,
    )
    assert flags["has_balcony"] == 1
    assert flags["has_lift"] == 1


def test_coalesce_listing_descriptions_fills_from_short() -> None:
    frame = pd.DataFrame([
        {"listing_id": "A", "description": None, "short_description": "Nice flat"},
        {"listing_id": "B", "description": "Keep me", "short_description": "Other"},
    ])
    out = coalesce_listing_descriptions(frame)
    assert out.loc[0, "description"] == "Nice flat"
    assert out.loc[1, "description"] == "Keep me"


def test_prepare_listings_for_model_aligns_schema() -> None:
    frame = pd.DataFrame([{"listing_id": "OLX-1", "price_pln": 100.0}])
    out = prepare_listings_for_model(frame)
    assert out.loc[0, "listing_active"] == 1
    assert "heating" in out.columns


def test_align_preserves_existing_values() -> None:
    row = align_buyer_listing_row({"heating": "gazowe", "has_garden": 1})
    assert row["heating"] == "gazowe"
    assert row["has_garden"] == 1
