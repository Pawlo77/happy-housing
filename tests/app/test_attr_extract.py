"""Tests for compact LM listing-attribute extraction."""

from __future__ import annotations

from happy_housing.app.attr_extract import (
    EXTRACT_FIELDS,
    ListingAttrExtract,
    build_extract_user_prompt,
    merge_structured_and_extract,
    parse_extract_line,
)


def test_parse_extract_line_numbers_and_n() -> None:
    line = "flat,3,0,1,0,1,1,1,full,0,1972,2019,1,N"
    parsed = parse_extract_line(line)
    assert parsed.object_type == "flat"
    assert parsed.floor_no == 3
    assert parsed.is_attic == 0
    assert parsed.has_garden == 1
    assert parsed.has_garage == 0
    assert parsed.has_outdoor_parking == 1
    assert parsed.has_basement == 1
    assert parsed.has_balcony == 1
    assert parsed.ownership == "full"
    assert parsed.needs_renovation == 0
    assert parsed.year_built == 1972
    assert parsed.year_renovated == 2019
    assert parsed.plumbing_replaced == 1
    assert parsed.wiring_replaced is None
    assert len(EXTRACT_FIELDS) == 14


def test_parse_extract_line_rejects_garbage() -> None:
    assert parse_extract_line("hello") is None
    assert parse_extract_line("") is None


def test_build_extract_user_prompt_truncates() -> None:
    text = "x" * 5000
    prompt = build_extract_user_prompt(text, max_chars=100)
    assert len(prompt) < 200
    assert "xxxx" in prompt or "x" * 50 in prompt


def test_merge_prefers_structured_over_extract() -> None:
    structured = {
        "estate": "FLAT",
        "floor_no": 5,
        "has_garage": 1,
        "has_outdoor_parking": 0,
        "has_basement": 0,
        "has_balcony": 1,
        "has_garden": 0,
        "is_attic": 0,
        "building_ownership": "usufruct",
        "construction_status": "to_renovation",
        "year_built": 1960.0,
    }
    extract = ListingAttrExtract(
        object_type="house",
        floor_no=1,
        has_garage=0,
        plumbing_replaced=1,
        wiring_replaced=0,
        year_renovated=2015,
    )
    merged = merge_structured_and_extract(structured, extract)
    assert merged["object_type"] == "flat"
    assert merged["floor_no"] == 5
    assert merged["has_garage"] == 1
    assert merged["ownership"] == "usufruct"
    assert merged["needs_renovation"] == 1
    assert merged["plumbing_replaced"] == 1
    assert merged["wiring_replaced"] == 0
    assert merged["year_renovated"] == 2015
