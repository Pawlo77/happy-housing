"""Tests for compact LM listing-attribute extraction."""

from typing import Any

import pandas as pd

from happy_housing.app.attr_extract import (
    EXTRACT_FIELDS,
    ListingAttrExtract,
    build_extract_user_prompt,
    enrich_frame_listing_attrs,
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


def test_merge_without_extract_preserves_prior_desc_fields() -> None:
    structured = {
        "estate": "FLAT",
        "has_basement": 1,
        "ownership": "full",
        "plumbing_replaced": 1,
        "wiring_replaced": 0,
        "year_renovated": 2018,
    }
    merged = merge_structured_and_extract(structured, None)
    assert merged["plumbing_replaced"] == 1
    assert merged["wiring_replaced"] == 0
    assert merged["year_renovated"] == 2018
    assert merged["ownership"] == "full"
    assert merged["has_basement"] == 1


def test_row_needs_attr_llm_skips_when_fingerprint_matches() -> None:
    from happy_housing.app.attr_extract import (
        description_fingerprint,
        row_needs_attr_llm,
    )

    desc = "Wymieniona instalacja wodna i elektryczna, piwnica."
    fp = description_fingerprint(desc[:900])
    row = {
        "description": desc,
        "ownership": "full",
        "has_basement": 1,
        "has_outdoor_parking": 0,
        "is_attic": 0,
        "object_type": "flat",
        "plumbing_replaced": None,
        "wiring_replaced": None,
        "year_renovated": None,
        "attr_extract_fp": fp,
    }
    cols = set(row) | {
        "plumbing_replaced",
        "wiring_replaced",
        "year_renovated",
        "attr_extract_fp",
    }
    assert row_needs_attr_llm(row, cols, desc_fingerprint=fp) is False


def test_row_needs_attr_llm_when_ownership_missing() -> None:
    from happy_housing.app.attr_extract import row_needs_attr_llm

    row = {
        "description": "Mieszkanie z balkonem",
        "has_basement": 0,
        "has_outdoor_parking": 0,
        "is_attic": 0,
        "object_type": "flat",
    }
    assert row_needs_attr_llm(row, set(row), desc_fingerprint="abc") is True


def test_enrich_frame_without_llm_keeps_plumbing() -> None:
    import pandas as pd

    from happy_housing.app.attr_extract import enrich_frame_listing_attrs

    frame = pd.DataFrame({
        "estate": ["FLAT"],
        "description": ["desc"],
        "has_basement": [1],
        "has_outdoor_parking": [0],
        "is_attic": [0],
        "ownership": ["full"],
        "plumbing_replaced": [1],
        "wiring_replaced": [0],
        "year_renovated": [2019],
        "attr_extract_fp": ["already"],
    })
    out = enrich_frame_listing_attrs(frame, use_llm=False)
    assert int(out.iloc[0]["plumbing_replaced"]) == 1
    assert int(out.iloc[0]["wiring_replaced"]) == 0


def test_enrich_frame_parallel_lm_workers(monkeypatch: Any) -> None:
    """Parallel workers fill the same cache as sequential calls."""
    calls: list[str] = []

    def fake_extract(desc: str, **_k: object) -> ListingAttrExtract:
        calls.append(desc)
        return ListingAttrExtract(
            object_type="flat",
            plumbing_replaced=1,
            wiring_replaced=0,
        )

    monkeypatch.setattr(
        "happy_housing.app.attr_extract.extract_attrs_from_description",
        fake_extract,
    )
    frame = pd.DataFrame({
        "description": ["same text", "same text", "other text"],
        "has_basement": [None, None, None],
        "has_outdoor_parking": [None, None, None],
        "is_attic": [None, None, None],
        "ownership": [None, None, None],
        "needs_renovation": [None, None, None],
        "year_built": [None, None, None],
        "year_renovated": [None, None, None],
        "plumbing_replaced": [None, None, None],
        "wiring_replaced": [None, None, None],
        "object_type": [None, None, None],
        "floor_no": [None, None, None],
        "has_garden": [None, None, None],
        "has_garage": [None, None, None],
        "has_balcony": [None, None, None],
    })
    out = enrich_frame_listing_attrs(frame, lm_workers=2)
    assert len(out) == 3
    assert sorted(calls) == ["other text", "same text"]
