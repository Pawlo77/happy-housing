"""Board table text search and selection-safe display prep."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.board_table import (
    filter_dataframe_text,
    prepare_board_display,
)


def test_filter_dataframe_text_empty_query_passthrough() -> None:
    frame = pd.DataFrame({"district": ["mokotów", "wola"], "listing_id": ["A", "B"]})
    out = filter_dataframe_text(frame, "  ")
    assert list(out["listing_id"]) == ["A", "B"]


def test_filter_dataframe_text_case_insensitive_substring() -> None:
    frame = pd.DataFrame({
        "district": ["Białołęka", "Mokotów"],
        "listing_id": ["RCN-1", "OTO-2"],
        "title": ["None", "Sunny flat"],
        "price_pln": [100.0, 200.0],
    })
    by_district = filter_dataframe_text(frame, "biało")
    assert list(by_district["listing_id"]) == ["RCN-1"]
    by_id = filter_dataframe_text(frame, "oto-2")
    assert list(by_id["listing_id"]) == ["OTO-2"]
    by_title = filter_dataframe_text(frame, "sunny")
    assert list(by_title["listing_id"]) == ["OTO-2"]
    by_price = filter_dataframe_text(frame, "200")
    assert list(by_price["listing_id"]) == ["OTO-2"]


def test_filter_dataframe_text_respects_column_subset() -> None:
    frame = pd.DataFrame({
        "district": ["wola", "wola"],
        "listing_id": ["HIT", "MISS"],
        "title": ["keep me", "HIT in title only"],
    })
    out = filter_dataframe_text(frame, "HIT", columns=("listing_id",))
    assert list(out["listing_id"]) == ["HIT"]


def test_prepare_board_display_marks_seen_and_rounds_delta() -> None:
    frame = pd.DataFrame({
        "seen": ["new", "opened", "seen x2"],
        "confidence": [0.7891, 0.5, 0.1],
        "delta_price_per_m2": [-1200.4, 50.0, 0.0],
        "listing_id": ["A", "B", "C"],
    })
    out = prepare_board_display(frame)
    assert list(out["seen"]) == ["● new", "○ opened", "◌ seen x2"]
    assert float(out.loc[0, "confidence"]) == 0.789
    assert float(out.loc[0, "delta_price_per_m2"]) == -1200.0
