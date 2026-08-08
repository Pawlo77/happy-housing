"""Tests for Explore board → Offer navigation helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.explore_nav import listing_id_from_board_selection


def test_listing_id_from_board_selection_reads_first_row() -> None:
    frame = pd.DataFrame({
        "listing_id": ["OTO-1", "OTO-2"],
        "district": ["wola", "mokotow"],
    })
    selection = {"selection": {"rows": [1]}}
    assert listing_id_from_board_selection(frame, selection) == "OTO-2"


def test_listing_id_from_board_selection_empty() -> None:
    frame = pd.DataFrame({"listing_id": ["OTO-1"]})
    assert listing_id_from_board_selection(frame, None) is None
    assert listing_id_from_board_selection(frame, {"selection": {"rows": []}}) is None
