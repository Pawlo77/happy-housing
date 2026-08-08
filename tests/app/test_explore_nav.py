"""Tests for Explore board → Offer navigation helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.explore_nav import (
    listing_id_from_board_selection,
    prepare_offer_select_state,
)


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


def test_prepare_offer_select_state_force_overrides_current() -> None:
    options = {"A — Flat": "A", "B — Loft": "B"}
    assert (
        prepare_offer_select_state(
            options=options,
            preferred_id="B",
            force=True,
            current_label="A — Flat",
        )
        == "B — Loft"
    )


def test_prepare_offer_select_state_keeps_user_choice_without_force() -> None:
    options = {"A — Flat": "A", "B — Loft": "B"}
    assert (
        prepare_offer_select_state(
            options=options,
            preferred_id="B",
            force=False,
            current_label="A — Flat",
        )
        == "A — Flat"
    )


def test_explore_sections_include_prefs() -> None:
    from happy_housing.app.explore_nav import EXPLORE_SECTIONS

    assert "Spatial" in EXPLORE_SECTIONS
    assert EXPLORE_SECTIONS[-1] == "Prefs"
    assert "News" in EXPLORE_SECTIONS
