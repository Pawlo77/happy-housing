"""Tests for Explore board → Offer navigation helpers."""

from unittest.mock import MagicMock, patch

import pandas as pd

from happy_housing.app.explore_nav import (
    listing_id_from_board_selection,
    listing_id_from_map_selection,
    open_explore_offer,
    prepare_offer_select_state,
    register_explore_page,
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


def test_listing_id_from_map_selection_reads_listings_layer() -> None:
    event = {
        "selection": {
            "objects": {"listings": [{"listing_id": "OTO-9", "tooltip_name": "OTO-9"}]}
        }
    }
    assert listing_id_from_map_selection(event) == "OTO-9"


def test_listing_id_from_map_selection_empty() -> None:
    assert listing_id_from_map_selection(None) is None
    assert listing_id_from_map_selection({"selection": {"objects": {}}}) is None


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
    assert "Chat" in EXPLORE_SECTIONS
    assert EXPLORE_SECTIONS[-1] == "Prefs"
    assert "News" in EXPLORE_SECTIONS


def test_normalize_listing_id() -> None:
    from happy_housing.app.explore_nav import normalize_listing_id

    assert normalize_listing_id("OTO-1") == "OTO-1"
    assert normalize_listing_id("  RCN-9  ") == "RCN-9"
    assert normalize_listing_id("") is None
    assert normalize_listing_id("   ") is None
    assert normalize_listing_id(None) is None


def test_explore_offer_path_encodes_listing_id() -> None:
    from happy_housing.app.explore_nav import explore_offer_path

    assert explore_offer_path("OTO-1") == "/explore?listing_id=OTO-1"
    assert explore_offer_path("A B") == "/explore?listing_id=A+B"


def test_apply_offer_from_query_params_hydrates_new_deep_link() -> None:
    from happy_housing.app.explore_nav import apply_offer_from_query_params

    session: dict[str, object] = {}
    mock_st = MagicMock()
    mock_st.session_state = session
    mock_st.query_params = {"listing_id": "OTO-42"}
    with patch("happy_housing.app.explore_nav.st", mock_st):
        assert apply_offer_from_query_params() == "OTO-42"
    assert session["explore_offer_id"] == "OTO-42"
    assert session["explore_offer_force"] is True
    assert session["explore_section_pending"] == "Offer"


def test_apply_offer_from_query_params_skips_repeat_force() -> None:
    from happy_housing.app.explore_nav import apply_offer_from_query_params

    session: dict[str, object] = {"explore_offer_id": "OTO-42"}
    mock_st = MagicMock()
    mock_st.session_state = session
    mock_st.query_params = {"listing_id": "OTO-42"}
    with patch("happy_housing.app.explore_nav.st", mock_st):
        assert apply_offer_from_query_params() == "OTO-42"
    assert "explore_offer_force" not in session
    assert "explore_section_pending" not in session


def test_sync_offer_query_param_writes_listing_id() -> None:
    from happy_housing.app.explore_nav import sync_offer_query_param

    mock_params = MagicMock()
    mock_st = MagicMock()
    mock_st.query_params = mock_params
    with patch("happy_housing.app.explore_nav.st", mock_st):
        sync_offer_query_param("OTO-7")
    mock_params.__setitem__.assert_called_with("listing_id", "OTO-7")


def test_open_explore_offer_sets_session_and_switches_page() -> None:
    session: dict[str, object] = {}
    mock_st = MagicMock()
    mock_st.session_state = session
    explore_page = object()
    register_explore_page(explore_page)
    with patch("happy_housing.app.explore_nav.st", mock_st):
        open_explore_offer("RCN-605")
    assert session["explore_offer_id"] == "RCN-605"
    assert session["explore_offer_force"] is True
    assert session["explore_section_pending"] == "Offer"
    assert "explore_section" not in session
    mock_st.switch_page.assert_called_once_with(
        explore_page,
        query_params={"listing_id": "RCN-605"},
    )


def test_open_explore_offer_requires_registered_page() -> None:
    register_explore_page(None)
    session: dict[str, object] = {}
    mock_st = MagicMock()
    mock_st.session_state = session
    with patch("happy_housing.app.explore_nav.st", mock_st):
        try:
            open_explore_offer("OTO-1")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "not registered" in str(exc)


def test_focus_explore_offer_sets_session_and_query() -> None:
    from happy_housing.app.explore_nav import focus_explore_offer

    session: dict[str, object] = {}
    mock_params = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = session
    mock_st.query_params = mock_params
    with patch("happy_housing.app.explore_nav.st", mock_st):
        focus_explore_offer("OTO-3")
    assert session["explore_offer_id"] == "OTO-3"
    assert session["explore_offer_force"] is True
    assert session["explore_section_pending"] == "Offer"
    mock_params.__setitem__.assert_called_with("listing_id", "OTO-3")


def test_apply_explore_section_pending_moves_into_section_key() -> None:
    from happy_housing.app.explore_nav import (
        apply_explore_section_pending,
        request_explore_section,
    )

    session: dict[str, object] = {}
    mock_st = MagicMock()
    mock_st.session_state = session
    with patch("happy_housing.app.explore_nav.st", mock_st):
        request_explore_section("Offer")
        assert session["explore_section_pending"] == "Offer"
        apply_explore_section_pending()
    assert session.get("explore_section") == "Offer"
    assert "explore_section_pending" not in session
