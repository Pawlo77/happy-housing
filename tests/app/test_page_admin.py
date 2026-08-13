"""Tests for Admin section navigation helpers."""

import pytest

from happy_housing.app.page_admin import ADMIN_SECTIONS, panel_for_section


def test_admin_sections_cover_planned_ia() -> None:
    """Top-nav pills match the short-panel IA (Ops hub first)."""
    assert ADMIN_SECTIONS == (
        "Ops",
        "Commute",
        "Sources",
        "Explore",
        "Price model",
        "Recommendations",
        "Offer AI",
        "Assistant",
        "Extract",
        "LMS metrics",
        "Stars & News",
        "Stats",
    )
    assert len(ADMIN_SECTIONS) == len(set(ADMIN_SECTIONS))


def test_panel_for_section_dispatches_every_label() -> None:
    for label in ADMIN_SECTIONS:
        fn = panel_for_section(label)
        assert callable(fn)
        assert fn.__name__.startswith("_panel_")


def test_panel_for_section_rejects_unknown() -> None:
    with pytest.raises(AssertionError, match="unreachable"):
        panel_for_section("Not a real section")
