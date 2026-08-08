"""Tests for Home About business copy."""

from __future__ import annotations

from happy_housing.app.home_about import ABOUT_SECTIONS, about_section_titles


def test_about_sections_cover_business_basics() -> None:
    titles = about_section_titles()
    assert titles[0] == "What this is"
    assert "Who it is for" in titles
    assert "What we model" in titles
    assert "What it is not" in titles
    assert "Where to go next" not in titles
    assert "Assistant" in ABOUT_SECTIONS[-1].body


def test_about_copy_stays_plain_and_nonempty() -> None:
    banned = (
        "leverage",
        "seamless",
        "empower",
        "cutting-edge",
        "revolutionize",
        "synergy",
        "holistic",
    )
    for section in ABOUT_SECTIONS:
        assert section.title.strip()
        assert len(section.body.strip()) > 40
        low = section.body.lower()
        for word in banned:
            assert word not in low, f"{section.title} contains '{word}'"
        assert "lightgbm" not in low
        assert "shap" not in low
