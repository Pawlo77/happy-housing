"""Tests for shared methodology docs."""

from happy_housing.app.methodology_docs import (
    chat_methodology_blurb,
    format_methodology_note,
    model_tab_methodology_markdown,
)


def test_methodology_uses_commute_label() -> None:
    note = format_methodology_note("Centralny Port Komunikacyjny")
    assert "Centralny Port Komunikacyjny" in note
    assert "Rondo Daszyńskiego" not in note


def test_model_tab_contains_mermaid() -> None:
    md = model_tab_methodology_markdown(commute_label="Test Hub", photo_cv_enabled=True)
    assert "```mermaid" in md
    assert "Test Hub" in md
    assert "Photo CV" in md


def test_chat_blurb_mentions_otodom_labels() -> None:
    blurb = chat_methodology_blurb("Somewhere")
    assert "Otodom" in blurb
    assert "Somewhere" in blurb
    assert "recommender" in blurb.lower() or "rule" in blurb.lower()
