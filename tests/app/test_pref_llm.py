"""Tests for NL → UserPreferences JSON parse (no live LM Studio)."""

from __future__ import annotations

import pytest

from happy_housing.app.pref_llm import (
    extract_preferences_json,
    merge_preferences,
    preferences_system_prompt,
)
from happy_housing.app.preferences import UserPreferences


def test_extract_preferences_json_from_fenced_block() -> None:
    text = """
Here you go:
```json
{"enabled": true, "districts": ["Wola"], "max_commute_min": 25, "bogus": 1}
```
"""
    prefs = extract_preferences_json(text)
    assert prefs.enabled is True
    assert prefs.districts == ["Wola"]
    assert prefs.max_commute_min == 25.0


def test_extract_preferences_json_rejects_empty() -> None:
    with pytest.raises(ValueError, match="JSON"):
        extract_preferences_json("no json here")


def test_merge_preferences_keeps_nl_and_overlays_fields() -> None:
    base = UserPreferences(enabled=False, districts=["mokotow"], nl_text="old")
    overlay = UserPreferences(
        enabled=True,
        max_commute_min=20.0,
        require_underpriced=True,
    )
    merged = merge_preferences(base, overlay, nl_text="near metro")
    assert merged.enabled is True
    assert merged.districts == ["mokotow"]
    assert merged.max_commute_min == 20.0
    assert merged.require_underpriced is True
    assert merged.nl_text == "near metro"


def test_preferences_system_prompt_lists_fields() -> None:
    prompt = preferences_system_prompt()
    assert "max_commute_min" in prompt
    assert "JSON" in prompt
