"""Tests for Otodom street-facing / quiet-side description hints."""

from __future__ import annotations

import pandas as pd

from happy_housing.gis.street_frontage import (
    attach_street_frontage_hints,
    quiet_side_hint_from_text,
    street_facing_hint_from_text,
)


def test_street_facing_and_quiet_phrases() -> None:
    assert street_facing_hint_from_text("Mieszkanie od ulicy, jasne.") == 1
    assert street_facing_hint_from_text("Okna na ulicę główną.") == 1
    assert quiet_side_hint_from_text("Pokój od podwórza, cicha strona.") == 1
    assert quiet_side_hint_from_text("Courtyard view.") == 1
    assert street_facing_hint_from_text("Ładne mieszkanie w centrum.") == 0
    assert quiet_side_hint_from_text("Ładne mieszkanie w centrum.") == 0


def test_attach_street_frontage_hints_on_frame() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a", "b", "c"],
        "description": [
            "Apartament frontem do ulicy.",
            "Sypialnia od podwórka.",
            "Brak szczegółów.",
        ],
        "short_description": ["", "cicha strona budynku", ""],
    })
    out = attach_street_frontage_hints(frame)
    assert int(out.loc[0, "street_facing_hint"]) == 1
    assert int(out.loc[0, "quiet_side_hint"]) == 0
    assert int(out.loc[1, "quiet_side_hint"]) == 1
    assert int(out.loc[2, "street_facing_hint"]) == 0
    assert int(out.loc[2, "quiet_side_hint"]) == 0
