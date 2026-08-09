"""Otodom description hints for street-facing vs quiet / courtyard side."""

from __future__ import annotations

import re

import pandas as pd

_STREET_FACING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"od\s+ulicy",
        r"od\s+strony\s+ulicy",
        r"frontem\s+do\s+ulicy",
        r"wystaw[aą]\s+na\s+ulic",
        r"okna\s+na\s+ulic",
        r"na\s+ulic[eę]",
    )
)
"""Polish / Latin phrases suggesting the unit faces a street."""

_QUIET_SIDE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"od\s+podwórz[ay]",
        r"od\s+podwórk",
        r"od\s+ogrodu",
        r"cicha\s+strona",
        r"od\s+strony\s+spokojnej",
        r"courtyard",
        r"ciche\s+podwór",
        r"okna\s+na\s+podwór",
        r"od\s+strony\s+ogrodu",
    )
)
"""Phrases suggesting courtyard / garden / quiet side exposure."""


def _combine_description(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("description", "short_description"):
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def text_matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Return True when any compiled pattern matches ``text``."""
    if not text:
        return False
    return any(pat.search(text) is not None for pat in patterns)


def street_facing_hint_from_text(text: str) -> int:
    """1 when description suggests street frontage, else 0."""
    return int(text_matches_any(text, _STREET_FACING_PATTERNS))


def quiet_side_hint_from_text(text: str) -> int:
    """1 when description suggests courtyard / quiet side, else 0."""
    return int(text_matches_any(text, _QUIET_SIDE_PATTERNS))


def attach_street_frontage_hints(listings: pd.DataFrame) -> pd.DataFrame:
    """Add ``street_facing_hint`` and ``quiet_side_hint`` from listing text.

    Both flags are 0 when no matching phrase is found (weak prior for the
    model; UI should treat 0/0 as unknown unless description is empty).
    """
    out = listings.copy()
    if out.empty:
        out["street_facing_hint"] = pd.Series(dtype=int)
        out["quiet_side_hint"] = pd.Series(dtype=int)
        return out

    texts = out.apply(_combine_description, axis=1)
    out["street_facing_hint"] = texts.map(street_facing_hint_from_text).astype(int)
    out["quiet_side_hint"] = texts.map(quiet_side_hint_from_text).astype(int)
    return out
