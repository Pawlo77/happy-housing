"""Parse and normalize Otodom / listing ``listed_at`` timestamps."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

_FALLBACK_LISTED_AT: str = "2024-01-01T00:00:00+00:00"
"""Neutral fallback when no usable listing date is present."""


def parse_listed_at(value: object) -> str | None:
    """Parse Otodom / ISO / epoch listing dates to UTC ISO-8601 strings.

    Returns:
        Normalized UTC timestamp string, or ``None`` when unparsable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()
    if isinstance(value, (int, float)):
        # Otodom sometimes emits ms epoch.
        number = float(value)
        if number != number:
            return None
        if number > 1e12:
            number /= 1000.0
        try:
            return datetime.fromtimestamp(number, tz=UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    # Strip trailing Z → +00:00 for fromisoformat.
    cleaned = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()
    except ValueError:
        pass
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return str(pd.Timestamp(parsed).isoformat())


def coalesce_listed_at(*candidates: object, fallback: str = _FALLBACK_LISTED_AT) -> str:
    """Return the first parseable listed_at among ``candidates``, else fallback."""
    for value in candidates:
        parsed = parse_listed_at(value)
        if parsed is not None:
            return parsed
    return fallback


def listed_at_series(values: Any) -> pd.Series:
    """Vectorized parse preserving index; unparsable → NaT then fillna fallback."""
    series = pd.Series(values)
    parsed = series.map(parse_listed_at)
    return parsed.fillna(_FALLBACK_LISTED_AT)
