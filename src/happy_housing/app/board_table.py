"""Explore Board table helpers: text search and selection-safe display prep.

Streamlit row selection does not work with Pandas ``Styler``, so visibility
comes from markers + ``column_config`` (progress / signed numbers), not CSS.
"""

from collections.abc import Sequence

import pandas as pd

_BOARD_SEARCH_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "title",
    "district",
    "source",
    "market",
    "seen",
    "url",
    "area_m2",
    "rooms",
    "year_built",
    "price_pln",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "commute_min",
    "listed_at",
)
"""Default columns scanned by Board text search when present."""


def filter_dataframe_text(
    frame: pd.DataFrame,
    query: str,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Keep rows where ``query`` appears in any scanned column (casefold)."""
    needle = str(query or "").strip().casefold()
    if not needle or frame.empty:
        return frame.reset_index(drop=True)
    if columns is None:
        cols = [c for c in frame.columns]
    else:
        cols = [c for c in columns if c in frame.columns]
    if not cols:
        return frame.iloc[0:0].reset_index(drop=True)
    mask = pd.Series(False, index=frame.index)
    for col in cols:
        series = frame[col].fillna("").astype(str).str.casefold()
        mask |= series.str.contains(needle, regex=False, na=False)
    return frame.loc[mask].reset_index(drop=True)


def _format_seen_label(label: object) -> str:
    text = str(label or "").strip()
    if text == "new":
        return "● new"
    if text == "opened":
        return "○ opened"
    if text.startswith("seen"):
        return f"◌ {text}"
    return text


def prepare_board_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Markers + rounded numerics for Board (keeps selection-compatible plain data)."""
    if frame.empty:
        return frame.reset_index(drop=True)
    out = frame.copy()
    if "seen" in out.columns:
        out["seen"] = out["seen"].map(_format_seen_label)
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").round(3)
    if "delta_price_per_m2" in out.columns:
        out["delta_price_per_m2"] = pd.to_numeric(
            out["delta_price_per_m2"], errors="coerce"
        ).round(0)
    for col in ("price_pln", "price_per_m2", "model_price_per_m2"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
    return out.reset_index(drop=True)


def board_search_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Intersect default search columns with the frame."""
    return tuple(c for c in _BOARD_SEARCH_COLUMNS if c in frame.columns)
