"""Navigation helpers for Explore Map / Offer / Board."""

from __future__ import annotations

from typing import Any

import pandas as pd

EXPLORE_SECTIONS: tuple[str, ...] = ("Map", "Offer", "Board", "News")
"""Top-level Explore section labels."""


def listing_id_from_board_selection(
    frame: pd.DataFrame,
    event: object | None,
) -> str | None:
    """Resolve ``listing_id`` from a Streamlit dataframe selection event."""
    if frame.empty or event is None:
        return None
    payload: dict[str, Any]
    if isinstance(event, dict):
        payload = event
    else:
        selection = getattr(event, "selection", None)
        if selection is None:
            return None
        rows = getattr(selection, "rows", None)
        if rows is None and isinstance(selection, dict):
            rows = selection.get("rows")
        payload = {"selection": {"rows": list(rows or [])}}
    rows = payload.get("selection", {}).get("rows") or []
    if not rows:
        return None
    index = int(rows[0])
    if index < 0 or index >= len(frame):
        return None
    return str(frame.iloc[index]["listing_id"])
