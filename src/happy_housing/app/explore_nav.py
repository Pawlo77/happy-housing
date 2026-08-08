"""Navigation helpers for Explore Map / Offer / Board."""

from __future__ import annotations

from typing import Any

import pandas as pd

EXPLORE_SECTIONS: tuple[str, ...] = (
    "Map",
    "Spatial",
    "Offer",
    "Board",
    "News",
    "Prefs",
)
"""Top-level Explore section labels."""

OFFER_SELECT_KEY: str = "explore_offer_select"
"""Session key for the Offer-tab listing selectbox."""


def listing_id_from_map_selection(event: object | None) -> str | None:
    """Resolve ``listing_id`` from a ``st.pydeck_chart`` selection event."""
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return None
    objects = getattr(selection, "objects", None)
    if objects is None and isinstance(selection, dict):
        objects = selection.get("objects")
    if not isinstance(objects, dict):
        return None
    # Prefer the listings scatter layer; fall back to any object with listing_id.
    candidates: list[Any] = []
    if "listings" in objects:
        candidates.extend(list(objects.get("listings") or []))
    else:
        for rows in objects.values():
            candidates.extend(list(rows or []))
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        lid = obj.get("listing_id") or obj.get("tooltip_name")
        if lid:
            return str(lid)
    return None


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


def label_for_listing_id(options: dict[str, str], listing_id: str) -> str | None:
    """Return the selectbox label whose value equals ``listing_id``."""
    for label, lid in options.items():
        if lid == listing_id:
            return label
    return None


def prepare_offer_select_state(
    *,
    options: dict[str, str],
    preferred_id: str | None,
    force: bool,
    current_label: str | None,
) -> str | None:
    """Resolve which selectbox label should be selected.

    When ``force`` is True (Board/News jump) or the widget has no value yet,
    prefer ``preferred_id``. Otherwise keep the user's current choice.
    """
    if not options:
        return None
    preferred_label = (
        label_for_listing_id(options, preferred_id) if preferred_id else None
    )
    if force and preferred_label is not None:
        return preferred_label
    if current_label in options:
        return current_label
    if preferred_label is not None:
        return preferred_label
    return next(iter(options))
