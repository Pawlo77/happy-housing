"""Navigation helpers for Explore Map / Offer / Board."""

from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

EXPLORE_SECTIONS: tuple[str, ...] = (
    "Map",
    "Spatial",
    "Offer",
    "Board",
    "News",
    "Chat",
    "Prefs",
)
"""Top-level Explore section labels."""

EXPLORE_PAGE_PATH: str = "explore"
"""``st.navigation`` url_path for the Explore page."""

OFFER_SELECT_KEY: str = "explore_offer_select"
"""Session key for the Offer-tab listing selectbox."""

OFFER_QUERY_KEY: str = "listing_id"
"""Query-param key for shareable / new-tab Explore offer URLs."""

MAP_PIN_SELECTION_KEY: str = "explore_map_selected_id"
"""Session key for the listing last clicked on the Explore Map pydeck."""

EXPLORE_SECTION_KEY: str = "explore_section"
"""Session key bound to the Explore ``st.tabs`` widget."""

EXPLORE_SECTION_PENDING_KEY: str = "explore_section_pending"
"""Queued section switch — applied before ``st.tabs`` is instantiated."""

_explore_page: object | None = None
"""Registered ``st.Page`` for Explore — required by ``st.switch_page``."""


def register_explore_page(page: object | None) -> None:
    """Store the Explore ``st.Page`` built in ``st.navigation``."""
    global _explore_page
    _explore_page = page


def get_explore_page() -> object | None:
    """Return the registered Explore page, if any."""
    return _explore_page


def normalize_listing_id(raw: object | None) -> str | None:
    """Strip a listing id; empty / missing → ``None``."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]
    text = str(raw).strip()
    return text or None


def explore_offer_path(listing_id: str) -> str:
    """Relative Explore URL that opens ``listing_id`` on the Offer tab."""
    lid = normalize_listing_id(listing_id)
    if lid is None:
        raise ValueError("listing_id must be non-empty")
    return f"/{EXPLORE_PAGE_PATH}?{urlencode({OFFER_QUERY_KEY: lid})}"


def sync_offer_query_param(listing_id: str | None) -> None:
    """Keep the browser URL query param aligned with the focused offer."""
    lid = normalize_listing_id(listing_id)
    if lid is None:
        if OFFER_QUERY_KEY in st.query_params:
            del st.query_params[OFFER_QUERY_KEY]
        return
    st.query_params[OFFER_QUERY_KEY] = lid


def apply_offer_from_query_params() -> str | None:
    """Hydrate Offer focus from ``?listing_id=``; force Offer only on change."""
    lid = normalize_listing_id(st.query_params.get(OFFER_QUERY_KEY))
    if lid is None:
        return None
    current = normalize_listing_id(st.session_state.get("explore_offer_id"))
    if current == lid:
        return lid
    st.session_state["explore_offer_id"] = lid
    st.session_state["explore_offer_force"] = True
    request_explore_section("Offer")
    return lid


def request_explore_section(section: str) -> None:
    """Queue an Explore tab switch (safe after ``st.tabs`` already ran)."""
    if section not in EXPLORE_SECTIONS:
        return
    st.session_state[EXPLORE_SECTION_PENDING_KEY] = section


def apply_explore_section_pending() -> None:
    """Flush a queued tab switch into ``explore_section`` before the tabs widget."""
    pending = st.session_state.pop(EXPLORE_SECTION_PENDING_KEY, None)
    if pending is None:
        return
    if pending in EXPLORE_SECTIONS:
        st.session_state[EXPLORE_SECTION_KEY] = pending


def open_explore_offer(listing_id: str) -> None:
    """Jump to Explore → Offer for ``listing_id`` (session keys + switch_page)."""
    lid = normalize_listing_id(listing_id)
    if lid is None:
        raise ValueError("listing_id must be non-empty")
    st.session_state["explore_offer_id"] = lid
    st.session_state["explore_offer_force"] = True
    request_explore_section("Offer")
    sync_offer_query_param(lid)
    page = _explore_page
    if page is None:
        msg = "Explore page not registered; call register_explore_page from main()"
        raise RuntimeError(msg)
    st.switch_page(page, query_params={OFFER_QUERY_KEY: lid})


def focus_explore_offer(listing_id: str) -> None:
    """Focus an offer on the current Explore page (session + URL + Offer tab)."""
    lid = normalize_listing_id(listing_id)
    if lid is None:
        raise ValueError("listing_id must be non-empty")
    st.session_state["explore_offer_id"] = lid
    st.session_state["explore_offer_force"] = True
    request_explore_section("Offer")
    sync_offer_query_param(lid)


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


def resolve_map_pin_selection(
    event: object | None,
    session: Any | None = None,
) -> str | None:
    """Persist map pin click in session; clear when selection is empty.

    Args:
        event: ``st.pydeck_chart`` return value (or dict-shaped test double).
        session: Mutable mapping (defaults to ``st.session_state``).

    Returns:
        Selected ``listing_id``, or ``None`` when nothing is selected.
    """
    store = st.session_state if session is None else session
    clicked = listing_id_from_map_selection(event)
    if clicked is not None:
        store[MAP_PIN_SELECTION_KEY] = clicked
        return clicked
    store.pop(MAP_PIN_SELECTION_KEY, None)
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
