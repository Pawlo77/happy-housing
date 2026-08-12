"""Session-persisted Admin banners (survive ``st.rerun`` after long jobs)."""

from __future__ import annotations

import logging
from typing import Literal

import streamlit as st

logger = logging.getLogger(__name__)

AdminFlashKind = Literal["success", "info", "warning", "error"]
"""Streamlit banner kinds used for Admin flash messages."""

_ADMIN_FLASH_KEY: str = "admin_flash_messages"
"""Session key holding a list of ``(kind, message)`` until consumed."""


def set_admin_flash(kind: AdminFlashKind, message: str) -> None:
    """Queue a banner for the next Admin render after ``st.rerun``."""
    text = str(message or "").strip()
    if not text:
        return
    queued = st.session_state.get(_ADMIN_FLASH_KEY)
    if not isinstance(queued, list):
        queued = []
    queued.append((str(kind), text))
    st.session_state[_ADMIN_FLASH_KEY] = queued
    logger.debug("Admin flash queued kind=%s (n=%s)", kind, len(queued))


def pop_admin_flash() -> None:
    """Render and clear any queued Admin flash banners."""
    raw = st.session_state.pop(_ADMIN_FLASH_KEY, None)
    if not isinstance(raw, list):
        return
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        kind, message = str(item[0]), str(item[1])
        if not message:
            continue
        match kind:
            case "success":
                st.success(message)
            case "info":
                st.info(message)
            case "warning":
                st.warning(message)
            case "error":
                st.error(message)
            case _:
                st.info(message)
