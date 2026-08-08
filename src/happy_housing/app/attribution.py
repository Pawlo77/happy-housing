"""Shared open-data attribution footer for Streamlit pages."""

from __future__ import annotations

import streamlit as st

ATTRIBUTION_LINE = (
    "Data: Otodom · UM Warszawa RCN · OpenStreetMap/OSMnx · ZTM GTFS (mkuran.pl) · "
    "GIOŚ · GUNB · Chopin OOU — free/public sources; verify on each provider site."
)
"""Single-line attribution shown under Explore / Stats / Admin / Home."""


def render_attribution_footer() -> None:
    """Render a quiet page footer with source attribution."""
    st.divider()
    st.caption(ATTRIBUTION_LINE)
