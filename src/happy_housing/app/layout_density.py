"""Streamlit layout density helpers (minimal padding only)."""

from __future__ import annotations

import streamlit as st

# Streamlit reserves a large top pad under the header / top nav; TOML cannot
# shrink it. Keep this CSS limited to padding-top — see no-custom-css.mdc.
_DENSE_TOP_CSS = """
<style>
section.stMain .block-container,
div.stMainBlockContainer,
.stAppViewBlockContainer {
  padding-top: 1rem;
}
</style>
"""


def apply_dense_top_layout() -> None:
    """Tighten main-area top padding under top page navigation."""
    st.markdown(_DENSE_TOP_CSS, unsafe_allow_html=True)
