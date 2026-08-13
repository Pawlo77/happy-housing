"""Streamlit layout density helpers (minimal padding only)."""

import streamlit as st

# Streamlit reserves a large top pad under the header / top nav; TOML cannot
# shrink it. Keep this CSS limited to padding-top — see no-custom-css.mdc.
# Must clear the top ``st.navigation`` bar (~3rem) plus a small gap so page
# tabs / first widgets are not clipped; 1rem was too aggressive.
_DENSE_TOP_PADDING = "4.5rem"
"""Main block-container padding-top under top page navigation."""

_DENSE_TOP_CSS = f"""
<style>
section.stMain .block-container,
div.stMainBlockContainer,
.stAppViewBlockContainer {{
  padding-top: {_DENSE_TOP_PADDING};
}}
</style>
"""


def apply_dense_top_layout() -> None:
    """Tighten main-area top padding under top page navigation."""
    st.markdown(_DENSE_TOP_CSS, unsafe_allow_html=True)
