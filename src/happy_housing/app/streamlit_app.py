"""Streamlit multipage entry: Home, Explore, Admin."""

from __future__ import annotations

import streamlit as st

from happy_housing.app.page_admin import render_admin
from happy_housing.app.page_explore import render_explore
from happy_housing.app.page_home import render_home


def main() -> None:
    """Configure navigation and run the selected page."""
    st.set_page_config(
        page_title="happy-housing",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    pages = [
        st.Page(render_home, title="Home", default=True),
        st.Page(render_explore, title="Explore"),
        st.Page(render_admin, title="Admin"),
    ]
    navigation = st.navigation(pages, position="top")
    navigation.run()


def render_app() -> None:
    """Back-compat alias used by tests and package exports."""
    main()


if __name__ == "__main__":
    main()
