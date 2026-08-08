"""Streamlit multipage entry: Home, Explore, Stats, Admin."""

from __future__ import annotations

import logging

import streamlit as st

from happy_housing.app.app_settings import load_app_settings, settings_path
from happy_housing.app.data import load_config, processed_dir
from happy_housing.app.layout_density import apply_dense_top_layout
from happy_housing.app.page_admin import render_admin
from happy_housing.app.page_explore import render_explore
from happy_housing.app.page_home import render_home
from happy_housing.app.page_stats import render_stats
from happy_housing.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _bootstrap_logging() -> None:
    """Attach package logging from experiment YAML, overridden by Admin settings."""
    level = "INFO"
    try:
        level = load_config().logging.level
    except (OSError, ValueError, TypeError) as exc:
        configure_logging(level, force=False)
        logger.warning("Experiment logging config unavailable: %s", exc)
    try:
        settings = load_app_settings(settings_path(processed_dir()))
        level = settings.log_level
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("App settings log level unavailable: %s", exc)
    configure_logging(level, force=False)


def main() -> None:
    """Configure navigation and run the selected page."""
    _bootstrap_logging()
    st.set_page_config(
        page_title="happy-housing",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_dense_top_layout()
    logger.debug("Streamlit navigation starting")
    pages = [
        st.Page(render_home, title="Home", default=True),
        st.Page(render_explore, title="Explore"),
        st.Page(render_stats, title="Stats"),
        st.Page(render_admin, title="Admin"),
    ]
    navigation = st.navigation(pages, position="top")
    navigation.run()


def render_app() -> None:
    """Back-compat alias used by tests and package exports."""
    main()


if __name__ == "__main__":
    main()
