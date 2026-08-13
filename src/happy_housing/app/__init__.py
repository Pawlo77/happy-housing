"""Streamlit application package for happy-housing."""

from typing import Any

__all__ = ["main", "render_app"]


def __getattr__(name: str) -> Any:
    """Lazy-load Streamlit entrypoints so CLI imports stay UI-free."""
    if name in {"main", "render_app"}:
        from happy_housing.app.streamlit_app import main, render_app

        return {"main": main, "render_app": render_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
