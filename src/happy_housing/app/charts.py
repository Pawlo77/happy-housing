"""Shared Altair chart helpers for Streamlit pages."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

_DEFAULT_BAR_COLOR = "#1B5E52"
"""Brand forest accent used by default bar marks."""


def bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str = _DEFAULT_BAR_COLOR,
    x_title: str | None = None,
) -> None:
    """Render a horizontal Altair bar chart, or an info stub when empty."""
    if frame.empty:
        st.info(f"No data for {title.lower()}.")
        return
    chart = (
        alt
        .Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=color)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title or x.replace("_", " ")),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            tooltip=list(frame.columns),
        )
        .properties(height=max(200, 26 * len(frame)), title=title)
    )
    st.altair_chart(chart, width="stretch")


def line_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str = _DEFAULT_BAR_COLOR,
) -> None:
    """Render a categorical-x line chart preserving row order."""
    if frame.empty:
        st.info("Not enough data for this cut.")
        return
    plot = frame.copy()
    plot[x] = plot[x].astype(str)
    chart = (
        alt
        .Chart(plot)
        .mark_line(point=True, color=color)
        .encode(
            x=alt.X(f"{x}:N", sort=list(plot[x]), title=x.replace("_", " ")),
            y=alt.Y(f"{y}:Q", title=y.replace("_", " ")),
            tooltip=list(plot.columns),
        )
        .properties(height=280, title=title)
    )
    st.altair_chart(chart, width="stretch")


def histogram(
    series: pd.Series,
    *,
    title: str,
    color: str = _DEFAULT_BAR_COLOR,
    maxbins: int = 36,
) -> None:
    """Render a numeric histogram for a single series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        st.info(f"No values for {title.lower()}.")
        return
    data = pd.DataFrame({"value": clean})
    chart = (
        alt
        .Chart(data)
        .mark_bar(color=color, opacity=0.85)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=maxbins), title=title),
            y=alt.Y("count()", title="Offers"),
            tooltip=[alt.Tooltip("count()", title="Offers")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
