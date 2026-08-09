"""Shared Altair chart helpers — one visual language across the app."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from happy_housing.app.feature_labels import feature_description

# Match `.streamlit/config.toml` brand accents.
PRIMARY = "#0F6B5C"
"""Ink teal — primary marks / underpriced."""

SECONDARY = "#2F6F7E"
"""Cool slate secondary series."""

DANGER = "#B42318"
"""Overpriced / raise-ask red."""

WARNING = "#B54708"
"""Ask premium / amber accent."""

MUTED = "#5A6670"
"""Axis / volume secondary."""

UNDERPRICED = PRIMARY
"""Alias for underpriced residual colour."""

OVERPRICED = DANGER
"""Alias for overpriced residual colour."""

ASK_COLOR = PRIMARY
"""Otodom ask series."""

INK_COLOR = WARNING
"""RCN ink / transaction series."""

CHART_HEIGHT = 280
"""Default chart height in pixels."""

_TOOLTIP_LIMIT = 12
"""Max columns included in default tooltips."""


def styled(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Apply shared axis / legend / title styling."""
    return (
        chart
        .configure_axis(
            labelColor="#3A4550",
            titleColor="#12181F",
            gridColor="#E8EDF1",
            domainColor="#C5CED6",
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight=600,
        )
        .configure_legend(
            labelColor="#3A4550",
            titleColor="#12181F",
            labelFontSize=12,
            titleFontSize=12,
            orient="top",
        )
        .configure_title(
            color="#12181F",
            fontSize=15,
            fontWeight=600,
            anchor="start",
            offset=8,
        )
    )


def _tooltip(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in list(frame.columns)[:_TOOLTIP_LIMIT]]


def _with_feature_descriptions(
    frame: pd.DataFrame,
    feature_col: str,
) -> pd.DataFrame:
    """Attach a ``description`` column for model-feature hover text."""
    if feature_col not in frame.columns:
        return frame
    out = frame.copy()
    out["description"] = out[feature_col].map(feature_description)
    return out


def _feature_axis_tooltip(
    frame: pd.DataFrame,
    *,
    feature_col: str,
    value_col: str,
    value_title: str,
    category_col: str | None = None,
) -> list[alt.Tooltip]:
    """Tooltip that surfaces the full feature name + human description."""
    tips: list[alt.Tooltip] = [
        alt.Tooltip(f"{feature_col}:N", title="Feature"),
        alt.Tooltip("description:N", title="Description"),
        alt.Tooltip(f"{value_col}:Q", title=value_title, format=".3f"),
    ]
    if category_col is not None and category_col in frame.columns:
        tips.append(alt.Tooltip(f"{category_col}:N", title="Effect"))
    return tips


def _show(chart: alt.Chart | alt.LayerChart) -> None:
    st.altair_chart(styled(chart), width="stretch")


def bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str = PRIMARY,
    x_title: str | None = None,
) -> None:
    """Render a horizontal Altair bar chart, or an info stub when empty."""
    if frame.empty:
        st.info(f"No data for {title.lower()}.")
        return
    plot = frame
    tip: list[str] | list[alt.Tooltip] = _tooltip(frame)
    if y == "feature":
        plot = _with_feature_descriptions(frame, "feature")
        tip = _feature_axis_tooltip(
            plot,
            feature_col="feature",
            value_col=x,
            value_title=x_title or x.replace("_", " "),
        )
    chart = (
        alt
        .Chart(plot)
        .mark_bar(cornerRadiusEnd=3, color=color)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title or x.replace("_", " ")),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            tooltip=tip,
        )
        .properties(height=max(200, 26 * len(frame)), title=title)
    )
    _show(chart)


def grouped_bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str,
    color_domain: list[str],
    color_range: list[str],
    x_title: str | None = None,
) -> None:
    """Horizontal grouped bars (e.g. ask vs ink)."""
    if frame.empty:
        st.info(f"No data for {title.lower()}.")
        return
    chart = (
        alt
        .Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title or x.replace("_", " ")),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            color=alt.Color(
                f"{color}:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title=None),
            ),
            tooltip=_tooltip(frame),
        )
        .properties(height=max(240, 34 * frame[y].nunique()), title=title)
    )
    _show(chart)


def line_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str = PRIMARY,
    y_title: str | None = None,
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
        .mark_line(point=True, color=color, strokeWidth=2.5)
        .encode(
            x=alt.X(f"{x}:N", sort=list(plot[x]), title=x.replace("_", " ")),
            y=alt.Y(f"{y}:Q", title=y_title or y.replace("_", " ")),
            tooltip=_tooltip(plot),
        )
        .properties(height=CHART_HEIGHT, title=title)
    )
    _show(chart)


def multi_line_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str,
    color_domain: list[str] | None = None,
    color_range: list[str] | None = None,
    y_title: str | None = None,
) -> None:
    """Multi-series line chart with shared brand palette."""
    if frame.empty:
        st.info("Not enough data for this cut.")
        return
    plot = frame.copy()
    plot[x] = plot[x].astype(str)
    if color_domain is not None and color_range is not None:
        color_enc = alt.Color(
            f"{color}:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title=None),
        )
    else:
        color_enc = alt.Color(f"{color}:N", legend=alt.Legend(title=None))
    chart = (
        alt
        .Chart(plot)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(f"{x}:N", sort=sorted(plot[x].unique()), title=x.replace("_", " ")),
            y=alt.Y(f"{y}:Q", title=y_title or y.replace("_", " ")),
            color=color_enc,
            tooltip=_tooltip(plot),
        )
        .properties(height=CHART_HEIGHT, title=title)
    )
    _show(chart)


def histogram(
    series: pd.Series,
    *,
    title: str,
    color: str = PRIMARY,
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
        .mark_bar(color=color, opacity=0.9, cornerRadiusEnd=2)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=maxbins), title=title),
            y=alt.Y("count()", title="Offers"),
            tooltip=[alt.Tooltip("count()", title="Offers")],
        )
        .properties(height=CHART_HEIGHT, title=title)
    )
    _show(chart)


def stacked_histogram(
    frame: pd.DataFrame,
    *,
    value: str,
    category: str,
    title: str,
    color_domain: list[str],
    color_range: list[str],
    maxbins: int = 40,
    x_title: str | None = None,
) -> None:
    """Histogram coloured by a categorical side (e.g. over/underpriced)."""
    if frame.empty:
        st.info(f"No data for {title.lower()}.")
        return
    chart = (
        alt
        .Chart(frame)
        .mark_bar(opacity=0.9, cornerRadiusEnd=2)
        .encode(
            x=alt.X(
                f"{value}:Q",
                bin=alt.Bin(maxbins=maxbins),
                title=x_title or value.replace("_", " "),
            ),
            y=alt.Y("count()", title="Offers"),
            color=alt.Color(
                f"{category}:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                category,
                alt.Tooltip("count()", title="Offers"),
            ],
        )
        .properties(height=CHART_HEIGHT, title=title)
    )
    _show(chart)


def scatter_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    size: str | None = None,
    color: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> None:
    """Bubble / scatter chart with brand sequential colouring."""
    if frame.empty:
        st.info(f"No data for {title.lower()}.")
        return
    encode: dict[str, Any] = {
        "x": alt.X(f"{x}:Q", title=x_title or x.replace("_", " ")),
        "y": alt.Y(f"{y}:Q", title=y_title or y.replace("_", " ")),
        "tooltip": _tooltip(frame),
    }
    if size:
        encode["size"] = alt.Size(
            f"{size}:Q", title=size.replace("_", " "), scale=alt.Scale(range=[60, 900])
        )
    if color:
        encode["color"] = alt.Color(
            f"{color}:Q",
            title=color.replace("_", " "),
            scale=alt.Scale(
                range=["#E8EDF1", "#96B5B0", "#4F8A7A", "#0F6B5C", "#06352E"]
            ),
        )
    else:
        encode["color"] = alt.value(PRIMARY)
    chart = (
        alt
        .Chart(frame)
        .mark_circle(opacity=0.8)
        .encode(**encode)
        .properties(height=360, title=title)
    )
    _show(chart)


def diverging_bar_chart(
    frame: pd.DataFrame,
    *,
    feature: str,
    value: str,
    category: str,
    title: str,
    color_domain: list[str],
    color_range: list[str],
    x_title: str | None = None,
) -> None:
    """Horizontal bars coloured by sign/category (SHAP-style)."""
    if frame.empty:
        st.info(f"No data for {title.lower()}.")
        return
    plot = frame
    tip: list[str] | list[alt.Tooltip] = _tooltip(frame)
    if feature == "feature":
        plot = _with_feature_descriptions(frame, "feature")
        tip = _feature_axis_tooltip(
            plot,
            feature_col="feature",
            value_col=value,
            value_title=x_title or value.replace("_", " "),
            category_col=category,
        )
    chart = (
        alt
        .Chart(plot)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{value}:Q", title=x_title or value.replace("_", " ")),
            y=alt.Y(f"{feature}:N", sort="-x", title=None),
            color=alt.Color(
                f"{category}:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title=None),
            ),
            tooltip=tip,
        )
        .properties(height=max(280, 28 * len(frame)), title=title)
    )
    _show(chart)
