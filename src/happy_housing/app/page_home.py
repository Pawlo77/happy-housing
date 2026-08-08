"""Home page: interactive market overview, sources with URLs, model story."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from happy_housing.app.charts import bar_chart, histogram
from happy_housing.app.data import load_metadata, load_scored, scored_path
from happy_housing.app.home_stats import counts_frame, market_summary
from happy_housing.app.metadata import (
    DEFAULT_METHODOLOGY,
    DEFAULT_SOURCE_CATALOG,
    METHODOLOGY_STEPS,
)


def _format_refresh(refreshed: str | None) -> str:
    if not refreshed:
        return "—"
    return refreshed[:19].replace("T", " ")


def _hero(refreshed: str | None, n_offers: int, r2: float | None) -> None:
    """Top brand strip with badges and headline metrics."""
    with st.container(border=True):
        left, right = st.columns([2.2, 1], vertical_alignment="center")
        with left:
            st.title("happy-housing")
            st.caption(
                "Warsaw fair-price intelligence — open data, LightGBM, "
                "explainable maps."
            )
            b1, b2, b3 = st.columns(3)
            b1.badge("Open data", color="green")
            b2.badge("LightGBM + SHAP", color="primary")
            b3.badge("Warsaw focus", color="gray")
        with right:
            st.metric("Scored offers", f"{n_offers:,}")
            st.metric(
                "Hold-out R²",
                f"{r2:.3f}" if r2 is not None else "—",
            )
            st.caption(f"Last refresh (UTC): {_format_refresh(refreshed)}")


def _kpi_strip(summary: dict[str, Any], duration_s: float | None) -> None:
    """Primary KPI row in bordered metric cards."""
    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Median zł/m²", f"{summary['median_price_per_m2']:,.0f}")
    with c2.container(border=True):
        delta = summary["median_delta"]
        st.metric(
            "Median delta",
            f"{delta:,.0f}",
            delta="listing - model",
            delta_color="off",
        )
    with c3.container(border=True):
        share = summary["share_overpriced"]
        st.metric(
            "Share overpriced",
            f"{100 * share:.0f}%" if share == share else "—",
        )
    with c4.container(border=True):
        if duration_s is not None:
            st.metric("Pipeline duration", f"{duration_s:.1f}s")
        else:
            st.metric("Pipeline duration", "—")


def _tab_overview(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    duration_s: float | None,
) -> None:
    """Interactive market pulse with filterable charts."""
    _kpi_strip(summary, duration_s)

    sources = ["(all)", *sorted(summary["by_source"])]
    districts = ["(all)", *sorted(summary["by_district"])]
    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
    with f1:
        source = st.selectbox("Source lens", sources, key="home_source")
    with f2:
        district = st.selectbox("District lens", districts, key="home_district")
    with f3:
        view = (
            st.segmented_control(
                "Chart focus",
                options=["By source", "By district", "Price mix"],
                default="By source",
                key="home_chart_focus",
            )
            or "By source"
        )

    mask = pd.Series(True, index=frame.index, dtype=bool)
    if source != "(all)" and "source" in frame.columns:
        mask &= frame["source"].astype(str).eq(source)
    if district != "(all)" and "district" in frame.columns:
        mask &= frame["district"].astype(str).eq(district)
    filtered = frame.loc[mask]

    local = market_summary(filtered)
    st.caption(
        f"Lens: **{len(filtered):,}** offers · median "
        f"**{local['median_price_per_m2']:,.0f}** zł/m² · "
        f"overpriced **{100 * local['share_overpriced']:.0f}%**"
        if local["n_offers"]
        else "Lens empty — widen filters."
    )

    if view == "By source":
        bar_chart(
            counts_frame(local["by_source"], category="source"),
            x="count",
            y="source",
            title="Offers by source",
        )
    elif view == "By district":
        bar_chart(
            counts_frame(local["by_district"], category="district"),
            x="count",
            y="district",
            title="Offers by district",
        )
    else:
        left, right = st.columns(2)
        with left:
            histogram(
                filtered.get("price_per_m2", pd.Series(dtype=float)),
                title="Listing zł/m²",
                color="#1B5E52",
            )
        with right:
            histogram(
                filtered.get("delta_price_per_m2", pd.Series(dtype=float)),
                title="Delta (listing - model)",
                color="#2F6F7E",
            )

    if not filtered.empty:
        preview_cols = [
            c
            for c in (
                "listing_id",
                "district",
                "source",
                "price_per_m2",
                "model_price_per_m2",
                "delta_price_per_m2",
            )
            if c in filtered.columns
        ]
        st.dataframe(
            filtered[preview_cols].head(40),
            hide_index=True,
            width="stretch",
            column_config={
                "price_per_m2": st.column_config.NumberColumn("zł/m²", format="%.0f"),
                "model_price_per_m2": st.column_config.NumberColumn(
                    "Model zł/m²", format="%.0f"
                ),
                "delta_price_per_m2": st.column_config.NumberColumn(
                    "Delta", format="%+.0f"
                ),
            },
        )


def _tab_model(meta_metrics: dict[str, float], feature_columns: list[str]) -> None:
    """Model performance cards + methodology steps."""
    if meta_metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1.container(border=True):
            st.metric("MAE", f"{meta_metrics.get('mae', float('nan')):,.0f} zł/m²")
        with m2.container(border=True):
            st.metric("R²", f"{meta_metrics.get('r2', float('nan')):.3f}")
        with m3.container(border=True):
            st.metric("Train rows", f"{int(meta_metrics.get('n_train', 0)):,}")
        with m4.container(border=True):
            st.metric("Test rows", f"{int(meta_metrics.get('n_test', 0)):,}")
    else:
        st.info("No training metrics yet — run the pipeline from Admin.")

    st.subheader("How the fair price is built")
    for idx, step in enumerate(METHODOLOGY_STEPS, start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}.** {step}")

    with st.expander("Full methodology note", expanded=False):
        st.write(DEFAULT_METHODOLOGY)

    if feature_columns:
        st.subheader("Feature set")
        st.pills(
            "Model inputs",
            options=feature_columns,
            selection_mode="multi",
            default=feature_columns,
            key="home_features",
            disabled=True,
        )


def _tab_sources() -> None:
    """Attributed sources with outbound links."""
    st.caption("Every feed is free / public. Open the source to verify attribution.")
    for src in DEFAULT_SOURCE_CATALOG:
        with st.container(border=True):
            left, right = st.columns([3.4, 1], vertical_alignment="center")
            with left:
                st.markdown(f"**{src.name}**")
                st.caption(src.role)
            with right:
                st.link_button("Open source", src.url, width="stretch")


def render_home() -> None:
    """Render the interactive Home overview."""
    meta = load_metadata()
    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        frame = pd.DataFrame()

    summary = market_summary(frame)
    refreshed = meta.refreshed_at if meta else None
    path = scored_path()
    if not refreshed and path.exists():
        refreshed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    r2 = meta.model_metrics.get("r2") if meta and meta.model_metrics else None
    duration = meta.duration_s if meta else None
    metrics = dict(meta.model_metrics) if meta and meta.model_metrics else {}
    features = list(meta.feature_columns) if meta and meta.feature_columns else []

    _hero(refreshed, summary["n_offers"], r2)

    overview, model_tab, sources_tab = st.tabs([
        "Market pulse",
        "Model",
        "Data sources",
    ])
    with overview:
        if frame.empty:
            st.info("No scored offers yet — run Admin → pipeline.")
        else:
            _tab_overview(frame, summary, duration)
    with model_tab:
        _tab_model(metrics, features)
    with sources_tab:
        _tab_sources()
