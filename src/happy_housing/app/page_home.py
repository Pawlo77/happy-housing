"""Home page: about, market overview, sources with URLs, model story."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from happy_housing.app.charts import PRIMARY, bar_chart, histogram
from happy_housing.app.data import (
    load_config,
    load_metadata,
    load_scored,
    processed_dir,
    scored_path,
)
from happy_housing.app.feature_labels import feature_catalog_frame
from happy_housing.app.home_about import ABOUT_SECTIONS
from happy_housing.app.home_hero import format_hero_stats
from happy_housing.app.home_stats import counts_frame, market_summary
from happy_housing.app.metadata import (
    DEFAULT_METHODOLOGY,
    DEFAULT_SOURCE_CATALOG,
    METHODOLOGY_STEPS,
)
from happy_housing.app.recommender import load_recommender_metrics
from happy_housing.app.source_inventory import (
    SourceInventoryRow,
    build_source_inventory,
)

logger = logging.getLogger(__name__)


def _hero(refreshed: str | None, n_offers: int, r2: float | None) -> None:
    """Top brand strip — title, tagline, and Streamlit metrics only."""
    stats = format_hero_stats(refreshed, n_offers, r2)
    st.title("happy-housing")
    st.caption("Warsaw fair-price intelligence from open data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Scored offers", stats.offers_label)
    c2.metric("Hold-out R²", stats.r2_label)
    c3.metric("Last refresh (UTC)", stats.refreshed_label)


def _tab_about() -> None:
    """Business purpose of the tool in plain language."""
    st.caption(
        "A short orientation before the charts — what the tool does and how to read it."
    )
    for section in ABOUT_SECTIONS:
        st.subheader(section.title)
        st.markdown(section.body)


def _kpi_strip(summary: dict[str, Any], duration_s: float | None) -> None:
    """Primary KPI row — equal-height metrics (no delta subtitle line)."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median zł/m²", f"{summary['median_price_per_m2']:,.0f}")
    delta = summary["median_delta"]
    c2.metric(
        "Median delta",
        f"{delta:,.0f}",
        help="Listing zł/m² minus model zł/m²",
    )
    share = summary["share_overpriced"]
    c3.metric(
        "Share overpriced",
        f"{100 * share:.0f}%" if share == share else "—",
    )
    if duration_s is not None:
        c4.metric("Pipeline duration", f"{duration_s:.1f}s")
    else:
        c4.metric("Pipeline duration", "—")


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
                color=PRIMARY,
            )
        with right:
            histogram(
                filtered.get("delta_price_per_m2", pd.Series(dtype=float)),
                title="Delta (listing - model)",
            )

    # Offer board lives on Explore → Board (row opens Offer).


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

    rec = load_recommender_metrics(processed_dir())
    st.subheader("Preference recommender")
    if rec:
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Candidates", f"{int(rec.get('n_candidates', 0)):,}")
        r2.metric("Recommended", f"{int(rec.get('n_recommended', 0)):,}")
        r3.metric("Mean confidence", f"{rec.get('mean_confidence', float('nan')):.3f}")
        r4.metric(
            "Share unseen",
            f"{100 * rec.get('share_unseen', float('nan')):.0f}%",
        )
        r5.metric(
            "Prefs on",
            "Yes" if rec.get("prefs_enabled", 0.0) >= 0.5 else "No",
        )
        st.caption(
            "Rule-based preference-fit + bargain scorer with soft demotion for "
            "already-seen offers (not the price LightGBM). "
            "Configure on Explore → Prefs; retrain from Admin → Recommendations."
        )
    else:
        st.info("No recommender metrics yet — open Explore → News or Admin → Retrain.")

    st.subheader("How the fair price is built")
    for idx, step in enumerate(METHODOLOGY_STEPS, start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}.** {step}")

    with st.expander("Full methodology note", expanded=False):
        st.write(DEFAULT_METHODOLOGY)

    if feature_columns:
        st.subheader("Feature set")
        st.caption("Model inputs — hover a row to read the full name and description.")
        catalog = feature_catalog_frame(feature_columns)
        st.dataframe(
            catalog,
            hide_index=True,
            width="stretch",
            height=min(420, 36 + 35 * min(len(catalog), 12)),
            column_config={
                "feature": st.column_config.TextColumn(
                    "Feature",
                    help="Technical column name used by the LightGBM model",
                    width="medium",
                ),
                "description": st.column_config.TextColumn(
                    "Description",
                    help="What this input measures for the fair-price model",
                    width="large",
                ),
            },
        )


def _tab_sources(rows: list[SourceInventoryRow]) -> None:
    """Attributed sources with live fetch counts and timestamps."""
    st.caption(
        "Every feed is free / public. Counts and fetch times come from "
        "local artifacts (processed / cache / fixtures)."
    )
    for row in rows:
        with st.container(border=True):
            left, right = st.columns([3.4, 1], vertical_alignment="center")
            with left:
                st.markdown(f"**{row.name}**")
                st.caption(row.role)
                st.caption(
                    f"{row.rows_label} · fetched {row.fetched_label} UTC"
                    f" · {row.origin_label}"
                )
            with right:
                st.link_button("Open source", row.url, width="stretch")


def render_home() -> None:
    """Render the interactive Home overview."""
    logger.info("Rendering Home")
    meta = load_metadata()
    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        logger.warning("Home missing scored listings: %s", exc)
        st.warning(str(exc))
        frame = pd.DataFrame()

    summary = market_summary(frame)
    logger.debug(
        "Home summary: n=%s median_ppm=%s",
        summary.get("n_offers"),
        summary.get("median_price_per_m2"),
    )
    refreshed = meta.refreshed_at if meta else None
    path = scored_path()
    if not refreshed and path.exists():
        refreshed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    r2 = meta.model_metrics.get("r2") if meta and meta.model_metrics else None
    duration = meta.duration_s if meta else None
    metrics = dict(meta.model_metrics) if meta and meta.model_metrics else {}
    features = list(meta.feature_columns) if meta and meta.feature_columns else []

    _hero(refreshed, summary["n_offers"], r2)

    about_tab, overview, model_tab, sources_tab = st.tabs([
        "About",
        "Market pulse",
        "Model",
        "Data sources",
    ])
    with about_tab:
        _tab_about()
    with overview:
        if frame.empty:
            st.info("No scored offers yet — run Admin → pipeline.")
        else:
            _tab_overview(frame, summary, duration)
    with model_tab:
        _tab_model(metrics, features)
    with sources_tab:
        config = load_config()
        inventory = build_source_inventory(
            DEFAULT_SOURCE_CATALOG,
            processed_dir=config.data.processed_path(),
            cache_dir=config.data.cache_path(),
            fixtures_dir=config.data.fixtures_path(),
        )
        _tab_sources(inventory)
