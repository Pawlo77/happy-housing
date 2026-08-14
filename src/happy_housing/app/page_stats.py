"""Stats page: filterable market analytics, livability, and model XAI."""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from happy_housing.app.app_settings import AppSettings, load_app_settings, settings_path
from happy_housing.app.charts import (
    ASK_COLOR,
    INK_COLOR,
    OVERPRICED,
    PRIMARY,
    SECONDARY,
    UNDERPRICED,
    bar_chart,
    diverging_bar_chart,
    grouped_bar_chart,
    line_chart,
    multi_line_chart,
    scatter_chart,
    stacked_histogram,
)
from happy_housing.app.data import load_scored, load_shap_bundle, processed_dir
from happy_housing.app.explore_nav import (
    listing_id_from_board_selection,
    open_explore_offer,
)
from happy_housing.app.property_filter_ui import render_property_attr_filters
from happy_housing.app.scored_views import drop_heavy_listing_columns
from happy_housing.app.stats_filters import StatsFilterState, apply_stats_filters
from happy_housing.app.stats_insights import (
    affordability_bands,
    air_quality_ladder,
    aircraft_noise_ladder,
    ask_vs_ink,
    chopin_surcharge,
    commute_cost_curve,
    delta_volcano,
    distance_price_curve,
    district_scoreboard,
    frontage_hint_price_table,
    high_noise_surcharge,
    highway_class_price_table,
    market_kpis,
    market_segment_mix,
    market_temperature,
    metro_price_curve,
    monthly_trends,
    noise_source_premiums,
    noise_total_ladder,
    normalize_district,
    price_band_inventory,
    room_ladder,
    rooms_inventory,
    safety_radius,
    size_sweet_spot,
    source_mix_summary,
    top_value_offers,
    transit_gravity,
    transit_price_curve,
    year_paradox,
)
from happy_housing.app.stats_xai import booster_importance_table, mean_abs_shap_table
from happy_housing.app.ui_help import STATS_FILTER_HELP
from happy_housing.ingest.listing_common import append_inactive_otodom_asks
from happy_housing.models.train import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    model_feature_columns,
)

logger = logging.getLogger(__name__)


def _section(title: str, blurb: str) -> None:
    st.subheader(title)
    st.caption(blurb)


def _unique_sorted(series: pd.Series) -> list[str]:
    return sorted({str(v) for v in series.dropna().astype(str) if str(v)})


def _sidebar_filters(frame: pd.DataFrame) -> StatsFilterState:
    """Render Stats filter lens in the sidebar."""
    st.sidebar.header("Stats filters")
    st.sidebar.caption("Applies to every tab and chart.")

    districts = sorted({
        normalize_district(v)
        for v in frame.get("district", pd.Series(dtype=str)).dropna()
        if str(v).strip()
    })
    sources = _unique_sorted(frame.get("source", pd.Series(dtype=str)))
    months = [
        m
        for m in _unique_sorted(frame.get("listed_month", pd.Series(dtype=str)))
        if m != "unknown"
    ]
    ppm = pd.to_numeric(frame.get("price_per_m2"), errors="coerce")
    area = pd.to_numeric(frame.get("area_m2"), errors="coerce")
    rooms = pd.to_numeric(frame.get("rooms"), errors="coerce")
    ppm_min = float(ppm.min()) if ppm.notna().any() else 0.0
    ppm_max = float(ppm.max()) if ppm.notna().any() else 100_000.0
    area_lo = float(area.min()) if area.notna().any() else 0.0
    area_hi = float(area.max()) if area.notna().any() else 400.0
    rooms_lo = int(rooms.min()) if rooms.notna().any() else 1
    rooms_hi = int(rooms.max()) if rooms.notna().any() else 6

    picked_districts = st.sidebar.multiselect(
        "Districts",
        options=districts,
        default=[],
        key="stats_districts",
        help=STATS_FILTER_HELP["districts"],
    )
    picked_sources = st.sidebar.multiselect(
        "Sources",
        options=sources,
        default=[],
        key="stats_sources",
        help=STATS_FILTER_HELP["sources"],
    )
    picked_months = st.sidebar.multiselect(
        "Listed months",
        options=months,
        default=[],
        key="stats_months",
        help=STATS_FILTER_HELP["months"],
    )
    valuation = st.sidebar.selectbox(
        "Valuation",
        options=["(all)", "underpriced", "overpriced"],
        key="stats_valuation",
        help=STATS_FILTER_HELP["valuation"],
    )
    oou = st.sidebar.selectbox(
        "Aircraft Lden ≥ 55 dB",
        options=["(all)", "inside", "outside"],
        key="stats_oou",
        help=STATS_FILTER_HELP["oou"],
    )
    high_noise = st.sidebar.selectbox(
        "Total noise Lden ≥ 55 dB",
        options=["(all)", "inside", "outside"],
        key="stats_high_noise",
        help=STATS_FILTER_HELP["high_noise"],
    )
    price_range = st.sidebar.slider(
        "Listing zł/m²",
        min_value=float(ppm_min),
        max_value=float(max(ppm_max, ppm_min + 1.0)),
        value=(float(ppm_min), float(ppm_max)),
        key="stats_price",
        help=STATS_FILTER_HELP["price"],
    )
    area_range = st.sidebar.slider(
        "Area m²",
        min_value=float(area_lo),
        max_value=float(max(area_hi, area_lo + 1.0)),
        value=(float(area_lo), float(area_hi)),
        key="stats_area",
        help=STATS_FILTER_HELP["area"],
    )
    rooms_range = st.sidebar.slider(
        "Rooms",
        min_value=int(rooms_lo),
        max_value=int(max(rooms_hi, rooms_lo)),
        value=(int(rooms_lo), int(rooms_hi)),
        key="stats_rooms",
        help=STATS_FILTER_HELP["rooms"],
    )

    with st.sidebar:
        prop = render_property_attr_filters("stats")

    return StatsFilterState(
        districts=list(picked_districts),
        sources=list(picked_sources),
        months=list(picked_months),
        oou=str(oou),
        high_noise=str(high_noise),
        valuation=str(valuation),
        price_min=float(price_range[0]),
        price_max=float(price_range[1]),
        area_min=float(area_range[0]),
        area_max=float(area_range[1]),
        rooms_min=int(rooms_range[0]),
        rooms_max=int(rooms_range[1]),
        **prop,
    )


def _kpi_row(kpis: dict[str, float], frame: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Offers in lens", f"{int(kpis['n_offers']):,}")
    med = kpis["median_price_per_m2"]
    c2.metric("Median zł/m²", f"{med:,.0f}" if med == med else "—")
    model = kpis["median_model"]
    c3.metric("Median model", f"{model:,.0f}" if model == model else "—")
    delta = kpis["median_delta"]
    c4.metric("Median delta", f"{delta:+,.0f}" if delta == delta else "—")
    share = kpis["share_overpriced"]
    c5.metric(
        "Share overpriced",
        f"{100 * share:.0f}%" if share == share else "—",
    )
    if "in_high_noise" in frame.columns:
        flag = pd.to_numeric(frame["in_high_noise"], errors="coerce")
        noisy = float(flag.eq(1).mean()) if flag.notna().any() else float("nan")
    else:
        noisy = float("nan")
    c6.metric(
        "Share high noise",
        f"{100 * noisy:.0f}%" if noisy == noisy else "—",
        help="Total Lden ≥ 55 dB (UM 2022 energy sum).",
    )


def _tab_overview(frame: pd.DataFrame) -> None:
    _section(
        "1 · Inventory mix",
        "Where supply clusters by zł/m², rooms, and ask vs transaction source.",
    )
    bands = price_band_inventory(frame)
    left, right = st.columns(2)
    with left:
        bar_chart(
            bands.assign(n_offers=bands["n"]),
            x="n",
            y="price_band",
            title="Offers by zł/m² band",
            color=PRIMARY,
            x_title="Offers",
        )
    with right:
        bar_chart(
            bands,
            x="median_delta",
            y="price_band",
            title="Median delta by zł/m² band",
            x_title="Median delta zł/m²",
        )

    c1, c2 = st.columns(2)
    with c1:
        bar_chart(
            rooms_inventory(frame),
            x="n",
            y="rooms",
            title="Supply by rooms",
            color=PRIMARY,
            x_title="Offers",
        )
    with c2:
        src = source_mix_summary(frame)
        if src.empty:
            st.info("No source mix in the lens.")
        else:
            bar_chart(
                src,
                x="median_ppm",
                y="source",
                title="Median zł/m² by source",
                x_title="zł/m²",
            )

    seg = market_segment_mix(frame)
    _section(
        "2 · Primary vs secondary",
        "New-build share vs resale — useful when comparing ask premiums.",
    )
    if seg.empty:
        st.info("No known primary/secondary market flag in the lens.")
    else:
        bar_chart(
            seg,
            x="n",
            y="segment",
            title="Offers by market segment",
            x_title="Offers",
        )

    _section(
        "3 · Budget ladder",
        "What total asking price buys — area and zł/m² by budget band.",
    )
    afford = affordability_bands(frame)
    if afford.empty:
        st.info("Need asking prices (price_pln) in the lens.")
        return
    a1, a2 = st.columns(2)
    with a1:
        bar_chart(
            afford,
            x="n",
            y="price_band",
            title="Offers by total price band",
            color=PRIMARY,
            x_title="Offers",
        )
    with a2:
        line_chart(
            afford,
            x="price_band",
            y="median_ppm",
            title="Median zł/m² by budget band",
            y_title="zł/m²",
        )


def _tab_market(frame: pd.DataFrame, *, min_offers: int, top_n: int) -> None:
    _section(
        "1 · Price & fair-value trends",
        "Listing vs model medians month-to-month in the current lens.",
    )
    trends = monthly_trends(frame)
    if trends.empty or len(trends) < 2:
        st.info("Need at least two listed months in the lens for trends.")
    else:
        melt = trends.melt(
            id_vars=["listed_month"],
            value_vars=["median_price_per_m2", "median_model"],
            var_name="series",
            value_name="value_ppm",
        )
        melt["series"] = melt["series"].map({
            "median_price_per_m2": "Listing median",
            "median_model": "Model median",
        })
        multi_line_chart(
            melt,
            x="listed_month",
            y="value_ppm",
            color="series",
            title="Median zł/m² over listed months",
            color_domain=["Listing median", "Model median"],
            color_range=[PRIMARY, SECONDARY],
            y_title="zł/m²",
        )
        left, right = st.columns(2)
        with left:
            line_chart(
                trends,
                x="listed_month",
                y="median_delta",
                title="Median delta (listing - model)",
                y_title="Delta zł/m²",
            )
        with right:
            line_chart(
                trends.rename(columns={"n": "offers"}),
                x="listed_month",
                y="offers",
                title="Offer volume by month",
                color=PRIMARY,
                y_title="Offers",
            )

    _section(
        "2 · Residual pressure",
        "Distribution of listing - model zł/m².",
    )
    stacked_histogram(
        delta_volcano(frame),
        value="delta",
        category="side",
        title="Delta distribution",
        color_domain=["underpriced", "overpriced"],
        color_range=[UNDERPRICED, OVERPRICED],
        x_title="Delta zł/m²",
    )

    _section(
        "3 · District value board",
        "Ask vs model by district. Negative median delta → more bargains.",
    )
    board = district_scoreboard(frame, min_offers=min_offers)
    if board.empty:
        st.info(f"Need districts with at least {min_offers} offers in the lens.")
    else:
        show = board.copy()
        show["bargain_share"] = (show["bargain_share"] * 100).round(0)
        show["share_overpriced"] = (show["share_overpriced"] * 100).round(0)
        left, right = st.columns([1.3, 1])
        with left:
            bar_chart(
                board.head(15),
                x="median_delta",
                y="district",
                title="Median delta by district",
                color=PRIMARY,
                x_title="Median delta zł/m²",
            )
        with right:
            st.dataframe(
                show.head(15),
                hide_index=True,
                width="stretch",
                column_config={
                    "bargain_share": st.column_config.NumberColumn(
                        "Bargain %", format="%.0f%%"
                    ),
                    "share_overpriced": st.column_config.NumberColumn(
                        "Overpriced %", format="%.0f%%"
                    ),
                    "median_delta": st.column_config.NumberColumn(
                        "Median delta", format="%+.0f"
                    ),
                    "median_ppm": st.column_config.NumberColumn(
                        "Ask zł/m²", format="%.0f"
                    ),
                    "median_model": st.column_config.NumberColumn(
                        "Model zł/m²", format="%.0f"
                    ),
                },
            )

    _section(
        "4 · Ask vs ink",
        "Otodom asking medians versus RCN transaction medians.",
    )
    gap = ask_vs_ink(frame, min_offers=max(2, min_offers // 2))
    if gap.empty:
        st.info("Need overlapping Otodom + RCN districts in the lens.")
    else:
        melt = gap.melt(
            id_vars=["district"],
            value_vars=["ask_median", "ink_median"],
            var_name="kind",
            value_name="median_ppm",
        )
        melt["kind"] = melt["kind"].map({
            "ask_median": "Otodom ask",
            "ink_median": "RCN ink",
        })
        grouped_bar_chart(
            melt,
            x="median_ppm",
            y="district",
            color="kind",
            title="Ask vs transaction medians",
            color_domain=["Otodom ask", "RCN ink"],
            color_range=[ASK_COLOR, INK_COLOR],
            x_title="Median zł/m²",
        )

    value_cols = {
        "price_per_m2": st.column_config.NumberColumn("Listing zł/m²", format="%.0f"),
        "model_price_per_m2": st.column_config.NumberColumn(
            "Model zł/m²", format="%.0f"
        ),
        "delta_price_per_m2": st.column_config.NumberColumn("Delta", format="%+.0f"),
        "commute_min": st.column_config.NumberColumn("Commute min", format="%.0f"),
    }

    _section(
        "5 · Hunt & watch",
        "Strongest underpriced / overpriced offers — click a row → Explore Offer.",
    )
    hunt_col, watch_col = st.columns(2)
    with hunt_col:
        st.markdown("**Hunt (underpriced)**")
        hunt = top_value_offers(frame, n=top_n, side="underpriced")
        if hunt.empty:
            st.info("No underpriced offers.")
        else:
            hunt_event = st.dataframe(
                hunt,
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                key="stats_hunt_table",
                column_config=value_cols,
            )
            hunt_id = listing_id_from_board_selection(hunt, hunt_event)
            if hunt_id is not None:
                open_explore_offer(hunt_id)
    with watch_col:
        st.markdown("**Watch (overpriced)**")
        watch = top_value_offers(frame, n=top_n, side="overpriced")
        if watch.empty:
            st.info("No overpriced offers.")
        else:
            watch_event = st.dataframe(
                watch,
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                key="stats_watch_table",
                column_config=value_cols,
            )
            watch_id = listing_id_from_board_selection(watch, watch_event)
            if watch_id is not None:
                open_explore_offer(watch_id)


def _tab_livability(frame: pd.DataFrame) -> None:
    chopin = chopin_surcharge(frame)
    noisy = high_noise_surcharge(frame)
    transit = transit_gravity(frame)

    _section(
        "1 · Noise premiums",
        "UM 2022 strategic maps — total energy-sum and per-source exposure "
        "(≥55 dB) vs quieter stock.",
    )
    a, b, c, d = st.columns(4)
    a.metric(
        "High-noise surcharge",
        f"{noisy['surcharge_pct']:+.1f}%"
        if noisy["surcharge_pct"] == noisy["surcharge_pct"]
        else "—",
        help="Median zł/m² inside total Lden ≥55 vs outside.",
    )
    denom = max(noisy["n_inside"] + noisy["n_outside"], 1.0)
    b.metric(
        "High-noise share",
        f"{100 * noisy['n_inside'] / denom:.0f}%",
    )
    c.metric(
        "Aircraft surcharge",
        f"{chopin['surcharge_pct']:+.1f}%"
        if chopin["surcharge_pct"] == chopin["surcharge_pct"]
        else "—",
    )
    d.metric(
        "Transit ≤300 m",
        f"{100 * transit['share_within']:.0f}%"
        if transit["share_within"] == transit["share_within"]
        else "—",
    )

    n1, n2 = st.columns(2)
    with n1:
        line_chart(
            noise_total_ladder(frame),
            x="lden_band",
            y="median_price_per_m2",
            title="Median zł/m² by total Lden",
            y_title="zł/m²",
        )
    with n2:
        line_chart(
            aircraft_noise_ladder(frame),
            x="lden_band",
            y="median_price_per_m2",
            title="Median zł/m² by aircraft Lden",
            y_title="zł/m²",
        )

    premiums = noise_source_premiums(frame)
    if not premiums.empty:
        bar_chart(
            premiums,
            x="premium_pct",
            y="source",
            title="zł/m² premium when source ≥ 55 dB",
            x_title="Premium vs quieter (%)",
        )
        st.dataframe(
            premiums,
            hide_index=True,
            width="stretch",
            column_config={
                "n_exposed": st.column_config.NumberColumn("Exposed", format="%.0f"),
                "n_quiet": st.column_config.NumberColumn("Quiet", format="%.0f"),
                "median_ppm_exposed": st.column_config.NumberColumn(
                    "Exposed zł/m²", format="%.0f"
                ),
                "median_ppm_quiet": st.column_config.NumberColumn(
                    "Quiet zł/m²", format="%.0f"
                ),
                "premium_pct": st.column_config.NumberColumn(
                    "Premium %", format="%+.1f"
                ),
            },
        )

    _section(
        "2 · Access & mobility",
        "Commute, transit, metro, bike paths — classic livability price curves.",
    )
    left, right = st.columns(2)
    with left:
        line_chart(
            commute_cost_curve(frame),
            x="commute_bin",
            y="median_price_per_m2",
            title="Median zł/m² by commute",
            y_title="zł/m²",
        )
        line_chart(
            metro_price_curve(frame),
            x="metro_band",
            y="median_price_per_m2",
            title="Median zł/m² by metro distance",
            color=PRIMARY,
            y_title="zł/m²",
        )
    with right:
        line_chart(
            transit_price_curve(frame),
            x="transit_band",
            y="median_price_per_m2",
            title="Median zł/m² by transit-stop distance",
            y_title="zł/m²",
        )
        line_chart(
            distance_price_curve(
                frame,
                dist_col="dist_bike_path_m",
                bins=[0, 100, 250, 500, 1000, 50_000],
                labels=["<100m", "100-250m", "250-500m", "500m-1km", "1km+"],
                band_col="bike_band",
            ),
            x="bike_band",
            y="median_price_per_m2",
            title="Median zł/m² by bike-path distance",
            y_title="zł/m²",
        )

    _section(
        "3 · Environment & restricted land",
        "Traffic proxy, parks, air quality, fire stations, and zone proximity.",
    )
    e1, e2 = st.columns(2)
    with e1:
        line_chart(
            distance_price_curve(
                frame,
                dist_col="dist_park_m",
                bins=[0, 200, 500, 1000, 2000, 50_000],
                labels=["<200m", "200-500m", "500m-1km", "1-2km", "2km+"],
                band_col="park_band",
            ),
            x="park_band",
            y="median_price_per_m2",
            title="Median zł/m² by park distance",
            y_title="zł/m²",
        )
        line_chart(
            safety_radius(frame),
            x="fire_band",
            y="median_price_per_m2",
            title="Median zł/m² by fire-station distance",
            y_title="zł/m²",
        )
        line_chart(
            air_quality_ladder(frame),
            x="aqi_band",
            y="median_price_per_m2",
            title="Median zł/m² by nearest AQI",
            y_title="zł/m²",
        )
    with e2:
        line_chart(
            distance_price_curve(
                frame,
                dist_col="dist_traffic_road_m",
                bins=[0, 50, 100, 200, 500, 50_000],
                labels=["<50m", "50-100m", "100-200m", "200-500m", "500m+"],
                band_col="traffic_band",
            ),
            x="traffic_band",
            y="median_price_per_m2",
            title="Median zł/m² by drive-road distance",
            y_title="zł/m²",
        )
        hwy = highway_class_price_table(frame)
        if not hwy.empty:
            bar_chart(
                hwy.head(12),
                x="nearest_highway",
                y="median_price_per_m2",
                title="Median zł/m² by nearest highway class",
            )
        frontage = frontage_hint_price_table(frame)
        if not frontage.empty:
            bar_chart(
                frontage,
                x="frontage",
                y="median_price_per_m2",
                title="Median zł/m² by street/quiet description hint",
            )
        line_chart(
            distance_price_curve(
                frame,
                dist_col="dist_rail_area_m",
                bins=[0, 200, 500, 1000, 2000, 50_000],
                labels=["<200m", "200-500m", "500m-1km", "1-2km", "2km+"],
                band_col="rail_band",
            ),
            x="rail_band",
            y="median_price_per_m2",
            title="Median zł/m² by rail-yard distance",
            y_title="zł/m²",
        )
        line_chart(
            distance_price_curve(
                frame,
                dist_col="dist_military_m",
                bins=[0, 500, 1000, 2000, 5000, 50_000],
                labels=["<500m", "500m-1km", "1-2km", "2-5km", "5km+"],
                band_col="military_band",
            ),
            x="military_band",
            y="median_price_per_m2",
            title="Median zł/m² by military-area distance",
            y_title="zł/m²",
        )

    _section(
        "4 · District mood",
        "Bubbles: x = median ask, y = median delta, size = offers.",
    )
    scatter_chart(
        market_temperature(frame),
        x="median_ppm",
        y="median_delta",
        size="n",
        color="polarization",
        title="District temperature",
        x_title="Median zł/m²",
        y_title="Median delta",
    )


def _tab_stock(frame: pd.DataFrame) -> None:
    _section(
        "Product mix",
        "How size, rooms, and build era relate to price and residuals.",
    )
    left, right = st.columns(2)
    with left:
        bar_chart(
            size_sweet_spot(frame),
            x="median_delta",
            y="area_bin",
            title="Median delta by area bucket",
            x_title="Median delta zł/m²",
        )
    with right:
        line_chart(
            room_ladder(frame),
            x="rooms",
            y="median_price_per_m2",
            title="Median zł/m² by rooms",
            y_title="zł/m²",
        )
    bar_chart(
        year_paradox(frame),
        x="median_price_per_m2",
        y="era",
        title="Median zł/m² by build era",
        color=PRIMARY,
        x_title="Median zł/m²",
    )


def _tab_model(frame: pd.DataFrame, settings: AppSettings) -> None:
    _section(
        "1 · What the model listens to",
        "LightGBM gain importance from the trained booster, plus global mean "
        "|SHAP| on a sample of the current lens.",
    )
    bundle = load_shap_bundle()
    if bundle is None:
        st.info("Model file missing — retrain from Admin → Pipeline.")
        return
    booster, explainer = bundle
    top_n = int(settings.stats_xai_top_n)
    sample_n = int(settings.stats_shap_sample_n)

    importance = booster_importance_table(booster, top_n=top_n)
    left, right = st.columns(2)
    with left:
        if importance.empty:
            st.info("No feature importance available.")
        else:
            bar_chart(
                importance,
                x="importance",
                y="feature",
                title=f"Top {len(importance)} features (gain)",
                color=PRIMARY,
                x_title="Gain",
            )
    with right:
        with st.spinner(f"Computing SHAP on up to {sample_n:,} offers…"):
            shap_table = mean_abs_shap_table(
                booster,
                explainer,
                frame,
                sample_n=sample_n,
                top_n=top_n,
            )
        if shap_table.empty:
            st.info("SHAP summary unavailable for this lens.")
        else:
            bar_chart(
                shap_table,
                x="mean_abs_shap",
                y="feature",
                title=f"Mean |SHAP| (sample ≤{sample_n})",
                x_title="Mean |SHAP| zł/m²",
            )
            signed = shap_table.copy()
            signed["side"] = np.where(
                signed["mean_shap"] >= 0, "raises ask", "lowers ask"
            )
            diverging_bar_chart(
                signed,
                feature="feature",
                value="mean_shap",
                category="side",
                title="Mean signed SHAP (direction)",
                color_domain=["raises ask", "lowers ask"],
                color_range=[OVERPRICED, UNDERPRICED],
                x_title="Mean SHAP zł/m²",
            )

    cols = model_feature_columns(booster)
    st.caption(
        f"Booster uses **{len(cols)}** features "
        f"(catalog has {len(FEATURE_COLUMNS)}). "
        "Tune sample size in Admin → Stats."
    )

    _section(
        "2 · Example offer explanation",
        "SHAP for the strongest underpriced offer in the lens "
        "(same idea as Explore → Offer → Explain).",
    )
    example = top_value_offers(frame, n=1, side="underpriced")
    if example.empty:
        st.info("No underpriced offer to explain in this lens.")
        return
    listing_id = str(example.iloc[0]["listing_id"])
    st.markdown(f"**Listing** `{listing_id}`")
    row = frame.loc[frame["listing_id"].astype(str) == listing_id]
    if row.empty:
        st.warning("Could not locate the example row in the lens.")
        return
    try:
        features = build_feature_matrix(row, cols)
        explanation = explainer(features)
        values = np.asarray(explanation.values[0], dtype=float)
        names = list(features.columns)
        order = np.argsort(np.abs(values))[::-1][:top_n]
        chart_df = pd.DataFrame({
            "feature": [names[i] for i in order],
            "shap": values[order],
            "side": ["raises ask" if values[i] >= 0 else "lowers ask" for i in order],
        })
        diverging_bar_chart(
            chart_df,
            feature="feature",
            value="shap",
            category="side",
            title="Feature contributions (SHAP)",
            color_domain=["raises ask", "lowers ask"],
            color_range=[OVERPRICED, UNDERPRICED],
            x_title="SHAP contribution to predicted zł/m²",
        )
        if st.button("Open this offer in Explore", key="stats_xai_open_offer"):
            open_explore_offer(listing_id)
    except Exception as exc:
        st.warning(f"SHAP unavailable: {exc}")


def render_stats() -> None:
    """Render filterable Stats analytics for buyers and investors."""
    logger.info("Rendering Stats")
    st.caption(
        "Five focused lenses: inventory → market value → livability "
        "(noise / access / environment) → stock mix → model XAI. "
        "Sidebar filters apply everywhere."
    )

    try:
        frame = append_inactive_otodom_asks(
            drop_heavy_listing_columns(load_scored()),
            processed_dir(),
        )
    except FileNotFoundError as exc:
        logger.warning("Stats missing scored listings: %s", exc)
        st.error(str(exc))
        st.stop()

    settings = load_app_settings(settings_path(processed_dir()))
    state = _sidebar_filters(frame)
    filtered = apply_stats_filters(frame, state)
    logger.debug(
        "Stats lens: %s/%s offers (districts=%s sources=%s valuation=%s)",
        len(filtered),
        len(frame),
        state.districts or "all",
        state.sources or "all",
        state.valuation,
    )
    if filtered.empty:
        st.warning("No offers match the current filters — widen the lens.")
        st.stop()

    _kpi_row(market_kpis(filtered), filtered)
    st.caption(
        f"Lens: **{len(filtered):,}** / {len(frame):,} scored offers · "
        "open outliers on Explore."
    )

    tab_overview, tab_market, tab_live, tab_stock, tab_model = st.tabs(
        ["Overview", "Market", "Livability", "Stock", "Model"],
        key="stats_main_tabs",
        on_change="rerun",
    )
    with tab_overview:
        if tab_overview.open:
            _tab_overview(filtered)
    with tab_market:
        if tab_market.open:
            _tab_market(
                filtered,
                min_offers=settings.stats_min_district_offers,
                top_n=settings.stats_top_n,
            )
    with tab_live:
        if tab_live.open:
            _tab_livability(filtered)
    with tab_stock:
        if tab_stock.open:
            _tab_stock(filtered)
    with tab_model:
        if tab_model.open:
            _tab_model(filtered, settings)
