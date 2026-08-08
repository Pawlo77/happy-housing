"""Stats page: filterable business analytics for Warsaw housing decisions."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from happy_housing.app.app_settings import load_app_settings, settings_path
from happy_housing.app.charts import (
    ASK_COLOR,
    INK_COLOR,
    OVERPRICED,
    PRIMARY,
    SECONDARY,
    UNDERPRICED,
    WARNING,
    bar_chart,
    grouped_bar_chart,
    line_chart,
    multi_line_chart,
    scatter_chart,
    stacked_histogram,
)
from happy_housing.app.data import load_scored, processed_dir
from happy_housing.app.stats_filters import StatsFilterState, apply_stats_filters
from happy_housing.app.stats_insights import (
    affordability_bands,
    aircraft_noise_ladder,
    ask_vs_ink,
    chopin_surcharge,
    commute_cost_curve,
    delta_volcano,
    district_scoreboard,
    market_kpis,
    market_segment_mix,
    market_temperature,
    metro_price_curve,
    monthly_trends,
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
from happy_housing.app.ui_help import STATS_FILTER_HELP

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

    return StatsFilterState(
        districts=list(picked_districts),
        sources=list(picked_sources),
        months=list(picked_months),
        oou=str(oou),
        valuation=str(valuation),
        price_min=float(price_range[0]),
        price_max=float(price_range[1]),
        area_min=float(area_range[0]),
        area_max=float(area_range[1]),
        rooms_min=int(rooms_range[0]),
        rooms_max=int(rooms_range[1]),
    )


def _kpi_row(kpis: dict[str, float]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
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


def _tab_overview(frame: pd.DataFrame) -> None:
    _section(
        "Inventory mix",
        "Proptech-style supply snapshot — where listings cluster by price, "
        "rooms, and source (ask vs ink).",
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
            color=SECONDARY,
            x_title="Median delta zł/m²",
        )

    c1, c2 = st.columns(2)
    with c1:
        rooms = rooms_inventory(frame)
        bar_chart(
            rooms,
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
                color=WARNING,
                x_title="zł/m²",
            )
            show = src.copy()
            show["share"] = (show["share"] * 100).round(0)
            st.dataframe(
                show,
                hide_index=True,
                width="stretch",
                column_config={
                    "share": st.column_config.NumberColumn("Share", format="%.0f%%"),
                    "median_ppm": st.column_config.NumberColumn(
                        "Median zł/m²", format="%.0f"
                    ),
                    "median_delta": st.column_config.NumberColumn(
                        "Median delta", format="%+.0f"
                    ),
                },
            )

    seg = market_segment_mix(frame)
    _section(
        "Primary vs secondary",
        "New-build share vs resale — useful when comparing ask premiums.",
    )
    if seg.empty:
        st.info("No market_primary flag in the scored frame.")
    else:
        bar_chart(
            seg,
            x="n",
            y="segment",
            title="Offers by market segment",
            color=SECONDARY,
            x_title="Offers",
        )


def _tab_trends(frame: pd.DataFrame) -> None:
    _section(
        "Price & fair-value trends",
        "How listing and model medians move month-to-month in the current lens.",
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
                color=SECONDARY,
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
        "Residual pressure",
        "Distribution of listing - model zł/m² — where the market sits vs fair value.",
    )
    volcano = delta_volcano(frame)
    stacked_histogram(
        volcano,
        value="delta",
        category="side",
        title="Delta distribution",
        color_domain=["underpriced", "overpriced"],
        color_range=[UNDERPRICED, OVERPRICED],
        x_title="Delta zł/m²",
    )


def _tab_value(frame: pd.DataFrame, *, min_offers: int, top_n: int) -> None:
    _section(
        "District value board",
        "Compare ask vs model by district. Negative median delta → model sees "
        "more bargains.",
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
                title="Median delta by district (most underpriced first)",
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
        "Ask vs ink",
        "Otodom asking medians versus RCN transaction medians — listing premium.",
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

    _section(
        "Hunt list",
        "Strongest model-underpriced offers in the lens — "
        "open them on Explore → Offer.",
    )
    hunt = top_value_offers(frame, n=top_n, side="underpriced")
    if hunt.empty:
        st.info("No underpriced offers in the current lens.")
    else:
        st.dataframe(
            hunt,
            hide_index=True,
            width="stretch",
            column_config={
                "price_per_m2": st.column_config.NumberColumn(
                    "Listing zł/m²", format="%.0f"
                ),
                "model_price_per_m2": st.column_config.NumberColumn(
                    "Model zł/m²", format="%.0f"
                ),
                "delta_price_per_m2": st.column_config.NumberColumn(
                    "Delta", format="%+.0f"
                ),
                "commute_min": st.column_config.NumberColumn(
                    "Commute min", format="%.0f"
                ),
            },
        )

    _section(
        "Watch list",
        "Most overpriced vs the model — useful for negotiation or avoidance.",
    )
    watch = top_value_offers(frame, n=top_n, side="overpriced")
    if watch.empty:
        st.info("No overpriced offers in the current lens.")
    else:
        st.dataframe(watch, hide_index=True, width="stretch")


def _tab_place(frame: pd.DataFrame) -> None:
    chopin = chopin_surcharge(frame)
    transit = transit_gravity(frame)

    _section(
        "Location premiums",
        "Commute, transit, metro, and aircraft-noise bands — classic livability "
        "price curves from market-intelligence dashboards.",
    )
    a, b, c, d = st.columns(4)
    surcharge = chopin["surcharge_pct"]
    a.metric(
        "Chopin surcharge",
        f"{surcharge:+.1f}%" if surcharge == surcharge else "—",
        help="Median zł/m² inside Lden ≥55 dB vs outside.",
    )
    b.metric(
        "Inside Lden≥55 median",
        f"{chopin['median_inside']:,.0f}"
        if chopin["median_inside"] == chopin["median_inside"]
        else "—",
    )
    c.metric(
        "Transit ≤300 m",
        f"{100 * transit['share_within']:.0f}%"
        if transit["share_within"] == transit["share_within"]
        else "—",
    )
    d.metric("Lden≥55 sample", f"{int(chopin['n_inside']):,}")

    left, right = st.columns(2)
    with left:
        line_chart(
            commute_cost_curve(frame),
            x="commute_bin",
            y="median_price_per_m2",
            title="Median zł/m² by commute to Daszyńskiego",
            y_title="zł/m²",
        )
    with right:
        line_chart(
            transit_price_curve(frame),
            x="transit_band",
            y="median_price_per_m2",
            title="Median zł/m² by transit-stop distance",
            color=SECONDARY,
            y_title="zł/m²",
        )

    m1, m2 = st.columns(2)
    with m1:
        line_chart(
            metro_price_curve(frame),
            x="metro_band",
            y="median_price_per_m2",
            title="Median zł/m² by metro distance",
            color=PRIMARY,
            y_title="zł/m²",
        )
    with m2:
        line_chart(
            aircraft_noise_ladder(frame),
            x="lden_band",
            y="median_price_per_m2",
            title="Median zł/m² by aircraft Lden band",
            color=WARNING,
            y_title="zł/m²",
        )

    line_chart(
        safety_radius(frame),
        x="fire_band",
        y="median_price_per_m2",
        title="Median zł/m² by fire-station distance",
        color=SECONDARY,
        y_title="zł/m²",
    )

    _section(
        "District mood",
        "Bubbles: x = median ask, y = median delta, size = offers, "
        "colour = residual IQR.",
    )
    temp = market_temperature(frame)
    scatter_chart(
        temp,
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
            color=SECONDARY,
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


def _tab_afford(frame: pd.DataFrame) -> None:
    _section(
        "Budget ladder",
        "What total asking price buys — area and zł/m² by budget band "
        "(affordability view common on buyer dashboards).",
    )
    afford = affordability_bands(frame)
    if afford.empty:
        st.info("Need asking prices (price_pln) in the lens.")
        return
    left, right = st.columns(2)
    with left:
        bar_chart(
            afford,
            x="n",
            y="price_band",
            title="Offers by total price band",
            color=PRIMARY,
            x_title="Offers",
        )
    with right:
        bar_chart(
            afford,
            x="median_area_m2",
            y="price_band",
            title="Median area by budget",
            color=SECONDARY,
            x_title="m²",
        )
    line_chart(
        afford,
        x="price_band",
        y="median_ppm",
        title="Median zł/m² by budget band",
        color=WARNING,
        y_title="zł/m²",
    )
    show = afford.copy()
    show["share"] = (show["share"] * 100).round(0)
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "share": st.column_config.NumberColumn("Share", format="%.0f%%"),
            "median_area_m2": st.column_config.NumberColumn("Median m²", format="%.0f"),
            "median_ppm": st.column_config.NumberColumn("Median zł/m²", format="%.0f"),
        },
    )


def render_stats() -> None:
    """Render filterable Stats analytics for buyers and investors."""
    logger.info("Rendering Stats")
    st.title("Market analytics")
    st.caption(
        "Inventory, trends, district value, livability premiums, stock mix, and "
        "affordability — sidebar filters apply to every chart."
    )

    try:
        frame = load_scored()
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

    _kpi_row(market_kpis(filtered))
    st.caption(
        f"Lens: **{len(filtered):,}** / {len(frame):,} scored offers · "
        "open outliers on Explore."
    )

    (
        tab_overview,
        tab_trends,
        tab_value,
        tab_place,
        tab_stock,
        tab_afford,
    ) = st.tabs(
        [
            "Overview",
            "Trends",
            "Value",
            "Place",
            "Stock",
            "Afford",
        ],
        key="stats_main_tabs",
        on_change="rerun",
    )
    with tab_overview:
        if tab_overview.open:
            _tab_overview(filtered)
    with tab_trends:
        if tab_trends.open:
            _tab_trends(filtered)
    with tab_value:
        if tab_value.open:
            _tab_value(
                filtered,
                min_offers=settings.stats_min_district_offers,
                top_n=settings.stats_top_n,
            )
    with tab_place:
        if tab_place.open:
            _tab_place(filtered)
    with tab_stock:
        if tab_stock.open:
            _tab_stock(filtered)
    with tab_afford:
        if tab_afford.open:
            _tab_afford(filtered)
