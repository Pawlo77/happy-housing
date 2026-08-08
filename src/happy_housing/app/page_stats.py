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
    ask_vs_ink,
    chopin_surcharge,
    commute_cost_curve,
    delta_volcano,
    district_scoreboard,
    market_kpis,
    market_temperature,
    monthly_trends,
    normalize_district,
    room_ladder,
    size_sweet_spot,
    top_value_offers,
    transit_gravity,
    transit_price_curve,
    year_paradox,
)

logger = logging.getLogger(__name__)


def _section(title: str, blurb: str) -> None:
    st.subheader(title)
    st.caption(blurb)


def _unique_sorted(series: pd.Series) -> list[str]:
    return sorted({str(v) for v in series.dropna().astype(str) if str(v)})


def _render_filters(frame: pd.DataFrame) -> StatsFilterState:
    """Top-of-page filter lens for Stats analytics."""
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

    with st.expander("Filters", expanded=False):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            picked_districts = st.multiselect(
                "Districts",
                options=districts,
                default=[],
                key="stats_districts",
            )
        with r1c2:
            picked_sources = st.multiselect(
                "Sources",
                options=sources,
                default=[],
                key="stats_sources",
            )
        with r1c3:
            picked_months = st.multiselect(
                "Listed months",
                options=months,
                default=[],
                key="stats_months",
            )
        with r1c4:
            valuation = st.selectbox(
                "Valuation",
                options=["(all)", "underpriced", "overpriced"],
                key="stats_valuation",
            )

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            oou = st.selectbox(
                "OOU / aircraft zone",
                options=["(all)", "inside", "outside"],
                key="stats_oou",
            )
        with r2c2:
            price_range = st.slider(
                "Listing zł/m²",
                min_value=float(ppm_min),
                max_value=float(max(ppm_max, ppm_min + 1.0)),
                value=(float(ppm_min), float(ppm_max)),
                key="stats_price",
            )
        with r2c3:
            area_range = st.slider(
                "Area m²",
                min_value=float(area_lo),
                max_value=float(max(area_hi, area_lo + 1.0)),
                value=(float(area_lo), float(area_hi)),
                key="stats_area",
            )
        with r2c4:
            rooms_range = st.slider(
                "Rooms",
                min_value=int(rooms_lo),
                max_value=int(max(rooms_hi, rooms_lo)),
                value=(int(rooms_lo), int(rooms_hi)),
                key="stats_rooms",
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
        "What proximity and restricted zones do to zł/m² in this lens.",
    )
    a, b, c, d = st.columns(4)
    surcharge = chopin["surcharge_pct"]
    a.metric(
        "Chopin surcharge",
        f"{surcharge:+.1f}%" if surcharge == surcharge else "—",
        help="Median zł/m² inside OOU vs outside.",
    )
    b.metric(
        "Inside OOU median",
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
    d.metric("OOU sample", f"{int(chopin['n_inside']):,}")

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


def render_stats() -> None:
    """Render filterable Stats analytics for buyers and investors."""
    logger.info("Rendering Stats")
    st.title("Market analytics")
    st.caption(
        "Explore trends, district value, location premiums, and stock mix — "
        "filters apply to every chart below."
    )

    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        logger.warning("Stats missing scored listings: %s", exc)
        st.error(str(exc))
        st.stop()

    settings = load_app_settings(settings_path(processed_dir()))
    state = _render_filters(frame)
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

    tab_trends, tab_value, tab_place, tab_stock = st.tabs([
        "Trends",
        "Value",
        "Place",
        "Stock",
    ])
    with tab_trends:
        _tab_trends(filtered)
    with tab_value:
        _tab_value(
            filtered,
            min_offers=settings.stats_min_district_offers,
            top_n=settings.stats_top_n,
        )
    with tab_place:
        _tab_place(filtered)
    with tab_stock:
        _tab_stock(filtered)
