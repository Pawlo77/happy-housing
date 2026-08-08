"""Stats page: catchy, unconventional market analyses and charts."""

from __future__ import annotations

import altair as alt
import streamlit as st

from happy_housing.app.attribution import render_attribution_footer
from happy_housing.app.charts import bar_chart, line_chart
from happy_housing.app.data import load_scored
from happy_housing.app.stats_insights import (
    ask_vs_ink,
    bargain_density,
    chopin_surcharge,
    commute_cost_curve,
    delta_volcano,
    market_temperature,
    room_ladder,
    safety_radius,
    size_sweet_spot,
    transit_gravity,
    year_paradox,
)


def _section(title: str, blurb: str) -> None:
    st.subheader(title)
    st.caption(blurb)


def render_stats() -> None:
    """Render the Stats laboratory of market oddities."""
    st.title("Market oddities")
    st.caption(
        "Unconventional cuts of the scored Warsaw universe — not dashboards, "
        "stories with charts."
    )

    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    chopin = chopin_surcharge(frame)
    transit = transit_gravity(frame)
    volcano = delta_volcano(frame)
    n_over = int((volcano["side"] == "overpriced").sum()) if not volcano.empty else 0
    n_under = int((volcano["side"] == "underpriced").sum()) if not volcano.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        surcharge = chopin["surcharge_pct"]
        st.metric(
            "Chopin surcharge",
            f"{surcharge:+.1f}%" if surcharge == surcharge else "—",
            help="Median zł/m² inside OOU vs outside.",
        )
    with c2.container(border=True):
        st.metric(
            "Transit gravity",
            f"{100 * transit['share_within']:.0f}%",
            help=f"Share of offers within {transit['threshold_m']:.0f}m of a stop.",
        )
    with c3.container(border=True):
        st.metric(
            "Green pins", f"{n_under:,}", help="Model says underpriced (delta < 0)."
        )
    with c4.container(border=True):
        st.metric("Red pins", f"{n_over:,}", help="Model says overpriced (delta ≥ 0).")

    tab_money, tab_place, tab_shape, tab_era = st.tabs([
        "Money games",
        "Place games",
        "Shape games",
        "Time games",
    ])

    with tab_money:
        _section(
            "Bargain dens",
            "Where underpriced offers cluster — high bargain share means the model "
            "thinks the street still has juice.",
        )
        bargains = bargain_density(frame, min_offers=5)
        left, right = st.columns([1.4, 1])
        with left:
            bar_chart(
                bargains.head(12),
                x="bargain_share",
                y="district",
                title="Bargain share by district",
                color="#1B7A4A",
                x_title="Share with delta < 0",
            )
        with right:
            show = bargains.head(12).copy()
            if not show.empty:
                show["bargain_share"] = (show["bargain_share"] * 100).round(0)
            st.dataframe(
                show,
                hide_index=True,
                width="stretch",
                column_config={
                    "bargain_share": st.column_config.NumberColumn(
                        "Bargain %", format="%.0f%%"
                    ),
                    "median_delta": st.column_config.NumberColumn(
                        "Median delta", format="%+.0f"
                    ),
                    "median_ppm": st.column_config.NumberColumn(
                        "Median zł/m²", format="%.0f"
                    ),
                },
            )

        _section(
            "Ask vs ink",
            "Otodom asking medians versus RCN transaction medians — the listing "
            "premium over deals that already closed.",
        )
        gap = ask_vs_ink(frame, min_offers=3)
        if gap.empty:
            st.info("Need overlapping Otodom + RCN districts for this story.")
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
            chart = (
                alt
                .Chart(melt)
                .mark_bar()
                .encode(
                    x=alt.X("median_ppm:Q", title="Median zł/m²"),
                    y=alt.Y("district:N", sort="-x"),
                    color=alt.Color(
                        "kind:N",
                        scale=alt.Scale(
                            domain=["Otodom ask", "RCN ink"],
                            range=["#1B5E52", "#B54708"],
                        ),
                    ),
                    tooltip=["district", "kind", "median_ppm"],
                )
                .properties(height=max(240, 34 * gap["district"].nunique()))
            )
            st.altair_chart(chart, width="stretch")
            st.dataframe(
                gap.assign(ask_premium_pct=lambda d: d["ask_premium_pct"].round(1)),
                hide_index=True,
                width="stretch",
            )

        _section(
            "Delta volcano",
            "The market's over/under pressure — two slopes of the residual mountain.",
        )
        if volcano.empty:
            st.info("No residuals to erupt.")
        else:
            hist = (
                alt
                .Chart(volcano)
                .mark_bar(opacity=0.85)
                .encode(
                    x=alt.X("delta:Q", bin=alt.Bin(maxbins=40), title="Delta zł/m²"),
                    y=alt.Y("count()", title="Offers"),
                    color=alt.Color(
                        "side:N",
                        scale=alt.Scale(
                            domain=["underpriced", "overpriced"],
                            range=["#1B7A4A", "#B42318"],
                        ),
                    ),
                    tooltip=["side", alt.Tooltip("count()", title="Offers")],
                )
                .properties(height=280)
            )
            st.altair_chart(hist, width="stretch")

    with tab_place:
        _section(
            "Chopin surcharge",
            "Aircraft restricted-use zone (OOU) vs the rest of the city — is the "
            "noise priced in, or ignored?",
        )
        a, b, c = st.columns(3)
        a.metric("Median inside OOU", f"{chopin['median_inside']:,.0f} zł/m²")
        b.metric("Median outside OOU", f"{chopin['median_outside']:,.0f} zł/m²")
        c.metric("Inside sample", f"{int(chopin['n_inside']):,}")

        _section(
            "Minutes that cost thousands",
            "Commute minutes to Rondo Daszyńskiego vs median zł/m² — "
            "the price of proximity.",
        )
        line_chart(
            commute_cost_curve(frame),
            x="commute_bin",
            y="median_price_per_m2",
            title="Median zł/m² by commute bin",
        )

        _section(
            "District mood ring",
            "Each bubble is a district: x = median zł/m², y = median model delta, "
            "size = offer count. Polarized markets sit high on residual IQR.",
        )
        temp = market_temperature(frame)
        if temp.empty:
            st.info("Need more districts for the mood ring.")
        else:
            bubbles = (
                alt
                .Chart(temp)
                .mark_circle(opacity=0.75)
                .encode(
                    x=alt.X("median_ppm:Q", title="Median zł/m²"),
                    y=alt.Y("median_delta:Q", title="Median delta (listing - model)"),
                    size=alt.Size(
                        "n:Q", title="Offers", scale=alt.Scale(range=[80, 1200])
                    ),
                    color=alt.Color(
                        "polarization:Q",
                        title="Delta IQR",
                        scale=alt.Scale(scheme="teals"),
                    ),
                    tooltip=[
                        "district",
                        "n",
                        "median_ppm",
                        "median_delta",
                        "polarization",
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(bubbles, width="stretch")

        _section(
            "Safety radius",
            "Does living closer to a fire station cost more — "
            "or is 'safe' just central?",
        )
        line_chart(
            safety_radius(frame),
            x="fire_band",
            y="median_price_per_m2",
            title="Median zł/m² by fire-station distance",
        )

    with tab_shape:
        _section(
            "The small-flat trap",
            "Residuals by size — micro flats often look 'expensive per metre' even "
            "when the model agrees.",
        )
        bar_chart(
            size_sweet_spot(frame),
            x="median_delta",
            y="area_bin",
            title="Median delta by area bucket",
            color="#2F6F7E",
            x_title="Median delta zł/m²",
        )

        _section(
            "Room ladder",
            "Median zł/m² by room count — when an extra room dilutes the metre price.",
        )
        line_chart(
            room_ladder(frame),
            x="rooms",
            y="median_price_per_m2",
            title="Median zł/m² by rooms",
        )

        with st.container(border=True):
            st.markdown("**Transit gravity well**")
            st.write(
                f"{100 * transit['share_within']:.1f}% of scored offers sit within "
                f"{transit['threshold_m']:.0f} m of a GTFS stop "
                f"(median distance "
                f"{transit.get('median_dist_m', float('nan')):,.0f} m)."
            )

    with tab_era:
        _section(
            "Year paradox",
            "Build era vs median zł/m² — vintage charm, PRL hangover, "
            "or new-build tax?",
        )
        bar_chart(
            year_paradox(frame),
            x="median_price_per_m2",
            y="era",
            title="Median zł/m² by build era",
            color="#7A4E3A",
            x_title="Median zł/m²",
        )

        st.caption(
            f"Universe: **{len(frame):,}** scored rows · switch to Explore to "
            "open any outlier on the map."
        )
    render_attribution_footer()
