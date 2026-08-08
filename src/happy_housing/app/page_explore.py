"""Explore page: sidebar filters + Map / Offer / Board sections."""

from __future__ import annotations

import logging

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

from happy_housing.app.ai_review import (
    DEFAULT_AI_MODEL,
    get_or_create_review,
    load_review,
    reviews_db_path,
)
from happy_housing.app.data import (
    load_amenities,
    load_config,
    load_gios_stations,
    load_gtfs_hourly,
    load_gtfs_stops,
    load_gunb_points,
    load_oou,
    load_scored,
    load_shap_bundle,
    load_shops,
    load_traffic_sample,
    processed_dir,
)
from happy_housing.app.explore_nav import (
    EXPLORE_SECTIONS,
    listing_id_from_board_selection,
)
from happy_housing.app.filters import FilterState, apply_filters
from happy_housing.app.map_layers import LayerFlags
from happy_housing.app.maps import (
    basemap_style,
    build_explore_layers,
    focus_marker_layer,
    render_deck,
)
from happy_housing.app.offer_display import (
    delta_hex,
    livability_rows,
    offer_fact_rows,
)
from happy_housing.ingest.otodom import fetch_listing_detail, otodom_session
from happy_housing.models.train import build_feature_matrix

logger = logging.getLogger(__name__)

_BOARD_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "title",
    "district",
    "source",
    "area_m2",
    "rooms",
    "price_pln",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "commute_min",
    "in_oou",
    "listed_at",
    "url",
)
"""Column order for the Explore board table when present."""


def _sidebar_filters(frame: pd.DataFrame) -> FilterState:
    """Render Explore filters in the sidebar and return state."""
    st.sidebar.header("Filters")
    districts = sorted(
        frame["district"].dropna().astype(str).unique().tolist()
        if "district" in frame.columns
        else []
    )
    sources = sorted(
        frame["source"].dropna().astype(str).unique().tolist()
        if "source" in frame.columns
        else []
    )
    months = sorted([
        m
        for m in frame
        .get("listed_month", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if m not in {"NaT", "unknown"}
    ])
    ppm_min = int(frame["price_per_m2"].min()) if not frame.empty else 0
    ppm_max = int(frame["price_per_m2"].max()) + 1 if not frame.empty else 1

    district = st.sidebar.selectbox("District", ["(all)", *districts])
    source = st.sidebar.selectbox("Source", ["(all)", *sources])
    oou = st.sidebar.selectbox(
        "OOU / aircraft zone",
        ["(all)", "inside", "outside"],
        help="Chopin restricted-use zone — noise/livability proxy, not measured dB.",
    )
    max_ppm = st.sidebar.slider(
        "Max listing zł/m²",
        min_value=ppm_min,
        max_value=max(ppm_max, ppm_min + 1),
        value=max(ppm_max, ppm_min + 1),
    )
    month = "(all)"
    if months:
        month = st.sidebar.select_slider(
            "Listed month",
            options=["(all)", *months],
            value="(all)",
        )
    return FilterState(
        district=district,
        source=source,
        oou=oou,
        max_price_per_m2=float(max_ppm),
        month=month,
    )


def _sidebar_map_controls() -> tuple[LayerFlags, int, str]:
    """Map basemap, hour-of-day, and per-layer toggles."""
    st.sidebar.header("Map layers")
    basemap = (
        st.sidebar.segmented_control(
            "Basemap",
            options=["Streets", "Satellite"],
            default="Streets",
            key="explore_basemap",
        )
        or "Streets"
    )
    hour = int(
        st.sidebar.slider(
            "Time of day (transit & traffic)",
            min_value=0,
            max_value=23,
            value=8,
            format="%02d:00",
            help=(
                "Filters GTFS departures by hour and scales the traffic-intensity "
                "proxy (typical diurnal curve — not live vehicle counts)."
            ),
        )
    )
    st.sidebar.caption("Toggle overlays. Hover any pin for details.")
    flags = LayerFlags(
        offers=st.sidebar.checkbox("Offers (Δ price)", value=True),
        oou=st.sidebar.checkbox("OOU / noise proxy zone", value=True),
        transit=st.sidebar.checkbox("Public transport (GTFS)", value=False),
        traffic=st.sidebar.checkbox("Traffic intensity proxy", value=False),
        shops=st.sidebar.checkbox("Shops", value=False),
        hospitals=st.sidebar.checkbox("Hospitals", value=False),
        fire_stations=st.sidebar.checkbox("Fire stations", value=False),
        police=st.sidebar.checkbox("Police", value=False),
        air=st.sidebar.checkbox("Air-quality stations", value=False),
        gunb=st.sidebar.checkbox("GUNB permits", value=False),
        commute_target=st.sidebar.checkbox("Commute target", value=False),
    )
    return flags, hour, basemap


def _tab_map(
    filtered: pd.DataFrame,
    *,
    oou: gpd.GeoDataFrame | None,
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    flags: LayerFlags,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
) -> None:
    """Market map with toggleable livability overlays."""
    st.caption(
        "Offers: green = model says underpriced, red = overpriced. "
        "Transit uses real GTFS departures at the selected hour. "
        "Traffic is a diurnal mobility proxy (not live sensors). "
        "OOU = aircraft/noise zone proxy. Hover pins for details."
    )
    if filtered.empty and flags.offers:
        st.warning("No offers match filters — disable Offers or widen filters.")
    layers = build_explore_layers(
        listings=filtered,
        flags=flags,
        oou=oou,
        amenities=amenities,
        shops=shops,
        stops=stops,
        hourly=hourly,
        traffic_sample=traffic_sample,
        air_stations=air_stations,
        gunb=gunb,
        commute_lon=commute_lon,
        commute_lat=commute_lat,
        hour=hour,
    )
    if not layers:
        st.info("Enable at least one map layer in the sidebar.")
        return
    if not filtered.empty:
        lat = float(filtered["lat"].mean())
        lon = float(filtered["lon"].mean())
    else:
        lat, lon = commute_lat, commute_lon
    render_deck(
        layers,
        latitude=lat,
        longitude=lon,
        zoom=10.5,
        map_style=basemap_style(basemap),
    )


def _ensure_description(row: pd.Series) -> str:
    """Return cached description or fetch from Otodom when a slug exists."""
    existing = row.get("description")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    slug = row.get("slug")
    if not isinstance(slug, str) or not slug.strip():
        return ""
    try:
        detail = fetch_listing_detail(otodom_session(), slug.strip())
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.warning("Description fetch failed for %s: %s", slug, exc)
        return ""
    text = detail.get("description")
    return text.strip() if isinstance(text, str) else ""


def _price_metrics(row: pd.Series) -> None:
    """Listing / model / coloured delta metrics."""
    listing = float(row["price_per_m2"])
    model = float(row["model_price_per_m2"])
    delta = float(row["delta_price_per_m2"])
    color = delta_hex(delta)
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.metric("Listing zł/m²", f"{listing:,.0f}")
    with c2.container(border=True):
        st.metric("Model zł/m²", f"{model:,.0f}")
    with c3.container(border=True):
        st.markdown("**Delta (listing - model)**")
        st.markdown(
            f"<p style='color:{color};font-size:1.75rem;font-weight:700;"
            f"margin:0'>{delta:+,.0f}</p>"
            f"<p style='color:#6B7280;font-size:0.85rem;margin:0'>"
            "green underpriced · yellow fair · red overpriced</p>",
            unsafe_allow_html=True,
        )


def _trigger_ai_review(row: pd.Series) -> None:
    """Load or generate AI review when Overview opens; cache in DuckDB."""
    db = reviews_db_path(processed_dir())
    listing_id = str(row.get("listing_id") or "")
    cached = load_review(db, listing_id, model=DEFAULT_AI_MODEL)
    if cached is not None:
        st.session_state[f"ai_review_{listing_id}"] = cached.review_text
        return
    with st.spinner(f"Asking {DEFAULT_AI_MODEL} for an offer review…"):
        try:
            review = get_or_create_review(db, row, model=DEFAULT_AI_MODEL)
            st.session_state[f"ai_review_{listing_id}"] = review.review_text
        except Exception as exc:
            logger.warning("AI review failed for %s: %s", listing_id, exc)
            st.session_state[f"ai_review_{listing_id}"] = ""
            st.session_state[f"ai_review_error_{listing_id}"] = str(exc)


def _render_shap(row_frame: pd.DataFrame) -> None:
    """Readable SHAP contribution chart (horizontal bars)."""
    bundle = load_shap_bundle()
    if bundle is None:
        st.info("Model file missing — retrain from Admin.")
        return
    try:
        _booster, explainer = bundle
        features = build_feature_matrix(row_frame)
        explanation = explainer(features)
        values = np.asarray(explanation.values[0], dtype=float)
        names = list(features.columns)
        order = np.argsort(np.abs(values))[::-1]
        chart_df = pd.DataFrame({
            "feature": [names[i] for i in order],
            "shap": values[order],
            "side": ["raises ask" if values[i] >= 0 else "lowers ask" for i in order],
        })
        chart = (
            alt
            .Chart(chart_df)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("shap:Q", title="SHAP contribution to predicted zł/m²"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color(
                    "side:N",
                    scale=alt.Scale(
                        domain=["raises ask", "lowers ask"],
                        range=["#B91C1C", "#15803D"],
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["feature", "shap", "side"],
            )
            .properties(height=max(280, 28 * len(chart_df)))
        )
        st.altair_chart(chart, width="stretch")
        with st.expander("Classic waterfall", expanded=False):
            fig, _ax = plt.subplots(figsize=(9, 5.5))
            shap.plots.waterfall(explanation[0], show=False, max_display=12)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    except Exception as exc:
        st.warning(f"SHAP unavailable: {exc}")


def _tab_offer(
    filtered: pd.DataFrame,
    *,
    oou: gpd.GeoDataFrame | None,
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    flags: LayerFlags,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
) -> None:
    """Single-offer deep dive with Overview / Context / Map / SHAP / AI subtabs."""
    if filtered.empty:
        st.warning("No offers match filters.")
        return
    labels = filtered["listing_id"].astype(str).tolist()
    preferred = st.session_state.get("explore_offer_id")
    if "title" in filtered.columns:
        options = {
            f"{lid} — {title}" if isinstance(title, str) and title else lid: lid
            for lid, title in zip(
                filtered["listing_id"],
                filtered["title"],
                strict=True,
            )
        }
        keys = list(options.keys())
        index = 0
        if preferred in labels:
            for i, key in enumerate(keys):
                if options[key] == preferred:
                    index = i
                    break
        choice = st.selectbox("Offer", keys, index=index)
        selected_id = options[choice]
    else:
        index = labels.index(preferred) if preferred in labels else 0
        selected_id = st.selectbox("Offer", labels, index=index)
    st.session_state["explore_offer_id"] = selected_id

    row = filtered.loc[filtered["listing_id"] == selected_id].iloc[0].copy()
    description = _ensure_description(row)
    if description:
        row["description"] = description

    overview, context, local_map, explain, ai_tab = st.tabs([
        "Overview",
        "Context",
        "Map",
        "Explain",
        "AI review",
    ])

    with overview:
        _trigger_ai_review(row)
        left, right = st.columns([1.1, 1.6])
        with left:
            if isinstance(row.get("image_url"), str) and row["image_url"]:
                st.image(row["image_url"], use_container_width=True)
            if isinstance(row.get("url"), str) and row["url"]:
                st.link_button("Open source listing", str(row["url"]), width="stretch")
        with right:
            title = row.get("title") or row.get("listing_id")
            st.subheader(str(title))
            _price_metrics(row)
            st.dataframe(offer_fact_rows(row), hide_index=True, width="stretch")
        st.markdown("#### Source description")
        if description:
            st.write(description)
        else:
            st.info("No listing description available for this offer.")

    with context:
        st.caption(
            "Distances rounded to whole metres. "
            "OOU is a zone flag (aircraft/noise proxy), not measured dB."
        )
        st.dataframe(livability_rows(row), hide_index=True, width="stretch")
        _price_metrics(row)

    with local_map:
        st.caption("Same layer stack as the main Explore map (sidebar toggles apply).")
        lat, lon = float(row["lat"]), float(row["lon"])
        layers = build_explore_layers(
            listings=filtered,
            flags=flags,
            oou=oou,
            amenities=amenities,
            shops=shops,
            stops=stops,
            hourly=hourly,
            traffic_sample=traffic_sample,
            air_stations=air_stations,
            gunb=gunb,
            commute_lon=commute_lon,
            commute_lat=commute_lat,
            hour=hour,
        )
        layers.append(focus_marker_layer(lat, lon))
        render_deck(
            layers,
            latitude=lat,
            longitude=lon,
            zoom=13.5,
            map_style=basemap_style(basemap),
        )

    with explain:
        st.caption(
            "Feature contributions to the model zł/m². "
            "Green lowers the fair price; red raises it."
        )
        one = filtered.loc[filtered["listing_id"] == selected_id]
        _render_shap(one)

    with ai_tab:
        listing_id = str(row.get("listing_id") or "")
        db = reviews_db_path(processed_dir())
        text = st.session_state.get(f"ai_review_{listing_id}", "")
        err = st.session_state.get(f"ai_review_error_{listing_id}", "")
        st.caption(f"Model: `{DEFAULT_AI_MODEL}` · cached in DuckDB after first ask.")
        if text:
            st.markdown(text)
        elif err:
            st.warning(f"AI review unavailable: {err}")
        else:
            st.info("Open Overview to request a review, or regenerate below.")
        if st.button("Regenerate AI review", key=f"regen_{listing_id}"):
            with st.spinner(f"Re-asking {DEFAULT_AI_MODEL}…"):
                try:
                    review = get_or_create_review(
                        db, row, model=DEFAULT_AI_MODEL, force=True
                    )
                    st.session_state[f"ai_review_{listing_id}"] = review.review_text
                    st.session_state.pop(f"ai_review_error_{listing_id}", None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _tab_board(filtered: pd.DataFrame) -> None:
    """Tabular offer board; selecting a row opens Offer for that listing."""
    cols = [c for c in _BOARD_COLUMNS if c in filtered.columns]
    board = filtered[cols].reset_index(drop=True)
    st.caption(f"{len(board)} offers after filters · click a row to open Offer")
    event = st.dataframe(
        board,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="explore_board",
    )
    listing_id = listing_id_from_board_selection(board, event)
    if listing_id is None:
        return
    st.session_state["explore_offer_id"] = listing_id
    st.session_state["explore_section"] = "Offer"
    st.rerun()


def render_explore() -> None:
    """Render Explore with sidebar filters and Map / Offer / Board sections."""
    st.title("Explore")
    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    state = _sidebar_filters(frame)
    flags, hour, basemap = _sidebar_map_controls()
    filtered = apply_filters(frame, state)
    config = load_config()

    oou = load_oou()
    amenities = load_amenities()
    shops = load_shops()
    stops = load_gtfs_stops()
    hourly = load_gtfs_hourly()
    traffic_sample = load_traffic_sample()
    air_stations = load_gios_stations()
    gunb = load_gunb_points()

    if "explore_section" not in st.session_state:
        st.session_state["explore_section"] = "Map"
    section = st.segmented_control(
        "Section",
        options=list(EXPLORE_SECTIONS),
        key="explore_section",
        label_visibility="collapsed",
    )
    if section is None:
        section = "Map"
    map_kwargs = {
        "oou": oou,
        "amenities": amenities,
        "shops": shops,
        "stops": stops,
        "hourly": hourly,
        "traffic_sample": traffic_sample,
        "air_stations": air_stations,
        "gunb": gunb,
        "flags": flags,
        "hour": hour,
        "basemap": basemap,
        "commute_lon": float(config.commute_target_lon),
        "commute_lat": float(config.commute_target_lat),
    }
    if section == "Map":
        _tab_map(filtered, **map_kwargs)
    elif section == "Offer":
        _tab_offer(filtered, **map_kwargs)
    else:
        _tab_board(filtered)
