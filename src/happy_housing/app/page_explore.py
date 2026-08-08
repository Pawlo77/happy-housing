"""Explore page: sidebar filters + Map / Offer / Board / News / Prefs tabs."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

from happy_housing.app.ai_review import (
    get_or_create_review,
    load_review,
    review_context_hash,
    reviews_db_path,
)
from happy_housing.app.app_settings import AppSettings, load_app_settings, settings_path
from happy_housing.app.board_sort import (
    BOARD_SORT_OPTIONS,
    BOARD_SORT_RECOMMENDATION,
    coerce_board_sort,
    enrich_board_recommendations,
    sort_board,
)
from happy_housing.app.board_table import (
    board_search_columns,
    filter_dataframe_text,
    prepare_board_display,
)
from happy_housing.app.charts import (
    OVERPRICED,
    UNDERPRICED,
    bar_chart,
    diverging_bar_chart,
)
from happy_housing.app.commute_target import resolve_commute_target
from happy_housing.app.data import (
    load_aircraft_lden,
    load_amenities,
    load_config,
    load_gios_stations,
    load_gtfs_hourly,
    load_gtfs_route_shapes,
    load_gtfs_stops,
    load_gunb_points,
    load_parks,
    load_scored,
    load_shap_bundle,
    load_shops,
    load_traffic_sample,
    processed_dir,
)
from happy_housing.app.explore_nav import (
    EXPLORE_SECTIONS,
    OFFER_SELECT_KEY,
    listing_id_from_board_selection,
    prepare_offer_select_state,
)
from happy_housing.app.filters import FilterState, apply_filters, apply_starred_only
from happy_housing.app.llm_metrics import llm_metrics_path
from happy_housing.app.map_layers import (
    MAX_GUNB_MAP_POINTS,
    LayerFlags,
)
from happy_housing.app.map_marker_shapes import map_shape_legend_caption
from happy_housing.app.maps import (
    basemap_style,
    build_explore_layers,
    build_spatial_layers,
    focus_marker_layer,
    render_deck,
)
from happy_housing.app.offer_display import (
    livability_rows,
    offer_fact_rows,
)
from happy_housing.app.pref_llm import merge_preferences, translate_preferences_nl
from happy_housing.app.preferences import (
    OOU_ANY,
    OOU_INSIDE_OK,
    OOU_OUTSIDE_ONLY,
    OOU_PREFER_OUTSIDE,
    UserPreferences,
    apply_preferences,
    load_preferences,
    preferences_path,
    save_preferences,
)
from happy_housing.app.recommender import (
    impressions_path,
    load_impressions,
    mark_opened,
    record_impressions,
    retrain_recommender,
    top_recommendations,
)
from happy_housing.app.spatial_insights import (
    SPATIAL_METRICS,
    district_spatial_summary,
    get_spatial_metric,
    spatial_kpis,
)
from happy_housing.app.star_news import (
    StarNewsItem,
    format_change_markdown,
    load_news,
    save_news,
    sync_starred_news,
)
from happy_housing.app.star_news import (
    news_path as starred_news_path,
)
from happy_housing.app.star_news import (
    snapshots_path as starred_snapshots_path,
)
from happy_housing.app.stars import (
    is_starred,
    load_stars,
    star_marker,
    stars_path,
    toggle_star,
)
from happy_housing.app.ui_help import (
    BASEMAP_HELP,
    FILTER_HELP,
    PREF_HELP,
    SPATIAL_METRIC_HELP,
    layer_help,
)
from happy_housing.ingest.otodom import fetch_listing_detail, otodom_session
from happy_housing.models.train import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    model_feature_columns,
)

logger = logging.getLogger(__name__)

_BOARD_COLUMNS: tuple[str, ...] = (
    "starred",
    "seen",
    "confidence",
    "listing_id",
    "title",
    "district",
    "source",
    "market",
    "area_m2",
    "rooms",
    "floor_no",
    "year_built",
    "price_pln",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "has_balcony",
    "has_garage",
    "has_lift",
    "commute_min",
    "in_oou",
    "listed_at",
    "url",
)
"""Column order for the Explore board table when present."""


def _sidebar_filters(frame: pd.DataFrame) -> tuple[FilterState, bool]:
    """Render Explore filters in the sidebar; return state + starred-only flag."""
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
    ppm = pd.to_numeric(frame.get("price_per_m2"), errors="coerce")
    finite = ppm[ppm.notna()]
    ppm_min = int(finite.min()) if not finite.empty else 0
    ppm_max = int(finite.max()) + 1 if not finite.empty else 1

    district = st.sidebar.selectbox(
        "District",
        ["(all)", *districts],
        help=FILTER_HELP["district"],
        key="explore_district",
    )
    source = st.sidebar.selectbox(
        "Source",
        ["(all)", *sources],
        help=FILTER_HELP["source"],
        key="explore_source",
    )
    oou = st.sidebar.selectbox(
        "Aircraft Lden ≥ 55 dB",
        ["(all)", "inside", "outside"],
        help=FILTER_HELP["oou"],
        key="explore_oou",
    )
    max_ppm = st.sidebar.slider(
        "Max listing zł/m²",
        min_value=ppm_min,
        max_value=max(ppm_max, ppm_min + 1),
        value=max(ppm_max, ppm_min + 1),
        help=FILTER_HELP["max_ppm"],
        key="explore_max_ppm",
    )
    month = "(all)"
    if months:
        month = st.sidebar.select_slider(
            "Listed month",
            options=["(all)", *months],
            value="(all)",
            help=FILTER_HELP["month"],
            key="explore_month",
        )
    starred_only = st.sidebar.checkbox(
        "Starred only",
        value=False,
        key="explore_starred_only",
        help=(
            "Map and Board show only starred offers. "
            "Offer tab stays unfiltered by star."
        ),
    )
    return (
        FilterState(
            district=district,
            source=source,
            oou=oou,
            max_price_per_m2=float(max_ppm),
            month=month,
        ),
        starred_only,
    )


def _sidebar_map_controls() -> tuple[LayerFlags, int, str]:
    """Map basemap, hour-of-day, and per-layer toggles."""
    st.sidebar.header("Map layers")
    basemap = (
        st.sidebar.segmented_control(
            "Basemap",
            options=["Streets", "Satellite"],
            default="Streets",
            help=BASEMAP_HELP,
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
                "proxy (typical diurnal curve — not live vehicle counts). "
                "Peak hours ≈ noisier corridors for living comfort."
            ),
            key="explore_hour",
        )
    )
    st.sidebar.caption(
        "Toggle overlays (hover ? for what / price / livability). "
        "Hover map pins for details."
    )
    flags = LayerFlags(
        offers=st.sidebar.checkbox(
            "Offers (Δ price)",
            value=True,
            help=layer_help("offers"),
            key="explore_ly_offers",
        ),
        oou=st.sidebar.checkbox(
            "Aircraft Lden contours",
            value=True,
            help=layer_help("oou"),
            key="explore_ly_oou",
        ),
        transit_intensity=st.sidebar.checkbox(
            "Transit intensity (hourly)",
            value=False,
            help=layer_help("transit_intensity"),
            key="explore_ly_transit_intensity",
        ),
        transit_network=st.sidebar.checkbox(
            "Transit lines & stops",
            value=False,
            help=layer_help("transit_network"),
            key="explore_ly_transit_network",
        ),
        transit_show_metro=st.sidebar.checkbox(
            "  Metro lines",
            value=True,
            help=layer_help("transit_show_metro"),
            key="explore_ly_transit_metro",
        ),
        transit_show_tram=st.sidebar.checkbox(
            "  Tram lines",
            value=True,
            help=layer_help("transit_show_tram"),
            key="explore_ly_transit_tram",
        ),
        transit_show_bus=st.sidebar.checkbox(
            "  Bus lines",
            value=True,
            help=layer_help("transit_show_bus"),
            key="explore_ly_transit_bus",
        ),
        transit_show_rail=st.sidebar.checkbox(
            "  Train / rail lines",
            value=True,
            help=layer_help("transit_show_rail"),
            key="explore_ly_transit_rail",
        ),
        traffic=st.sidebar.checkbox(
            "Traffic intensity proxy",
            value=False,
            help=layer_help("traffic"),
            key="explore_ly_traffic",
        ),
        shops=st.sidebar.checkbox(
            "Shops",
            value=False,
            help=layer_help("shops"),
            key="explore_ly_shops",
        ),
        hospitals=st.sidebar.checkbox(
            "Hospitals",
            value=False,
            help=layer_help("hospitals"),
            key="explore_ly_hospitals",
        ),
        fire_stations=st.sidebar.checkbox(
            "Fire stations",
            value=False,
            help=layer_help("fire_stations"),
            key="explore_ly_fire",
        ),
        police=st.sidebar.checkbox(
            "Police",
            value=False,
            help=layer_help("police"),
            key="explore_ly_police",
        ),
        air=st.sidebar.checkbox(
            "Air-quality stations",
            value=False,
            help=layer_help("air"),
            key="explore_ly_air",
        ),
        gunb=st.sidebar.checkbox(
            "GUNB permits",
            value=False,
            help=layer_help("gunb"),
            key="explore_ly_gunb",
        ),
        parks=st.sidebar.checkbox(
            "Parks & squares (UM)",
            value=False,
            help=layer_help("parks"),
            key="explore_ly_parks",
        ),
        commute_target=st.sidebar.checkbox(
            "Commute target",
            value=False,
            help=layer_help("commute_target"),
            key="explore_ly_commute",
        ),
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
    route_shapes: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    parks: gpd.GeoDataFrame,
    flags: LayerFlags,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
    commute_label: str = "Commute target",
) -> None:
    """Market map with toggleable livability overlays."""
    st.caption(
        "Offers: green = model says underpriced, red = overpriced. "
        "Transit uses real GTFS departures at the selected hour. "
        "Traffic is a diurnal mobility proxy (not live sensors). "
        "Parks from UM Warszawa PLACE_SKWERY (testmapa). "
        "Lden bands from EEA END major-airport contours. Hover pins for details. "
        f"GUNB shows a Warsaw sample (max {MAX_GUNB_MAP_POINTS:,} points) so the "
        "map stays under Streamlit message limits. "
        "Transit lines: metro red · tram dark red · bus blue · rail green. "
        + map_shape_legend_caption()
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
        route_shapes=route_shapes,
        traffic_sample=traffic_sample,
        air_stations=air_stations,
        gunb=gunb,
        parks=parks,
        commute_lon=commute_lon,
        commute_lat=commute_lat,
        hour=hour,
        commute_label=commute_label,
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


def _tab_spatial(
    filtered: pd.DataFrame,
    *,
    traffic_sample: pd.DataFrame,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
) -> None:
    """Spatial analytics: metric heatmaps + district charts (not map-only)."""
    st.caption(
        "Pick a market signal — map shows spatial intensity; KPIs and district "
        "bars explain where it concentrates. Uses the same sidebar filters as Map."
    )
    labels = [m.label for m in SPATIAL_METRICS]
    keys = [m.key for m in SPATIAL_METRICS]
    prev_label = st.session_state.get("explore_spatial_metric", labels[0])
    prev_key = keys[labels.index(str(prev_label))] if prev_label in labels else keys[0]
    choice = st.selectbox(
        "Spatial metric",
        options=labels,
        index=0,
        key="explore_spatial_metric",
        help=SPATIAL_METRIC_HELP[prev_key],
    )
    metric_key = keys[labels.index(str(choice))]
    metric = get_spatial_metric(metric_key)
    show_hex = st.checkbox(
        "Show hex elevation (3D)",
        value=False,
        key="explore_spatial_hex",
        help=(
            "Extrude hex bins by the same metric weight. Listing metrics only — "
            "disabled for traffic proxy. Visual density ≠ price change by itself."
        ),
        disabled=metric_key == "traffic",
    )

    kpis = spatial_kpis(filtered, metric_key)
    k1, k2, k3 = st.columns(3)
    k1.metric("Mapped points", f"{int(kpis['n_points']):,}")
    headline = float(kpis["headline_value"])
    k2.metric(
        str(kpis["headline"]),
        f"{headline:,.0f}" if headline == headline else "—",
    )
    k3.metric("Districts ranked", f"{int(kpis['districts']):,}")
    st.caption(SPATIAL_METRIC_HELP.get(metric_key, metric.blurb))

    map_col, chart_col = st.columns([1.35, 1.0], gap="large")
    with map_col:
        layers = build_spatial_layers(
            listings=filtered,
            metric_key=metric_key,
            traffic_sample=traffic_sample,
            hour=hour,
            show_hex=bool(show_hex) and metric_key != "traffic",
        )
        if not layers:
            st.info(
                "No spatial points for this metric under current filters "
                "(or traffic sample missing)."
            )
        else:
            if not filtered.empty and {"lat", "lon"} <= set(filtered.columns):
                lat = float(pd.to_numeric(filtered["lat"], errors="coerce").mean())
                lon = float(pd.to_numeric(filtered["lon"], errors="coerce").mean())
                if lat != lat or lon != lon:
                    lat, lon = commute_lat, commute_lon
            else:
                lat, lon = commute_lat, commute_lon
            render_deck(
                layers,
                latitude=lat,
                longitude=lon,
                zoom=10.5 if not show_hex else 11.0,
                map_style=basemap_style(basemap),
            )
    with chart_col:
        if metric_key == "traffic":
            st.info(
                "Traffic uses the OSM diurnal sample — pick hour in the sidebar. "
                "District bars are not defined for this proxy."
            )
            st.metric("Hour", f"{hour:02d}:00")
        else:
            summary = district_spatial_summary(filtered, metric_key)
            if summary.empty:
                st.info("Not enough district coverage for a leaderboard.")
            else:
                bar_chart(
                    summary.head(12),
                    x="value",
                    y="district",
                    title=f"Top districts · {metric.value_title}",
                    x_title=metric.value_title,
                    color=UNDERPRICED,
                )
                st.dataframe(
                    summary.rename(
                        columns={"n": "Offers", "value": metric.value_title}
                    ),
                    width="stretch",
                    hide_index=True,
                )

    st.subheader("How to read this")
    st.markdown(SPATIAL_METRIC_HELP.get(metric_key, metric.blurb))


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _fetch_otodom_description(slug: str) -> str:
    """Cached Otodom detail description fetch keyed by listing slug."""
    detail = fetch_listing_detail(otodom_session(), slug.strip())
    text = detail.get("description")
    return text.strip() if isinstance(text, str) else ""


def _ensure_description(row: pd.Series) -> str:
    """Return stored description or a cached Otodom fetch when a slug exists."""
    existing = row.get("description")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    slug = row.get("slug")
    if not isinstance(slug, str) or not slug.strip():
        return ""
    try:
        return _fetch_otodom_description(slug.strip())  # type: ignore[no-any-return]
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.warning("Description fetch failed for %s: %s", slug, exc)
        return ""


_DELTA_COLORS_HELP = "listing - model zł/m² (negative = model sees a bargain)"
"""Hover-only key for the delta metric (st.metric help)."""


def _price_metrics(row: pd.Series) -> None:
    """Listing / model / delta metrics in equal-height bordered columns."""
    listing_raw = pd.to_numeric(row.get("price_per_m2"), errors="coerce")
    model_raw = pd.to_numeric(row.get("model_price_per_m2"), errors="coerce")
    delta_raw = pd.to_numeric(row.get("delta_price_per_m2"), errors="coerce")
    listing = float(listing_raw) if listing_raw == listing_raw else 0.0
    model = float(model_raw) if model_raw == model_raw else 0.0
    delta = float(delta_raw) if delta_raw == delta_raw else 0.0
    c1, c2, c3 = st.columns(3, border=True)
    with c1:
        st.metric(
            "Listing zł/m²",
            f"{listing:,.0f}",
            help="Asking price per square metre",
            height="stretch",
        )
    with c2:
        st.metric(
            "Model zł/m²",
            f"{model:,.0f}",
            help="Model fair price per square metre",
            height="stretch",
        )
    with c3:
        st.metric(
            "Delta zł/m²",
            f"{delta:+,.0f}",
            delta=float(delta),
            delta_color="inverse",
            help=_DELTA_COLORS_HELP,
            height="stretch",
        )


def _trigger_ai_review(
    row: pd.Series,
    *,
    model: str,
    enabled: bool,
    settings: AppSettings,
) -> None:
    """Load or generate AI review when the AI review tab is open; cache in DuckDB."""
    if not enabled:
        return
    proc = processed_dir()
    db = reviews_db_path(proc)
    listing_id = str(row.get("listing_id") or "")
    ctx = review_context_hash(settings, row)
    cached = load_review(db, listing_id, model=model, context_hash=ctx)
    if cached is not None:
        st.session_state[f"ai_review_{listing_id}"] = cached.review_text
        return
    with st.spinner(f"Asking {model} for an offer review…"):
        try:
            review = get_or_create_review(
                db,
                row,
                model=model,
                settings=settings,
                metrics_path=llm_metrics_path(proc),
            )
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
        booster, explainer = bundle
        cols = model_feature_columns(booster)
        features = build_feature_matrix(row_frame, cols)
        explanation = explainer(features)
        values = np.asarray(explanation.values[0], dtype=float)
        names = list(features.columns)
        order = np.argsort(np.abs(values))[::-1]
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
        if len(cols) < len(FEATURE_COLUMNS):
            st.caption(
                f"Model trained on {len(cols)} features (current catalog has "
                f"{len(FEATURE_COLUMNS)}). Retrain in Admin to include new "
                "signals such as park distance."
            )
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
    starred: frozenset[str],
    stars_file: Path,
    lm_model: str,
    ai_review_on_open: bool,
    app_settings: AppSettings,
    oou: gpd.GeoDataFrame | None,
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    route_shapes: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    parks: gpd.GeoDataFrame,
    flags: LayerFlags,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
    commute_label: str = "Commute target",
) -> None:
    """Single-offer deep dive with Overview / Context / Map / SHAP / AI subtabs."""
    if filtered.empty:
        st.warning("No offers match filters.")
        return
    labels = filtered["listing_id"].astype(str).tolist()
    preferred = st.session_state.get("explore_offer_id")
    force = bool(st.session_state.pop("explore_offer_force", False))
    pick, star_col = st.columns([4.2, 1], vertical_alignment="bottom")
    with pick:
        if "title" in filtered.columns:
            options = {
                f"{lid} — {title}" if isinstance(title, str) and title else lid: lid
                for lid, title in zip(
                    filtered["listing_id"].astype(str),
                    filtered["title"],
                    strict=True,
                )
            }
        else:
            options = {lid: lid for lid in labels}
        desired = prepare_offer_select_state(
            options=options,
            preferred_id=str(preferred) if preferred is not None else None,
            force=force,
            current_label=st.session_state.get(OFFER_SELECT_KEY),
        )
        if desired is not None:
            st.session_state[OFFER_SELECT_KEY] = desired
        choice = st.selectbox("Offer", list(options.keys()), key=OFFER_SELECT_KEY)
        selected_id = options[str(choice)]
    st.session_state["explore_offer_id"] = selected_id
    starred_now = is_starred(starred, selected_id)
    with star_col:
        label = "★ Starred" if starred_now else "☆ Star"
        if st.button(label, key=f"star_btn_{selected_id}", width="stretch"):
            toggle_star(stars_file, selected_id)
            st.rerun()

    row = filtered.loc[filtered["listing_id"] == selected_id].iloc[0].copy()
    description = _ensure_description(row)
    if description:
        row["description"] = description

    overview, context, local_map, explain, ai_tab = st.tabs(
        [
            "Overview",
            "Context",
            "Map",
            "Explain",
            "AI review",
        ],
        key="offer_subtab",
        on_change="rerun",
    )

    with overview:
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
            "Aircraft Lden/Lnight from EEA END contour bands (dB lower edge)."
        )
        st.dataframe(
            livability_rows(row, commute_label=commute_label),
            hide_index=True,
            width="stretch",
        )
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
            route_shapes=route_shapes,
            traffic_sample=traffic_sample,
            air_stations=air_stations,
            gunb=gunb,
            parks=parks,
            commute_lon=commute_lon,
            commute_lat=commute_lat,
            hour=hour,
            commute_label=commute_label,
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
        if ai_tab.open:
            _trigger_ai_review(
                row,
                model=lm_model,
                enabled=ai_review_on_open,
                settings=app_settings,
            )
        text = st.session_state.get(f"ai_review_{listing_id}", "")
        err = st.session_state.get(f"ai_review_error_{listing_id}", "")
        st.caption(f"Model: `{lm_model}` · cached in DuckDB after first ask.")
        if text:
            st.markdown(text)
        elif err:
            st.warning(f"AI review unavailable: {err}")
        elif not ai_review_on_open:
            st.info("Auto-review is off — use Regenerate below to ask the model.")
        else:
            st.info("Switch to this tab to request a review.")
        if st.button("Regenerate AI review", key=f"regen_{listing_id}"):
            with st.spinner(f"Re-asking {lm_model}…"):
                try:
                    review = get_or_create_review(
                        db,
                        row,
                        model=lm_model,
                        force=True,
                        settings=app_settings,
                        metrics_path=llm_metrics_path(processed_dir()),
                    )
                    st.session_state[f"ai_review_{listing_id}"] = review.review_text
                    st.session_state.pop(f"ai_review_error_{listing_id}", None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _tab_board(
    filtered: pd.DataFrame,
    *,
    starred: frozenset[str],
    app_settings: AppSettings,
    impressions_file: Path,
    prefs: UserPreferences,
) -> None:
    """Tabular offer board; selecting a row opens Offer for that listing."""
    impressions = load_impressions(impressions_file)
    enriched = enrich_board_recommendations(
        filtered,
        impressions=impressions,
        demotion_alpha=float(app_settings.recommender_demotion_alpha),
        prefs=prefs,
    )
    labels = [label for _, label in BOARD_SORT_OPTIONS]
    keys = [key for key, _ in BOARD_SORT_OPTIONS]
    default_idx = keys.index(BOARD_SORT_RECOMMENDATION)
    sort_label = st.selectbox(
        "Sort board",
        options=labels,
        index=default_idx,
        key="explore_board_sort",
        help=(
            "Default ranks by recommender score with new / unseen offers first. "
            "Pick another key to reorder by price, district, posted time, etc."
        ),
    )
    sort_key = coerce_board_sort(keys[labels.index(str(sort_label))])
    board = sort_board(enriched, sort_key)
    board = board.copy()
    board.insert(
        0,
        "starred",
        board["listing_id"].astype(str).map(lambda lid: star_marker(lid, starred)),
    )
    cols = [c for c in _BOARD_COLUMNS if c in board.columns]
    board = board[cols]
    search = st.text_input(
        "Search board",
        key="explore_board_search",
        placeholder="id, district, title, price…",
        help=(
            "Case-insensitive substring across listing fields. "
            "Clears when empty. Row click still opens Offer."
        ),
    )
    before_n = len(board)
    board = filter_dataframe_text(board, search, columns=board_search_columns(board))
    board = prepare_board_display(board)
    match_note = (
        f"{len(board)} of {before_n} offers match search"
        if str(search or "").strip()
        else f"{len(board)} offers after filters"
    )
    st.caption(
        f"{match_note} · sort: **{sort_label}** · "
        "click a row to open Offer · star on Offer · "
        "● new / ○ opened · confidence bar · signed delta zł/m²"
    )
    event = st.dataframe(
        board,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="explore_board",
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence",
                help="Recommender score (higher = stronger fit / bargain).",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                color="primary",
            ),
            "seen": st.column_config.TextColumn(
                "seen",
                help="● new · ◌ impressed · ○ opened on Offer.",
            ),
            "delta_price_per_m2": st.column_config.NumberColumn(
                "delta_price_per_m2",
                help="Ask minus model zł/m² (negative = underpriced).",
                format="%+.0f",
            ),
            "price_pln": st.column_config.NumberColumn("price_pln", format="%.0f"),
            "price_per_m2": st.column_config.NumberColumn(
                "price_per_m2", format="%.0f"
            ),
            "model_price_per_m2": st.column_config.NumberColumn(
                "model_price_per_m2", format="%.0f"
            ),
            "url": st.column_config.LinkColumn("url", display_text="Open"),
        },
    )
    listing_id = listing_id_from_board_selection(board, event)
    if listing_id is None:
        return
    mark_opened(impressions_file, listing_id)
    st.session_state["explore_offer_id"] = listing_id
    st.session_state["explore_offer_force"] = True
    st.session_state["explore_section"] = "Offer"
    st.rerun()


def _format_news_time(iso: str) -> str:
    if not iso:
        return "—"
    return iso[:19].replace("T", " ") + " UTC"


def _tab_news(
    *,
    starred: frozenset[str],
    items: list[StarNewsItem],
    news_file: Path,
    scored: pd.DataFrame,
    prefs: UserPreferences,
    app_settings: AppSettings,
    proc: Path,
) -> None:
    """Promising recommendations + starred-offer change feed."""
    st.subheader("Promising for you")
    if not prefs.enabled:
        st.caption(
            "Enable preferences on the Prefs tab to rank offers by your fit + bargain."
        )
    else:
        st.caption(
            "Preference-fit + bargain score with soft demotion for already-seen "
            "listings. Opening an offer marks it seen."
        )
        rec_ids_key = "explore_news_rec_ids"
        previous = st.session_state.get(rec_ids_key)
        promising = top_recommendations(
            scored,
            processed_dir=proc,
            top_n=int(app_settings.recommender_top_n),
            demotion_alpha=float(app_settings.recommender_demotion_alpha),
            prefs=prefs,
            record=False,
        )
        current_ids = (
            tuple(promising["listing_id"].astype(str).tolist())
            if not promising.empty
            else ()
        )
        if current_ids and current_ids != previous:
            record_impressions(impressions_path(proc), list(current_ids))
            retrain_recommender(
                scored,
                processed_dir=proc,
                top_n=int(app_settings.recommender_top_n),
                demotion_alpha=float(app_settings.recommender_demotion_alpha),
                prefs=prefs,
            )
            st.session_state[rec_ids_key] = current_ids
        if promising.empty:
            st.info("No listings match your enabled preferences.")
        else:
            for _, row in promising.iterrows():
                with st.container(border=True):
                    head, action = st.columns([3.5, 1], vertical_alignment="center")
                    with head:
                        title = str(row.get("title") or row["listing_id"])
                        st.markdown(f"**{title}**")
                        conf = row.get("confidence")
                        try:
                            conf_txt = f"{float(conf):.2f}"
                        except (TypeError, ValueError):
                            conf_txt = "—"
                        if conf_txt != "—" and conf != conf:  # NaN
                            conf_txt = "—"
                        st.caption(
                            f"{row['listing_id']} · confidence {conf_txt} · "
                            f"{row.get('district', '—')}"
                        )
                    with action:
                        if st.button(
                            "Open offer",
                            key=f"news_rec_open_{row['listing_id']}",
                            width="stretch",
                        ):
                            st.session_state["explore_offer_id"] = str(
                                row["listing_id"]
                            )
                            st.session_state["explore_offer_force"] = True
                            st.session_state["explore_section"] = "Offer"
                            st.rerun()

    st.subheader("Starred changes")
    st.caption(
        "Tracks price and listing field changes on starred offers. "
        "Old values are struck through; new values are emphasized. "
        "Baselines when you first star; updates when Explore reloads."
    )
    if not starred:
        st.info("Star offers on the Offer tab to start watching them here.")
        return
    st.caption(f"Watching {len(starred)} starred offer(s)")
    if not items:
        st.info("No changes detected yet for your starred offers.")
        return
    top, _ = st.columns([1, 3])
    with top:
        if st.button("Clear news", key="explore_clear_star_news"):
            save_news(news_file, [])
            st.rerun()
    for item in reversed(items):
        with st.container(border=True):
            head, action = st.columns([3.5, 1], vertical_alignment="center")
            with head:
                st.markdown(f"**{item.title}**")
                st.caption(f"{item.listing_id} · {_format_news_time(item.detected_at)}")
            with action:
                if st.button(
                    "Open offer",
                    key=f"news_open_{item.id}",
                    width="stretch",
                ):
                    st.session_state["explore_offer_id"] = item.listing_id
                    st.session_state["explore_offer_force"] = True
                    st.session_state["explore_section"] = "Offer"
                    st.rerun()
            if item.kind == "removed":
                st.warning(
                    "Removed from the scored universe "
                    "(delisted or dropped by the latest pipeline)."
                )
            else:
                for change in item.changes:
                    st.markdown(
                        format_change_markdown(
                            change.field,
                            change.old,
                            change.new,
                        )
                    )


def _opt_num(
    value: float | int | None,
    *,
    as_int: bool = False,
) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    if number <= 0:
        return None
    if as_int:
        return round(number)
    return number


def _tab_prefs(
    *,
    prefs: UserPreferences,
    prefs_file: Path,
    frame: pd.DataFrame,
    lm_model: str,
    metrics_file: Path,
) -> None:
    """Structured preference form + optional LM Studio NL translate."""
    st.caption(
        "Hard constraints filter Map and Board when enabled. "
        "News recommendations use the same lens with bargain/proximity scoring."
    )
    enabled = st.checkbox(
        "Apply preferences to Map/Board",
        value=prefs.enabled,
        key="prefs_enabled",
        help=PREF_HELP["enabled"],
    )
    districts_default = sorted({
        str(d) for d in frame.get("district", pd.Series(dtype=str)).dropna().unique()
    })
    sources_default = sorted({
        str(s) for s in frame.get("source", pd.Series(dtype=str)).dropna().unique()
    })
    districts = st.multiselect(
        "Districts",
        options=districts_default,
        default=[d for d in prefs.districts if d in districts_default],
        key="prefs_districts",
        help=PREF_HELP["districts"],
    )
    sources = st.multiselect(
        "Sources",
        options=sources_default,
        default=[s for s in prefs.sources if s in sources_default],
        key="prefs_sources",
        help=PREF_HELP["sources"],
    )
    oou_labels = {
        OOU_ANY: "Any noise contour",
        OOU_PREFER_OUTSIDE: "Prefer outside Lden ≥55",
        OOU_OUTSIDE_ONLY: "Outside Lden ≥55 only",
        OOU_INSIDE_OK: "Inside Lden ≥55 OK",
    }
    oou_keys = list(oou_labels.keys())
    oou_idx = oou_keys.index(prefs.oou) if prefs.oou in oou_keys else 0
    oou_label = st.selectbox(
        "Aircraft noise",
        options=[oou_labels[k] for k in oou_keys],
        index=oou_idx,
        key="prefs_oou",
        help=PREF_HELP["oou"],
    )
    oou = oou_keys[[oou_labels[k] for k in oou_keys].index(str(oou_label))]

    c1, c2, c3 = st.columns(3)
    with c1:
        min_price = st.number_input(
            "Min price PLN",
            min_value=0.0,
            value=float(prefs.min_price_pln or 0.0),
            step=10_000.0,
            key="prefs_min_price",
            help=PREF_HELP["min_price"],
        )
        max_price = st.number_input(
            "Max price PLN",
            min_value=0.0,
            value=float(prefs.max_price_pln or 0.0),
            step=10_000.0,
            key="prefs_max_price",
            help=PREF_HELP["max_price"],
        )
        min_ppm = st.number_input(
            "Min zł/m²",
            min_value=0.0,
            value=float(prefs.min_price_per_m2 or 0.0),
            step=500.0,
            key="prefs_min_ppm",
            help=PREF_HELP["min_ppm"],
        )
        max_ppm = st.number_input(
            "Max zł/m²",
            min_value=0.0,
            value=float(prefs.max_price_per_m2 or 0.0),
            step=500.0,
            key="prefs_max_ppm",
            help=PREF_HELP["max_ppm"],
        )
    with c2:
        min_area = st.number_input(
            "Min area m²",
            min_value=0.0,
            value=float(prefs.min_area_m2 or 0.0),
            step=1.0,
            key="prefs_min_area",
            help=PREF_HELP["min_area"],
        )
        max_area = st.number_input(
            "Max area m²",
            min_value=0.0,
            value=float(prefs.max_area_m2 or 0.0),
            step=1.0,
            key="prefs_max_area",
            help=PREF_HELP["max_area"],
        )
        min_rooms = st.number_input(
            "Min rooms",
            min_value=0,
            value=int(prefs.min_rooms or 0),
            step=1,
            key="prefs_min_rooms",
            help=PREF_HELP["min_rooms"],
        )
        max_rooms = st.number_input(
            "Max rooms",
            min_value=0,
            value=int(prefs.max_rooms or 0),
            step=1,
            key="prefs_max_rooms",
            help=PREF_HELP["max_rooms"],
        )
    with c3:
        min_year = st.number_input(
            "Min year built",
            min_value=0,
            value=int(prefs.min_year_built or 0),
            step=1,
            key="prefs_min_year",
            help=PREF_HELP["min_year"],
        )
        max_commute = st.number_input(
            "Max commute min",
            min_value=0.0,
            value=float(prefs.max_commute_min or 0.0),
            step=1.0,
            key="prefs_max_commute",
            help=PREF_HELP["max_commute"],
        )
        max_transit = st.number_input(
            "Max transit m",
            min_value=0.0,
            value=float(prefs.max_dist_transit_m or 0.0),
            step=50.0,
            key="prefs_max_transit",
            help=PREF_HELP["max_transit"],
        )
        max_delta = st.number_input(
            "Max delta zł/m²",
            min_value=-5_000.0,
            value=float(
                prefs.max_delta_price_per_m2
                if prefs.max_delta_price_per_m2 is not None
                else 0.0
            ),
            step=50.0,
            key="prefs_max_delta",
            help=PREF_HELP["max_delta"],
        )
    require_under = st.checkbox(
        "Require underpriced (delta < 0)",
        value=prefs.require_underpriced,
        key="prefs_require_under",
        help=PREF_HELP["require_under"],
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        max_hospital = st.number_input(
            "Max hospital m",
            min_value=0.0,
            value=float(prefs.max_dist_hospital_m or 0.0),
            step=100.0,
            key="prefs_max_hospital",
            help=PREF_HELP["max_hospital"],
        )
        max_fire = st.number_input(
            "Max fire station m",
            min_value=0.0,
            value=float(prefs.max_dist_fire_m or 0.0),
            step=100.0,
            key="prefs_max_fire",
            help=PREF_HELP["max_fire"],
        )
    with d2:
        max_police = st.number_input(
            "Max police m",
            min_value=0.0,
            value=float(prefs.max_dist_police_m or 0.0),
            step=100.0,
            key="prefs_max_police",
            help=PREF_HELP["max_police"],
        )
        max_air = st.number_input(
            "Max air station m",
            min_value=0.0,
            value=float(prefs.max_dist_air_m or 0.0),
            step=100.0,
            key="prefs_max_air",
            help=PREF_HELP["max_air"],
        )
    with d3:
        max_gunb = st.number_input(
            "Max GUNB m",
            min_value=0.0,
            value=float(prefs.max_dist_gunb_m or 0.0),
            step=100.0,
            key="prefs_max_gunb",
            help=PREF_HELP["max_gunb"],
        )
        max_park = st.number_input(
            "Max park m",
            min_value=0.0,
            value=float(prefs.max_dist_park_m or 0.0),
            step=50.0,
            key="prefs_max_park",
            help=PREF_HELP["max_park"],
        )

    nl_text = st.text_area(
        "Natural-language preferences",
        value=prefs.nl_text,
        height=100,
        key="prefs_nl",
        help="Describe what you want; translate with LM Studio into the form fields.",
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save preferences", type="primary", key="prefs_save"):
            max_delta_val: float | None = (
                float(max_delta) if abs(float(max_delta)) > 1e-9 else None
            )
            updated = UserPreferences(
                enabled=bool(enabled),
                districts=list(districts),
                sources=list(sources),
                oou=str(oou),
                min_price_pln=_opt_num(min_price),
                max_price_pln=_opt_num(max_price),
                min_price_per_m2=_opt_num(min_ppm),
                max_price_per_m2=_opt_num(max_ppm),
                min_area_m2=_opt_num(min_area),
                max_area_m2=_opt_num(max_area),
                min_rooms=int(_opt_num(min_rooms, as_int=True) or 0) or None,
                max_rooms=int(_opt_num(max_rooms, as_int=True) or 0) or None,
                min_year_built=int(_opt_num(min_year, as_int=True) or 0) or None,
                max_commute_min=_opt_num(max_commute),
                max_dist_transit_m=_opt_num(max_transit),
                max_dist_hospital_m=_opt_num(max_hospital),
                max_dist_fire_m=_opt_num(max_fire),
                max_dist_police_m=_opt_num(max_police),
                max_dist_air_m=_opt_num(max_air),
                max_dist_gunb_m=_opt_num(max_gunb),
                max_dist_park_m=_opt_num(max_park),
                max_delta_price_per_m2=max_delta_val,
                require_underpriced=bool(require_under),
                nl_text=str(nl_text or ""),
            )
            save_preferences(prefs_file, updated)
            st.success("Preferences saved.")
            st.rerun()
    with b2:
        if st.button("Translate with LM Studio", key="prefs_translate"):
            try:
                with st.spinner(f"Asking {lm_model}…"):
                    parsed = translate_preferences_nl(
                        str(nl_text or ""),
                        model=lm_model,
                        metrics_path=metrics_file,
                    )
                current = UserPreferences(
                    enabled=bool(enabled),
                    districts=list(districts),
                    sources=list(sources),
                    oou=str(oou),
                    nl_text=str(nl_text or ""),
                )
                merged = merge_preferences(current, parsed, nl_text=str(nl_text or ""))
                merged.enabled = True
                save_preferences(prefs_file, merged)
                st.success("Translated and saved — form refreshed.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def render_explore() -> None:
    """Render Explore with sidebar filters and Map / Offer / Board / News / Prefs."""
    logger.info("Rendering Explore")
    st.title("Explore")
    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        logger.warning("Explore missing scored listings: %s", exc)
        st.error(str(exc))
        st.stop()

    state, starred_only = _sidebar_filters(frame)
    flags, hour, basemap = _sidebar_map_controls()
    filtered = apply_filters(frame, state)
    logger.debug(
        "Explore filters → %s/%s offers (district=%s source=%s starred_only=%s)",
        len(filtered),
        len(frame),
        state.district,
        state.source,
        starred_only,
    )
    proc = processed_dir()
    stars_file = stars_path(proc)
    starred = load_stars(stars_file)
    app_settings = load_app_settings(settings_path(proc))
    prefs_file = preferences_path(proc)
    prefs = load_preferences(prefs_file)
    snaps_file = starred_snapshots_path(proc)
    news_file = starred_news_path(proc)
    # Sync against the full scored universe so sidebar filters never fake removals.
    sync_starred_news(
        starred,
        frame,
        snapshots_path=snaps_file,
        news_path=news_file,
        history_cap=app_settings.news_history_cap,
    )
    news_items = load_news(news_file)
    map_board = apply_starred_only(filtered, starred) if starred_only else filtered
    if prefs.enabled:
        map_board = apply_preferences(map_board, prefs)
    config = load_config()

    oou = load_aircraft_lden()
    amenities = load_amenities()
    shops = load_shops()
    stops = load_gtfs_stops()
    hourly = load_gtfs_hourly()
    route_shapes = load_gtfs_route_shapes()
    traffic_sample = load_traffic_sample()
    air_stations = load_gios_stations()
    gunb = load_gunb_points()
    parks = load_parks()

    if "explore_section" not in st.session_state:
        st.session_state["explore_section"] = "Map"
    commute = resolve_commute_target(app_settings, config)
    map_kwargs = {
        "oou": oou,
        "amenities": amenities,
        "shops": shops,
        "stops": stops,
        "hourly": hourly,
        "route_shapes": route_shapes,
        "traffic_sample": traffic_sample,
        "air_stations": air_stations,
        "gunb": gunb,
        "parks": parks,
        "flags": flags,
        "hour": hour,
        "basemap": basemap,
        "commute_lon": float(commute.lon),
        "commute_lat": float(commute.lat),
        "commute_label": commute.label,
    }
    tab_map, tab_spatial, tab_offer, tab_board, tab_news, tab_prefs = st.tabs(
        list(EXPLORE_SECTIONS),
        key="explore_section",
        on_change="rerun",
    )
    with tab_map:
        if tab_map.open:
            captions: list[str] = []
            if starred_only:
                captions.append("Starred only")
            if prefs.enabled:
                captions.append("Prefs on")
            if captions:
                st.caption(f"{' · '.join(captions)} · {len(map_board)} offers")
            _tab_map(map_board, **map_kwargs)
    with tab_spatial:
        if tab_spatial.open:
            captions_s: list[str] = []
            if starred_only:
                captions_s.append("Starred only")
            if prefs.enabled:
                captions_s.append("Prefs on")
            if captions_s:
                st.caption(f"{' · '.join(captions_s)} · {len(map_board)} offers")
            _tab_spatial(
                map_board,
                traffic_sample=traffic_sample,
                hour=hour,
                basemap=basemap,
                commute_lon=float(commute.lon),
                commute_lat=float(commute.lat),
            )
    with tab_offer:
        if tab_offer.open:
            _tab_offer(
                filtered,
                starred=starred,
                stars_file=stars_file,
                lm_model=app_settings.lm_model,
                ai_review_on_open=app_settings.ai_review_on_offer_open,
                app_settings=app_settings,
                **map_kwargs,
            )
    with tab_board:
        if tab_board.open:
            captions_b: list[str] = []
            if starred_only:
                captions_b.append("Starred only")
            if prefs.enabled:
                captions_b.append("Prefs on")
            if captions_b:
                st.caption(f"{' · '.join(captions_b)} · {len(map_board)} offers")
            _tab_board(
                map_board,
                starred=starred,
                app_settings=app_settings,
                impressions_file=impressions_path(proc),
                prefs=prefs,
            )
    with tab_news:
        if tab_news.open:
            _tab_news(
                starred=starred,
                items=news_items,
                news_file=news_file,
                scored=frame,
                prefs=prefs,
                app_settings=app_settings,
                proc=proc,
            )
    with tab_prefs:
        if tab_prefs.open:
            _tab_prefs(
                prefs=prefs,
                prefs_file=prefs_file,
                frame=frame,
                lm_model=app_settings.lm_model,
                metrics_file=llm_metrics_path(proc),
            )
