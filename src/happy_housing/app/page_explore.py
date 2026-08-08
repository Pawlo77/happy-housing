"""Explore page: sidebar filters + Map / Offer / Board / News tabs."""

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
from happy_housing.app.charts import OVERPRICED, UNDERPRICED, diverging_bar_chart
from happy_housing.app.data import (
    load_aircraft_lden,
    load_amenities,
    load_config,
    load_gios_stations,
    load_gtfs_hourly,
    load_gtfs_stops,
    load_gunb_points,
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
from happy_housing.app.map_layers import LayerFlags
from happy_housing.app.maps import (
    basemap_style,
    build_explore_layers,
    focus_marker_layer,
    render_deck,
)
from happy_housing.app.offer_display import (
    livability_rows,
    offer_fact_rows,
)
from happy_housing.app.recommender import (
    impressions_path,
    load_impressions,
    mark_opened,
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
from happy_housing.ingest.otodom import fetch_listing_detail, otodom_session
from happy_housing.models.train import build_feature_matrix

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
        key="explore_district",
    )
    source = st.sidebar.selectbox(
        "Source",
        ["(all)", *sources],
        key="explore_source",
    )
    oou = st.sidebar.selectbox(
        "Aircraft Lden ≥ 55 dB",
        ["(all)", "inside", "outside"],
        help="EEA END major-airport Lden band (≥55 dB). Binary filter on exposure.",
        key="explore_oou",
    )
    max_ppm = st.sidebar.slider(
        "Max listing zł/m²",
        min_value=ppm_min,
        max_value=max(ppm_max, ppm_min + 1),
        value=max(ppm_max, ppm_min + 1),
        key="explore_max_ppm",
    )
    month = "(all)"
    if months:
        month = st.sidebar.select_slider(
            "Listed month",
            options=["(all)", *months],
            value="(all)",
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
            key="explore_hour",
        )
    )
    st.sidebar.caption("Toggle overlays. Hover any pin for details.")
    flags = LayerFlags(
        offers=st.sidebar.checkbox(
            "Offers (Δ price)", value=True, key="explore_ly_offers"
        ),
        oou=st.sidebar.checkbox(
            "Aircraft Lden contours", value=True, key="explore_ly_oou"
        ),
        transit=st.sidebar.checkbox(
            "Public transport (GTFS)", value=False, key="explore_ly_transit"
        ),
        traffic=st.sidebar.checkbox(
            "Traffic intensity proxy", value=False, key="explore_ly_traffic"
        ),
        shops=st.sidebar.checkbox("Shops", value=False, key="explore_ly_shops"),
        hospitals=st.sidebar.checkbox(
            "Hospitals", value=False, key="explore_ly_hospitals"
        ),
        fire_stations=st.sidebar.checkbox(
            "Fire stations", value=False, key="explore_ly_fire"
        ),
        police=st.sidebar.checkbox("Police", value=False, key="explore_ly_police"),
        air=st.sidebar.checkbox(
            "Air-quality stations", value=False, key="explore_ly_air"
        ),
        gunb=st.sidebar.checkbox("GUNB permits", value=False, key="explore_ly_gunb"),
        commute_target=st.sidebar.checkbox(
            "Commute target", value=False, key="explore_ly_commute"
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
        "Lden bands from EEA END major-airport contours. Hover pins for details."
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
) -> None:
    """Tabular offer board; selecting a row opens Offer for that listing."""
    impressions = load_impressions(impressions_file)
    enriched = enrich_board_recommendations(
        filtered,
        impressions=impressions,
        demotion_alpha=float(app_settings.recommender_demotion_alpha),
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
    if "confidence" in board.columns:
        board["confidence"] = pd.to_numeric(board["confidence"], errors="coerce").round(
            3
        )
    cols = [c for c in _BOARD_COLUMNS if c in board.columns]
    board = board[cols]
    st.caption(
        f"{len(board)} offers after filters · sort: **{sort_label}** · "
        "click a row to open Offer · star offers on the Offer tab"
    )
    event = st.dataframe(
        board,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="explore_board",
        column_config={
            "confidence": st.column_config.NumberColumn("confidence", format="%.3f"),
            "seen": st.column_config.TextColumn("seen"),
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
) -> None:
    """News feed of changes to starred offers since last Explore sync."""
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


def render_explore() -> None:
    """Render Explore with sidebar filters and Map / Offer / Board / News."""
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
    config = load_config()

    oou = load_aircraft_lden()
    amenities = load_amenities()
    shops = load_shops()
    stops = load_gtfs_stops()
    hourly = load_gtfs_hourly()
    traffic_sample = load_traffic_sample()
    air_stations = load_gios_stations()
    gunb = load_gunb_points()

    if "explore_section" not in st.session_state:
        st.session_state["explore_section"] = "Map"
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
    tab_map, tab_offer, tab_board, tab_news = st.tabs(
        list(EXPLORE_SECTIONS),
        key="explore_section",
        on_change="rerun",
    )
    with tab_map:
        if tab_map.open:
            if starred_only:
                st.caption(f"Starred only · {len(map_board)} offers")
            _tab_map(map_board, **map_kwargs)
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
            if starred_only:
                st.caption(f"Starred only · {len(map_board)} offers")
            _tab_board(
                map_board,
                starred=starred,
                app_settings=app_settings,
                impressions_file=impressions_path(proc),
            )
    with tab_news:
        if tab_news.open:
            _tab_news(starred=starred, items=news_items, news_file=news_file)
