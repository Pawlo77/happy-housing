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
from happy_housing.app.chat_ui import render_assistant_popover, render_assistant_tab
from happy_housing.app.commute_target import resolve_commute_target
from happy_housing.app.data import (
    load_airport_areas,
    load_amenities,
    load_bike_paths,
    load_city_boundary,
    load_city_pois,
    load_config,
    load_gios_aq_overlay,
    load_gios_stations,
    load_gtfs_hourly,
    load_gtfs_route_shapes,
    load_gtfs_stops,
    load_gunb_points,
    load_military_areas,
    load_parks,
    load_rail_areas,
    load_scored,
    load_shap_bundle,
    load_shops,
    load_traffic_sample,
    load_warsaw_noise_overlay,
    processed_dir,
)
from happy_housing.app.explore_nav import (
    EXPLORE_SECTIONS,
    OFFER_SELECT_KEY,
    apply_explore_section_pending,
    listing_id_from_board_selection,
    listing_id_from_map_selection,
    prepare_offer_select_state,
    request_explore_section,
)
from happy_housing.app.filters import (
    FilterState,
    apply_filters,
    apply_starred_only,
    offer_age_months,
)
from happy_housing.app.geocode import (
    format_place_option_labels,
    live_search_ready,
    resolve_place_pick,
    search_places,
)
from happy_housing.app.llm_metrics import llm_metrics_path
from happy_housing.app.map_layers import LayerFlags, explore_datasets_needed
from happy_housing.app.maps import (
    ADDRESS_FOCUS_ZOOM,
    MapAddressFocus,
    address_search_layer,
    basemap_style,
    build_explore_layers,
    build_spatial_layers,
    explore_map_camera,
    explore_map_selection_key,
    focus_marker_layer,
    render_deck,
)
from happy_housing.app.offer_display import (
    floor_plan_urls_from_row,
    image_urls_from_row,
    livability_rows,
    offer_fact_rows,
    sanitize_listing_html,
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
from happy_housing.app.property_filter_ui import render_property_attr_filters
from happy_housing.app.recommender import (
    impressions_path,
    load_impressions,
    mark_opened,
    record_impressions,
    retrain_recommender,
    top_recommendations,
)
from happy_housing.app.spatial_insights import (
    SPATIAL_FIELD_MODE_MODEL,
    SPATIAL_FIELD_MODES,
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
from happy_housing.gis.amenity_catalog import (
    AMENITY_TYPES,
    DEFAULT_MAP_OBJECT_TYPES,
    city_map_objects_by_group,
    coerce_amenity_weights,
)
from happy_housing.gis.gios_aq_field import AQ_MAP_OVERLAYS, AqMapOverlay
from happy_housing.gis.warsaw_noise import NOISE_MAP_OVERLAYS, NoiseMapOverlay
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

EXPLORE_SIDEBAR_TABS: tuple[str, str] = ("Filters", "Map layers")
"""Sidebar tab labels: filters vs map overlays (avoids long stacked scroll)."""


def _sidebar_filters(frame: pd.DataFrame) -> tuple[FilterState, bool]:
    """Render Explore filters; call inside the Filters sidebar tab."""
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
    ppm = pd.to_numeric(frame.get("price_per_m2"), errors="coerce")
    finite = ppm[ppm.notna()]
    ppm_min = int(finite.min()) if not finite.empty else 0
    ppm_max = int(finite.max()) + 1 if not finite.empty else 1
    ages = offer_age_months(frame)
    age_finite = ages.dropna()
    age_max = int(age_finite.max()) if not age_finite.empty else 0

    district = st.selectbox(
        "District",
        ["(all)", *districts],
        help=FILTER_HELP["district"],
        key="explore_district",
    )
    source = st.selectbox(
        "Source",
        ["(all)", *sources],
        help=FILTER_HELP["source"],
        key="explore_source",
    )
    oou = st.selectbox(
        "Aircraft Lden ≥ 55 dB",
        ["(all)", "inside", "outside"],
        help=FILTER_HELP["oou"],
        key="explore_oou",
    )
    high_noise = st.selectbox(
        "Total noise Lden ≥ 55 dB",
        ["(all)", "inside", "outside"],
        help=FILTER_HELP["high_noise"],
        key="explore_high_noise",
    )
    poor_air = st.selectbox(
        "Air quality AQI ≥ 3 (IDW)",
        ["(all)", "inside", "outside"],
        help=FILTER_HELP["poor_air"],
        key="explore_poor_air",
    )
    noise_cap = st.number_input(
        "Max total Lden (dB, 0 = off)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help=FILTER_HELP["max_noise_total"],
        key="explore_max_noise_total",
    )
    max_ppm = st.slider(
        "Max listing zł/m²",
        min_value=ppm_min,
        max_value=max(ppm_max, ppm_min + 1),
        value=max(ppm_max, ppm_min + 1),
        help=FILTER_HELP["max_ppm"],
        key="explore_max_ppm",
    )
    max_age_months: int | None = None
    if age_max > 0:
        picked_age = st.slider(
            "Offer age (months)",
            min_value=0,
            max_value=age_max,
            value=age_max,
            help=FILTER_HELP["max_age_months"],
            key="explore_max_age_months",
        )
        if picked_age < age_max:
            max_age_months = int(picked_age)
    starred_only = st.checkbox(
        "Starred only",
        value=False,
        key="explore_starred_only",
        help=(
            "Map and Board show only starred offers. "
            "Offer tab stays unfiltered by star."
        ),
    )
    prop = render_property_attr_filters("explore")
    return (
        FilterState(
            district=district,
            source=source,
            oou=oou,
            high_noise=high_noise,
            poor_air=poor_air,
            max_noise_total_lden_db=float(noise_cap) if float(noise_cap) > 0 else None,
            max_price_per_m2=float(max_ppm),
            max_age_months=max_age_months,
            **prop,
        ),
        starred_only,
    )


def _sidebar_map_poi_multiselect(
    title: str,
    keys: tuple[str, ...],
    *,
    help_text: str,
    widget_key: str,
) -> tuple[str, ...]:
    """Multiselect for city POIs without group prefixes on labels."""
    label_by_key = {t.key: t.label for t in AMENITY_TYPES if t.key in set(keys)}
    default_keys = [k for k in DEFAULT_MAP_OBJECT_TYPES if k in label_by_key]
    options = [label_by_key[k] for k in keys if k in label_by_key]
    selected_labels = st.multiselect(
        title,
        options=options,
        default=[label_by_key[k] for k in default_keys],
        help=help_text,
        key=widget_key,
    )
    label_to_key = {v: k for k, v in label_by_key.items()}
    return tuple(
        label_to_key[label] for label in selected_labels if label in label_to_key
    )


def _sidebar_map_controls() -> tuple[LayerFlags, int, str]:
    """Map basemap, hour-of-day, and per-layer toggles; call inside Map layers tab."""
    basemap = (
        st.segmented_control(
            "Basemap",
            options=["Streets", "Satellite"],
            default="Streets",
            help=BASEMAP_HELP,
            key="explore_basemap",
        )
        or "Streets"
    )

    st.markdown("##### Listings & context")
    offers = st.checkbox(
        "Offers (Δ price)",
        value=True,
        help=layer_help("offers"),
        key="explore_ly_offers",
    )
    city_boundary = st.checkbox(
        "City boundary",
        value=False,
        help=layer_help("city_boundary"),
        key="explore_ly_city_boundary",
    )
    commute_target = st.checkbox(
        "Commute target",
        value=False,
        help=layer_help("commute_target"),
        key="explore_ly_commute",
    )

    st.markdown("##### Transport")
    hour = int(
        st.slider(
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
    transit_network = st.checkbox(
        "Transit lines & stops",
        value=False,
        help=layer_help("transit_network"),
        key="explore_ly_transit_network",
    )
    if transit_network:
        transit_show_metro = st.checkbox(
            "  Metro lines",
            value=True,
            help=layer_help("transit_show_metro"),
            key="explore_ly_transit_metro",
        )
        transit_show_tram = st.checkbox(
            "  Tram lines",
            value=True,
            help=layer_help("transit_show_tram"),
            key="explore_ly_transit_tram",
        )
        transit_show_bus = st.checkbox(
            "  Bus lines",
            value=True,
            help=layer_help("transit_show_bus"),
            key="explore_ly_transit_bus",
        )
        transit_show_rail = st.checkbox(
            "  Train / rail lines",
            value=True,
            help=layer_help("transit_show_rail"),
            key="explore_ly_transit_rail",
        )
    else:
        transit_show_metro = False
        transit_show_tram = False
        transit_show_bus = False
        transit_show_rail = False
    transit_intensity = st.checkbox(
        "Transit intensity (hourly)",
        value=False,
        help=layer_help("transit_intensity"),
        key="explore_ly_transit_intensity",
    )
    traffic = st.checkbox(
        "Traffic intensity proxy",
        value=False,
        help=layer_help("traffic"),
        key="explore_ly_traffic",
    )
    bike_paths = st.checkbox(
        "Bike paths (OSM)",
        value=False,
        help=layer_help("bike_paths"),
        key="explore_ly_bike_paths",
    )
    charging_stations = st.checkbox(
        "EV charging",
        value=False,
        help=layer_help("charging_stations"),
        key="explore_ly_charging",
    )
    gas_stations = st.checkbox(
        "Gas stations",
        value=False,
        help=layer_help("gas_stations"),
        key="explore_ly_gas",
    )

    st.markdown("##### Shopping")
    shops_on = st.checkbox(
        "Shops",
        value=False,
        help=layer_help("shops"),
        key="explore_ly_shops",
    )
    if shops_on:
        shop_show_grocery = st.checkbox(
            "  Groceries",
            value=True,
            help=layer_help("shop_show_grocery"),
            key="explore_ly_shop_grocery",
        )
        shop_show_health_beauty = st.checkbox(
            "  Health & beauty",
            value=True,
            help=layer_help("shop_show_health_beauty"),
            key="explore_ly_shop_health_beauty",
        )
        shop_show_fashion = st.checkbox(
            "  Fashion",
            value=True,
            help=layer_help("shop_show_fashion"),
            key="explore_ly_shop_fashion",
        )
        shop_show_home = st.checkbox(
            "  Home & electronics",
            value=True,
            help=layer_help("shop_show_home"),
            key="explore_ly_shop_home",
        )
        shop_show_mobility = st.checkbox(
            "  Car & bike",
            value=True,
            help=layer_help("shop_show_mobility"),
            key="explore_ly_shop_mobility",
        )
        shop_show_other = st.checkbox(
            "  Other shops",
            value=True,
            help=layer_help("shop_show_other"),
            key="explore_ly_shop_other",
        )
    else:
        shop_show_grocery = False
        shop_show_health_beauty = False
        shop_show_fashion = False
        shop_show_home = False
        shop_show_mobility = False
        shop_show_other = False

    st.markdown("##### Public services")
    hospitals = st.checkbox(
        "Hospitals",
        value=False,
        help=layer_help("hospitals"),
        key="explore_ly_hospitals",
    )
    fire_stations = st.checkbox(
        "Fire stations",
        value=False,
        help=layer_help("fire_stations"),
        key="explore_ly_fire",
    )
    police = st.checkbox(
        "Police",
        value=False,
        help=layer_help("police"),
        key="explore_ly_police",
    )

    st.markdown("##### Environment")
    military = st.checkbox(
        "Military areas",
        value=True,
        help=layer_help("military"),
        key="explore_ly_military",
    )
    rail_areas = st.checkbox(
        "Rail land / yards",
        value=True,
        help=layer_help("rail_areas"),
        key="explore_ly_rail_areas",
    )
    airport = st.checkbox(
        "Airport footprints",
        value=True,
        help=layer_help("airport"),
        key="explore_ly_airport",
    )
    parks = st.checkbox(
        "Parks & squares (UM)",
        value=False,
        help=layer_help("parks"),
        key="explore_ly_parks",
    )
    air = st.checkbox(
        "Air-quality stations",
        value=False,
        help=layer_help("air"),
        key="explore_ly_air",
    )
    gunb = st.checkbox(
        "GUNB permits",
        value=False,
        help=layer_help("gunb"),
        key="explore_ly_gunb",
    )
    noise_labels = [label for label, _ in NOISE_MAP_OVERLAYS]
    noise_by_label = {label: key for label, key in NOISE_MAP_OVERLAYS}
    default_noise_label = next(
        (label for label, key in NOISE_MAP_OVERLAYS if key == "off"),
        noise_labels[0],
    )
    noise_label = st.selectbox(
        "Strategic noise (UM 2022)",
        options=noise_labels,
        index=noise_labels.index(default_noise_label),
        help=layer_help("noise_overlay"),
        key="explore_ly_noise_overlay",
    )
    aq_labels = [label for label, _ in AQ_MAP_OVERLAYS]
    aq_by_label = {label: key for label, key in AQ_MAP_OVERLAYS}
    default_aq_label = next(
        (label for label, key in AQ_MAP_OVERLAYS if key == "off"),
        aq_labels[0],
    )
    aq_label = st.selectbox(
        "Air quality field (GIOŚ IDW)",
        options=aq_labels,
        index=aq_labels.index(default_aq_label),
        help=layer_help("aq_overlay"),
        key="explore_ly_aq_overlay",
    )

    st.markdown("##### Sport")
    sport_keys = _sidebar_map_poi_multiselect(
        "Sports facilities (UM + OSM gyms)",
        city_map_objects_by_group("Sports"),
        help_text=(
            "UM sports (mapa.um.warszawa.pl/obiekty-sportowe) plus OSM "
            "commercial fitness clubs under Gym / fitness club. "
            "Default: gyms, fitness studios, swimming pools."
        ),
        widget_key="explore_ly_city_pois_sports",
    )

    st.markdown("##### Civic & culture")
    civic_keys = _sidebar_map_poi_multiselect(
        "Civic points (UM city map)",
        city_map_objects_by_group("Civic"),
        help_text=(
            "UM civic WFS (schools, libraries, pharmacies, hotels, …). "
            "Police / hospitals / fire / parks / shops / EV / gas use the "
            "dedicated checkboxes in other sections — not listed here."
        ),
        widget_key="explore_ly_city_pois_civic",
    )

    flags = LayerFlags(
        offers=offers,
        city_boundary=city_boundary,
        military=military,
        rail_areas=rail_areas,
        airport=airport,
        transit_intensity=transit_intensity,
        transit_network=transit_network,
        transit_show_metro=transit_show_metro,
        transit_show_tram=transit_show_tram,
        transit_show_bus=transit_show_bus,
        transit_show_rail=transit_show_rail,
        traffic=traffic,
        bike_paths=bike_paths,
        shops=shops_on,
        shop_show_grocery=shop_show_grocery,
        shop_show_health_beauty=shop_show_health_beauty,
        shop_show_fashion=shop_show_fashion,
        shop_show_home=shop_show_home,
        shop_show_mobility=shop_show_mobility,
        shop_show_other=shop_show_other,
        hospitals=hospitals,
        fire_stations=fire_stations,
        police=police,
        charging_stations=charging_stations,
        gas_stations=gas_stations,
        air=air,
        gunb=gunb,
        parks=parks,
        commute_target=commute_target,
        city_poi_types=sport_keys + civic_keys,
        noise_overlay=noise_by_label.get(str(noise_label), "off"),
        aq_overlay=aq_by_label.get(str(aq_label), "off"),
    )
    return flags, hour, basemap


def _render_explore_sidebar(
    frame: pd.DataFrame,
) -> tuple[FilterState, bool, LayerFlags, int, str]:
    """Sidebar tabs for filters and map layers; return Explore control state."""
    tab_filters, tab_layers = st.sidebar.tabs(list(EXPLORE_SIDEBAR_TABS))
    with tab_filters:
        state, starred_only = _sidebar_filters(frame)
    with tab_layers:
        flags, hour, basemap = _sidebar_map_controls()
    return state, starred_only, flags, hour, basemap


@st.cache_data(ttl=300, show_spinner=False)  # type: ignore[misc]
def _cached_explore_place_search(
    query: str,
) -> tuple[dict[str, float | str | int], ...]:
    """Cached Nominatim hits for Explore Map address search."""
    hits = search_places(query)
    return tuple(
        {
            "label": h.label,
            "lon": h.lon,
            "lat": h.lat,
            "osm_id": h.osm_id,
            "osm_type": h.osm_type,
        }
        for h in hits
    )


def _read_explore_map_focus() -> MapAddressFocus | None:
    """Load address focus from session, or ``None`` when cleared / invalid."""
    lon = st.session_state.get("explore_map_focus_lon")
    lat = st.session_state.get("explore_map_focus_lat")
    label = st.session_state.get("explore_map_focus_label")
    if lon is None or lat is None:
        return None
    try:
        return MapAddressFocus(
            lon=float(lon),
            lat=float(lat),
            label=str(label or "Address").strip() or "Address",
        )
    except (TypeError, ValueError):
        return None


def _clear_explore_map_focus() -> None:
    """Drop Explore Map address pin + search seed."""
    for key in (
        "explore_map_focus_lon",
        "explore_map_focus_lat",
        "explore_map_focus_label",
        "explore_map_search_seed",
    ):
        st.session_state.pop(key, None)


def _render_explore_address_search() -> MapAddressFocus | None:
    """Nominatim address search above the Explore Map; centers + zooms on pick."""
    search_seed = str(st.session_state.get("explore_map_search_seed") or "").strip()
    hit_rows: tuple[dict[str, float | str | int], ...] = ()
    if live_search_ready(search_seed):
        try:
            hit_rows = _cached_explore_place_search(search_seed)
        except Exception as exc:
            st.error(f"Address search failed: {exc}")
            hit_rows = ()
    option_labels = format_place_option_labels(hit_rows)
    c_search, c_clear = st.columns([5, 1])
    with c_search:
        place_pick = st.selectbox(
            "Search address",
            options=option_labels,
            index=None,
            placeholder="Type a Warsaw address, press Enter, then pick a match…",
            accept_new_options=True,
            key="explore_map_place_combo",
            help=(
                "OpenStreetMap Nominatim (Poland / Warsaw-biased). "
                "Picking a match centers the map and zooms to street level."
            ),
        )
    with c_clear:
        st.write("")  # align with selectbox
        if st.button(
            "Clear",
            key="explore_map_clear_address",
            help="Remove the address pin and restore the default map view.",
            disabled=_read_explore_map_focus() is None and not search_seed,
        ):
            _clear_explore_map_focus()
            st.rerun()

    if place_pick:
        chosen = resolve_place_pick(str(place_pick), hit_rows, option_labels)
        if chosen is not None and isinstance(chosen, dict):
            new_lon = float(str(chosen["lon"]))
            new_lat = float(str(chosen["lat"]))
            new_label = str(chosen["label"])[:160]
            current = _read_explore_map_focus()
            changed = (
                current is None
                or abs(current.lon - new_lon) > 1e-6
                or abs(current.lat - new_lat) > 1e-6
                or current.label != new_label
            )
            if changed:
                st.session_state["explore_map_focus_lon"] = new_lon
                st.session_state["explore_map_focus_lat"] = new_lat
                st.session_state["explore_map_focus_label"] = new_label
                st.rerun()
        elif live_search_ready(place_pick):
            needle = str(place_pick).strip()
            if needle[:3].rstrip(". ").isdigit() and ". " in needle:
                needle = needle.split(". ", 1)[-1].strip()
            if needle != search_seed:
                st.session_state["explore_map_search_seed"] = needle
                st.rerun()
            elif not hit_rows:
                st.caption("No matches for this query.")
            else:
                st.caption(f"{len(hit_rows)} matches — pick one from the list.")

    focus = _read_explore_map_focus()
    if focus is not None:
        st.caption(f"Centered on **{focus.label}** · zoom {ADDRESS_FOCUS_ZOOM:g}")
    return focus


def _tab_map(
    filtered: pd.DataFrame,
    *,
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    route_shapes: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    parks: gpd.GeoDataFrame,
    bike_paths: gpd.GeoDataFrame,
    city_boundary: gpd.GeoDataFrame,
    military: gpd.GeoDataFrame,
    rail_areas: gpd.GeoDataFrame,
    airport: gpd.GeoDataFrame,
    city_pois: gpd.GeoDataFrame,
    noise_overlay_data: NoiseMapOverlay | None,
    aq_overlay_data: AqMapOverlay | None,
    flags: LayerFlags,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
    commute_label: str = "Commute target",
) -> None:
    """Market map with toggleable livability overlays."""
    if filtered.empty and flags.offers:
        st.warning("No offers match filters — disable Offers or widen filters.")
    focus = _render_explore_address_search()
    layers = build_explore_layers(
        listings=filtered,
        flags=flags,
        amenities=amenities,
        shops=shops,
        stops=stops,
        hourly=hourly,
        route_shapes=route_shapes,
        traffic_sample=traffic_sample,
        air_stations=air_stations,
        gunb=gunb,
        parks=parks,
        bike_paths=bike_paths,
        city_boundary=city_boundary,
        military=military,
        rail_areas=rail_areas,
        airport=airport,
        city_pois=city_pois,
        noise_overlay=noise_overlay_data,
        aq_overlay=aq_overlay_data,
        commute_lon=commute_lon,
        commute_lat=commute_lat,
        hour=hour,
        commute_label=commute_label,
    )
    if focus is not None:
        layers.append(address_search_layer(focus.lon, focus.lat, label=focus.label))
    if not layers:
        st.info("Enable at least one map layer in the sidebar.")
        return
    lat, lon, zoom = explore_map_camera(
        filtered,
        focus=focus,
        fallback_lat=commute_lat,
        fallback_lon=commute_lon,
    )
    event = render_deck(
        layers,
        latitude=lat,
        longitude=lon,
        zoom=zoom,
        map_style=basemap_style(basemap),
        selection_key=explore_map_selection_key(focus),
    )
    if not flags.offers or filtered.empty:
        return
    clicked = listing_id_from_map_selection(event)
    if clicked is None:
        st.caption(
            "Click an offer pin to preview it and open the Offer tab."
            if focus is None
            else "Address pin centered · click an offer pin to preview."
        )
        return
    hit = filtered.loc[filtered["listing_id"].astype(str) == clicked]
    if hit.empty:
        return
    row = hit.iloc[0]
    st.subheader(f"Selected · {clicked}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("District", str(row.get("district") or "—"))
    ppm = pd.to_numeric(row.get("price_per_m2"), errors="coerce")
    c2.metric("zł/m²", f"{ppm:,.0f}" if pd.notna(ppm) else "—")
    delta = pd.to_numeric(row.get("delta_price_per_m2"), errors="coerce")
    c3.metric("Δ zł/m²", f"{delta:+,.0f}" if pd.notna(delta) else "—")
    area = pd.to_numeric(row.get("area_m2"), errors="coerce")
    c4.metric("Area", f"{area:.0f} m²" if pd.notna(area) else "—")
    rooms = pd.to_numeric(row.get("rooms"), errors="coerce")
    c5.metric("Rooms", f"{rooms:g}" if pd.notna(rooms) else "—")
    price = pd.to_numeric(row.get("price_pln"), errors="coerce")
    commute = pd.to_numeric(row.get("commute_min"), errors="coerce")
    model = pd.to_numeric(row.get("model_price_per_m2"), errors="coerce")
    st.caption(
        f"Total {price:,.0f} zł · model {model:,.0f} zł/m² · commute {commute:.0f} min"
        if pd.notna(price) and pd.notna(model) and pd.notna(commute)
        else "Click Open in Offer for the full listing card."
    )
    if st.button("Open in Offer", type="primary", key=f"map_open_offer_{clicked}"):
        st.session_state["explore_offer_id"] = clicked
        st.session_state["explore_offer_force"] = True
        request_explore_section("Offer")
        st.rerun()


def _tab_spatial(
    filtered: pd.DataFrame,
    *,
    traffic_sample: pd.DataFrame,
    hour: int,
    basemap: str,
    commute_lon: float,
    commute_lat: float,
    city_boundary: gpd.GeoDataFrame,
    military: gpd.GeoDataFrame,
    rail_areas: gpd.GeoDataFrame,
    airport: gpd.GeoDataFrame,
    flags: LayerFlags,
) -> None:
    """Spatial analytics: metric heatmaps + district charts (not map-only)."""
    st.caption(
        "Pick a market signal — continuous city field plus historical listing "
        "pins. Toggle Model vs Historical surface. Same sidebar filters as Map."
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
    field_mode = SPATIAL_FIELD_MODE_MODEL
    if metric_key != "traffic":
        mode_labels = [label for label, _ in SPATIAL_FIELD_MODES]
        mode_by_label = {label: key for label, key in SPATIAL_FIELD_MODES}
        mode_choice = (
            st.segmented_control(
                "Surface field",
                options=mode_labels,
                default=mode_labels[0],
                key="explore_spatial_field_mode",
                help=(
                    "Model predictions = continuous fair-price / model-derived "
                    "field. Historical = interpolated observed asks (replaces "
                    "the model surface). Pins always mark historical listings."
                ),
            )
            or mode_labels[0]
        )
        field_mode = mode_by_label.get(str(mode_choice), SPATIAL_FIELD_MODE_MODEL)
    show_hex = st.checkbox(
        "Show hex elevation (3D)",
        value=False,
        key="explore_spatial_hex",
        help=(
            "Extrude hex bins by observed metric weight. Listing metrics only — "
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
            city_boundary=city_boundary,
            military=military,
            rail_areas=rail_areas,
            airport=airport,
            flags=flags,
            field_mode=field_mode,
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
def _fetch_otodom_offer_media(slug: str) -> dict[str, object]:
    """Cached Otodom detail media (description + photos + floor plans)."""
    detail = fetch_listing_detail(otodom_session(), slug.strip())
    description = detail.get("description")
    text = description.strip() if isinstance(description, str) else ""
    return {
        "description": text,
        "image_url": detail.get("image_url"),
        "image_urls_json": detail.get("image_urls_json") or "[]",
        "floor_plan_urls_json": detail.get("floor_plan_urls_json") or "[]",
    }


def _ensure_offer_media(row: pd.Series) -> pd.Series:
    """Fill description / multi-photo / floor-plan fields from Otodom when missing."""
    out = row.copy()
    existing = out.get("description")
    has_desc = isinstance(existing, str) and existing.strip()
    photos = image_urls_from_row(out)
    plans = floor_plan_urls_from_row(out)
    needs_media = len(photos) <= 1 or not plans
    if has_desc and not needs_media:
        return out
    slug = out.get("slug")
    if not isinstance(slug, str) or not slug.strip():
        return out
    try:
        media = _fetch_otodom_offer_media(slug.strip())
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.warning("Offer media fetch failed for %s: %s", slug, exc)
        return out
    desc = media.get("description")
    if (not has_desc) and isinstance(desc, str) and desc.strip():
        out["description"] = desc.strip()
    fetched_photos = image_urls_from_row(media)
    if len(fetched_photos) > len(photos):
        out["image_urls_json"] = media.get("image_urls_json") or "[]"
        if isinstance(media.get("image_url"), str) and media["image_url"]:
            out["image_url"] = media["image_url"]
    fetched_plans = floor_plan_urls_from_row(media)
    if fetched_plans and not plans:
        out["floor_plan_urls_json"] = media.get("floor_plan_urls_json") or "[]"
    return out


def _render_offer_gallery(row: pd.Series) -> None:
    """Photo carousel + optional floor-plan images under the gallery."""
    photos = image_urls_from_row(row)
    if photos:
        if len(photos) == 1:
            st.image(photos[0], width="stretch")
        else:
            labels = [f"Photo {i + 1} / {len(photos)}" for i in range(len(photos))]
            pick = st.selectbox(
                "Photos",
                labels,
                key=f"offer_photo_{row.get('listing_id')}",
                label_visibility="collapsed",
            )
            idx = labels.index(str(pick))
            st.image(photos[idx], width="stretch")
            st.caption(f"{idx + 1} of {len(photos)} photos")
    plans = floor_plan_urls_from_row(row)
    if plans:
        st.markdown("##### Floor plan")
        for i, url in enumerate(plans):
            st.image(url, width="stretch", caption=f"Layout {i + 1}")


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
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    route_shapes: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    parks: gpd.GeoDataFrame,
    bike_paths: gpd.GeoDataFrame,
    city_boundary: gpd.GeoDataFrame,
    military: gpd.GeoDataFrame,
    rail_areas: gpd.GeoDataFrame,
    airport: gpd.GeoDataFrame,
    city_pois: gpd.GeoDataFrame,
    noise_overlay_data: NoiseMapOverlay | None,
    aq_overlay_data: AqMapOverlay | None,
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
        if overview.open:
            row = _ensure_offer_media(row)
            description = str(row.get("description") or "").strip()
            left, right = st.columns([1.1, 1.6])
            with left:
                _render_offer_gallery(row)
                if isinstance(row.get("url"), str) and row["url"]:
                    st.link_button(
                        "Open source listing", str(row["url"]), width="stretch"
                    )
            with right:
                title = row.get("title") or row.get("listing_id")
                st.subheader(str(title))
                _price_metrics(row)
                st.dataframe(offer_fact_rows(row), hide_index=True, width="stretch")
            st.markdown("#### Source description")
            if description:
                st.markdown(sanitize_listing_html(description), unsafe_allow_html=True)
            else:
                st.info("No listing description available for this offer.")

    with context:
        if context.open:
            st.caption(
                "Distances rounded to whole metres. "
                "Noise bands from UM Warszawa 2022 strategic maps (dB lower edge)."
            )
            st.dataframe(
                livability_rows(row, commute_label=commute_label),
                hide_index=True,
                width="stretch",
            )
            _price_metrics(row)

    with local_map:
        if local_map.open:
            st.caption(
                "Same layer stack as the main Explore map (sidebar toggles apply)."
            )
            lat, lon = float(row["lat"]), float(row["lon"])
            layers = build_explore_layers(
                listings=filtered,
                flags=flags,
                amenities=amenities,
                shops=shops,
                stops=stops,
                hourly=hourly,
                route_shapes=route_shapes,
                traffic_sample=traffic_sample,
                air_stations=air_stations,
                gunb=gunb,
                parks=parks,
                bike_paths=bike_paths,
                city_boundary=city_boundary,
                military=military,
                rail_areas=rail_areas,
                airport=airport,
                city_pois=city_pois,
                noise_overlay=noise_overlay_data,
                aq_overlay=aq_overlay_data,
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
        if explain.open:
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
        amenity_weights=coerce_amenity_weights(
            app_settings.recommender_amenity_weights
        ),
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
    request_explore_section("Offer")
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
            amenity_weights=coerce_amenity_weights(
                app_settings.recommender_amenity_weights
            ),
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
                amenity_weights=coerce_amenity_weights(
                    app_settings.recommender_amenity_weights
                ),
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
                            request_explore_section("Offer")
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
                    request_explore_section("Offer")
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
        max_bike_path = st.number_input(
            "Max bike path m",
            min_value=0.0,
            value=float(prefs.max_dist_bike_path_m or 0.0),
            step=50.0,
            key="prefs_max_bike_path",
            help=PREF_HELP["max_bike_path"],
        )
        max_noise_total = st.number_input(
            "Max total Lden (dB)",
            min_value=0.0,
            value=float(prefs.max_noise_total_lden_db or 0.0),
            step=1.0,
            key="prefs_max_noise_total",
            help=PREF_HELP["max_noise_total"],
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
                max_dist_bike_path_m=_opt_num(max_bike_path),
                max_noise_total_lden_db=_opt_num(max_noise_total),
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
    """Render Explore with sidebar Filter/Map-layer tabs + main content tabs."""
    logger.info("Rendering Explore")
    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        logger.warning("Explore missing scored listings: %s", exc)
        st.error(str(exc))
        st.stop()

    state, starred_only, flags, hour, basemap = _render_explore_sidebar(frame)
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
    map_board = apply_starred_only(filtered, starred) if starred_only else filtered
    if prefs.enabled:
        map_board = apply_preferences(map_board, prefs)
    config = load_config()

    apply_explore_section_pending()
    if "explore_section" not in st.session_state:
        st.session_state["explore_section"] = "Map"
    section = str(st.session_state.get("explore_section") or "Map")
    offer_subtab = str(st.session_state.get("offer_subtab") or "Overview")
    needed = explore_datasets_needed(
        section=section,
        flags=flags,
        offer_subtab=offer_subtab,
    )

    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    empty_df = pd.DataFrame()
    amenities = load_amenities() if "amenities" in needed else empty_gdf
    shops = load_shops() if "shops" in needed else empty_df
    stops = load_gtfs_stops() if "stops" in needed else empty_df
    hourly = load_gtfs_hourly() if "hourly" in needed else empty_df
    route_shapes = load_gtfs_route_shapes() if "route_shapes" in needed else empty_df
    traffic_sample = load_traffic_sample() if "traffic" in needed else empty_df
    bike_paths = load_bike_paths() if "bike_paths" in needed else empty_gdf
    air_stations = load_gios_stations() if "air" in needed else empty_df
    gunb = load_gunb_points() if "gunb" in needed else empty_df
    parks = load_parks() if "parks" in needed else empty_gdf
    city_pois = load_city_pois() if "city_pois" in needed else empty_gdf
    city_boundary = load_city_boundary() if "city_boundary" in needed else empty_gdf
    military = load_military_areas() if "military" in needed else empty_gdf
    rail_areas = load_rail_areas() if "rail_areas" in needed else empty_gdf
    airport = load_airport_areas() if "airport" in needed else empty_gdf
    noise_overlay_data = (
        load_warsaw_noise_overlay(flags.noise_overlay)
        if "warsaw_noise" in needed
        else None
    )
    aq_overlay_data = (
        load_gios_aq_overlay(flags.aq_overlay) if "gios_aq" in needed else None
    )

    commute = resolve_commute_target(app_settings, config)
    render_assistant_popover(
        scored=frame,
        processed=proc,
        prefs=prefs,
        settings=app_settings,
    )
    map_kwargs = {
        "amenities": amenities,
        "shops": shops,
        "stops": stops,
        "hourly": hourly,
        "route_shapes": route_shapes,
        "traffic_sample": traffic_sample,
        "air_stations": air_stations,
        "gunb": gunb,
        "parks": parks,
        "bike_paths": bike_paths,
        "city_pois": city_pois,
        "city_boundary": city_boundary,
        "military": military,
        "rail_areas": rail_areas,
        "airport": airport,
        "noise_overlay_data": noise_overlay_data,
        "aq_overlay_data": aq_overlay_data,
        "flags": flags,
        "hour": hour,
        "basemap": basemap,
        "commute_lon": float(commute.lon),
        "commute_lat": float(commute.lat),
        "commute_label": commute.label,
    }
    tab_map, tab_spatial, tab_offer, tab_board, tab_news, tab_chat, tab_prefs = st.tabs(
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
                city_boundary=city_boundary,
                military=military,
                rail_areas=rail_areas,
                airport=airport,
                flags=flags,
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
            # Sync against full scored universe so sidebar filters never fake removals.
            sync_starred_news(
                starred,
                frame,
                snapshots_path=snaps_file,
                news_path=news_file,
                history_cap=app_settings.news_history_cap,
            )
            _tab_news(
                starred=starred,
                items=load_news(news_file),
                news_file=news_file,
                scored=frame,
                prefs=prefs,
                app_settings=app_settings,
                proc=proc,
            )
    with tab_chat:
        if tab_chat.open:
            render_assistant_tab(
                scored=frame,
                processed=proc,
                prefs=prefs,
                settings=app_settings,
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
