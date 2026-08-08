"""Admin page: section nav + short settings panels."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 — Admin runs fixed local pipeline/scrape commands
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Never

import pandas as pd
import streamlit as st

from happy_housing.app.ai_review import (
    DEFAULT_SYSTEM_PROMPT,
    clear_ai_review_session_keys,
    clear_ai_reviews,
    reviews_db_path,
)
from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.attr_extract import enrich_frame_listing_attrs
from happy_housing.app.chat_store import chat_db_path, clear_all_conversations
from happy_housing.app.commute_target import (
    COORD_EPS,
    CommuteTarget,
    commute_compare_view,
    commute_target_stale,
    recompute_commute_features,
    resolve_commute_target,
)
from happy_housing.app.data import (
    clear_data_caches,
    load_config,
    load_metadata,
    load_scored,
    processed_dir,
)
from happy_housing.app.geocode import (
    format_place_option_labels,
    live_search_ready,
    resolve_place_pick,
    search_places,
)
from happy_housing.app.listing_attrs import promote_listing_flags
from happy_housing.app.llm_metrics import (
    clear_llm_metrics,
    llm_metrics_path,
    purpose_stats_frame,
    summarize_llm_calls,
)
from happy_housing.app.maps import commute_compare_layers, render_deck
from happy_housing.app.pipeline_ingest import (
    resolve_ingest_flags,
    resolve_ingest_limits,
    resolve_ingest_urls,
)
from happy_housing.app.preferences import load_preferences, preferences_path
from happy_housing.app.recommender import (
    clear_recommendation_memory,
    impressions_path,
    load_impressions,
    load_recommender_metrics,
    retrain_recommender,
)
from happy_housing.app.star_news import clear_star_news_memory, load_news, news_path
from happy_housing.app.stars import clear_stars, load_stars, stars_path
from happy_housing.logging_config import configure_logging
from happy_housing.models.metrics import (
    OPT_METRICS,
    coerce_opt_metric,
    metric_direction,
)

logger = logging.getLogger(__name__)

ADMIN_SECTIONS: tuple[str, ...] = (
    "Overview",
    "Commute",
    "Run",
    "Sources",
    "Price model",
    "Recommendations",
    "Offer AI",
    "Assistant",
    "Extract",
    "LMS metrics",
    "Stars & News",
    "Stats",
)
"""Left-nav section labels (task frequency order)."""


def _assert_never(value: object) -> Never:
    """Exhaustive ``match`` guard."""
    raise AssertionError(f"unreachable admin section: {value!r}")


def _run_command(command: list[str], timeout_s: int) -> tuple[int, str]:
    """Run a subprocess and return ``(returncode, combined_output)``."""
    completed = subprocess.run(  # nosec B603 — argv built in-process, shell=False
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(Path.cwd()),
    )
    out = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, out


@st.cache_data(ttl=300, show_spinner=False)  # type: ignore[misc]
def _cached_place_search(query: str) -> tuple[dict[str, float | str | int], ...]:
    """Cached Nominatim hits for live Admin place search."""
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


def _panel_overview(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Paths, last refresh, log level, cache clear."""
    del proc  # unused; signature shared with other panels
    config = load_config()
    meta = load_metadata()
    st.caption("Experiment paths and last pipeline refresh.")
    c1, c2 = st.columns(2)
    c1.metric("Processed dir", config.data.processed_path().name)
    c2.metric(
        "Last refresh",
        (
            meta.refreshed_at[:19].replace("T", " ")
            if meta and meta.refreshed_at
            else "—"
        ),
    )
    st.caption(
        "Config file: `configs/experiment.yaml` · fixtures: "
        f"`{config.data.fixtures_path()}`"
    )

    st.subheader("Logging")
    log_level = st.selectbox(
        "App log level",
        options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
            settings.log_level
            if settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            else "INFO"
        ),
        help="Applies to happy_housing.* loggers (stderr). Env "
        "HAPPY_HOUSING_LOG_LEVEL overrides when set at process start.",
        key="settings_log_level",
    )
    g1, g2, _g3 = st.columns([1, 1, 3])
    with g1:
        if st.button("Save logging settings", key="save_logging_settings"):
            settings.log_level = str(log_level)
            save_app_settings(settings_file, settings)
            configure_logging(settings.log_level, force=True)
            logger.info("App log level set to %s", settings.log_level)
            st.success(f"Log level saved: {settings.log_level}")
    with g2:
        if st.button("Clear Streamlit data caches", key="admin_clear_caches"):
            clear_data_caches()
            logger.info("Cleared Streamlit data caches from Admin")
            st.success("Caches cleared.")


def _panel_commute(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Commute destination search, map compare, save / recompute."""
    config = load_config()
    target = resolve_commute_target(settings, config)
    st.caption(
        "Destination for dist_daszynskiego_m / commute_min / OTP transit. "
        "Search uses OpenStreetMap Nominatim (open geocoder). "
        "After changing the place, recompute features (or run the full pipeline)."
    )
    if "admin_commute_lon" not in st.session_state:
        st.session_state["admin_commute_lon"] = float(target.lon)
        st.session_state["admin_commute_lat"] = float(target.lat)
        st.session_state["admin_commute_label"] = str(target.label)

    search_seed = str(st.session_state.get("admin_commute_search_seed") or "").strip()
    hit_rows: tuple[dict[str, float | str | int], ...] = ()
    if live_search_ready(search_seed):
        try:
            hit_rows = _cached_place_search(search_seed)
        except Exception as exc:
            st.error(f"Nominatim search failed: {exc}")
            hit_rows = ()
    option_labels = format_place_option_labels(hit_rows)
    place_pick = st.selectbox(
        "Search place",
        options=option_labels,
        index=None,
        placeholder="Type a place, press Enter, then pick a match…",
        accept_new_options=True,
        key="admin_commute_place_combo",
        help=(
            "Live OpenStreetMap Nominatim search (Poland / Warsaw-biased). "
            "Type a query and press Enter to load matches in this dropdown."
        ),
    )
    if place_pick:
        chosen = resolve_place_pick(str(place_pick), hit_rows, option_labels)
        if chosen is not None and isinstance(chosen, dict):
            new_lon = float(str(chosen["lon"]))
            new_lat = float(str(chosen["lat"]))
            new_label = str(chosen["label"])[:120]
            changed = (
                abs(float(st.session_state["admin_commute_lon"]) - new_lon) > 1e-6
                or abs(float(st.session_state["admin_commute_lat"]) - new_lat) > 1e-6
                or str(st.session_state["admin_commute_label"]) != new_label
            )
            if changed:
                st.session_state["admin_commute_lon"] = new_lon
                st.session_state["admin_commute_lat"] = new_lat
                st.session_state["admin_commute_label"] = new_label
                st.rerun()
        elif live_search_ready(place_pick):
            needle = str(place_pick).strip()
            if needle[:3].rstrip(". ").isdigit() and ". " in needle:
                needle = needle.split(". ", 1)[-1].strip()
            if needle != search_seed:
                st.session_state["admin_commute_search_seed"] = needle
                st.rerun()
            elif not hit_rows:
                st.caption("No Nominatim matches for this query.")
            else:
                st.caption(f"{len(hit_rows)} matches — pick one from the list.")

    lon_in = st.number_input(
        "Longitude",
        format="%.6f",
        key="admin_commute_lon",
    )
    lat_in = st.number_input(
        "Latitude",
        format="%.6f",
        key="admin_commute_lat",
    )
    label_in = st.text_input(
        "Label",
        key="admin_commute_label",
    )
    draft = CommuteTarget(
        lon=float(lon_in),
        lat=float(lat_in),
        label=str(label_in).strip() or target.label,
    )
    view_lat, view_lon, view_zoom = commute_compare_view(saved=target, draft=draft)
    render_deck(
        commute_compare_layers(saved=target, draft=draft),
        latitude=view_lat,
        longitude=view_lon,
        zoom=view_zoom,
        chart_key=(
            f"admin_commute_map_{target.lon:.5f}_{target.lat:.5f}_"
            f"{draft.lon:.5f}_{draft.lat:.5f}_{view_zoom:.1f}"
        ),
    )
    same_pin = (
        abs(float(target.lon) - float(draft.lon)) < COORD_EPS
        and abs(float(target.lat) - float(draft.lat)) < COORD_EPS
    )
    if same_pin:
        st.caption("Pins overlap — draft equals saved target (slate under teal).")
    else:
        st.caption("Slate pin = saved target · Teal pin = new (draft) target")
    st.caption(f"Active: **{target.label}** · {target.lon:.5f}, {target.lat:.5f}")
    if commute_target_stale(settings):
        st.warning(
            "Scored listings still use the previous commute target. "
            "Recompute commute features (fast) or run the full pipeline."
        )
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Save commute target", type="primary", key="save_commute_target"):
            prev = resolve_commute_target(settings, config)
            if settings.commute_features_lon is None:
                settings.commute_features_lon = float(prev.lon)
                settings.commute_features_lat = float(prev.lat)
            settings.commute_target_lon = float(lon_in)
            settings.commute_target_lat = float(lat_in)
            settings.commute_target_label = str(label_in).strip() or prev.label
            save_app_settings(settings_file, settings)
            st.success(
                f"Saved commute target: {settings.commute_target_label} "
                f"({lon_in:.5f}, {lat_in:.5f})"
            )
            if commute_target_stale(settings):
                st.info("Next: click Recompute commute + rescore.")
            st.rerun()
    with b2:
        if st.button("Recompute commute + rescore", key="recompute_commute"):
            dest_label = str(label_in).strip() or target.label
            with st.spinner("Updating distances, commute minutes, and scores…"):
                try:
                    result = recompute_commute_features(
                        proc,
                        dest_lon=float(lon_in),
                        dest_lat=float(lat_in),
                        label=dest_label,
                        config=config,
                        projected_crs=config.bbox.projected_crs,
                        osrm_url=config.routing.osrm_url,
                        use_otp_transit=bool(config.routing.use_otp_transit),
                        otp_url=config.routing.otp_url,
                    )
                    clear_data_caches()
                    st.success(
                        f"Updated {result['n_offers']} offers for "
                        f"{result['label']} "
                        f"({result['lon']:.5f}, {result['lat']:.5f}). "
                        "Full model retrain still recommended after big "
                        "target moves."
                    )
                    st.rerun()
                except Exception as exc:
                    logger.exception("Commute recompute failed")
                    st.error(str(exc))
    with b3:
        if st.button("Reset to YAML default", key="reset_commute_yaml"):
            settings.commute_target_lon = float(config.commute_target_lon)
            settings.commute_target_lat = float(config.commute_target_lat)
            settings.commute_target_label = "Rondo Daszyńskiego"
            save_app_settings(settings_file, settings)
            st.session_state["admin_commute_lon"] = settings.commute_target_lon
            st.session_state["admin_commute_lat"] = settings.commute_target_lat
            st.session_state["admin_commute_label"] = settings.commute_target_label
            st.session_state.pop("admin_commute_search_seed", None)
            st.session_state.pop("admin_commute_place_combo", None)
            st.success("Reset override to experiment.yaml coordinates.")
            st.rerun()


def _apply_pipeline_widgets(
    settings: AppSettings,
    *,
    otodom_unlimited: bool,
    max_pages: int | None,
    max_listings: int | None,
    incremental: bool,
    download_rcn: bool,
    rcn_unlimited: bool,
    rcn_max_rows: int | None,
    rcn_zip_url: str,
    download_gtfs: bool,
    gtfs_zip_url: str,
    download_gios: bool,
    gios_stations_url: str,
    download_gunb: bool,
    gunb_zip_url: str,
    download_crime: bool,
    crime_panorama_7z_url: str,
    use_osm: bool,
    scrape_in_pipe: bool,
    snap_out_of_parks: bool,
    bike_path_coverage_radius_m: float,
) -> None:
    """Copy Pipeline Sources widget values onto ``settings`` (overrides on)."""
    settings.pipeline_overrides_enabled = True
    settings.otodom_max_pages = None if otodom_unlimited else max_pages
    settings.otodom_max_listings = None if otodom_unlimited else max_listings
    settings.otodom_incremental = bool(incremental)
    settings.download_rcn = bool(download_rcn)
    settings.rcn_max_rows = None if rcn_unlimited else rcn_max_rows
    settings.rcn_zip_url = str(rcn_zip_url).strip()
    settings.download_gtfs = bool(download_gtfs)
    settings.gtfs_zip_url = str(gtfs_zip_url).strip()
    settings.download_gios = bool(download_gios)
    settings.gios_stations_url = str(gios_stations_url).strip()
    settings.download_gunb = bool(download_gunb)
    settings.gunb_zip_url = str(gunb_zip_url).strip()
    settings.download_crime = bool(download_crime)
    settings.crime_panorama_7z_url = str(crime_panorama_7z_url).strip()
    settings.use_osm = bool(use_osm)
    settings.scrape_otodom_in_pipeline = bool(scrape_in_pipe)
    settings.snap_listings_out_of_parks = bool(snap_out_of_parks)
    settings.bike_path_coverage_radius_m = float(bike_path_coverage_radius_m)


def _panel_run(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Primary scrape / pipeline actions using saved ingest settings."""
    del proc
    config = load_config()
    limits = resolve_ingest_limits(settings, config)
    flags = resolve_ingest_flags(settings, config)

    pages = "∞" if limits.otodom_max_pages is None else str(limits.otodom_max_pages)
    listings = (
        "∞" if limits.otodom_max_listings is None else str(limits.otodom_max_listings)
    )
    rcn = "∞" if limits.rcn_max_rows is None else f"{limits.rcn_max_rows:,}"
    st.caption(
        f"Uses **saved** Sources caps · Otodom pages `{pages}` · "
        f"listings `{listings}` · incremental "
        f"`{'on' if limits.otodom_incremental else 'off'}` · RCN rows `{rcn}`. "
        "Edit under **Sources**."
    )

    scrape_in_pipe = st.checkbox(
        "Scrape Otodom inside pipeline",
        value=bool(flags.scrape_otodom),
        key="pipe_scrape_otodom",
    )

    r1, r2 = st.columns(2)
    with r1:
        if st.button("Run Otodom scrape", type="primary", key="run_otodom"):
            if scrape_in_pipe != bool(settings.scrape_otodom_in_pipeline):
                settings.scrape_otodom_in_pipeline = bool(scrape_in_pipe)
                settings.pipeline_overrides_enabled = True
                save_app_settings(settings_file, settings)
            out_path = processed_dir() / "otodom_listings.parquet"
            pages_arg = (
                "None"
                if limits.otodom_max_pages is None
                else str(int(limits.otodom_max_pages))
            )
            listings_arg = (
                "None"
                if limits.otodom_max_listings is None
                else str(int(limits.otodom_max_listings))
            )
            incr_arg = "True" if limits.otodom_incremental else "False"
            cmd = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from happy_housing.logging_config import configure_logging; "
                    "from happy_housing.ingest.otodom import scrape_otodom_warsaw; "
                    "configure_logging(); "
                    f"scrape_otodom_warsaw(Path({str(out_path)!r}), "
                    f"max_pages={pages_arg}, max_listings={listings_arg}, "
                    f"incremental={incr_arg})"
                ),
            ]
            logger.info(
                "Admin starting Otodom scrape → %s (max_pages=%s max_listings=%s "
                "incremental=%s)",
                out_path,
                pages_arg,
                listings_arg,
                incr_arg,
            )
            with st.spinner("Scraping Otodom (full run may take hours)…"):
                try:
                    code, log = _run_command(cmd, timeout_s=24 * 3600)
                except subprocess.TimeoutExpired:
                    logger.error("Otodom scrape timed out")
                    st.error("Scrape timed out.")
                else:
                    st.code(log[-8000:] or "(no output)")
                    if code == 0:
                        clear_data_caches()
                        logger.info("Otodom scrape finished → %s", out_path)
                        st.success(f"Scrape finished → {out_path}")
                    else:
                        logger.error("Otodom scrape failed (exit %s)", code)
                        st.error(f"Scrape failed (exit {code})")
    with r2:
        if st.button(
            "Run pipeline (enrich + train + score)",
            type="primary",
            key="run_pipe",
        ):
            settings.scrape_otodom_in_pipeline = bool(scrape_in_pipe)
            settings.pipeline_overrides_enabled = True
            save_app_settings(settings_file, settings)
            cmd = [
                sys.executable,
                "-c",
                "from happy_housing.pipeline import run_pipeline; run_pipeline()",
            ]
            logger.info(
                "Admin starting pipeline (overrides saved; resolve via AppSettings)"
            )
            with st.spinner("Running pipeline (full RCN/Otodom may take hours)…"):
                try:
                    code, log = _run_command(cmd, timeout_s=24 * 3600)
                except subprocess.TimeoutExpired:
                    logger.error("Pipeline timed out")
                    st.error("Pipeline timed out.")
                else:
                    st.code(log[-12000:] or "(no output)")
                    if code == 0:
                        clear_data_caches()
                        logger.info("Pipeline finished successfully")
                        st.success("Pipeline finished — Home metrics refreshed.")
                    else:
                        logger.error("Pipeline failed (exit %s)", code)
                        st.error(f"Pipeline failed (exit {code})")


def _panel_sources(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Ingest caps, URLs, OSM flags — save overrides only."""
    del proc
    config = load_config()
    limits = resolve_ingest_limits(settings, config)
    urls = resolve_ingest_urls(settings, config)
    flags = resolve_ingest_flags(settings, config)

    st.caption(
        "Admin caps override YAML when saved. Full-source downloads can take hours."
    )

    st.subheader("Otodom")
    otodom_unlimited = st.checkbox(
        "Unlimited (ignore page/listing caps)",
        value=limits.otodom_max_pages is None and limits.otodom_max_listings is None,
        help="Uses every search page Otodom reports. Rate-limited; can take hours.",
        key="pipe_otodom_unlimited",
    )
    max_pages: int | None = None
    max_listings: int | None = None
    if not otodom_unlimited:
        max_pages = int(
            st.number_input(
                "Max search pages",
                min_value=1,
                max_value=5000,
                value=int(limits.otodom_max_pages or 50),
                key="pipe_otodom_max_pages",
            )
        )
        max_listings = int(
            st.number_input(
                "Max listings",
                min_value=10,
                max_value=100_000,
                value=int(limits.otodom_max_listings or 1200),
                key="pipe_otodom_max_listings",
            )
        )
    incremental = st.checkbox(
        "Incremental (new/updated only)",
        value=bool(limits.otodom_incremental),
        help=(
            "Sort newest-first, stop once known pages repeat, skip unchanged "
            "detail fetches, merge into existing parquet."
        ),
        key="pipe_otodom_incremental",
    )

    st.subheader("RCN")
    download_rcn = st.checkbox(
        "Download RCN",
        value=bool(flags.download_rcn),
        help="RCN GML is multi-GB; full parse can take a long time.",
        key="pipe_download_rcn",
    )
    rcn_unlimited = st.checkbox(
        "Unlimited RCN rows",
        value=limits.rcn_max_rows is None,
        key="pipe_rcn_unlimited",
    )
    rcn_max_rows: int | None = None
    if not rcn_unlimited:
        rcn_max_rows = int(
            st.number_input(
                "Max RCN rows",
                min_value=1,
                max_value=10_000_000,
                value=int(limits.rcn_max_rows or 50_000),
                key="pipe_rcn_max_rows",
            )
        )

    st.subheader("Feeds")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        download_gtfs = st.checkbox(
            "Download GTFS",
            value=bool(flags.download_gtfs),
            key="pipe_download_gtfs",
        )
    with f2:
        download_gios = st.checkbox(
            "Download GIOŚ",
            value=bool(flags.download_gios),
            key="pipe_download_gios",
        )
    with f3:
        download_gunb = st.checkbox(
            "Download GUNB",
            value=bool(flags.download_gunb),
            key="pipe_download_gunb",
        )
    with f4:
        download_crime = st.checkbox(
            "Download crime",
            value=bool(flags.download_crime),
            help="GUS Panorama dzielnic — police crime rates by district.",
            key="pipe_download_crime",
        )

    st.subheader("OSM / enrich")
    use_osm = st.checkbox(
        "Fetch OSM amenities + shops (else fixture/cache)",
        value=bool(flags.use_osm),
        key="pipe_use_osm",
    )
    snap_out_of_parks = st.checkbox(
        "Snap listing pins out of parks (OSM buildings)",
        value=bool(settings.snap_listings_out_of_parks),
        help=(
            "Otodom/fixture/RCN pins that land inside UM park polygons move to "
            "the nearest OpenStreetMap building centroid outside the park."
        ),
        key="pipe_snap_out_of_parks",
    )
    bike_path_coverage_radius_m = float(
        st.number_input(
            "Bike-path coverage radius (m)",
            min_value=50.0,
            max_value=2_000.0,
            value=float(settings.bike_path_coverage_radius_m),
            step=50.0,
            help=(
                "Neighbourhood radius for bike_path_coverage_m (metres of OSM "
                "cycle infrastructure near each listing). Used at enrich/retrain."
            ),
            key="pipe_bike_path_coverage_radius_m",
        )
    )

    # Preserve Run-panel toggle if present in session; else current settings.
    scrape_in_pipe = bool(
        st.session_state.get("pipe_scrape_otodom", flags.scrape_otodom)
    )
    with st.expander("Source URLs", expanded=False):
        rcn_zip_url = st.text_input(
            "RCN zip URL",
            value=urls.rcn_zip,
            key="pipe_rcn_zip_url",
        )
        gtfs_zip_url = st.text_input(
            "GTFS zip URL",
            value=urls.gtfs_zip,
            key="pipe_gtfs_zip_url",
        )
        gios_stations_url = st.text_input(
            "GIOŚ stations URL",
            value=urls.gios_stations,
            key="pipe_gios_stations_url",
        )
        gunb_zip_url = st.text_input(
            "GUNB zip URL",
            value=urls.gunb_zip,
            key="pipe_gunb_zip_url",
        )
        crime_panorama_7z_url = st.text_input(
            "Crime Panorama .7z URL",
            value=urls.crime_panorama_7z,
            key="pipe_crime_panorama_7z_url",
        )

    if st.button(
        "Save Pipeline settings",
        type="primary",
        key="save_pipeline_settings",
    ):
        _apply_pipeline_widgets(
            settings,
            otodom_unlimited=otodom_unlimited,
            max_pages=max_pages,
            max_listings=max_listings,
            incremental=incremental,
            download_rcn=download_rcn,
            rcn_unlimited=rcn_unlimited,
            rcn_max_rows=rcn_max_rows,
            rcn_zip_url=rcn_zip_url,
            download_gtfs=download_gtfs,
            gtfs_zip_url=gtfs_zip_url,
            download_gios=download_gios,
            gios_stations_url=gios_stations_url,
            download_gunb=download_gunb,
            gunb_zip_url=gunb_zip_url,
            download_crime=download_crime,
            crime_panorama_7z_url=crime_panorama_7z_url,
            use_osm=use_osm,
            scrape_in_pipe=scrape_in_pipe,
            snap_out_of_parks=snap_out_of_parks,
            bike_path_coverage_radius_m=bike_path_coverage_radius_m,
        )
        save_app_settings(settings_file, settings)
        st.success("Pipeline settings saved (overrides YAML ingest).")


def _panel_model(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Price-model HPO metric, validation split, neighborhood comps."""
    del proc
    from happy_housing.models.train import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS

    n_core = len(CORE_FEATURE_COLUMNS)
    n_full = len(FEATURE_COLUMNS)
    nbhd_note = (
        f" · **{n_full}** with RCN nbhd" if settings.model_use_neighborhood else ""
    )
    st.caption(
        f"Fair-price LightGBM · **{n_core}** core features{nbhd_note}. "
        "HPO on validation; test held out. "
        "Retrain via Admin → Run after feature or data changes. "
        "Scored deltas use out-of-fold predictions for train labels."
    )
    metric = st.selectbox(
        "Optimization metric (validation)",
        options=list(OPT_METRICS),
        index=list(OPT_METRICS).index(coerce_opt_metric(settings.model_opt_metric)),
        help=(
            "mae/rmse/MAPE/medae → minimize · r2 → maximize. "
            f"Direction: "
            f"{metric_direction(coerce_opt_metric(settings.model_opt_metric))}."
        ),
        key="settings_model_opt_metric",
    )
    val_size = st.slider(
        "Validation fraction",
        min_value=0.05,
        max_value=0.35,
        value=float(settings.model_val_size),
        step=0.01,
        key="settings_model_val_size",
    )
    trials = st.number_input(
        "HPO trials",
        min_value=1,
        max_value=64,
        value=int(settings.model_hpo_trials),
        help="1 = use YAML defaults only (no search).",
        key="settings_model_hpo_trials",
    )
    train_asks_only = st.checkbox(
        "Train on Otodom/fixture asks only (RCN = comps, not labels)",
        value=bool(settings.train_otodom_labels_only),
        help=(
            "Avoids mixing deed prices with asks and reduces RCN self-leak "
            "when neighborhood features are on."
        ),
        key="settings_train_otodom_only",
    )
    use_nbhd = st.checkbox(
        "Use RCN neighborhood historical prices",
        value=bool(settings.model_use_neighborhood),
        help=(
            "Adds local comparable-sale features (median/mean/IDW zł/m²) and "
            "logs an IDW baseline model in MLflow for comparison. "
            "Self-sales are excluded from each listing's comps."
        ),
        key="settings_model_use_nbhd",
    )
    radius = st.number_input(
        "Neighborhood radius (m)",
        min_value=100.0,
        max_value=3000.0,
        value=float(settings.model_nbhd_radius_m),
        step=50.0,
        key="settings_model_nbhd_radius",
    )
    if st.button("Save Model settings", type="primary", key="save_model_settings"):
        settings.model_opt_metric = coerce_opt_metric(metric)
        settings.model_val_size = float(val_size)
        settings.model_hpo_trials = int(trials)
        settings.model_use_neighborhood = bool(use_nbhd)
        settings.model_nbhd_radius_m = float(radius)
        settings.train_otodom_labels_only = bool(train_asks_only)
        save_app_settings(settings_file, settings)
        st.success(
            f"Model settings saved · optimize {settings.model_opt_metric} "
            f"({metric_direction(coerce_opt_metric(settings.model_opt_metric))}) "
            f"· {settings.model_hpo_trials} trials"
            f"{' · Otodom labels only' if train_asks_only else ''}."
        )


def _panel_recommender(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Bargain recommender retrain and impression memory."""
    from happy_housing.gis.amenity_catalog import (
        AMENITY_TYPES,
        coerce_amenity_weights,
    )

    metrics = load_recommender_metrics(proc)
    impressions = load_impressions(impressions_path(proc))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates", f"{int(metrics.get('n_candidates', 0)):,}")
    m2.metric("Recommended", f"{int(metrics.get('n_recommended', 0)):,}")
    m3.metric(
        "Mean confidence",
        f"{metrics.get('mean_confidence', float('nan')):.3f}" if metrics else "—",
    )
    m4.metric("Impressions tracked", f"{len(impressions):,}")
    top_n = st.number_input(
        "Top-N recommendations",
        min_value=1,
        max_value=100,
        value=int(settings.recommender_top_n),
        key="settings_rec_top_n",
    )
    alpha = st.number_input(
        "Seen demotion alpha",
        min_value=0.0,
        max_value=10.0,
        value=float(settings.recommender_demotion_alpha),
        step=0.1,
        key="settings_rec_alpha",
        help="Higher → already-seen offers rank lower (soft, not excluded).",
    )
    st.markdown("#### Amenity polarity & weight")
    st.caption(
        "Positive = prefer offers near this type; negative = prefer far. "
        "Magnitude -1..+1. Distance soft-cap 2 km."
    )
    weights = coerce_amenity_weights(settings.recommender_amenity_weights)
    weight_rows = [
        {
            "group": t.group,
            "type": t.key,
            "label": t.label,
            "weight": float(weights.get(t.key, 0.0)),
        }
        for t in AMENITY_TYPES
    ]
    edited = st.data_editor(
        pd.DataFrame(weight_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "group": st.column_config.TextColumn("Group", disabled=True),
            "type": st.column_config.TextColumn("Key", disabled=True),
            "label": st.column_config.TextColumn("Object type", disabled=True),
            "weight": st.column_config.NumberColumn(
                "Weight (-1..+1)",
                min_value=-1.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
            ),
        },
        disabled=["group", "type", "label"],
        key="settings_rec_amenity_weights",
    )
    new_weights = {
        str(row["type"]): float(row["weight"])
        for _, row in edited.iterrows()
        if "type" in row and "weight" in row
    }
    new_weights = coerce_amenity_weights(new_weights)
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("Save recommender settings", key="save_rec_settings"):
            settings.recommender_top_n = int(top_n)
            settings.recommender_demotion_alpha = float(alpha)
            settings.recommender_amenity_weights = new_weights
            save_app_settings(settings_file, settings)
            st.success("Recommender settings saved.")
    with r2:
        if st.button("Retrain recommendation model", type="primary", key="retrain_rec"):
            settings.recommender_top_n = int(top_n)
            settings.recommender_demotion_alpha = float(alpha)
            settings.recommender_amenity_weights = new_weights
            save_app_settings(settings_file, settings)
            try:
                frame = load_scored()
            except FileNotFoundError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Rescoring offers…"):
                    prefs = load_preferences(preferences_path(proc))
                    out = retrain_recommender(
                        frame,
                        processed_dir=proc,
                        top_n=int(top_n),
                        demotion_alpha=float(alpha),
                        prefs=prefs,
                        amenity_weights=new_weights,
                    )
                st.success(
                    f"Retrained · {int(out['n_recommended'])} of "
                    f"{int(out['n_candidates'])} candidates · "
                    f"mean confidence {out['mean_confidence']:.3f}"
                )
                st.rerun()
    with r3:
        if st.button("Delete recommendation memory", key="clear_rec_memory"):
            clear_recommendation_memory(proc)
            st.success("Recommendation impressions + metrics deleted.")
            st.rerun()


def _panel_offer_ai(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Offer AI review model, buyer prefs, prompt, cache clear."""
    st.caption("Used for Offer AI reviews (LM Studio must be running locally).")
    lm_model = st.text_input(
        "LM Studio model id (reviews / prefs)",
        value=settings.lm_model,
        help="Must match a model loaded in LM Studio.",
        key="settings_lm_model",
    )
    ai_on_open = st.checkbox(
        "Auto-review when opening Offer → AI review",
        value=settings.ai_review_on_offer_open,
        key="settings_ai_on_open",
    )

    st.subheader("Buyer preferences")
    st.caption("Included in every offer review prompt when the toggle below is on.")
    pref_notes = st.text_area(
        "Preference notes",
        value=settings.buyer_pref_notes,
        height=100,
        key="settings_buyer_notes",
        help="Free-text goals, lifestyle, deal-breakers.",
    )
    must_haves = st.text_input(
        "Must-haves",
        value=settings.buyer_must_haves,
        key="settings_buyer_must",
        help="e.g. balcony, lift, quiet street",
    )
    districts = st.text_input(
        "Preferred districts",
        value=settings.buyer_preferred_districts,
        key="settings_buyer_districts",
        help="Comma-separated, e.g. Wola, Żoliborz",
    )
    p1, p2, p3 = st.columns(3)
    with p1:
        max_price = st.number_input(
            "Max asking price (PLN)",
            min_value=0.0,
            value=float(settings.buyer_max_price_pln or 0.0),
            step=10_000.0,
            key="settings_buyer_max_price",
            help="0 = unset",
        )
        min_area = st.number_input(
            "Min area (m²)",
            min_value=0.0,
            value=float(settings.buyer_min_area_m2 or 0.0),
            step=1.0,
            key="settings_buyer_min_area",
            help="0 = unset",
        )
    with p2:
        max_ppm = st.number_input(
            "Max zł/m²",
            min_value=0.0,
            value=float(settings.buyer_max_price_per_m2 or 0.0),
            step=100.0,
            key="settings_buyer_max_ppm",
            help="0 = unset",
        )
        max_commute = st.number_input(
            "Max commute (min)",
            min_value=0.0,
            value=float(settings.buyer_max_commute_min or 0.0),
            step=1.0,
            key="settings_buyer_max_commute",
            help="0 = unset",
        )
    with p3:
        rooms_min = st.number_input(
            "Min rooms",
            min_value=0,
            value=int(settings.buyer_rooms_min or 0),
            step=1,
            key="settings_buyer_rooms_min",
            help="0 = unset",
        )
        rooms_max = st.number_input(
            "Max rooms",
            min_value=0,
            value=int(settings.buyer_rooms_max or 0),
            step=1,
            key="settings_buyer_rooms_max",
            help="0 = unset",
        )
    avoid_oou = st.checkbox(
        "Avoid aircraft Lden ≥ 55 dB",
        value=settings.buyer_avoid_oou,
        key="settings_buyer_avoid_oou",
        help="Prefer offers outside EEA END major-airport Lden ≥55 dB contours.",
    )

    st.subheader("Review prompt")
    system_prompt = st.text_area(
        "System prompt",
        value=settings.ai_review_system_prompt or DEFAULT_SYSTEM_PROMPT,
        height=160,
        key="settings_ai_system_prompt",
    )
    if st.button("Reset system prompt to default", key="reset_ai_system_prompt"):
        st.session_state["settings_ai_system_prompt"] = DEFAULT_SYSTEM_PROMPT
        st.rerun()

    with st.expander("Prompt sections & generation limits", expanded=False):
        st.caption("Sections included in the user prompt sent with each offer.")
        inc_prefs = st.checkbox(
            "Buyer preferences",
            value=settings.ai_review_include_preferences,
            key="settings_ai_inc_prefs",
        )
        inc_core = st.checkbox(
            "Offer facts (price, rooms, building, …)",
            value=settings.ai_review_include_core_facts,
            key="settings_ai_inc_core",
        )
        inc_model = st.checkbox(
            "Model valuation (listing / model / delta zł/m²)",
            value=settings.ai_review_include_model_valuation,
            key="settings_ai_inc_model",
        )
        inc_prox = st.checkbox(
            "Proximity & livability (transit, metro, hospital, aircraft Lden, …)",
            value=settings.ai_review_include_proximity,
            key="settings_ai_inc_prox",
        )
        inc_nbhd = st.checkbox(
            "Neighborhood price history (RCN comps)",
            value=settings.ai_review_include_neighborhood_prices,
            key="settings_ai_inc_nbhd",
        )
        inc_desc = st.checkbox(
            "Source description",
            value=settings.ai_review_include_description,
            key="settings_ai_inc_desc",
        )
        inc_raw = st.checkbox(
            "Raw Otodom characteristics JSON",
            value=settings.ai_review_include_raw_characteristics,
            key="settings_ai_inc_raw",
        )
        inc_all = st.checkbox(
            "All other stored offer fields",
            value=settings.ai_review_include_all_remaining,
            key="settings_ai_inc_all",
        )
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            max_desc = st.number_input(
                "Max description chars",
                min_value=200,
                max_value=20_000,
                value=int(settings.ai_review_max_description_chars),
                key="settings_ai_max_desc",
            )
        with t2:
            max_field = st.number_input(
                "Max field chars",
                min_value=100,
                max_value=10_000,
                value=int(settings.ai_review_max_field_chars),
                key="settings_ai_max_field",
            )
        with t3:
            temperature = st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=1.5,
                value=float(settings.ai_review_temperature),
                step=0.05,
                key="settings_ai_temp",
            )
        with t4:
            max_tokens = st.number_input(
                "Max tokens",
                min_value=64,
                max_value=4000,
                value=int(settings.ai_review_max_tokens),
                step=50,
                key="settings_ai_max_tokens",
            )

    if st.button(
        "Save Offer AI settings",
        type="primary",
        key="save_offer_ai_settings",
    ):
        settings.lm_model = lm_model.strip() or settings.lm_model
        settings.ai_review_on_offer_open = bool(ai_on_open)
        settings.ai_review_system_prompt = (
            str(system_prompt).strip() or DEFAULT_SYSTEM_PROMPT
        )
        settings.ai_review_include_preferences = bool(inc_prefs)
        settings.ai_review_include_core_facts = bool(inc_core)
        settings.ai_review_include_model_valuation = bool(inc_model)
        settings.ai_review_include_proximity = bool(inc_prox)
        settings.ai_review_include_neighborhood_prices = bool(inc_nbhd)
        settings.ai_review_include_description = bool(inc_desc)
        settings.ai_review_include_raw_characteristics = bool(inc_raw)
        settings.ai_review_include_all_remaining = bool(inc_all)
        settings.ai_review_max_description_chars = int(max_desc)
        settings.ai_review_max_field_chars = int(max_field)
        settings.ai_review_temperature = float(temperature)
        settings.ai_review_max_tokens = int(max_tokens)
        settings.buyer_pref_notes = str(pref_notes)
        settings.buyer_must_haves = str(must_haves)
        settings.buyer_preferred_districts = str(districts)
        settings.buyer_max_price_pln = float(max_price) if max_price > 0 else None
        settings.buyer_max_price_per_m2 = float(max_ppm) if max_ppm > 0 else None
        settings.buyer_min_area_m2 = float(min_area) if min_area > 0 else None
        settings.buyer_max_commute_min = float(max_commute) if max_commute > 0 else None
        settings.buyer_rooms_min = int(rooms_min) if rooms_min > 0 else None
        settings.buyer_rooms_max = int(rooms_max) if rooms_max > 0 else None
        settings.buyer_avoid_oou = bool(avoid_oou)
        save_app_settings(settings_file, settings)
        st.success(
            "Offer AI settings saved (stale reviews auto-regenerate on next ask)."
        )

    db = reviews_db_path(proc)
    st.caption(f"Cached reviews DB: `{'present' if db.exists() else 'empty'}`")
    if st.button("Clear AI review cache", key="clear_ai_reviews"):
        clear_ai_reviews(db)
        cleared = clear_ai_review_session_keys(st.session_state)
        st.success(
            f"AI review cache cleared{f' · {cleared} session keys' if cleared else ''}."
        )


def _panel_assistant(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Explore Assistant chat settings and conversation clear."""
    st.caption(
        "Popover + Chat tab. LMS tool calls over scored listings, prefs, "
        "recommendations, stars, reviews, and web search."
    )
    chat_on = st.checkbox(
        "Enable Assistant",
        value=settings.chat_enabled,
        key="settings_chat_enabled",
    )
    chat_prompt = st.text_area(
        "Assistant system prompt (optional override)",
        value=settings.chat_system_prompt,
        height=100,
        key="settings_chat_prompt",
        help="Leave blank to use the built-in Warsaw housing assistant prompt.",
    )
    c_a, c_b, c_c, c_d = st.columns(4)
    with c_a:
        chat_temp = st.number_input(
            "Chat temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(settings.chat_temperature),
            step=0.05,
            key="settings_chat_temp",
        )
    with c_b:
        chat_tokens = st.number_input(
            "Chat max tokens",
            min_value=64,
            max_value=4000,
            value=int(settings.chat_max_tokens),
            step=50,
            key="settings_chat_tokens",
        )
    with c_c:
        chat_hist = st.number_input(
            "History messages",
            min_value=4,
            max_value=80,
            value=int(settings.chat_history_max_messages),
            key="settings_chat_hist",
        )
    with c_d:
        chat_rounds = st.number_input(
            "Max tool rounds",
            min_value=1,
            max_value=12,
            value=int(settings.chat_max_tool_rounds),
            key="settings_chat_rounds",
        )
    if st.button(
        "Save Assistant settings",
        type="primary",
        key="save_assistant_settings",
    ):
        settings.chat_enabled = bool(chat_on)
        settings.chat_system_prompt = str(chat_prompt or "")
        settings.chat_temperature = float(chat_temp)
        settings.chat_max_tokens = int(chat_tokens)
        settings.chat_history_max_messages = int(chat_hist)
        settings.chat_max_tool_rounds = int(chat_rounds)
        save_app_settings(settings_file, settings)
        st.success("Assistant settings saved.")

    chat_db = chat_db_path(proc)
    st.caption(f"Assistant chats DB: `{'present' if chat_db.exists() else 'empty'}`")
    if st.button("Clear all Assistant conversations", key="clear_chat_convs"):
        clear_all_conversations(chat_db)
        for key in list(st.session_state.keys()):
            if str(key).startswith("assistant_"):
                st.session_state.pop(key, None)
        st.success("Assistant conversations cleared.")


def _panel_extract(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Listing attribute extract from truncated descriptions."""
    st.caption(
        "Lightweight model fills gaps from truncated descriptions "
        "(CSV line, N if unknown). Structured Otodom fields always win."
    )
    extract_model = st.text_input(
        "Extract LM Studio model id",
        value=settings.extract_lm_model,
        help="Default gemma-3-4b-it — keep small for speed.",
        key="settings_extract_lm_model",
    )
    extract_on = st.checkbox(
        "Enable description extract in pipeline",
        value=settings.extract_attrs_enabled,
        key="settings_extract_enabled",
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        extract_chars = st.number_input(
            "Max description chars",
            min_value=200,
            max_value=4000,
            value=int(settings.extract_max_chars),
            key="settings_extract_chars",
        )
    with e2:
        extract_tokens = st.number_input(
            "Max extract tokens",
            min_value=16,
            max_value=128,
            value=int(settings.extract_max_tokens),
            key="settings_extract_tokens",
            help="CSV line only — keep low.",
        )
    with e3:
        extract_cap = st.number_input(
            "Max LM calls / run",
            min_value=0,
            max_value=50_000,
            value=int(settings.extract_max_listings),
            key="settings_extract_cap",
            help="0 = unlimited. Prefer a cap for first runs.",
        )
    s1, s2 = st.columns(2)
    with s1:
        if st.button(
            "Save Extract settings",
            type="primary",
            key="save_extract_settings",
        ):
            settings.extract_lm_model = (
                str(extract_model).strip() or settings.extract_lm_model
            )
            settings.extract_attrs_enabled = bool(extract_on)
            settings.extract_max_chars = int(extract_chars)
            settings.extract_max_tokens = int(extract_tokens)
            settings.extract_max_listings = int(extract_cap)
            save_app_settings(settings_file, settings)
            st.success("Extract settings saved.")
    with s2:
        if st.button("Backfill attrs on scored listings", key="admin_extract_backfill"):
            try:
                frame = load_scored()
            except FileNotFoundError as exc:
                st.error(str(exc))
            else:
                cap = int(extract_cap) or None
                with st.spinner(
                    f"Extracting attrs with {extract_model} (cap={cap or '∞'})…"
                ):
                    work = promote_listing_flags(frame)
                    work = enrich_frame_listing_attrs(
                        work,
                        model=str(extract_model).strip() or settings.extract_lm_model,
                        max_chars=int(extract_chars),
                        max_tokens=int(extract_tokens),
                        use_llm=True,
                        metrics_path=llm_metrics_path(proc),
                        max_llm_calls=cap,
                    )
                    out_path = processed_dir() / "listings_scored.parquet"
                    work.to_parquet(out_path, index=False)
                    clear_data_caches()
                st.success(f"Wrote attrs → {out_path}")
                st.rerun()


def _panel_lms_metrics(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """LM Studio call metrics table."""
    del settings, settings_file
    st.caption(
        "Per-purpose totals from real LMS calls (duration, prompt/response size, "
        "errors). Cached reviews do not count."
    )
    metrics_file = llm_metrics_path(proc)
    stats_rows = purpose_stats_frame(summarize_llm_calls(metrics_file))
    if stats_rows:
        st.dataframe(stats_rows, hide_index=True, width="stretch")
        total_calls = sum(int(r["calls"]) for r in stats_rows)
        total_errors = sum(int(r["errors"]) for r in stats_rows)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total LMS calls", f"{total_calls:,}")
        m2.metric("Total errors", f"{total_errors:,}")
        m3.metric(
            "Overall error ratio",
            f"{(total_errors / total_calls):.1%}" if total_calls else "—",
        )
    else:
        st.info("No LMS calls recorded yet.")
    if st.button("Clear LMS call metrics", key="clear_llm_metrics"):
        clear_llm_metrics(metrics_file)
        st.success("LMS call metrics cleared.")
        st.rerun()


def _panel_stars_news(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Starred offers and News feed memory."""
    stars_file = stars_path(proc)
    starred = load_stars(stars_file)
    news_items = load_news(news_path(proc))
    s1, s2 = st.columns(2)
    s1.metric("Starred offers", f"{len(starred):,}")
    s2.metric("News events", f"{len(news_items):,}")
    cap = st.number_input(
        "News history cap",
        min_value=10,
        max_value=2000,
        value=int(settings.news_history_cap),
        key="settings_news_cap",
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Save news settings", key="save_news_settings"):
            settings.news_history_cap = int(cap)
            save_app_settings(settings_file, settings)
            st.success("News settings saved.")
    with a2:
        if st.button("Clear starred offers", key="clear_stars"):
            clear_stars(stars_file)
            st.success("Starred offers cleared.")
    with a3:
        if st.button("Clear starred news + snapshots", key="clear_star_news"):
            clear_star_news_memory(proc)
            st.success("Starred news memory cleared.")


def _panel_stats(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Stats page aggregation thresholds."""
    del proc
    st.caption(
        "Controls district scoreboards, hunt/watch length, and Model XAI "
        "sample size. Recalculate clears Streamlit caches so Stats reloads "
        "scored listings."
    )
    min_offers = st.number_input(
        "Min offers per district",
        min_value=1,
        max_value=200,
        value=int(settings.stats_min_district_offers),
        key="settings_stats_min_offers",
        help="Districts below this count are hidden on Stats → Market.",
    )
    top_n = st.number_input(
        "Hunt / watch list size",
        min_value=1,
        max_value=100,
        value=int(settings.stats_top_n),
        key="settings_stats_top_n",
    )
    shap_n = st.number_input(
        "SHAP sample size (Model tab)",
        min_value=20,
        max_value=2000,
        value=int(settings.stats_shap_sample_n),
        key="settings_stats_shap_sample_n",
        help="Larger = stabler global SHAP, slower page.",
    )
    xai_top = st.number_input(
        "Top features on XAI charts",
        min_value=5,
        max_value=80,
        value=int(settings.stats_xai_top_n),
        key="settings_stats_xai_top_n",
    )
    s1, s2, _s3 = st.columns([1, 1, 3])
    with s1:
        if st.button("Save Stats settings", type="primary", key="save_stats_settings"):
            settings.stats_min_district_offers = int(min_offers)
            settings.stats_top_n = int(top_n)
            settings.stats_shap_sample_n = int(shap_n)
            settings.stats_xai_top_n = int(xai_top)
            save_app_settings(settings_file, settings)
            st.success("Stats settings saved.")
    with s2:
        if st.button("Recalculate stats", key="recalc_stats"):
            with st.spinner("Reloading scored listings…"):
                clear_data_caches()
                try:
                    frame = load_scored()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                else:
                    logger.info(
                        "Admin stats recalculation: %s offers after cache clear",
                        len(frame),
                    )
                    st.success(
                        f"Stats refreshed · {len(frame):,} scored offers loaded."
                    )


PanelFn = Callable[[AppSettings, Path, Path], None]
"""Admin panel renderer signature."""


def panel_for_section(section: str) -> PanelFn:
    """Map Admin section label to its panel renderer."""
    match section:
        case "Overview":
            return _panel_overview
        case "Commute":
            return _panel_commute
        case "Run":
            return _panel_run
        case "Sources":
            return _panel_sources
        case "Price model":
            return _panel_model
        case "Recommendations":
            return _panel_recommender
        case "Offer AI":
            return _panel_offer_ai
        case "Assistant":
            return _panel_assistant
        case "Extract":
            return _panel_extract
        case "LMS metrics":
            return _panel_lms_metrics
        case "Stars & News":
            return _panel_stars_news
        case "Stats":
            return _panel_stats
        case _:
            _assert_never(section)


def render_admin() -> None:
    """Render Admin with left section nav (one short panel at a time)."""
    logger.info("Rendering Admin")
    st.caption("Pick a section — one panel at a time. No raw settings dump.")

    proc = processed_dir()
    settings_file = settings_path(proc)
    settings = load_app_settings(settings_file)

    nav, body = st.columns([1, 3], gap="large")
    with nav:
        st.markdown("**Section**")
        section = st.radio(
            "Admin section",
            options=list(ADMIN_SECTIONS),
            key="admin_section",
            label_visibility="collapsed",
        )
    with body:
        st.subheader(str(section))
        panel_for_section(str(section))(settings, settings_file, proc)
