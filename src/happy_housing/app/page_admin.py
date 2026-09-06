"""Admin page: section nav + short settings panels."""

import logging
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Never

import pandas as pd
import streamlit as st

from happy_housing.app.admin_flash import pop_admin_flash, set_admin_flash
from happy_housing.app.admin_jobs import (
    format_log_lines_markdown,
    job_elapsed_s,
    load_active_job,
    read_log_tail,
    refresh_admin_job,
    start_admin_job,
    stop_admin_job,
)
from happy_housing.app.admin_ops import clear_pipeline_step_cache
from happy_housing.app.ai_review import (
    DEFAULT_SYSTEM_PROMPT,
    clear_ai_review_session_keys,
    clear_ai_reviews,
    reviews_db_path,
)
from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    recommender_kwargs_from_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.attr_extract import enrich_frame_listing_attrs
from happy_housing.app.chat_store import chat_db_path, clear_all_conversations
from happy_housing.app.commute_modes import (
    COMMUTE_DISPLAY_MODES,
    COMMUTE_MODE_LABELS,
    coerce_commute_display_mode,
)
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
    load_rcn_transactions,
    load_scored,
    processed_dir,
)
from happy_housing.app.detail_backfill_jobs import (
    detail_backfill_command,
    stop_auto_detail_backfill,
)
from happy_housing.app.filter_presets import (
    clear_filter_presets,
    filter_presets_path,
    load_filter_presets,
)
from happy_housing.app.filters import (
    LISTING_KIND_OPTIONS,
    LISTING_STATUS_LABELS,
    LISTING_STATUS_OPTIONS,
    coerce_listing_kind,
    coerce_listing_status,
)
from happy_housing.app.freshness import (
    build_freshness_report,
    derived_to_frame,
    sources_to_frame,
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
from happy_housing.app.lmstudio_models import list_lmstudio_llm_keys
from happy_housing.app.map_view_presets import (
    clear_map_view_presets,
    load_map_view_presets,
    map_view_presets_path,
)
from happy_housing.app.maps import commute_compare_layers, render_deck
from happy_housing.app.offer_comps import (
    format_rcn_coord_quality,
    rcn_coord_source_counts,
)
from happy_housing.app.ops_services import check_pipeline_services
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
from happy_housing.ingest.detail_backfill import count_pending_detail_backfill
from happy_housing.ingest.fx import (
    clear_fx_cache,
    fx_cache_path,
    fx_cache_summary,
    get_pln_rate,
)
from happy_housing.logging_config import configure_logging
from happy_housing.models.metrics import (
    OPT_METRICS,
    coerce_opt_metric,
    metric_direction,
)

logger = logging.getLogger(__name__)

ADMIN_SECTIONS: tuple[str, ...] = (
    "Ops",
    "Freshness",
    "Commute",
    "Sources",
    "Explore",
    "Price model",
    "Recommendations",
    "Offer AI",
    "Assistant",
    "Extract",
    "LMS metrics",
    "Stars & News",
    "Stats",
)
"""Top-nav section labels (Ops hub first, Freshness next, then tunables)."""


def _assert_never(value: object) -> Never:
    """Exhaustive ``match`` guard."""
    raise AssertionError(f"unreachable admin section: {value!r}")


def _launch_ops_job(
    *,
    name: str,
    command: list[str],
    processed: Path,
    log_level: str,
) -> None:
    """Start a background Ops job and force a UI refresh."""
    active = load_active_job(processed)
    if active is not None:
        active = refresh_admin_job(processed, active.job_id)
        if active.status == "running":
            st.warning(
                f"Job `{active.name}` still running (pid {active.pid}). "
                "Stop it before starting another."
            )
            return
    start_admin_job(
        name=name,
        command=command,
        processed_dir=processed,
        env={"HAPPY_HOUSING_LOG_LEVEL": str(log_level)},
    )
    st.session_state.pop("_ops_cleared_job", None)
    st.rerun()


def _render_ops_services(config: object) -> None:
    """Docker / OSRM / OTP readiness row."""
    routing = getattr(config, "routing", None)
    osrm_url = str(getattr(routing, "osrm_url", "http://127.0.0.1:5001"))
    osrm_foot_url = str(getattr(routing, "osrm_foot_url", "http://127.0.0.1:5002"))
    osrm_bike_url = str(getattr(routing, "osrm_bike_url", "http://127.0.0.1:5003"))
    otp_url = str(getattr(routing, "otp_url", "http://127.0.0.1:8080"))
    st.subheader("Routing services")
    st.caption(
        "Car OSRM required for pipeline commute. Foot/bike graphs optional "
        "(`make osrm-prepare` → `make osrm-up`: :5001/:5002/:5003). "
        "OTP optional for transit (`make otp-prepare` → `make otp-up`; "
        "`routing.use_otp_transit` in configs/experiment.yaml)."
    )
    checks = check_pipeline_services(
        osrm_url=osrm_url,
        otp_url=otp_url,
        osrm_foot_url=osrm_foot_url,
        osrm_bike_url=osrm_bike_url,
    )
    cols = st.columns(len(checks))
    for col, check in zip(cols, checks, strict=True):
        with col, st.container(border=True):
            st.metric(
                check.label,
                "up" if check.ok else "down",
            )
            st.caption(check.detail)
            if check.latency_ms is not None:
                st.caption(f"{check.latency_ms:.0f} ms")


def _render_ops_job_monitor(settings: AppSettings, processed: Path) -> None:
    """Live log window for the active background job."""
    st.subheader("Live job log")
    poll_s = float(settings.ops_log_poll_s)
    tail_n = int(settings.ops_log_tail_lines)

    @st.fragment(run_every=timedelta(seconds=poll_s))  # type: ignore[misc]  # Streamlit stub
    def _monitor() -> None:
        job = load_active_job(processed)
        if job is None:
            st.caption("No active Ops job. Start a scrape or pipeline below.")
            return
        job = refresh_admin_job(processed, job.job_id)
        elapsed = job_elapsed_s(job)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Job", job.name)
        m2.metric("Status", job.status)
        m3.metric("Elapsed", f"{elapsed:.0f}s")
        m4.metric("PID", str(job.pid))
        if job.status == "running":
            if st.button("Stop job", key=f"ops_stop_{job.job_id}"):
                stop_admin_job(processed, job.job_id)
                st.rerun()
        elif job.status == "succeeded":
            if st.session_state.get("_ops_cleared_job") != job.job_id:
                clear_data_caches()
                st.session_state["_ops_cleared_job"] = job.job_id
            st.success(f"Finished OK in {elapsed:.0f}s · returncode={job.returncode}")
        else:
            st.error(
                f"Job {job.status} · returncode={job.returncode} · {job.error or ''}"
            )
        log_text = read_log_tail(Path(job.log_path), max_lines=tail_n)
        # Newest first so the tile's default (top) scroll keeps live lines visible
        # without custom CSS/JS auto-scroll.
        lines = list(reversed(log_text.splitlines())) if log_text else []
        st.caption(
            f"Newest {len(lines)} / {tail_n} lines (top) · refresh every "
            f"{poll_s:.0f}s · `{job.log_path}`"
        )
        with st.container(height=320, border=True):
            st.markdown(format_log_lines_markdown(lines))

    _monitor()


@st.cache_data(ttl=300, show_spinner=False)  # type: ignore[misc]  # Streamlit decorator stub
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


def _panel_ops(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Recompute / redownload, logging, and caches."""
    config = load_config()
    meta = load_metadata()
    st.caption(
        "Ops hub — scrape / pipeline jobs, logging, and caches. "
        "Source + model freshness → **Freshness**. "
        "Tune ingest URLs and caps under **Sources**. Jobs run in the "
        "background; watch the live log window (not a blocking spinner)."
    )
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed dir", config.data.processed_path().name)
        c2.metric(
            "Last pipeline refresh",
            (
                meta.refreshed_at[:19].replace("T", " ")
                if meta and meta.refreshed_at
                else "—"
            ),
        )
        c3.metric(
            "Fixtures",
            config.data.fixtures_path().name,
        )
        st.caption(
            f"Config: `configs/experiment.yaml` · `{config.data.fixtures_path()}`"
        )

    st.divider()
    _render_ops_services(config)
    st.divider()
    st.info(
        "Ops jobs use `posix_spawn` + a background worker started by `make app` "
        "(Streamlit cannot fork children on macOS). "
        "CLI fallback: `uv run python -m happy_housing.pipeline`."
    )
    _render_ops_job_monitor(settings, proc)
    st.divider()

    limits = resolve_ingest_limits(settings, config)
    flags = resolve_ingest_flags(settings, config)
    pages = "∞" if limits.otodom_max_pages is None else str(limits.otodom_max_pages)
    listings = (
        "∞" if limits.otodom_max_listings is None else str(limits.otodom_max_listings)
    )
    rcn = "∞" if limits.rcn_max_rows is None else f"{limits.rcn_max_rows:,}"
    olx_listings = (
        "∞" if limits.olx_max_listings is None else f"{limits.olx_max_listings:,}"
    )
    nol_listings = (
        "∞" if limits.nol_max_listings is None else f"{limits.nol_max_listings:,}"
    )

    st.subheader("Jobs")
    pending = count_pending_detail_backfill(proc)
    pending_otodom = int(pending.get("otodom", 0))
    pending_nol = int(pending.get("nol", 0))
    st.caption(
        f"Caps from **Sources** · Otodom `{pages}` pages / `{listings}` listings "
        f"(incr `{'on' if limits.otodom_incremental else 'off'}`) · "
        f"detail pending **{pending_otodom}** Otodom / **{pending_nol}** NOL · "
        f"proxies **`{'on' if settings.http_free_proxies_enabled else 'off'}`** "
        f"(Sources → Detail backfill) · "
        f"NOL `{nol_listings}` "
        f"(incr `{'on' if limits.nol_incremental else 'off'}`, "
        f"rent `{'on' if limits.nol_include_rent else 'off'}`) · "
        f"OLX `{olx_listings}` "
        f"(incr `{'on' if limits.olx_incremental else 'off'}`, "
        f"rent `{'on' if limits.olx_include_rent else 'off'}`) · "
        f"RCN `{rcn}`."
    )

    scrape_col, pipe_col = st.columns(2)
    with scrape_col, st.container(border=True):
        st.markdown("**Listing scrapes**")
        st.caption("Background jobs · watch the live log above.")
        scrape_in_pipe = st.checkbox(
            "Include Otodom in pipeline",
            value=bool(flags.scrape_otodom),
            key="pipe_scrape_otodom",
        )
        scrape_olx_in_pipe = st.checkbox(
            "Include OLX in pipeline",
            value=bool(flags.scrape_olx),
            key="pipe_scrape_olx",
        )
        scrape_nol_in_pipe = st.checkbox(
            "Include NOL in pipeline",
            value=bool(flags.scrape_nol),
            key="pipe_scrape_nol",
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            run_otodom = st.button(
                "Otodom scrape",
                type="primary",
                key="run_otodom",
                use_container_width=True,
            )
        with s2:
            run_olx = st.button(
                "OLX scrape",
                type="primary",
                key="run_olx",
                use_container_width=True,
            )
        with s3:
            run_nol = st.button(
                "NOL scrape",
                type="primary",
                key="run_nol",
                use_container_width=True,
            )
        auto_bf = "on" if settings.auto_detail_backfill_enabled else "off"
        st.caption(
            f"Auto detail backfill **{auto_bf}** "
            "(Sources → Detail backfill; stop below halts schedule + running batch)."
        )
        b1, b2 = st.columns(2)
        with b1:
            run_detail = st.button(
                "Detail backfill batch",
                key="run_detail_backfill",
                help=(
                    "Fetch Otodom/NOL detail pages for listings missing coords/attrs "
                    "(one batch; auto-schedules while pending when enabled)."
                ),
                use_container_width=True,
            )
        with b2:
            stop_auto_detail = st.button(
                "Stop auto detail backfill",
                key="stop_auto_detail_backfill",
                help=(
                    "Turn off auto-scheduled detail downloads and stop a running "
                    "detail_backfill job. Re-enable under Sources → Detail backfill."
                ),
                use_container_width=True,
            )
    with pipe_col, st.container(border=True):
        st.markdown("**Pipeline**")
        st.caption(
            "Incremental: scrape new/changed listings; GIS/OSRM/OTP/photo only "
            "for new or moved pins. Force full recompute re-enriches every row. "
            "Force GIS re-downloads archives."
        )
        force_refresh = st.checkbox(
            "Force full recompute",
            value=bool(settings.pipeline_force_refresh),
            help=(
                "Re-run enrich → train → score → recommend even when step "
                "fingerprints match. Does not re-download GUNB / RCN / GTFS / "
                "OSM — tick Force GIS / remote refresh for that."
            ),
            key="pipe_force_refresh",
        )
        force_gis = st.checkbox(
            "Force GIS / remote refresh",
            value=False,
            help=(
                "Re-download open-data archives and map caches: GUNB zip + WFS, "
                "RCN zip, GTFS, GIOŚ, crime, OSM / Warsaw layers. Use this to "
                "re-extract RCN parcel geometry when Offer Comps is empty "
                "(centroid-only deeds). Leave off for day-to-day listing + "
                "model runs (reuse local files)."
            ),
            key="pipe_force_gis",
        )
        p1, p2 = st.columns(2)
        with p1:
            run_pipe = st.button(
                "Run pipeline",
                type="primary",
                key="run_pipe",
                use_container_width=True,
            )
        with p2:
            clear_steps = st.button(
                "Clear step cache",
                key="admin_clear_step_cache",
                use_container_width=True,
            )

    if run_otodom:
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
        proxy_flag = bool(settings.http_free_proxies_enabled)
        level = str(settings.log_level)
        cmd = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from happy_housing.logging_config import configure_logging; "
                "from happy_housing.ingest.proxy_pool import "
                "configure_http_proxies; "
                "from happy_housing.ingest.otodom import scrape_otodom_warsaw; "
                f"configure_logging({level!r}, force=True); "
                f"configure_http_proxies(enabled={proxy_flag!r}); "
                f"scrape_otodom_warsaw(Path({str(out_path)!r}), "
                f"max_pages={pages_arg}, max_listings={listings_arg}, "
                f"incremental={incr_arg}, fetch_details=False)"
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
        _launch_ops_job(
            name="otodom_scrape",
            command=cmd,
            processed=proc,
            log_level=level,
        )

    if run_olx:
        if scrape_olx_in_pipe != bool(settings.scrape_olx_in_pipeline):
            settings.scrape_olx_in_pipeline = bool(scrape_olx_in_pipe)
            settings.pipeline_overrides_enabled = True
            save_app_settings(settings_file, settings)
        out_path = processed_dir() / "olx_listings.parquet"
        listings_arg = (
            "None"
            if limits.olx_max_listings is None
            else str(int(limits.olx_max_listings))
        )
        incr_arg = "True" if limits.olx_incremental else "False"
        rent_arg = "True" if limits.olx_include_rent else "False"
        proxy_flag = bool(settings.http_free_proxies_enabled)
        level = str(settings.log_level)
        cmd = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from happy_housing.logging_config import configure_logging; "
                "from happy_housing.ingest.proxy_pool import "
                "configure_http_proxies; "
                "from happy_housing.ingest.olx import scrape_olx_warsaw; "
                f"configure_logging({level!r}, force=True); "
                f"configure_http_proxies(enabled={proxy_flag!r}); "
                f"scrape_olx_warsaw(Path({str(out_path)!r}), "
                f"max_listings={listings_arg}, incremental={incr_arg}, "
                f"include_rent={rent_arg})"
            ),
        ]
        logger.info(
            "Admin starting OLX scrape → %s (max_listings=%s incremental=%s rent=%s)",
            out_path,
            listings_arg,
            incr_arg,
            rent_arg,
        )
        _launch_ops_job(
            name="olx_scrape",
            command=cmd,
            processed=proc,
            log_level=level,
        )

    if run_nol:
        if scrape_nol_in_pipe != bool(settings.scrape_nol_in_pipeline):
            settings.scrape_nol_in_pipeline = bool(scrape_nol_in_pipe)
            settings.pipeline_overrides_enabled = True
            save_app_settings(settings_file, settings)
        out_path = processed_dir() / "nol_listings.parquet"
        pages_arg = (
            "None" if limits.nol_max_pages is None else str(int(limits.nol_max_pages))
        )
        listings_arg = (
            "None"
            if limits.nol_max_listings is None
            else str(int(limits.nol_max_listings))
        )
        incr_arg = "True" if limits.nol_incremental else "False"
        rent_arg = "True" if limits.nol_include_rent else "False"
        proxy_flag = bool(settings.http_free_proxies_enabled)
        level = str(settings.log_level)
        cmd = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from happy_housing.logging_config import configure_logging; "
                "from happy_housing.ingest.proxy_pool import "
                "configure_http_proxies; "
                "from happy_housing.ingest.nol import scrape_nol_warsaw; "
                f"configure_logging({level!r}, force=True); "
                f"configure_http_proxies(enabled={proxy_flag!r}); "
                f"scrape_nol_warsaw(Path({str(out_path)!r}), "
                f"max_pages={pages_arg}, max_listings={listings_arg}, "
                f"incremental={incr_arg}, include_rent={rent_arg}, "
                f"fetch_details=False)"
            ),
        ]
        logger.info(
            "Admin starting NOL scrape → %s (max_pages=%s max_listings=%s "
            "incremental=%s rent=%s)",
            out_path,
            pages_arg,
            listings_arg,
            incr_arg,
            rent_arg,
        )
        _launch_ops_job(
            name="nol_scrape",
            command=cmd,
            processed=proc,
            log_level=level,
        )

    if run_detail:
        level = str(settings.log_level)
        _launch_ops_job(
            name="detail_backfill",
            command=detail_backfill_command(proc, settings),
            processed=proc,
            log_level=level,
        )

    if stop_auto_detail:
        result = stop_auto_detail_backfill(proc)
        parts: list[str] = []
        if result.auto_disabled:
            parts.append("auto-schedule off")
        elif not result.auto_was_enabled:
            parts.append("auto-schedule already off")
        if result.job_stopped:
            parts.append(f"stopped job `{result.job_id}`")
        else:
            parts.append("no running detail_backfill job")
        set_admin_flash(
            "success",
            " · ".join(parts) + ". Re-enable under Sources → Detail backfill.",
        )
        st.rerun()

    if run_pipe:
        settings.scrape_otodom_in_pipeline = bool(scrape_in_pipe)
        settings.scrape_olx_in_pipeline = bool(scrape_olx_in_pipe)
        settings.scrape_nol_in_pipeline = bool(scrape_nol_in_pipe)
        settings.pipeline_force_refresh = bool(force_refresh)
        settings.pipeline_overrides_enabled = True
        save_app_settings(settings_file, settings)
        gis_flag = bool(force_gis)
        level = str(settings.log_level)
        cmd = [
            sys.executable,
            "-c",
            (
                "from happy_housing.logging_config import configure_logging; "
                "from happy_housing.pipeline import run_pipeline; "
                f"configure_logging({level!r}, force=True); "
                f"run_pipeline(force_gis_refresh={gis_flag!r})"
            ),
        ]
        logger.info(
            "Admin starting pipeline (overrides saved; force_gis=%s)",
            gis_flag,
        )
        _launch_ops_job(
            name="pipeline",
            command=cmd,
            processed=proc,
            log_level=level,
        )

    if clear_steps:
        removed = clear_pipeline_step_cache(proc)
        clear_data_caches()
        if removed:
            st.success("Deleted pipeline_steps.json — next run recomputes steps.")
        else:
            st.info("No pipeline_steps.json present.")

    st.divider()
    st.subheader("Logging & caches")
    with st.container(border=True):
        log_level = st.selectbox(
            "App log level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                settings.log_level
                if settings.log_level
                in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
                else "INFO"
            ),
            help=(
                "Applies to happy_housing.* loggers (stderr). Env "
                "HAPPY_HOUSING_LOG_LEVEL overrides when set at process start."
            ),
            key="settings_log_level",
        )
        l1, l2 = st.columns(2)
        with l1:
            poll_s = st.number_input(
                "Ops log refresh (s)",
                min_value=1.0,
                max_value=30.0,
                value=float(settings.ops_log_poll_s),
                step=1.0,
                key="settings_ops_log_poll_s",
            )
        with l2:
            tail_n = st.number_input(
                "Ops log tail lines",
                min_value=20,
                max_value=2000,
                value=int(settings.ops_log_tail_lines),
                step=20,
                key="settings_ops_log_tail_lines",
            )
        g1, g2 = st.columns(2)
        with g1:
            if st.button(
                "Save logging settings",
                key="save_logging_settings",
                use_container_width=True,
            ):
                settings.log_level = str(log_level)
                settings.ops_log_poll_s = float(poll_s)
                settings.ops_log_tail_lines = int(tail_n)
                save_app_settings(settings_file, settings)
                configure_logging(settings.log_level, force=True)
                logger.info("App log level set to %s", settings.log_level)
                st.success(f"Log level saved: {settings.log_level}")
                st.rerun()
        with g2:
            if st.button(
                "Clear Streamlit caches",
                key="admin_clear_caches",
                use_container_width=True,
            ):
                clear_data_caches()
                logger.info("Cleared Streamlit data caches from Admin")
                st.success("Caches cleared.")


def _panel_freshness(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Local-disk source + derived product freshness dashboard."""
    config = load_config()
    st.caption(
        "Local disk freshness — preferred ``processed``/``cache`` artifacts vs "
        "fixture fallbacks, last mtime, and whether scored listings / model / "
        "recommender / run metadata lag upstream sources. "
        "Remote completeness audit: ``docs/data-sources/``. "
        "Scrape / pipeline actions stay under **Ops**."
    )
    report = build_freshness_report(
        processed_dir=proc,
        cache_dir=config.data.cache_path(),
        fixtures_dir=config.data.fixtures_path(),
        source_warn_h=float(settings.freshness_source_warn_h),
        derived_warn_h=float(settings.freshness_derived_warn_h),
    )
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sources ok", report.n_sources_ok)
        m2.metric("Fixture only", report.n_sources_fixture)
        m3.metric(
            "Missing / empty",
            report.n_sources_missing + report.n_sources_empty,
        )
        m4.metric(
            "Derived ok / stale / missing",
            (
                f"{report.n_derived_ok} / {report.n_derived_stale} / "
                f"{report.n_derived_missing}"
            ),
        )

    st.divider()
    src_col, der_col = st.columns(2)
    with src_col, st.container(border=True):
        st.markdown("**Data sources**")
        st.dataframe(
            sources_to_frame(report.sources),
            hide_index=True,
            width="stretch",
        )
    with der_col, st.container(border=True):
        st.markdown("**Derived products**")
        st.dataframe(
            derived_to_frame(report.derived),
            hide_index=True,
            width="stretch",
        )

    st.divider()
    with st.container(border=True):
        st.markdown("**Warn thresholds**")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            source_warn = st.number_input(
                "Source age warn (hours)",
                min_value=1.0,
                max_value=8760.0,
                value=float(settings.freshness_source_warn_h),
                step=24.0,
                help="Marks age_warn when a source mtime is older than this.",
                key="settings_freshness_source_warn_h",
            )
        with c2:
            derived_warn = st.number_input(
                "Derived age warn (hours)",
                min_value=1.0,
                max_value=8760.0,
                value=float(settings.freshness_derived_warn_h),
                step=24.0,
                help=(
                    "Marks derived products stale when older than this, "
                    "even if newer than sources."
                ),
                key="settings_freshness_derived_warn_h",
            )
        with c3:
            st.write("")
            st.write("")
            if st.button(
                "Save",
                key="save_freshness_settings",
                use_container_width=True,
                type="primary",
            ):
                settings.freshness_source_warn_h = float(source_warn)
                settings.freshness_derived_warn_h = float(derived_warn)
                save_app_settings(settings_file, settings)
                st.success("Freshness warn thresholds saved.")
                st.rerun()


def _panel_commute(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Commute destination search, map compare, save / recompute."""
    config = load_config()
    target = resolve_commute_target(settings, config)
    st.caption(
        "Destination for car / transit / bike / walk commute minutes. "
        "Search uses OpenStreetMap Nominatim (open geocoder). "
        "After changing the place, recompute features (or run the full pipeline). "
        "OSRM: car :5001 · foot :5002 · bike :5003 (`make osrm-up`). "
        f"OTP transit: {'on' if config.routing.use_otp_transit else 'off'} "
        f"({config.routing.otp_url}) — `make otp-up` + "
        "`routing.use_otp_transit: true` in experiment.yaml."
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
    mode_labels = [COMMUTE_MODE_LABELS[m] for m in COMMUTE_DISPLAY_MODES]
    mode_by_label = {COMMUTE_MODE_LABELS[m]: m for m in COMMUTE_DISPLAY_MODES}
    current_mode = coerce_commute_display_mode(settings.commute_display_mode)
    mode_choice = st.selectbox(
        "Offer hover commute mode",
        options=mode_labels,
        index=mode_labels.index(COMMUTE_MODE_LABELS[current_mode]),
        key="admin_commute_display_mode",
        help=(
            "Which commute minutes appear on Explore map pin hover and as the "
            "primary livability row. Default: public transport. "
            "Car / bike / walk use OSRM (bike/foot graphs optional; else speed "
            "proxy). Transit uses OTP when enabled."
        ),
    )
    settings.commute_display_mode = mode_by_label[str(mode_choice)]
    if settings.commute_display_mode != current_mode:
        save_app_settings(settings_file, settings)
        st.rerun()
    show_route = st.checkbox(
        "Show commute route on Offer Map",
        value=bool(settings.offer_map_show_commute_route),
        key="admin_offer_map_show_commute_route",
        help=(
            "Offer → Map draws the door-to-door path for the selected mode "
            "(OTP transit itinerary or OSRM car/bike/walk). Departure hour "
            "comes from Explore Layers → hour slider."
        ),
    )
    if bool(show_route) != bool(settings.offer_map_show_commute_route):
        settings.offer_map_show_commute_route = bool(show_route)
        save_app_settings(settings_file, settings)
        st.rerun()
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
            logger.info(
                "Saved commute target %s (%.5f, %.5f)",
                settings.commute_target_label,
                float(lon_in),
                float(lat_in),
            )
            set_admin_flash(
                "success",
                f"Saved commute target: {settings.commute_target_label} "
                f"({lon_in:.5f}, {lat_in:.5f})",
            )
            if commute_target_stale(settings):
                set_admin_flash(
                    "info",
                    "Commute target saved. Next: click Recompute commute + rescore.",
                )
            st.rerun()
    with b2:
        if st.button("Recompute commute + rescore", key="recompute_commute"):
            dest_label = str(label_in).strip() or target.label
            logger.info(
                "Admin commute recompute clicked → %s (%.5f, %.5f)",
                dest_label,
                float(lon_in),
                float(lat_in),
            )
            with st.status(
                "Recomputing commute + rescore…",
                expanded=True,
            ) as status:
                try:

                    def _progress(message: str) -> None:
                        st.write(message)

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
                        on_progress=_progress,
                    )
                    st.write("Clearing Streamlit data caches…")
                    clear_data_caches()
                    status.update(
                        label="Commute recompute complete",
                        state="complete",
                    )
                    set_admin_flash(
                        "success",
                        f"Updated {result['n_offers']:,} offers for "
                        f"{result['label']} "
                        f"({result['lon']:.5f}, {result['lat']:.5f}). "
                        "Full model retrain still recommended after big "
                        "target moves.",
                    )
                    st.rerun()
                except Exception as exc:
                    logger.exception("Commute recompute failed")
                    status.update(
                        label="Commute recompute failed",
                        state="error",
                    )
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
            logger.info("Reset commute target to YAML default")
            set_admin_flash(
                "success",
                "Reset override to experiment.yaml coordinates.",
            )
            st.rerun()


def _apply_pipeline_widgets(
    settings: AppSettings,
    *,
    otodom_unlimited: bool,
    max_pages: int | None,
    max_listings: int | None,
    incremental: bool,
    otodom_fetch_details: bool,
    auto_detail_backfill_enabled: bool,
    detail_backfill_batch_size: int,
    detail_backfill_workers: int,
    detail_backfill_sleep_s: float,
    olx_unlimited: bool,
    olx_max_listings: int | None,
    olx_incremental: bool,
    olx_include_rent: bool,
    nol_unlimited: bool,
    nol_max_pages: int | None,
    nol_max_listings: int | None,
    nol_incremental: bool,
    nol_include_rent: bool,
    nol_fetch_details: bool,
    http_free_proxies_enabled: bool,
    download_rcn: bool,
    rcn_unlimited: bool,
    rcn_max_rows: int | None,
    rcn_zip_url: str,
    download_gtfs: bool,
    gtfs_zip_url: str,
    download_gios: bool,
    gios_stations_url: str,
    aq_field_enabled: bool,
    aq_idw_power: float,
    download_gunb: bool,
    gunb_zip_url: str,
    gunb_wfs_full_history: bool,
    download_crime: bool,
    crime_panorama_7z_url: str,
    use_osm: bool,
    scrape_in_pipe: bool,
    scrape_olx_in_pipe: bool,
    scrape_nol_in_pipe: bool,
    snap_out_of_parks: bool,
    bike_path_coverage_radius_m: float,
    traffic_neighbourhood_radius_m: float,
) -> None:
    """Copy Pipeline Sources widget values onto ``settings`` (overrides on)."""
    settings.pipeline_overrides_enabled = True
    settings.otodom_max_pages = None if otodom_unlimited else max_pages
    settings.otodom_max_listings = None if otodom_unlimited else max_listings
    settings.otodom_incremental = bool(incremental)
    settings.otodom_fetch_details = bool(otodom_fetch_details)
    settings.auto_detail_backfill_enabled = bool(auto_detail_backfill_enabled)
    settings.detail_backfill_batch_size = max(1, int(detail_backfill_batch_size))
    settings.detail_backfill_workers = max(1, int(detail_backfill_workers))
    settings.detail_backfill_sleep_s = float(detail_backfill_sleep_s)
    settings.olx_max_listings = None if olx_unlimited else olx_max_listings
    settings.olx_incremental = bool(olx_incremental)
    settings.olx_include_rent = bool(olx_include_rent)
    settings.nol_max_pages = None if nol_unlimited else nol_max_pages
    settings.nol_max_listings = None if nol_unlimited else nol_max_listings
    settings.nol_incremental = bool(nol_incremental)
    settings.nol_include_rent = bool(nol_include_rent)
    settings.nol_fetch_details = bool(nol_fetch_details)
    settings.http_free_proxies_enabled = bool(http_free_proxies_enabled)
    settings.download_rcn = bool(download_rcn)
    settings.rcn_max_rows = None if rcn_unlimited else rcn_max_rows
    settings.rcn_zip_url = str(rcn_zip_url).strip()
    settings.download_gtfs = bool(download_gtfs)
    settings.gtfs_zip_url = str(gtfs_zip_url).strip()
    settings.download_gios = bool(download_gios)
    settings.gios_stations_url = str(gios_stations_url).strip()
    settings.aq_field_enabled = bool(aq_field_enabled)
    settings.aq_idw_power = float(aq_idw_power)
    settings.download_gunb = bool(download_gunb)
    settings.gunb_zip_url = str(gunb_zip_url).strip()
    settings.gunb_wfs_full_history = bool(gunb_wfs_full_history)
    settings.download_crime = bool(download_crime)
    settings.crime_panorama_7z_url = str(crime_panorama_7z_url).strip()
    settings.use_osm = bool(use_osm)
    settings.scrape_otodom_in_pipeline = bool(scrape_in_pipe)
    settings.scrape_olx_in_pipeline = bool(scrape_olx_in_pipe)
    settings.scrape_nol_in_pipeline = bool(scrape_nol_in_pipe)
    settings.snap_listings_out_of_parks = bool(snap_out_of_parks)
    settings.bike_path_coverage_radius_m = float(bike_path_coverage_radius_m)
    settings.traffic_neighbourhood_radius_m = float(traffic_neighbourhood_radius_m)


def _panel_sources(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Ingest caps, URLs, OSM flags — save overrides only."""
    config = load_config()
    limits = resolve_ingest_limits(settings, config)
    urls = resolve_ingest_urls(settings, config)
    flags = resolve_ingest_flags(settings, config)

    st.caption(
        "Admin caps override YAML when saved. Full-source downloads can take hours."
    )

    st.subheader("Otodom")
    scrape_in_pipe = st.checkbox(
        "Scrape Otodom in pipeline",
        value=bool(flags.scrape_otodom),
        help="Pull Warsaw sale listings during full pipeline runs.",
        key="pipe_scrape_otodom",
    )
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
            "Sort by=LATEST&direction=DESC, stop after known newest pages, "
            "skip unchanged detail fetches, merge into existing parquet. "
            "Unlimited page caps still early-stop — uncheck only for a full "
            "rebuild."
        ),
        key="pipe_otodom_incremental",
    )
    otodom_fetch_details = st.checkbox(
        "Fetch Otodom details during scrape/pipeline",
        value=bool(limits.otodom_fetch_details),
        help=(
            "Off (default): search cards only; background detail-backfill job "
            "fills coords and rich attrs in batches."
        ),
        key="pipe_otodom_fetch_details",
    )

    st.subheader("Nieruchomości Online")
    scrape_nol_in_pipe = st.checkbox(
        "Scrape NOL in pipeline",
        value=bool(flags.scrape_nol),
        help="JSON-LD Warsaw sale listings during full pipeline runs.",
        key="pipe_scrape_nol",
    )
    nol_unlimited = st.checkbox(
        "Unlimited (ignore NOL page/listing caps)",
        value=limits.nol_max_pages is None and limits.nol_max_listings is None,
        help="Walks every search page until empty or incremental early-stop.",
        key="pipe_nol_unlimited",
    )
    nol_max_pages: int | None = None
    nol_max_listings: int | None = None
    if not nol_unlimited:
        nol_max_pages = int(
            st.number_input(
                "Max NOL search pages",
                min_value=1,
                max_value=5000,
                value=int(limits.nol_max_pages or 50),
                key="pipe_nol_max_pages",
            )
        )
        nol_max_listings = int(
            st.number_input(
                "Max NOL listings",
                min_value=10,
                max_value=100_000,
                value=int(limits.nol_max_listings or 2000),
                key="pipe_nol_max_listings",
            )
        )
    nol_incremental = st.checkbox(
        "NOL incremental (new/updated only)",
        value=bool(limits.nol_incremental),
        help=(
            "Newest-first search; stop after known newest pages; merge into "
            "existing parquet. Uncheck only for a full rebuild."
        ),
        key="pipe_nol_incremental",
    )
    nol_include_rent = st.checkbox(
        "Include NOL rentals",
        value=bool(limits.nol_include_rent),
        help="Rent rows stay in Explore; price-model labels use sale only.",
        key="pipe_nol_include_rent",
    )
    nol_fetch_details = st.checkbox(
        "Fetch NOL details during scrape/pipeline",
        value=bool(limits.nol_fetch_details),
        help=(
            "Off (default): search JSON-LD + tilesData; background detail "
            "backfill fills remaining attrs."
        ),
        key="pipe_nol_fetch_details",
    )

    st.subheader("Detail backfill")
    auto_detail_backfill_enabled = st.checkbox(
        "Auto-schedule detail backfill",
        value=bool(settings.auto_detail_backfill_enabled),
        help=(
            "Ops worker queues detail-backfill batches while Otodom/NOL rows lack "
            "coords/attrs or search price changed."
        ),
        key="pipe_auto_detail_backfill",
    )
    db1, db2, db3 = st.columns(3)
    with db1:
        detail_backfill_batch_size = int(
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=5000,
                value=int(settings.detail_backfill_batch_size),
                key="pipe_detail_backfill_batch",
            )
        )
    with db2:
        detail_backfill_workers = int(
            st.number_input(
                "Detail workers",
                min_value=1,
                max_value=8,
                value=int(settings.detail_backfill_workers),
                key="pipe_detail_backfill_workers",
            )
        )
    with db3:
        detail_backfill_sleep_s = float(
            st.number_input(
                "Detail sleep (s)",
                min_value=0.0,
                max_value=10.0,
                value=float(settings.detail_backfill_sleep_s),
                step=0.25,
                key="pipe_detail_backfill_sleep",
            )
        )

    http_free_proxies_enabled = st.checkbox(
        "Use free HTTP proxies on 403/429",
        value=bool(settings.http_free_proxies_enabled),
        help=(
            "When off (default), Otodom/OLX/NOL/detail jobs use direct HTTP. Turn on "
            "only if your IP is blocked — free proxy lists are slow and often "
            "fail on listing portals."
        ),
        key="pipe_http_free_proxies",
    )

    st.subheader("OLX")
    scrape_olx_in_pipe = st.checkbox(
        "Scrape OLX in pipeline",
        value=bool(flags.scrape_olx),
        help="District-tiled Warsaw offers during full pipeline runs.",
        key="pipe_scrape_olx",
    )
    olx_unlimited = st.checkbox(
        "Unlimited OLX listings",
        value=limits.olx_max_listings is None,
        help=(
            "District-tiled public offers API. City-wide offset is unreliable; "
            "we walk each Warsaw district newest-first."
        ),
        key="pipe_olx_unlimited",
    )
    olx_max_listings: int | None = None
    if not olx_unlimited:
        olx_max_listings = int(
            st.number_input(
                "Max OLX listings",
                min_value=10,
                max_value=100_000,
                value=int(limits.olx_max_listings or 2000),
                key="pipe_olx_max_listings",
            )
        )
    olx_incremental = st.checkbox(
        "OLX incremental (new/updated only)",
        value=bool(limits.olx_incremental),
        help="Newest-first per district; early-stop when pages are already known.",
        key="pipe_olx_incremental",
    )
    olx_include_rent = st.checkbox(
        "Include OLX rentals",
        value=bool(limits.olx_include_rent),
        help="Rent rows stay in Explore; price-model labels use sale only.",
        key="pipe_olx_include_rent",
    )

    st.subheader("FX (NBP → PLN)")
    st.caption(
        "Non-PLN Otodom asks convert via NBP table A mid rates before "
        "filters/model. Original currency kept for display."
    )
    cache = fx_cache_path(proc)
    st.write(fx_cache_summary(cache))
    fx_cols = st.columns(2)
    with fx_cols[0]:
        if st.button("Refresh EUR rate", key="fx_refresh_eur"):
            try:
                rate, effective = get_pln_rate("EUR", cache, force_refresh=True)
                st.success(f"EUR→PLN {rate} (effective {effective})")
            except Exception as exc:
                st.error(f"NBP fetch failed: {exc}")
    with fx_cols[1]:
        if st.button("Clear FX cache", key="fx_clear_cache"):
            clear_fx_cache(cache)
            st.info("FX cache cleared.")

    st.subheader("RCN")
    download_rcn = st.checkbox(
        "Download RCN",
        value=bool(flags.download_rcn),
        help="RCN GML is multi-GB; full parse can take a long time.",
        key="pipe_download_rcn",
    )
    rcn_frame = load_rcn_transactions()
    rcn_quality = format_rcn_coord_quality(rcn_coord_source_counts(rcn_frame))
    st.caption(
        f"On disk: {rcn_quality}. Offer Comps needs parcel coords — "
        "if parcel count is 0, Run → Force GIS / remote refresh + pipeline."
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
    gunb_wfs_full_history = st.checkbox(
        "GUNB WFS full history (all year layers + pozostale)",
        value=bool(settings.gunb_wfs_full_history),
        help=(
            "On (default): scans every national bucket incl. pozostale "
            "(2M+ features; can take 12h+). Off: recent 3 year layers only "
            "(~1h first fetch). Cached per-layer parquets under data/raw/."
        ),
        key="pipe_gunb_wfs_full_history",
    )

    aq_field_enabled = st.checkbox(
        "Build GIOŚ air-quality IDW field (model + map)",
        value=bool(settings.aq_field_enabled),
        help=(
            "Interpolate continuous AQI/PM/NO₂ from Warsaw GIOŚ stations "
            "(same pattern as strategic noise rasters). Disable to keep "
            "nearest-station values only."
        ),
        key="pipe_aq_field_enabled",
    )
    aq_idw_power = float(
        st.number_input(
            "AQ IDW power",
            min_value=0.5,
            max_value=4.0,
            value=float(settings.aq_idw_power),
            step=0.5,
            help="Higher = sharper local influence around stations.",
            key="pipe_aq_idw_power",
        )
    )

    st.subheader("OSM / enrich")
    use_osm = st.checkbox(
        "Fetch OSM amenities + shops (else fixture/cache)",
        value=bool(flags.use_osm),
        key="pipe_use_osm",
    )
    snap_out_of_parks = st.checkbox(
        "Snap listing pins onto OSM buildings",
        value=bool(settings.snap_listings_out_of_parks),
        help=(
            "Otodom/fixture/RCN pins that land in parks, on roads, or otherwise "
            "outside building footprints move to the nearest OpenStreetMap "
            "residential building centroid (outside park polygons)."
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
    traffic_neighbourhood_radius_m = float(
        st.number_input(
            "Traffic neighbourhood radius (m)",
            min_value=50.0,
            max_value=2_000.0,
            value=float(settings.traffic_neighbourhood_radius_m),
            step=50.0,
            help=(
                "Radius for traffic_weight_nearby and major_traffic_weight_nearby "
                "(OSM highway-class proxy, not live counts). Street/quiet-side "
                "hints come from Otodom description phrases."
            ),
            key="pipe_traffic_neighbourhood_radius_m",
        )
    )

    # Ops panel shares the same widget keys for scrape toggles.
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
            otodom_fetch_details=otodom_fetch_details,
            auto_detail_backfill_enabled=auto_detail_backfill_enabled,
            detail_backfill_batch_size=detail_backfill_batch_size,
            detail_backfill_workers=detail_backfill_workers,
            detail_backfill_sleep_s=detail_backfill_sleep_s,
            olx_unlimited=olx_unlimited,
            olx_max_listings=olx_max_listings,
            olx_incremental=olx_incremental,
            olx_include_rent=olx_include_rent,
            nol_unlimited=nol_unlimited,
            nol_max_pages=nol_max_pages,
            nol_max_listings=nol_max_listings,
            nol_incremental=nol_incremental,
            nol_include_rent=nol_include_rent,
            nol_fetch_details=nol_fetch_details,
            http_free_proxies_enabled=http_free_proxies_enabled,
            download_rcn=download_rcn,
            rcn_unlimited=rcn_unlimited,
            rcn_max_rows=rcn_max_rows,
            rcn_zip_url=rcn_zip_url,
            download_gtfs=download_gtfs,
            gtfs_zip_url=gtfs_zip_url,
            download_gios=download_gios,
            gios_stations_url=gios_stations_url,
            aq_field_enabled=aq_field_enabled,
            aq_idw_power=aq_idw_power,
            download_gunb=download_gunb,
            gunb_zip_url=gunb_zip_url,
            gunb_wfs_full_history=gunb_wfs_full_history,
            download_crime=download_crime,
            crime_panorama_7z_url=crime_panorama_7z_url,
            use_osm=use_osm,
            scrape_in_pipe=scrape_in_pipe,
            scrape_olx_in_pipe=scrape_olx_in_pipe,
            scrape_nol_in_pipe=scrape_nol_in_pipe,
            snap_out_of_parks=snap_out_of_parks,
            bike_path_coverage_radius_m=bike_path_coverage_radius_m,
            traffic_neighbourhood_radius_m=traffic_neighbourhood_radius_m,
        )
        save_app_settings(settings_file, settings)
        st.success("Pipeline settings saved (overrides YAML ingest).")


def _panel_explore(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Explore Map defaults: hour, basemap, pin colour scale, point caps."""
    del proc
    st.caption(
        "Defaults for Explore Map when session widgets are empty. "
        "Point caps keep pydeck payloads under Streamlit message limits."
    )
    hour = st.slider(
        "Default time of day",
        min_value=0,
        max_value=23,
        value=int(settings.explore_default_hour),
        format="%02d:00",
        key="settings_explore_default_hour",
    )
    basemap = (
        st.segmented_control(
            "Default basemap",
            options=["Streets", "Satellite"],
            default=(
                "Satellite"
                if settings.explore_default_basemap == "Satellite"
                else "Streets"
            ),
            key="settings_explore_default_basemap",
        )
        or "Streets"
    )
    kind_default = coerce_listing_kind(settings.filter_default_listing_kind)
    listing_kind = st.selectbox(
        "Default listing kind",
        options=list(LISTING_KIND_OPTIONS),
        index=list(LISTING_KIND_OPTIONS).index(kind_default),
        key="settings_filter_default_listing_kind",
        help=(
            "Seed Explore and Stats sale/rent filters when the session "
            "widget is empty. Missing listing_kind still counts as sale."
        ),
    )
    status_default = coerce_listing_status(settings.filter_default_listing_status)
    listing_status = st.selectbox(
        "Default listing status",
        options=list(LISTING_STATUS_OPTIONS),
        index=list(LISTING_STATUS_OPTIONS).index(status_default),
        format_func=lambda key: LISTING_STATUS_LABELS.get(str(key), str(key)),
        key="settings_filter_default_listing_status",
        help=(
            "Seed Explore live vs closed asks. Historical pins are exact "
            "closed rows from scored buyer sources (otodom, NOL, OLX), "
            "not Spatial interpolation."
        ),
    )
    delta_scale = st.number_input(
        "Listing pin Δ colour scale (zł/m²)",
        min_value=100.0,
        max_value=10_000.0,
        value=float(settings.map_delta_color_scale_zl),
        step=50.0,
        help="Tanh soft-cap for green→amber→red offer pin colours.",
        key="settings_map_delta_scale",
    )
    g1, g2 = st.columns(2)
    max_gunb = g1.number_input(
        "Max GUNB pins",
        min_value=100,
        max_value=50_000,
        value=int(settings.map_max_gunb_points),
        step=100,
        key="settings_map_max_gunb",
    )
    max_shop = g2.number_input(
        "Max shop pins",
        min_value=100,
        max_value=50_000,
        value=int(settings.map_max_shop_points),
        step=100,
        key="settings_map_max_shop",
    )
    st.markdown("**Offer comps**")
    st.caption(
        "Local RCN membership on Offer → Comps. Radius defaults to Price model "
        "neighborhood search (metres)."
    )
    c1, c2 = st.columns(2)
    max_pins = c1.number_input(
        "Max comps pins",
        min_value=20,
        max_value=3_000,
        value=int(settings.offer_comps_max_pins),
        step=20,
        key="settings_offer_comps_max_pins",
        help="Cap pydeck RCN / nearby-ask pins (GUNB-style payload bound).",
    )
    months = c2.number_input(
        "Deed recency (months, 0 = all)",
        min_value=0,
        max_value=120,
        value=int(settings.offer_comps_months),
        key="settings_offer_comps_months",
    )
    include_metro = st.checkbox(
        "Include metro / unlabelled RCN",
        value=bool(settings.offer_comps_include_metro),
        key="settings_offer_comps_include_metro",
        help=(
            "Deeds outside the 18 dzielnice or GIS bbox plot as metro, "
            "not gmina layers."
        ),
    )
    show_asks = st.checkbox(
        "Show nearby live asks on comps map",
        value=bool(settings.offer_comps_show_nearby_asks),
        key="settings_offer_comps_show_asks",
    )
    if st.button("Save Explore settings", type="primary", key="save_explore_settings"):
        settings.explore_default_hour = int(hour)
        settings.explore_default_basemap = (
            "Satellite" if str(basemap) == "Satellite" else "Streets"
        )
        settings.filter_default_listing_kind = coerce_listing_kind(listing_kind)
        settings.filter_default_listing_status = coerce_listing_status(listing_status)
        settings.map_delta_color_scale_zl = float(delta_scale)
        settings.map_max_gunb_points = int(max_gunb)
        settings.map_max_shop_points = int(max_shop)
        settings.offer_comps_max_pins = int(max_pins)
        settings.offer_comps_months = int(months)
        settings.offer_comps_include_metro = bool(include_metro)
        settings.offer_comps_show_nearby_asks = bool(show_asks)
        save_app_settings(settings_file, settings)
        st.success("Explore map settings saved.")


def _panel_model(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Price-model HPO metric, validation split, neighborhood comps."""
    del proc
    from happy_housing.models.feature_eng import ENGINEERED_FEATURE_COLUMNS
    from happy_housing.models.train import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS

    n_core = len(CORE_FEATURE_COLUMNS)
    n_full = len(FEATURE_COLUMNS)
    n_eng = len(ENGINEERED_FEATURE_COLUMNS)
    nbhd_note = (
        f" · **{n_full}** with RCN nbhd" if settings.model_use_neighborhood else ""
    )
    eng_note = f" · **{n_eng}** engineered" if settings.model_use_engineered else ""
    st.caption(
        f"Fair-price LightGBM · **{n_core}** core features{nbhd_note}{eng_note}. "
        "HPO on validation; test held out. Target ≤1% MAPE. "
        "Retrain via Admin → Ops after feature or data changes. "
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
        max_value=200,
        value=int(settings.model_hpo_trials),
        help="1 = use YAML defaults only (no search). Wider space needs ≥80.",
        key="settings_model_hpo_trials",
    )
    early_stop = st.number_input(
        "Early stopping rounds (HPO val)",
        min_value=0,
        max_value=300,
        value=int(settings.model_early_stopping_rounds),
        help="0 disables early stopping during HPO trials.",
        key="settings_model_early_stop",
    )
    log_target = st.checkbox(
        "Log-target training (log1p zł/m²)",
        value=bool(settings.model_log_target),
        key="settings_model_log_target",
    )
    district_cal = st.checkbox(
        "District residual calibration",
        value=bool(settings.model_district_calibration),
        key="settings_model_district_cal",
    )
    use_eng = st.checkbox(
        "Engineered features (ratios / logs / district train stats)",
        value=bool(settings.model_use_engineered),
        help=(
            "Adds area/room ratios, log distances, listing calendar, and "
            "train-fold district median/mean zł/m² (no test leakage)."
        ),
        key="settings_model_use_engineered",
    )
    photo_cv = st.checkbox(
        "Photo CV features (pretrained CLIP / heuristic)",
        value=bool(settings.photo_cv_enabled),
        key="settings_photo_cv_enabled",
    )
    photo_max = st.number_input(
        "Max photos per listing (CV)",
        min_value=1,
        max_value=32,
        value=int(settings.photo_cv_max_photos),
        key="settings_photo_cv_max",
    )
    photo_model = st.text_input(
        "Photo CV model id (Hugging Face / CLIP)",
        value=str(settings.photo_cv_model_id),
        help="Used when photo CV is enabled during pipeline enrich.",
        key="settings_photo_cv_model_id",
    )
    train_asks_only = st.checkbox(
        "Train on buyer asks only (Otodom + NOL + OLX + fixture; RCN = comps)",
        value=bool(settings.train_buyer_asks_only),
        help=(
            "Uses all live buyer portals equally as ask-price labels; excludes "
            "RCN deed transactions and rental rows."
        ),
        key="settings_train_buyer_asks_only",
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
        settings.model_early_stopping_rounds = int(early_stop)
        settings.model_use_neighborhood = bool(use_nbhd)
        settings.model_use_engineered = bool(use_eng)
        settings.model_nbhd_radius_m = float(radius)
        settings.train_buyer_asks_only = bool(train_asks_only)
        settings.model_log_target = bool(log_target)
        settings.model_district_calibration = bool(district_cal)
        settings.photo_cv_enabled = bool(photo_cv)
        settings.photo_cv_max_photos = int(photo_max)
        settings.photo_cv_model_id = (
            str(photo_model).strip() or "openai/clip-vit-base-patch32"
        )
        save_app_settings(settings_file, settings)
        st.success(
            f"Model settings saved · optimize {settings.model_opt_metric} "
            f"({metric_direction(coerce_opt_metric(settings.model_opt_metric))}) "
            f"· {settings.model_hpo_trials} trials"
            f"{' · engineered' if use_eng else ''}"
            f"{' · buyer asks only' if train_asks_only else ''}."
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
    w1, w2, w3, w4 = st.columns(4)
    w_bargain = w1.number_input(
        "w bargain",
        min_value=0.0,
        max_value=2.0,
        value=float(settings.recommender_w_bargain),
        step=0.05,
        key="settings_rec_w_bargain",
    )
    w_commute = w2.number_input(
        "w commute",
        min_value=0.0,
        max_value=2.0,
        value=float(settings.recommender_w_commute),
        step=0.05,
        key="settings_rec_w_commute",
    )
    w_transit = w3.number_input(
        "w transit",
        min_value=0.0,
        max_value=2.0,
        value=float(settings.recommender_w_transit),
        step=0.05,
        key="settings_rec_w_transit",
    )
    w_overprice = w4.number_input(
        "w overprice",
        min_value=0.0,
        max_value=2.0,
        value=float(settings.recommender_w_overprice),
        step=0.05,
        key="settings_rec_w_overprice",
    )
    otodom_pool = st.checkbox(
        "Default pool = Otodom/fixture only (when prefs sources empty)",
        value=bool(settings.recommender_otodom_pool_default),
        key="settings_rec_otodom_pool",
    )
    s1, s2 = st.columns(2)
    delta_scale = s1.number_input(
        "Δ soft-cap (zł/m²)",
        min_value=100.0,
        max_value=20_000.0,
        value=float(settings.recommender_delta_scale_zl),
        step=100.0,
        key="settings_rec_delta_scale",
        help="Bargain / overprice score terms saturate beyond this Δ.",
    )
    dist_cap = s2.number_input(
        "Amenity distance soft-cap (m)",
        min_value=100.0,
        max_value=20_000.0,
        value=float(settings.recommender_dist_cap_m),
        step=100.0,
        key="settings_rec_dist_cap",
    )
    p1, p2 = st.columns(2)
    opened_pen = p1.number_input(
        "Opened demotion weight",
        min_value=0.0,
        max_value=20.0,
        value=float(settings.recommender_opened_penalty),
        step=0.1,
        key="settings_rec_opened_pen",
    )
    skipped_pen = p2.number_input(
        "Skipped demotion weight",
        min_value=0.0,
        max_value=20.0,
        value=float(settings.recommender_skipped_penalty),
        step=0.1,
        key="settings_rec_skipped_pen",
    )
    st.markdown("#### Amenity polarity & weight")
    st.caption(
        "Positive = prefer offers near this type; negative = prefer far. "
        f"Magnitude -1..+1. Distance soft-cap {float(dist_cap):,.0f} m."
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
            settings.recommender_w_bargain = float(w_bargain)
            settings.recommender_w_commute = float(w_commute)
            settings.recommender_w_transit = float(w_transit)
            settings.recommender_w_overprice = float(w_overprice)
            settings.recommender_otodom_pool_default = bool(otodom_pool)
            settings.recommender_delta_scale_zl = float(delta_scale)
            settings.recommender_dist_cap_m = float(dist_cap)
            settings.recommender_opened_penalty = float(opened_pen)
            settings.recommender_skipped_penalty = float(skipped_pen)
            save_app_settings(settings_file, settings)
            st.success("Recommender settings saved.")
    with r2:
        if st.button("Retrain recommendation model", type="primary", key="retrain_rec"):
            settings.recommender_top_n = int(top_n)
            settings.recommender_demotion_alpha = float(alpha)
            settings.recommender_amenity_weights = new_weights
            settings.recommender_w_bargain = float(w_bargain)
            settings.recommender_w_commute = float(w_commute)
            settings.recommender_w_transit = float(w_transit)
            settings.recommender_w_overprice = float(w_overprice)
            settings.recommender_otodom_pool_default = bool(otodom_pool)
            settings.recommender_delta_scale_zl = float(delta_scale)
            settings.recommender_dist_cap_m = float(dist_cap)
            settings.recommender_opened_penalty = float(opened_pen)
            settings.recommender_skipped_penalty = float(skipped_pen)
            save_app_settings(settings_file, settings)
            logger.info("Admin retrain recommender clicked (top_n=%s)", int(top_n))
            try:
                frame = load_scored()
            except FileNotFoundError as exc:
                logger.error("Retrain recommender missing scored: %s", exc)
                st.error(str(exc))
            else:
                with st.status(
                    "Retraining recommendation model…", expanded=True
                ) as status:
                    try:
                        st.write(f"Scoring {len(frame):,} offers…")
                        prefs = load_preferences(preferences_path(proc))
                        out = retrain_recommender(
                            frame,
                            processed_dir=proc,
                            top_n=int(top_n),
                            prefs=prefs,
                            amenity_weights=new_weights,
                            commute_mode=settings.commute_display_mode,
                            **recommender_kwargs_from_settings(settings),
                        )
                        status.update(
                            label="Recommendation model retrained",
                            state="complete",
                        )
                        set_admin_flash(
                            "success",
                            f"Retrained · {int(out['n_recommended'])} of "
                            f"{int(out['n_candidates'])} candidates · "
                            f"mean confidence {out['mean_confidence']:.3f}",
                        )
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Retrain recommender failed")
                        status.update(
                            label="Recommendation retrain failed",
                            state="error",
                        )
                        st.error(str(exc))
    with r3:
        if st.button("Delete recommendation memory", key="clear_rec_memory"):
            clear_recommendation_memory(proc)
            logger.info("Cleared recommendation memory from Admin")
            set_admin_flash(
                "success",
                "Recommendation impressions + metrics deleted.",
            )
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
        "(CSV line, N if unknown). Structured Otodom fields always win. "
        "On by default in `make pipeline` (even when GIS enrich is skipped). "
        "0 LM calls / run = unlimited."
    )
    lms_keys = list_lmstudio_llm_keys()
    if lms_keys:
        st.caption("LMS downloaded LLMs: `" + "`, `".join(lms_keys[:16]) + "`")
    else:
        st.caption(
            "No LMS LLM keys listed — start LM Studio local server, then refresh."
        )
    extract_model = st.text_input(
        "Extract LM Studio model id",
        value=settings.extract_lm_model,
        help=(
            "Must match an LMS download key (see list above). "
            "Aliases like gemma-3-4b-it ↔ google/gemma-3-4b resolve automatically."
        ),
        key="settings_extract_lm_model",
    )
    extract_on = st.checkbox(
        "Enable description extract in pipeline",
        value=settings.extract_attrs_enabled,
        key="settings_extract_enabled",
    )
    e1, e2, e3, e4 = st.columns(4)
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
    with e4:
        extract_workers = st.number_input(
            "Parallel LMS workers",
            min_value=1,
            max_value=8,
            value=int(settings.extract_lm_workers),
            key="settings_extract_workers",
            help="Concurrent attr-extract calls to LM Studio (2-4 typical).",
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
            settings.extract_lm_workers = int(extract_workers)
            save_app_settings(settings_file, settings)
            st.success("Extract settings saved.")
    with s2:
        if st.button("Backfill attrs on scored listings", key="admin_extract_backfill"):
            logger.info(
                "Admin attr backfill clicked model=%s cap=%s",
                extract_model,
                int(extract_cap) or None,
            )
            try:
                frame = load_scored()
            except FileNotFoundError as exc:
                logger.error("Attr backfill missing scored: %s", exc)
                st.error(str(exc))
            else:
                cap = int(extract_cap) or None
                with st.status(
                    f"Extracting attrs with {extract_model}…",
                    expanded=True,
                ) as status:
                    try:
                        st.write(f"Loaded {len(frame):,} scored offers")
                        st.write(
                            f"Running LM extract (cap={cap or '∞'}, "
                            f"workers={int(extract_workers)})…"
                        )
                        work = promote_listing_flags(frame)
                        work = enrich_frame_listing_attrs(
                            work,
                            model=str(extract_model).strip()
                            or settings.extract_lm_model,
                            max_chars=int(extract_chars),
                            max_tokens=int(extract_tokens),
                            use_llm=True,
                            metrics_path=llm_metrics_path(proc),
                            max_llm_calls=cap,
                            lm_workers=int(extract_workers),
                        )
                        out_path = processed_dir() / "listings_scored.parquet"
                        st.write(f"Writing {out_path.name}…")
                        work.to_parquet(out_path, index=False)
                        clear_data_caches()
                        status.update(label="Attr backfill complete", state="complete")
                        logger.info("Attr backfill wrote %s", out_path)
                        set_admin_flash("success", f"Wrote attrs → {out_path}")
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Attr backfill failed")
                        status.update(label="Attr backfill failed", state="error")
                        st.error(str(exc))


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
    """Starred offers, News feed memory, and Explore presets."""
    stars_file = stars_path(proc)
    starred = load_stars(stars_file)
    news_items = load_news(news_path(proc))
    filter_file = filter_presets_path(proc)
    filter_presets = load_filter_presets(filter_file)
    map_file = map_view_presets_path(proc)
    map_presets = load_map_view_presets(map_file)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Starred offers", f"{len(starred):,}")
    s2.metric("News events", f"{len(news_items):,}")
    s3.metric("Filter presets", f"{len(filter_presets):,}")
    s4.metric("Map views", f"{len(map_presets):,}")
    cap = st.number_input(
        "News history cap",
        min_value=10,
        max_value=2000,
        value=int(settings.news_history_cap),
        key="settings_news_cap",
    )
    filter_cap = st.number_input(
        "Max saved Explore filter presets",
        min_value=1,
        max_value=100,
        value=int(settings.filter_presets_max),
        key="settings_filter_presets_max",
        help="Oldest filter presets drop when saving beyond this cap.",
    )
    map_cap = st.number_input(
        "Max saved Explore map views",
        min_value=1,
        max_value=100,
        value=int(settings.map_view_presets_max),
        key="settings_map_view_presets_max",
        help="Oldest map-view presets drop when saving beyond this cap.",
    )
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Save news / preset settings", key="save_news_settings"):
            settings.news_history_cap = int(cap)
            settings.filter_presets_max = int(filter_cap)
            settings.map_view_presets_max = int(map_cap)
            save_app_settings(settings_file, settings)
            st.success("Stars & News settings saved.")
    with a2:
        if st.button("Clear starred offers", key="clear_stars"):
            clear_stars(stars_file)
            st.success("Starred offers cleared.")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Clear starred news + snapshots", key="clear_star_news"):
            clear_star_news_memory(proc)
            st.success("Starred news memory cleared.")
    with b2:
        if st.button("Clear filter presets", key="clear_filter_presets"):
            clear_filter_presets(filter_file)
            st.success("Explore filter presets cleared.")
            st.rerun()
    with b3:
        if st.button("Clear map views", key="clear_map_view_presets"):
            clear_map_view_presets(map_file)
            st.success("Explore map-view presets cleared.")
            st.rerun()


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
    trend_min = st.number_input(
        "Min n per month-slice (trends)",
        min_value=1,
        max_value=200,
        value=int(settings.stats_trend_min_n),
        key="settings_stats_trend_min_n",
        help="Hide rooms/era/district month series below this count.",
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
            settings.stats_trend_min_n = int(trend_min)
            settings.stats_shap_sample_n = int(shap_n)
            settings.stats_xai_top_n = int(xai_top)
            save_app_settings(settings_file, settings)
            st.success("Stats settings saved.")
    with s2:
        if st.button("Recalculate stats", key="recalc_stats"):
            logger.info("Admin stats recalculation clicked")
            with st.status("Recalculating stats…", expanded=True) as status:
                try:
                    st.write("Clearing Streamlit data caches…")
                    clear_data_caches()
                    st.write("Reloading scored listings…")
                    frame = load_scored()
                    status.update(label="Stats refreshed", state="complete")
                    logger.info(
                        "Admin stats recalculation: %s offers after cache clear",
                        len(frame),
                    )
                    set_admin_flash(
                        "success",
                        f"Stats refreshed · {len(frame):,} scored offers loaded.",
                    )
                    st.rerun()
                except FileNotFoundError as exc:
                    logger.error("Stats recalculation missing scored: %s", exc)
                    status.update(label="Stats refresh failed", state="error")
                    st.error(str(exc))


PanelFn = Callable[[AppSettings, Path, Path], None]
"""Admin panel renderer signature."""


def panel_for_section(section: str) -> PanelFn:
    """Map Admin section label to its panel renderer."""
    match section:
        case "Ops":
            return _panel_ops
        case "Freshness":
            return _panel_freshness
        case "Commute":
            return _panel_commute
        case "Sources":
            return _panel_sources
        case "Explore":
            return _panel_explore
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
    """Render Admin with top section pills (one short panel at a time)."""
    logger.info("Rendering Admin")
    pop_admin_flash()

    proc = processed_dir()
    settings_file = settings_path(proc)
    settings = load_app_settings(settings_file)

    section = st.pills(
        "Admin section",
        options=list(ADMIN_SECTIONS),
        default=ADMIN_SECTIONS[0],
        key="admin_section",
        label_visibility="collapsed",
    )
    if section is None:
        section = ADMIN_SECTIONS[0]
    logger.debug("Admin section=%s", section)
    st.subheader(str(section))
    panel_for_section(str(section))(settings, settings_file, proc)
