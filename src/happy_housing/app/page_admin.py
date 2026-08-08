"""Admin page: tabbed settings and scrape / retrain triggers."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 — Admin runs fixed local pipeline/scrape commands
import sys
from pathlib import Path

import streamlit as st

from happy_housing.app.ai_review import clear_ai_reviews, reviews_db_path
from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.attribution import render_attribution_footer
from happy_housing.app.data import (
    clear_data_caches,
    load_config,
    load_metadata,
    load_scored,
    processed_dir,
)
from happy_housing.app.recommender import (
    clear_recommendation_memory,
    impressions_path,
    load_impressions,
    load_recommender_metrics,
    retrain_recommender,
)
from happy_housing.app.star_news import clear_star_news_memory, load_news, news_path
from happy_housing.app.stars import clear_stars, load_stars, stars_path

logger = logging.getLogger(__name__)


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


def _tab_general(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Paths, ingest caps (read-only summary), and cache clear."""
    config = load_config()
    meta = load_metadata()
    st.caption("Experiment paths and last pipeline refresh (edit YAML for caps).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Processed dir", config.data.processed_path().name)
    c2.metric(
        "Otodom page cap",
        "∞"
        if config.ingest.otodom_max_pages is None
        else str(config.ingest.otodom_max_pages),
    )
    c3.metric(
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
    lon = config.commute_target_lon
    lat = config.commute_target_lat
    st.caption(f"Commute target: {lon:.4f}, {lat:.4f}")
    if st.button("Clear Streamlit data caches", key="admin_clear_caches"):
        clear_data_caches()
        st.success("Caches cleared.")


def _tab_ai(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """LM Studio model + AI review behaviour."""
    st.caption("Used for Offer AI reviews (LM Studio must be running locally).")
    lm_model = st.text_input(
        "LM Studio model id",
        value=settings.lm_model,
        help="Must match a model loaded in LM Studio.",
        key="settings_lm_model",
    )
    ai_on_open = st.checkbox(
        "Auto-review when opening Offer → Overview",
        value=settings.ai_review_on_offer_open,
        key="settings_ai_on_open",
    )
    if st.button("Save AI settings", type="primary", key="save_ai_settings"):
        settings.lm_model = lm_model.strip() or settings.lm_model
        settings.ai_review_on_offer_open = bool(ai_on_open)
        save_app_settings(settings_file, settings)
        st.success("AI settings saved.")
    db = reviews_db_path(proc)
    st.caption(f"Cached reviews DB: `{'present' if db.exists() else 'empty'}`")
    if st.button("Clear AI review cache", key="clear_ai_reviews"):
        clear_ai_reviews(db)
        st.success("AI review cache cleared.")


def _tab_stars_news(settings: AppSettings, settings_file: Path, proc: Path) -> None:
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
    if st.button("Save news settings", key="save_news_settings"):
        settings.news_history_cap = int(cap)
        save_app_settings(settings_file, settings)
        st.success("News settings saved.")
    if st.button("Clear starred offers", key="clear_stars"):
        clear_stars(stars_file)
        st.success("Starred offers cleared.")
    if st.button("Clear starred news + snapshots", key="clear_star_news"):
        clear_star_news_memory(proc)
        st.success("Starred news memory cleared.")


def _tab_recommender(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """Bargain recommender retrain and impression memory."""
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
    if st.button("Save recommender settings", key="save_rec_settings"):
        settings.recommender_top_n = int(top_n)
        settings.recommender_demotion_alpha = float(alpha)
        save_app_settings(settings_file, settings)
        st.success("Recommender settings saved.")
    if st.button("Retrain recommendation model", type="primary", key="retrain_rec"):
        try:
            frame = load_scored()
        except FileNotFoundError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Rescoring offers…"):
                out = retrain_recommender(
                    frame,
                    processed_dir=proc,
                    top_n=int(settings.recommender_top_n),
                    demotion_alpha=float(settings.recommender_demotion_alpha),
                )
            st.success(
                f"Retrained · {int(out['n_recommended'])} of "
                f"{int(out['n_candidates'])} candidates · "
                f"mean confidence {out['mean_confidence']:.3f}"
            )
            st.rerun()
    if st.button("Delete recommendation memory", key="clear_rec_memory"):
        clear_recommendation_memory(proc)
        st.success("Recommendation impressions + metrics deleted.")
        st.rerun()


def _tab_stats(settings: AppSettings, settings_file: Path) -> None:
    """Stats page aggregation thresholds."""
    st.caption("Controls district scoreboards and hunt/watch list length.")
    min_offers = st.number_input(
        "Min offers per district",
        min_value=1,
        max_value=200,
        value=int(settings.stats_min_district_offers),
        key="settings_stats_min_offers",
        help="Districts below this count are hidden on Stats → Value.",
    )
    top_n = st.number_input(
        "Hunt / watch list size",
        min_value=1,
        max_value=100,
        value=int(settings.stats_top_n),
        key="settings_stats_top_n",
    )
    if st.button("Save Stats settings", type="primary", key="save_stats_settings"):
        settings.stats_min_district_offers = int(min_offers)
        settings.stats_top_n = int(top_n)
        save_app_settings(settings_file, settings)
        st.success("Stats settings saved.")


def _tab_pipeline() -> None:
    """Otodom scrape and full remodel triggers."""
    st.caption(
        "Full-source downloads can take hours (Otodom detail crawl + multi-GB RCN)."
    )
    st.subheader("Otodom scrape")
    download_all = st.checkbox(
        "Download all Otodom Warsaw flats (ignore page/listing caps)",
        value=True,
        help="Uses every search page Otodom reports. Rate-limited; can take hours.",
    )
    max_pages = None
    max_listings = None
    if not download_all:
        max_pages = int(
            st.number_input("Max search pages", min_value=1, max_value=5000, value=50)
        )
        max_listings = int(
            st.number_input("Max listings", min_value=10, max_value=100_000, value=1200)
        )
    if st.button("Run Otodom scrape", type="primary", key="run_otodom"):
        out_path = processed_dir() / "otodom_listings.parquet"
        pages_arg = "None" if max_pages is None else str(int(max_pages))
        listings_arg = "None" if max_listings is None else str(int(max_listings))
        cmd = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from happy_housing.ingest.otodom import scrape_otodom_warsaw; "
                f"scrape_otodom_warsaw(Path({str(out_path)!r}), "
                f"max_pages={pages_arg}, max_listings={listings_arg})"
            ),
        ]
        with st.spinner("Scraping Otodom (full run may take hours)…"):
            try:
                code, log = _run_command(cmd, timeout_s=24 * 3600)
            except subprocess.TimeoutExpired:
                st.error("Scrape timed out.")
            else:
                st.code(log[-8000:] or "(no output)")
                if code == 0:
                    clear_data_caches()
                    st.success(f"Scrape finished → {out_path}")
                else:
                    st.error(f"Scrape failed (exit {code})")

    st.subheader("Remodel / full pipeline")
    download_remote = st.checkbox(
        "Re-download GTFS / GIOŚ / RCN / GUNB (full sources)",
        value=False,
        help="RCN GML is multi-GB; full parse can take a long time.",
    )
    scrape_otodom = st.checkbox(
        "Scrape Otodom inside pipeline (full, per config ingest limits)",
        value=False,
    )
    use_osm = st.checkbox(
        "Fetch OSM amenities + shops (else fixture/cache)", value=True
    )
    if st.button(
        "Run pipeline (enrich + train + score)", type="primary", key="run_pipe"
    ):
        cmd = [
            sys.executable,
            "-c",
            (
                "from happy_housing.pipeline import run_pipeline; "
                f"run_pipeline(download_remote={download_remote!r}, "
                f"use_osm={use_osm!r}, scrape_otodom={scrape_otodom!r})"
            ),
        ]
        with st.spinner("Running pipeline (full RCN/Otodom may take hours)…"):
            try:
                code, log = _run_command(cmd, timeout_s=24 * 3600)
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out.")
            else:
                st.code(log[-12000:] or "(no output)")
                if code == 0:
                    clear_data_caches()
                    st.success("Pipeline finished — Home metrics refreshed.")
                else:
                    st.error(f"Pipeline failed (exit {code})")
    st.caption(
        "Default ingest caps in configs/experiment.yaml are null (unlimited). "
        "Set numeric caps there only if you need a smaller local slice."
    )


def render_admin() -> None:
    """Render Admin with tabbed settings (no raw JSON dump)."""
    st.title("Admin")
    st.caption("Tweaks and maintenance — grouped by concern. No raw settings dump.")

    proc = processed_dir()
    settings_file = settings_path(proc)
    settings = load_app_settings(settings_file)

    general, ai_tab, stars_tab, rec_tab, stats_tab, pipeline_tab = st.tabs([
        "General",
        "AI / LM Studio",
        "Stars & News",
        "Recommendations",
        "Stats",
        "Pipeline",
    ])
    with general:
        _tab_general(settings, settings_file, proc)
    with ai_tab:
        _tab_ai(settings, settings_file, proc)
    with stars_tab:
        _tab_stars_news(settings, settings_file, proc)
    with rec_tab:
        _tab_recommender(settings, settings_file, proc)
    with stats_tab:
        _tab_stats(settings, settings_file)
    with pipeline_tab:
        _tab_pipeline()

    render_attribution_footer()
