"""Admin page: tabbed settings and scrape / retrain triggers."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 — Admin runs fixed local pipeline/scrape commands
import sys
from pathlib import Path

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
from happy_housing.app.commute_target import (
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
from happy_housing.app.geocode import search_places
from happy_housing.app.listing_attrs import promote_listing_flags
from happy_housing.app.llm_metrics import (
    clear_llm_metrics,
    llm_metrics_path,
    purpose_stats_frame,
    summarize_llm_calls,
)
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
    """Paths, commute target, cache clear, and log level."""
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

    target = resolve_commute_target(settings, config)
    st.subheader("Commute target")
    st.caption(
        "Destination for dist_daszynskiego_m / commute_min / OTP transit. "
        "Search uses OpenStreetMap Nominatim (open geocoder). "
        "After changing the place, recompute features (or run the full pipeline)."
    )
    if "admin_commute_lon" not in st.session_state:
        st.session_state["admin_commute_lon"] = float(target.lon)
        st.session_state["admin_commute_lat"] = float(target.lat)
        st.session_state["admin_commute_label"] = str(target.label)

    search_q = st.text_input(
        "Search place",
        value="",
        placeholder="e.g. Plac Bankowy, Warszawa",
        key="admin_commute_search_q",
        help="OpenStreetMap Nominatim suggestions (Poland / Warsaw-biased).",
    )
    if st.button("Suggest places", key="admin_commute_search_btn"):
        try:
            with st.spinner("Querying Nominatim…"):
                hits = search_places(search_q)
            st.session_state["admin_commute_suggestions"] = [
                {"label": h.label, "lon": h.lon, "lat": h.lat} for h in hits
            ]
            if not hits:
                st.info("No Nominatim hits — try a more specific query.")
        except Exception as exc:
            st.error(f"Nominatim search failed: {exc}")
    suggestions = st.session_state.get("admin_commute_suggestions") or []
    if suggestions:
        labels = [str(row["label"]) for row in suggestions]
        pick = st.selectbox(
            "Suggestions",
            options=labels,
            key="admin_commute_suggestion_pick",
        )
        chosen = suggestions[labels.index(str(pick))]
        if st.button("Use selected suggestion", key="admin_commute_use_suggestion"):
            st.session_state["admin_commute_lon"] = float(chosen["lon"])
            st.session_state["admin_commute_lat"] = float(chosen["lat"])
            st.session_state["admin_commute_label"] = str(chosen["label"])[:120]
            st.rerun()

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
            st.success("Reset override to experiment.yaml coordinates.")
            st.rerun()

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


def _tab_ai(settings: AppSettings, settings_file: Path, proc: Path) -> None:
    """LM Studio model, buyer prefs, and AI review prompt controls."""
    st.caption("Used for Offer AI reviews (LM Studio must be running locally).")
    lm_model = st.text_input(
        "LM Studio model id (reviews / prefs)",
        value=settings.lm_model,
        help="Must match a model loaded in LM Studio.",
        key="settings_lm_model",
    )
    st.subheader("Listing attribute extract")
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

    if st.button("Save AI settings", type="primary", key="save_ai_settings"):
        settings.lm_model = lm_model.strip() or settings.lm_model
        settings.extract_lm_model = (
            str(extract_model).strip() or settings.extract_lm_model
        )
        settings.extract_attrs_enabled = bool(extract_on)
        settings.extract_max_chars = int(extract_chars)
        settings.extract_max_tokens = int(extract_tokens)
        settings.extract_max_listings = int(extract_cap)
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
        st.success("AI settings saved (stale reviews auto-regenerate on next ask).")

    db = reviews_db_path(proc)
    st.caption(f"Cached reviews DB: `{'present' if db.exists() else 'empty'}`")
    if st.button("Clear AI review cache", key="clear_ai_reviews"):
        clear_ai_reviews(db)
        cleared = clear_ai_review_session_keys(st.session_state)
        st.success(
            f"AI review cache cleared{f' · {cleared} session keys' if cleared else ''}."
        )

    st.subheader("LM Studio call metrics")
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
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("Save recommender settings", key="save_rec_settings"):
            settings.recommender_top_n = int(top_n)
            settings.recommender_demotion_alpha = float(alpha)
            save_app_settings(settings_file, settings)
            st.success("Recommender settings saved.")
    with r2:
        if st.button("Retrain recommendation model", type="primary", key="retrain_rec"):
            settings.recommender_top_n = int(top_n)
            settings.recommender_demotion_alpha = float(alpha)
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


def _tab_stats(settings: AppSettings, settings_file: Path) -> None:
    """Stats page aggregation thresholds."""
    st.caption(
        "Controls district scoreboards and hunt/watch list length. "
        "Recalculate clears Streamlit caches so Stats reloads scored listings."
    )
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
    s1, s2, _s3 = st.columns([1, 1, 3])
    with s1:
        if st.button("Save Stats settings", type="primary", key="save_stats_settings"):
            settings.stats_min_district_offers = int(min_offers)
            settings.stats_top_n = int(top_n)
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
    use_osm: bool,
    scrape_in_pipe: bool,
) -> None:
    """Copy Pipeline tab widget values onto ``settings`` (overrides on)."""
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
    settings.use_osm = bool(use_osm)
    settings.scrape_otodom_in_pipeline = bool(scrape_in_pipe)


def _tab_pipeline(settings: AppSettings, settings_file: Path) -> None:
    """Otodom scrape and full remodel with Admin ingest overrides."""
    config = load_config()
    limits = resolve_ingest_limits(settings, config)
    urls = resolve_ingest_urls(settings, config)
    flags = resolve_ingest_flags(settings, config)

    st.caption(
        "Full-source downloads can take hours (Otodom detail crawl + multi-GB RCN). "
        "UM parks WFS + GIOŚ AQ; strategic noise WMS-only / EEA aircraft; "
        "Admin caps override YAML when saved."
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
    if st.button("Run Otodom scrape", type="primary", key="run_otodom"):
        out_path = processed_dir() / "otodom_listings.parquet"
        pages_arg = "None" if max_pages is None else str(int(max_pages))
        listings_arg = "None" if max_listings is None else str(int(max_listings))
        incr_arg = "True" if incremental else "False"
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
    rcn_zip_url = st.text_input(
        "RCN zip URL",
        value=urls.rcn_zip,
        key="pipe_rcn_zip_url",
    )

    st.subheader("GTFS")
    download_gtfs = st.checkbox(
        "Download GTFS",
        value=bool(flags.download_gtfs),
        key="pipe_download_gtfs",
    )
    gtfs_zip_url = st.text_input(
        "GTFS zip URL",
        value=urls.gtfs_zip,
        key="pipe_gtfs_zip_url",
    )

    st.subheader("GIOŚ")
    download_gios = st.checkbox(
        "Download GIOŚ",
        value=bool(flags.download_gios),
        key="pipe_download_gios",
    )
    gios_stations_url = st.text_input(
        "GIOŚ stations URL",
        value=urls.gios_stations,
        key="pipe_gios_stations_url",
    )

    st.subheader("GUNB")
    download_gunb = st.checkbox(
        "Download GUNB",
        value=bool(flags.download_gunb),
        key="pipe_download_gunb",
    )
    gunb_zip_url = st.text_input(
        "GUNB zip URL",
        value=urls.gunb_zip,
        key="pipe_gunb_zip_url",
    )

    st.subheader("OSM")
    use_osm = st.checkbox(
        "Fetch OSM amenities + shops (else fixture/cache)",
        value=bool(flags.use_osm),
        key="pipe_use_osm",
    )

    st.subheader("Remodel")
    scrape_in_pipe = st.checkbox(
        "Scrape Otodom inside pipeline",
        value=bool(flags.scrape_otodom),
        key="pipe_scrape_otodom",
    )
    p1, p2, _p3 = st.columns([1, 1, 3])
    with p1:
        if st.button("Save Pipeline settings", key="save_pipeline_settings"):
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
                use_osm=use_osm,
                scrape_in_pipe=scrape_in_pipe,
            )
            save_app_settings(settings_file, settings)
            st.success("Pipeline settings saved (overrides YAML ingest).")
    with p2:
        if st.button(
            "Run pipeline (enrich + train + score)",
            type="primary",
            key="run_pipe",
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
                use_osm=use_osm,
                scrape_in_pipe=scrape_in_pipe,
            )
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


def _tab_model(settings: AppSettings, settings_file: Path) -> None:
    """Price-model HPO metric, validation split, neighborhood comps."""
    from happy_housing.models.train import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS

    n_core = len(CORE_FEATURE_COLUMNS)
    n_full = len(FEATURE_COLUMNS)
    nbhd_note = (
        f" · **{n_full}** with RCN nbhd" if settings.model_use_neighborhood else ""
    )
    st.caption(
        f"Fair-price LightGBM · **{n_core}** core features{nbhd_note}. "
        "HPO on validation; test held out. "
        "Retrain via Admin → Pipeline after feature or data changes. "
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


def render_admin() -> None:
    """Render Admin with tabbed settings (no raw JSON dump)."""
    logger.info("Rendering Admin")
    st.caption("Tweaks and maintenance — grouped by concern. No raw settings dump.")

    proc = processed_dir()
    settings_file = settings_path(proc)
    settings = load_app_settings(settings_file)

    general, ai_tab, stars_tab, rec_tab, stats_tab, model_tab, pipeline_tab = st.tabs([
        "General",
        "AI / LM Studio",
        "Stars & News",
        "Recommendations",
        "Stats",
        "Model",
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
    with model_tab:
        _tab_model(settings, settings_file)
    with pipeline_tab:
        _tab_pipeline(settings, settings_file)
