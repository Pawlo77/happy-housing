"""Admin page: settings and scrape / retrain triggers."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 — Admin runs fixed local pipeline/scrape commands
import sys
from pathlib import Path

import streamlit as st

from happy_housing.app.data import (
    clear_data_caches,
    load_config,
    load_metadata,
    processed_dir,
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


def render_admin() -> None:
    """Render Admin controls for scrape and remodel."""
    st.title("Admin")
    st.caption(
        "Trigger Otodom scrape and full remodel. Full-source downloads can take "
        "hours (Otodom detail crawl + multi-GB RCN GML) — watch the log panel."
    )

    config = load_config()
    meta = load_metadata()

    st.subheader("Settings")
    st.write({
        "config": "configs/experiment.yaml",
        "processed": str(config.data.processed_path()),
        "fixtures": str(config.data.fixtures_path()),
        "ingest_limits": {
            "otodom_max_pages": config.ingest.otodom_max_pages,
            "otodom_max_listings": config.ingest.otodom_max_listings,
            "rcn_max_rows": config.ingest.rcn_max_rows,
        },
        "commute_target": (
            config.commute_target_lon,
            config.commute_target_lat,
        ),
        "model": {
            "n_estimators": config.model.n_estimators,
            "learning_rate": config.model.learning_rate,
            "num_leaves": config.model.num_leaves,
            "test_size": config.model.test_size,
        },
        "last_refresh": meta.refreshed_at if meta else None,
    })

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
    if st.button("Run Otodom scrape", type="primary"):
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
    if st.button("Run pipeline (enrich + train + score)", type="primary"):
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

    st.subheader("Notes")
    st.caption(
        "Default ingest caps in configs/experiment.yaml are null (unlimited). "
        "Set numeric caps there only if you need a smaller local slice. "
        "Reload the app after pipeline finishes to see new Home metrics."
    )
