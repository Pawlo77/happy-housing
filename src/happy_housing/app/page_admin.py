"""Admin page: settings and scrape / retrain triggers."""

from __future__ import annotations

import logging
import subprocess  # nosec B404 — Admin runs fixed local pipeline/scrape commands
import sys
from pathlib import Path

import streamlit as st

from happy_housing.app.data import load_config, load_metadata, processed_dir
from happy_housing.config import load_experiment_config

logger = logging.getLogger(__name__)


def _run_command(command: list[str], *, timeout_s: int) -> tuple[int, str]:
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
        "Trigger Otodom scrape and full remodel. Long jobs block this page "
        "until finished — watch the log panel."
    )

    config = load_config()
    meta = load_metadata()

    st.subheader("Settings")
    st.write({
        "config": "configs/experiment.yaml",
        "processed": str(config.data.processed_path()),
        "fixtures": str(config.data.fixtures_path()),
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
    max_pages = st.number_input(
        "Max search pages", min_value=1, max_value=100, value=20
    )
    max_listings = st.number_input(
        "Max listings", min_value=10, max_value=2000, value=400
    )
    if st.button("Run Otodom scrape", type="primary"):
        out_path = processed_dir() / "otodom_listings.parquet"
        cmd = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from happy_housing.ingest.otodom import scrape_otodom_warsaw; "
                f"scrape_otodom_warsaw(Path({str(out_path)!r}), "
                f"max_pages={int(max_pages)}, max_listings={int(max_listings)})"
            ),
        ]
        with st.spinner("Scraping Otodom…"):
            try:
                code, log = _run_command(cmd, timeout_s=3600)
            except subprocess.TimeoutExpired:
                st.error("Scrape timed out.")
            else:
                st.code(log[-8000:] or "(no output)")
                if code == 0:
                    st.success(f"Scrape finished → {out_path}")
                else:
                    st.error(f"Scrape failed (exit {code})")

    st.subheader("Remodel / full pipeline")
    download_remote = st.checkbox("Re-download GTFS / GIOŚ / RCN / GUNB", value=False)
    scrape_otodom = st.checkbox("Scrape Otodom inside pipeline", value=False)
    use_osm = st.checkbox("Fetch OSM amenities (else fixture/cache)", value=True)
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
        with st.spinner("Running pipeline…"):
            try:
                code, log = _run_command(cmd, timeout_s=7200)
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out.")
            else:
                st.code(log[-12000:] or "(no output)")
                if code == 0:
                    st.success("Pipeline finished — Home metrics refreshed.")
                    _ = load_experiment_config(Path("configs/experiment.yaml"))
                else:
                    st.error(f"Pipeline failed (exit {code})")

    st.subheader("Notes")
    st.caption(
        "Edit configs/experiment.yaml for model hyperparameters and data paths. "
        "Reload the app after pipeline finishes to see new Home metrics."
    )
