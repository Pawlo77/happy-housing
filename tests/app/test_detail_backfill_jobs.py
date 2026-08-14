"""Tests for auto-scheduled detail backfill Ops jobs."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from happy_housing.app.app_settings import AppSettings, save_app_settings, settings_path
from happy_housing.app.detail_backfill_jobs import maybe_schedule_detail_backfill


def _search_card(listing_id: str = "1") -> dict[str, Any]:
    return {
        "listing_id": listing_id,
        "slug": "offer-a",
        "price_pln": 500_000.0,
        "area_m2": 50.0,
        "rooms": 2,
        "image_urls_json": '["https://example.com/a.jpg"]',
        "lat": None,
        "lon": None,
    }


def test_maybe_schedule_detail_backfill_when_pending(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    jobs = proc / "admin_jobs"
    jobs.mkdir()
    pd.DataFrame([_search_card("1")]).to_parquet(
        proc / "otodom_listings.parquet",
        index=False,
    )
    settings = AppSettings(auto_detail_backfill_enabled=True)
    save_app_settings(settings_path(proc), settings)

    import happy_housing.app.detail_backfill_jobs as jobs_mod

    real_start = jobs_mod.start_admin_job
    calls: list[str] = []

    def _fake_start(
        name: str,
        command: list[str],
        processed_dir: Path,
        env: dict[str, str] | None = None,
    ) -> MagicMock:
        _ = command, processed_dir, env
        calls.append(name)
        return MagicMock()

    jobs_mod.start_admin_job = _fake_start  # type: ignore[method-assign]
    try:
        assert maybe_schedule_detail_backfill(proc) is True
        assert calls == ["detail_backfill"]
    finally:
        jobs_mod.start_admin_job = real_start  # type: ignore[method-assign]


def test_maybe_schedule_detail_backfill_when_nol_pending(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    jobs = proc / "admin_jobs"
    jobs.mkdir()
    pd.DataFrame([_search_card("NOL-1")]).to_parquet(
        proc / "nol_listings.parquet",
        index=False,
    )
    settings = AppSettings(auto_detail_backfill_enabled=True)
    save_app_settings(settings_path(proc), settings)

    import happy_housing.app.detail_backfill_jobs as jobs_mod

    calls: list[str] = []
    real_start = jobs_mod.start_admin_job

    def _fake_start(
        name: str,
        command: list[str],
        processed_dir: Path,
        env: dict[str, str] | None = None,
    ) -> MagicMock:
        _ = command, processed_dir, env
        calls.append(name)
        return MagicMock()

    jobs_mod.start_admin_job = _fake_start  # type: ignore[method-assign]
    try:
        assert maybe_schedule_detail_backfill(proc) is True
        assert calls == ["detail_backfill"]
    finally:
        jobs_mod.start_admin_job = real_start  # type: ignore[method-assign]


def test_maybe_schedule_skips_when_disabled(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    pd.DataFrame([_search_card("1")]).to_parquet(
        proc / "otodom_listings.parquet",
        index=False,
    )
    settings = AppSettings(auto_detail_backfill_enabled=False)
    save_app_settings(settings_path(proc), settings)
    assert maybe_schedule_detail_backfill(proc) is False
