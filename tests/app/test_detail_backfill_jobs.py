"""Tests for auto-scheduled detail backfill Ops jobs."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.detail_backfill_jobs import (
    maybe_schedule_detail_backfill,
    stop_auto_detail_backfill,
)


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


def test_stop_auto_detail_backfill_disables_and_stops_job(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "admin_jobs").mkdir()
    save_app_settings(
        settings_path(proc),
        AppSettings(auto_detail_backfill_enabled=True),
    )

    import happy_housing.app.detail_backfill_jobs as jobs_mod

    stop_calls: list[str] = []
    real_load = jobs_mod.load_active_job
    real_refresh = jobs_mod.refresh_admin_job
    real_stop = jobs_mod.stop_admin_job

    active = MagicMock()
    active.job_id = "bf-1"
    active.name = "detail_backfill"
    active.status = "running"

    jobs_mod.load_active_job = lambda _p: active  # type: ignore[method-assign]
    jobs_mod.refresh_admin_job = (  # type: ignore[method-assign]
        lambda _p, job_id: active if job_id == "bf-1" else None
    )

    def _fake_stop(processed_dir: Path, job_id: str) -> MagicMock:
        _ = processed_dir
        stop_calls.append(job_id)
        active.status = "killed"
        return active

    jobs_mod.stop_admin_job = _fake_stop  # type: ignore[method-assign]
    try:
        result = stop_auto_detail_backfill(proc)
    finally:
        jobs_mod.load_active_job = real_load  # type: ignore[method-assign]
        jobs_mod.refresh_admin_job = real_refresh  # type: ignore[method-assign]
        jobs_mod.stop_admin_job = real_stop  # type: ignore[method-assign]

    assert result.auto_was_enabled is True
    assert result.auto_disabled is True
    assert result.job_stopped is True
    assert result.job_id == "bf-1"
    assert stop_calls == ["bf-1"]
    loaded = load_app_settings(settings_path(proc))
    assert loaded.auto_detail_backfill_enabled is False
    assert maybe_schedule_detail_backfill(proc) is False


def test_stop_auto_detail_backfill_noop_when_already_off(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "admin_jobs").mkdir()
    save_app_settings(
        settings_path(proc),
        AppSettings(auto_detail_backfill_enabled=False),
    )
    result = stop_auto_detail_backfill(proc)
    assert result.auto_was_enabled is False
    assert result.auto_disabled is False
    assert result.job_stopped is False
    assert result.job_id is None


def test_ops_panel_has_stop_auto_detail_backfill_button() -> None:
    import inspect

    from happy_housing.app import page_admin

    source = inspect.getsource(page_admin._panel_ops)
    assert "Stop auto detail backfill" in source
    assert "stop_auto_detail_backfill" in source
    assert 'key="stop_auto_detail_backfill"' in source
