"""Admin Ops helpers for auto-scheduled listing detail backfill."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from happy_housing.app.admin_jobs import (
    load_active_job,
    refresh_admin_job,
    start_admin_job,
    stop_admin_job,
)
from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.ingest.detail_backfill import count_pending_detail_backfill

logger = logging.getLogger(__name__)

DETAIL_BACKFILL_JOB_NAME: str = "detail_backfill"
"""Ops job name for listing detail backfill batches."""


@dataclass(frozen=True, slots=True)
class StopAutoDetailBackfillResult:
    """Outcome of stopping auto-scheduled detail backfill from Ops."""

    auto_was_enabled: bool
    """Whether auto-schedule was on before this call."""
    auto_disabled: bool
    """Whether settings were updated to disable auto-schedule."""
    job_stopped: bool
    """Whether a running ``detail_backfill`` job was stopped."""
    job_id: str | None
    """Stopped job id when ``job_stopped``; else ``None``."""


def detail_backfill_command(processed_dir: Path, settings: AppSettings) -> list[str]:
    """Build argv for one detail-backfill Ops job."""
    level = str(settings.log_level)
    proxy_flag = bool(settings.http_free_proxies_enabled)
    batch = int(settings.detail_backfill_batch_size)
    workers = int(settings.detail_backfill_workers)
    sleep_s = float(settings.detail_backfill_sleep_s)
    return [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "from happy_housing.logging_config import configure_logging; "
            "from happy_housing.ingest.proxy_pool import configure_http_proxies; "
            "from happy_housing.ingest.detail_backfill import ("
            "run_detail_backfill_batch); "
            "from happy_housing.app.app_settings import ("
            "load_app_settings, settings_path); "
            f"configure_logging({level!r}, force=True); "
            f"p = Path({str(processed_dir.resolve())!r}); "
            "s = load_app_settings(settings_path(p)); "
            f"configure_http_proxies(enabled={proxy_flag!r}); "
            f"run_detail_backfill_batch(p, batch_size={batch}, "
            f"detail_workers={workers}, sleep_s={sleep_s!r})"
        ),
    ]


def maybe_schedule_detail_backfill(processed_dir: Path) -> bool:
    """Queue a detail-backfill job when idle and pending rows exist.

    Returns:
        True when a new job was queued.
    """
    settings = load_app_settings(settings_path(processed_dir))
    if not settings.auto_detail_backfill_enabled:
        return False
    active = load_active_job(processed_dir)
    if active is not None:
        active = refresh_admin_job(processed_dir, active.job_id)
        if active.status == "running":
            return False
    pending = count_pending_detail_backfill(processed_dir)
    total = sum(int(v) for v in pending.values())
    if total <= 0:
        return False
    start_admin_job(
        name=DETAIL_BACKFILL_JOB_NAME,
        command=detail_backfill_command(processed_dir, settings),
        processed_dir=processed_dir,
        env={"HAPPY_HOUSING_LOG_LEVEL": str(settings.log_level)},
    )
    logger.info("Auto-scheduled detail_backfill (pending=%s)", pending)
    return True


def stop_auto_detail_backfill(processed_dir: Path) -> StopAutoDetailBackfillResult:
    """Disable auto-schedule and stop a running detail-backfill job if any.

    Saves ``auto_detail_backfill_enabled=False``. Does not stop other Ops jobs.
    """
    settings_file = settings_path(processed_dir)
    settings = load_app_settings(settings_file)
    auto_was_enabled = bool(settings.auto_detail_backfill_enabled)
    auto_disabled = False
    if auto_was_enabled:
        settings.auto_detail_backfill_enabled = False
        save_app_settings(settings_file, settings)
        auto_disabled = True
        logger.info("Disabled auto_detail_backfill_enabled via Ops stop")

    job_stopped = False
    job_id: str | None = None
    active = load_active_job(processed_dir)
    if active is not None:
        active = refresh_admin_job(processed_dir, active.job_id)
        if active.name == DETAIL_BACKFILL_JOB_NAME and active.status == "running":
            stop_admin_job(processed_dir, active.job_id)
            job_stopped = True
            job_id = active.job_id
            logger.info("Stopped detail_backfill job %s via Ops", job_id)

    return StopAutoDetailBackfillResult(
        auto_was_enabled=auto_was_enabled,
        auto_disabled=auto_disabled,
        job_stopped=job_stopped,
        job_id=job_id,
    )
