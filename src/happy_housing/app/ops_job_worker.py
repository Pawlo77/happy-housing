"""Watch Admin Ops job requests and run them outside Streamlit."""

import argparse
import logging
import os
import time
from pathlib import Path

from happy_housing.app.admin_jobs import (
    jobs_dir,
    load_admin_job,
    run_queued_admin_job,
)
from happy_housing.app.detail_backfill_jobs import maybe_schedule_detail_backfill
from happy_housing.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _processed_default() -> Path:
    return Path("data/processed").resolve()


def run_worker_loop(processed_dir: Path, poll_s: float = 0.5) -> None:
    """Poll ``*.request`` files and execute queued Admin jobs."""
    root = jobs_dir(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    logger.info("Ops job worker watching %s", root)
    while True:
        ran_request = False
        for request in sorted(root.glob("*.request")):
            job_id = request.stem
            try:
                state = load_admin_job(processed_dir, job_id)
                if state is None:
                    request.unlink(missing_ok=True)
                    continue
                ran = run_queued_admin_job(processed_dir, job_id)
                if ran:
                    logger.info("Worker started job %s", job_id)
                ran_request = True
            except Exception:
                logger.exception("Worker failed on job %s", job_id)
            finally:
                request.unlink(missing_ok=True)
        if not ran_request:
            maybe_schedule_detail_backfill(processed_dir)
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Ops job worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed",
        type=Path,
        default=_processed_default(),
        help="Processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="Poll interval seconds",
    )
    args = parser.parse_args(argv)
    configure_logging(os.environ.get("HAPPY_HOUSING_LOG_LEVEL", "INFO"), force=True)
    run_worker_loop(args.processed.resolve(), poll_s=float(args.poll))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
