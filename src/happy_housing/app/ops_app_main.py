"""Launch Ops job worker + Streamlit so jobs never fork from Streamlit."""

import logging
import os
import signal
import subprocess  # nosec B404 — supervisor spawns worker + streamlit only
import sys
import time
from pathlib import Path

from happy_housing.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Start the Ops worker, then Streamlit; stop worker on exit."""
    configure_logging(os.environ.get("HAPPY_HOUSING_LOG_LEVEL", "INFO"), force=True)
    processed = Path("data/processed").resolve()
    processed.mkdir(parents=True, exist_ok=True)
    worker_cmd = [
        sys.executable,
        "-m",
        "happy_housing.app.ops_job_worker",
        "--processed",
        str(processed),
    ]
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/happy_housing/app/streamlit_app.py",
    ]
    worker = subprocess.Popen(  # nosec B603 — fixed argv
        worker_cmd,
        start_new_session=True,
    )
    logger.info("Ops job worker pid=%s", worker.pid)
    streamlit_rc = 1
    try:
        streamlit_rc = int(
            subprocess.call(streamlit_cmd)  # nosec B603 — fixed argv
        )
    finally:
        if worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                worker.terminate()
            deadline = time.time() + 5.0
            while time.time() < deadline and worker.poll() is None:
                time.sleep(0.1)
            if worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    worker.kill()
        logger.info("Ops job worker stopped")
    return streamlit_rc


if __name__ == "__main__":
    raise SystemExit(main())
