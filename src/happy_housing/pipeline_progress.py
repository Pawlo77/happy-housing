"""CLI progress bars and step timing for long pipeline loops."""

import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm


def tqdm_disabled() -> bool:
    """True when progress bars should stay off (pytest, non-TTY, explicit env)."""
    if os.environ.get("TQDM_DISABLE") == "1":
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return not sys.stderr.isatty()


def pipeline_tqdm[T](
    iterable: Iterable[T] | None = None,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
) -> "tqdm[Any]":
    """``tqdm`` wrapper for pipeline / ingest loops."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        disable=tqdm_disabled(),
    )


@contextmanager
def log_step(logger: logging.Logger, name: str) -> Iterator[None]:
    """Log ``STEP start/done`` with wall duration for Admin log streaming."""
    logger.info("STEP start %s", name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        logger.exception("STEP fail %s after %.1fs", name, dt)
        raise
    else:
        dt = time.perf_counter() - t0
        logger.info("STEP done %s in %.1fs (%.2f min)", name, dt, dt / 60.0)
