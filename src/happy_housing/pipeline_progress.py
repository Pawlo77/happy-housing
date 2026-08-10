"""CLI progress bars for long pipeline loops (tqdm, pytest-safe)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
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
) -> tqdm[Any]:
    """``tqdm`` wrapper for pipeline / ingest loops."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        disable=tqdm_disabled(),
    )
