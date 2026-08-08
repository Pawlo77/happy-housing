"""Central logging setup for happy-housing entrypoints.

Library modules use ``logging.getLogger(__name__)`` only — no handlers.
Call ``configure_logging`` once from CLI / Streamlit / pipeline mains.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

PACKAGE_LOGGER_NAME: Final[str] = "happy_housing"
"""Root logger name for all package modules."""

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
"""Default human-readable log line format."""

DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Timestamp format for log records."""

_ENV_LEVEL: Final[str] = "HAPPY_HOUSING_LOG_LEVEL"
"""Optional env override for log level (e.g. ``DEBUG``)."""

_configured: bool = False
"""Whether ``configure_logging`` has successfully attached handlers."""


def resolve_log_level(level: str | int) -> int:
    """Map a level name or int to a ``logging`` level constant.

    Args:
        level: Level name (case-insensitive) or numeric level.

    Returns:
        Numeric logging level.

    Raises:
        ValueError: If ``level`` is not a known name or int.
    """
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    if not name:
        raise ValueError("log level must not be empty")
    resolved = logging.getLevelNamesMapping().get(name)
    if resolved is None:
        raise ValueError(f"unknown log level: {level!r}")
    return int(resolved)


def configure_logging(
    level: str | int | None = None,
    *,
    force: bool = False,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure the ``happy_housing`` package logger (idempotent).

    Does not touch the process root logger. Level defaults to env
    ``HAPPY_HOUSING_LOG_LEVEL`` or ``INFO``.

    Args:
        level: Level name/int; ``None`` → env or INFO.
        force: Drop existing package handlers and reattach.
        handler: Optional handler; default is stderr ``StreamHandler``.

    Returns:
        The configured ``happy_housing`` logger.
    """
    global _configured
    package = logging.getLogger(PACKAGE_LOGGER_NAME)
    if level is None:
        level = os.environ.get(_ENV_LEVEL, "INFO")
    numeric = resolve_log_level(level)

    if _configured and not force and package.handlers:
        package.setLevel(numeric)
        for existing in package.handlers:
            existing.setLevel(numeric)
        return package

    if force:
        for existing in list(package.handlers):
            package.removeHandler(existing)
            existing.close()

    package.setLevel(numeric)
    package.propagate = False

    if not package.handlers:
        attached = handler if handler is not None else logging.StreamHandler(sys.stderr)
        attached.setLevel(numeric)
        attached.setFormatter(
            logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        package.addHandler(attached)
    else:
        for existing in package.handlers:
            existing.setLevel(numeric)

    _configured = True
    package.debug("Logging configured at level %s", logging.getLevelName(numeric))
    return package
