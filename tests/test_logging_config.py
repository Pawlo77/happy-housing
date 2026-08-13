"""Tests for package logging configuration."""

import logging

import pytest

from happy_housing.logging_config import (
    PACKAGE_LOGGER_NAME,
    configure_logging,
    resolve_log_level,
)


def test_resolve_log_level_accepts_names_and_ints() -> None:
    assert resolve_log_level("DEBUG") == logging.DEBUG
    assert resolve_log_level("info") == logging.INFO
    assert resolve_log_level(logging.WARNING) == logging.WARNING


def test_resolve_log_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        resolve_log_level("nope")


def test_configure_logging_attaches_handler_and_is_idempotent() -> None:
    records: list[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    configure_logging("DEBUG", force=True, handler=ListHandler())
    package = logging.getLogger(PACKAGE_LOGGER_NAME)
    assert package.level == logging.DEBUG
    assert package.handlers
    child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.pipeline")
    child.info("pipeline-ready")
    assert any("pipeline-ready" in record.getMessage() for record in records)

    handler_count = len(package.handlers)
    configure_logging("WARNING", force=False)
    assert len(package.handlers) == handler_count
    assert package.level == logging.WARNING
