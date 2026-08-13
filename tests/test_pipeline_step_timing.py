"""Tests for pipeline step timing helper."""

import logging

from happy_housing.pipeline_progress import log_step


def test_log_step_emits_start_and_done(caplog) -> None:
    logger = logging.getLogger("happy_housing.test_timing")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        log_step(logger, "enrich"),
    ):
        pass
    messages = [r.getMessage() for r in caplog.records]
    assert any("STEP start enrich" in m for m in messages)
    assert any("STEP done enrich" in m and "s" in m for m in messages)
