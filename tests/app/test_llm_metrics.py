"""Tests for LM Studio call metrics store and purpose summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from happy_housing.app.llm_metrics import (
    LlmCallEvent,
    append_llm_call,
    clear_llm_metrics,
    llm_metrics_path,
    load_llm_calls,
    record_timed_llm_call,
    summarize_llm_calls,
)


def test_append_and_summarize_by_purpose(tmp_path: Path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    append_llm_call(
        path,
        LlmCallEvent(
            purpose="offer_review",
            model="m1",
            duration_ms=100.0,
            input_chars=200,
            output_chars=50,
            ok=True,
            error="",
            created_at="2026-08-08T12:00:00+00:00",
        ),
    )
    append_llm_call(
        path,
        LlmCallEvent(
            purpose="offer_review",
            model="m1",
            duration_ms=300.0,
            input_chars=400,
            output_chars=100,
            ok=True,
            error="",
            created_at="2026-08-08T12:01:00+00:00",
        ),
    )
    append_llm_call(
        path,
        LlmCallEvent(
            purpose="offer_review",
            model="m1",
            duration_ms=50.0,
            input_chars=100,
            output_chars=0,
            ok=False,
            error="timeout",
            created_at="2026-08-08T12:02:00+00:00",
        ),
    )
    append_llm_call(
        path,
        LlmCallEvent(
            purpose="other_tool",
            model="m1",
            duration_ms=80.0,
            input_chars=10,
            output_chars=5,
            ok=True,
            error="",
            created_at="2026-08-08T12:03:00+00:00",
        ),
    )
    rows = summarize_llm_calls(path)
    by_purpose = {r.purpose: r for r in rows}
    assert set(by_purpose) == {"offer_review", "other_tool"}
    offer = by_purpose["offer_review"]
    assert offer.n_calls == 3
    assert offer.n_errors == 1
    assert offer.error_ratio == pytest.approx(1 / 3)
    assert offer.avg_duration_ms == pytest.approx(150.0)
    assert offer.std_duration_ms == pytest.approx(132.2875648, rel=1e-5)
    assert offer.avg_input_chars == pytest.approx(233.333333, rel=1e-5)
    assert offer.avg_output_chars == pytest.approx(50.0)
    other = by_purpose["other_tool"]
    assert other.n_calls == 1
    assert other.n_errors == 0
    assert other.error_ratio == 0.0
    assert other.std_duration_ms != other.std_duration_ms  # NaN when n < 2


def test_clear_llm_metrics(tmp_path: Path) -> None:
    path = llm_metrics_path(tmp_path)
    append_llm_call(
        path,
        LlmCallEvent(
            purpose="offer_review",
            model="m1",
            duration_ms=10.0,
            input_chars=1,
            output_chars=1,
            ok=True,
            error="",
            created_at="2026-08-08T12:00:00+00:00",
        ),
    )
    assert load_llm_calls(path)
    clear_llm_metrics(path)
    assert load_llm_calls(path) == []


def test_record_timed_llm_call_success_and_error(tmp_path: Path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    text = record_timed_llm_call(
        path,
        purpose="offer_review",
        model="m1",
        input_chars=12,
        runner=lambda: "hello world",
    )
    assert text == "hello world"
    with pytest.raises(RuntimeError, match="boom"):
        record_timed_llm_call(
            path,
            purpose="offer_review",
            model="m1",
            input_chars=12,
            runner=_raise,
        )
    events = load_llm_calls(path)
    assert len(events) == 2
    assert events[0].ok is True
    assert events[0].output_chars == len("hello world")
    assert events[1].ok is False
    assert "boom" in events[1].error


def _raise() -> str:
    raise RuntimeError("boom")
