"""Persist LM Studio call events and summarize metrics by purpose."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PURPOSE_OFFER_REVIEW: str = "offer_review"
"""Explore Offer AI review generation."""

PURPOSE_PREF_PARSE: str = "pref_parse"
"""Explore Prefs natural-language → JSON translation."""

_MAX_STORED_CALLS: int = 5_000
"""Soft cap — oldest events drop when appending past this size."""


@dataclass(frozen=True, slots=True)
class LlmCallEvent:
    """One LM Studio request outcome."""

    purpose: str
    """Call purpose key (e.g. ``offer_review``)."""
    model: str
    """LM Studio model id."""
    duration_ms: float
    """Wall time from request start to finish."""
    input_chars: int
    """System + user prompt character count."""
    output_chars: int
    """Assistant text character count (0 on failure)."""
    ok: bool
    """True when the call returned text without raising."""
    error: str
    """Error message when ``ok`` is False; else empty."""
    created_at: str
    """ISO-8601 UTC timestamp."""


@dataclass(frozen=True, slots=True)
class LlmPurposeStats:
    """Aggregated LMS metrics for one purpose."""

    purpose: str
    """Call purpose key."""
    n_calls: int
    """Total recorded calls."""
    n_errors: int
    """Failed calls."""
    error_ratio: float
    """``n_errors / n_calls`` (0 when empty)."""
    avg_duration_ms: float
    """Mean duration in milliseconds."""
    std_duration_ms: float
    """Sample std of duration (NaN when ``n_calls < 2``)."""
    avg_input_chars: float
    """Mean input size in characters."""
    std_input_chars: float
    """Sample std of input size (NaN when ``n_calls < 2``)."""
    avg_output_chars: float
    """Mean output size in characters."""
    std_output_chars: float
    """Sample std of output size (NaN when ``n_calls < 2``)."""


def llm_metrics_path(processed: Path) -> Path:
    """JSONL log of LMS call events under processed data."""
    return processed / "llm_call_metrics.jsonl"


def append_llm_call(path: Path, event: LlmCallEvent) -> None:
    """Append one call event; trim oldest rows past the soft cap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    _trim_if_needed(path)


def load_llm_calls(path: Path) -> list[LlmCallEvent]:
    """Load all call events; missing/invalid → empty list."""
    if not path.exists():
        return []
    out: list[LlmCallEvent] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read LLM metrics %s: %s", path, exc)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = _event_from_payload(payload)
        if event is not None:
            out.append(event)
    return out


def clear_llm_metrics(path: Path) -> None:
    """Delete the LMS metrics log when present."""
    if path.exists():
        path.unlink()
        logger.info("Cleared LLM metrics at %s", path)


def summarize_llm_calls(path: Path) -> list[LlmPurposeStats]:
    """Aggregate call metrics grouped by purpose (sorted by purpose)."""
    events = load_llm_calls(path)
    by_purpose: dict[str, list[LlmCallEvent]] = {}
    for event in events:
        by_purpose.setdefault(event.purpose, []).append(event)
    rows: list[LlmPurposeStats] = []
    for purpose in sorted(by_purpose):
        rows.append(_summarize_group(purpose, by_purpose[purpose]))
    return rows


def record_timed_llm_call(
    path: Path,
    *,
    purpose: str,
    model: str,
    input_chars: int,
    runner: Callable[[], str],
) -> str:
    """Run an LMS callable, persist duration/size/error metrics, return text."""
    started = time.perf_counter()
    created_at = datetime.now(tz=UTC).isoformat()
    try:
        text = runner()
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        append_llm_call(
            path,
            LlmCallEvent(
                purpose=purpose,
                model=model,
                duration_ms=duration_ms,
                input_chars=int(input_chars),
                output_chars=0,
                ok=False,
                error=str(exc)[:500],
                created_at=created_at,
            ),
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000.0
    out = str(text).strip()
    append_llm_call(
        path,
        LlmCallEvent(
            purpose=purpose,
            model=model,
            duration_ms=duration_ms,
            input_chars=int(input_chars),
            output_chars=len(out),
            ok=True,
            error="",
            created_at=created_at,
        ),
    )
    return out


def purpose_stats_frame(rows: list[LlmPurposeStats]) -> list[dict[str, Any]]:
    """Admin-friendly dict rows (NaN → None for Streamlit display)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({
            "purpose": row.purpose,
            "calls": row.n_calls,
            "errors": row.n_errors,
            "error_ratio": round(row.error_ratio, 4),
            "avg_duration_ms": _finite_or_none(row.avg_duration_ms),
            "std_duration_ms": _finite_or_none(row.std_duration_ms),
            "avg_input_chars": _finite_or_none(row.avg_input_chars),
            "std_input_chars": _finite_or_none(row.std_input_chars),
            "avg_output_chars": _finite_or_none(row.avg_output_chars),
            "std_output_chars": _finite_or_none(row.std_output_chars),
        })
    return out


def _finite_or_none(value: float) -> float | None:
    if value != value or math.isinf(value):
        return None
    return float(value)


def _sample_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _summarize_group(purpose: str, events: list[LlmCallEvent]) -> LlmPurposeStats:
    n = len(events)
    n_errors = sum(1 for e in events if not e.ok)
    avg_d, std_d = _sample_mean_std([e.duration_ms for e in events])
    avg_i, std_i = _sample_mean_std([float(e.input_chars) for e in events])
    avg_o, std_o = _sample_mean_std([float(e.output_chars) for e in events])
    return LlmPurposeStats(
        purpose=purpose,
        n_calls=n,
        n_errors=n_errors,
        error_ratio=(n_errors / n) if n else 0.0,
        avg_duration_ms=avg_d,
        std_duration_ms=std_d,
        avg_input_chars=avg_i,
        std_input_chars=std_i,
        avg_output_chars=avg_o,
        std_output_chars=std_o,
    )


def _event_from_payload(payload: Mapping[str, Any]) -> LlmCallEvent | None:
    try:
        return LlmCallEvent(
            purpose=str(payload.get("purpose") or "").strip() or "unknown",
            model=str(payload.get("model") or ""),
            duration_ms=float(payload.get("duration_ms", 0.0)),
            input_chars=int(payload.get("input_chars", 0)),
            output_chars=int(payload.get("output_chars", 0)),
            ok=bool(payload.get("ok", False)),
            error=str(payload.get("error") or ""),
            created_at=str(payload.get("created_at") or ""),
        )
    except (TypeError, ValueError):
        return None


def _trim_if_needed(path: Path) -> None:
    events = load_llm_calls(path)
    if len(events) <= _MAX_STORED_CALLS:
        return
    keep = events[-_MAX_STORED_CALLS:]
    path.write_text(
        "".join(json.dumps(asdict(e), ensure_ascii=False) + "\n" for e in keep),
        encoding="utf-8",
    )
