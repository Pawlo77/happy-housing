"""Fingerprint-based pipeline step cache for incremental runs."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PIPELINE_STEPS_SCHEMA: int = 1
"""Bump when step ids or fingerprint meaning change."""


def fingerprint(*parts: str) -> str:
    """Stable short hash of ordered string parts."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def file_token(path: Path) -> str:
    """Size + mtime token for a path (``missing`` when absent)."""
    if not path.exists():
        return f"{path.name}:missing"
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def payload_token(payload: Any) -> str:
    """Hash JSON-serializable settings / knobs."""
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return fingerprint(text)


def outputs_fresh_vs_inputs(
    outputs: list[Path],
    inputs: list[Path],
) -> bool:
    """True when every output exists and is at least as new as every input."""
    if not outputs:
        return False
    if not all(path.exists() and path.stat().st_size > 0 for path in outputs):
        return False
    input_mtimes = [path.stat().st_mtime_ns for path in inputs if path.exists()]
    if not input_mtimes:
        # No inputs on disk → treat existing outputs as fresh.
        return True
    newest_input = max(input_mtimes)
    return all(path.stat().st_mtime_ns >= newest_input for path in outputs)


@dataclass
class StepManifest:
    """Persisted per-step input fingerprints under ``processed/``."""

    path: Path
    """JSON manifest path."""
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    """step_id → {inputs_fp, schema, written_at}."""

    @classmethod
    def load(cls, path: Path) -> StepManifest:
        """Load manifest; missing/invalid → empty."""
        if not path.exists():
            return cls(path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load pipeline steps %s: %s", path, exc)
            return cls(path=path)
        if not isinstance(payload, dict):
            return cls(path=path)
        steps = payload.get("steps")
        if not isinstance(steps, dict):
            return cls(path=path)
        cleaned: dict[str, dict[str, Any]] = {}
        for key, value in steps.items():
            if isinstance(value, dict):
                cleaned[str(key)] = dict(value)
        return cls(path=path, steps=cleaned)

    def save(self) -> None:
        """Write manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": PIPELINE_STEPS_SCHEMA,
            "steps": self.steps,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def should_skip(
        self,
        step_id: str,
        inputs_fp: str,
        outputs: list[Path],
        *,
        force: bool = False,
    ) -> bool:
        """True when fingerprint matches and all outputs exist (unless forced)."""
        if force:
            return False
        if not outputs:
            return False
        if not all(path.exists() and path.stat().st_size > 0 for path in outputs):
            return False
        entry = self.steps.get(step_id) or {}
        if int(entry.get("schema") or 0) != PIPELINE_STEPS_SCHEMA:
            return False
        return str(entry.get("inputs_fp") or "") == inputs_fp

    def mark(self, step_id: str, inputs_fp: str) -> None:
        """Record a successful step fingerprint (call ``save`` after)."""
        from datetime import UTC, datetime

        self.steps[step_id] = {
            "inputs_fp": inputs_fp,
            "schema": PIPELINE_STEPS_SCHEMA,
            "written_at": datetime.now(tz=UTC).isoformat(),
        }

    def mark_and_save(self, step_id: str, inputs_fp: str) -> None:
        """Record a step fingerprint and persist immediately (safe on interrupt)."""
        self.mark(step_id, inputs_fp)
        self.save()


def pipeline_steps_path(processed: Path) -> Path:
    """Default manifest path under processed data."""
    return processed / "pipeline_steps.json"
