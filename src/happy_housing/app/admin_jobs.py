"""Background Admin jobs with streamed log files (Ops scrape / pipeline)."""

import json
import logging
import os
import re
import signal
import subprocess  # nosec B404 — Admin spawns fixed local pipeline/scrape argv
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["running", "succeeded", "failed", "killed"]

_LEVEL_RE = re.compile(
    r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b",
    re.IGNORECASE,
)
"""Match stdlib logging level tokens in a line."""

_ACTIVE_NAME: str = "active.json"
"""Pointer file naming the current Ops job id."""


@dataclass(slots=True)
class AdminJobState:
    """Persisted Admin subprocess job."""

    job_id: str
    """Unique id (timestamp + short name)."""
    name: str
    """Human label (``otodom_scrape``, ``pipeline``, …)."""
    status: JobStatus
    """Lifecycle status."""
    pid: int
    """Child process id."""
    command: list[str]
    """Argv passed to ``Popen``."""
    started_at: str
    """ISO-8601 UTC start."""
    finished_at: str | None
    """ISO-8601 UTC end when finished."""
    returncode: int | None
    """Process exit code when finished."""
    log_path: str
    """Absolute path to combined stdout/stderr log."""
    error: str | None
    """Optional error message."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdminJobState":
        """Rebuild from JSON mapping."""
        return cls(
            job_id=str(payload["job_id"]),
            name=str(payload["name"]),
            status=str(payload["status"]),  # type: ignore[arg-type]  # JSON status
            pid=int(payload["pid"]),
            command=[str(c) for c in list(payload.get("command") or [])],
            started_at=str(payload["started_at"]),
            finished_at=(
                str(payload["finished_at"])
                if payload.get("finished_at") is not None
                else None
            ),
            returncode=(
                int(payload["returncode"])
                if payload.get("returncode") is not None
                else None
            ),
            log_path=str(payload["log_path"]),
            error=(str(payload["error"]) if payload.get("error") is not None else None),
        )


def jobs_dir(processed_dir: Path) -> Path:
    """Directory for Admin job JSON + log files."""
    return processed_dir / "admin_jobs"


def _state_path(processed_dir: Path, job_id: str) -> Path:
    return jobs_dir(processed_dir) / f"{job_id}.json"


def _active_path(processed_dir: Path) -> Path:
    return jobs_dir(processed_dir) / _ACTIVE_NAME


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _write_state(processed_dir: Path, state: AdminJobState) -> None:
    root = jobs_dir(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _state_path(processed_dir, state.job_id)
    path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _set_active(processed_dir: Path, job_id: str) -> None:
    root = jobs_dir(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    _active_path(processed_dir).write_text(
        json.dumps({"job_id": job_id}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_admin_job(processed_dir: Path, job_id: str) -> AdminJobState | None:
    """Load job JSON, or ``None`` if missing/invalid."""
    path = _state_path(processed_dir, job_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load admin job %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    return AdminJobState.from_dict(payload)


def load_active_job(processed_dir: Path) -> AdminJobState | None:
    """Load the current Ops job pointed by ``active.json``."""
    path = _active_path(processed_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        return None
    return load_admin_job(processed_dir, job_id)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refresh_admin_job(processed_dir: Path, job_id: str) -> AdminJobState:
    """Poll child process and update status JSON."""
    state = load_admin_job(processed_dir, job_id)
    if state is None:
        raise FileNotFoundError(f"admin job not found: {job_id}")
    if state.status != "running":
        return state

    proc = _CHILDREN.get(job_id)
    returncode: int | None = None
    if proc is not None:
        returncode = proc.poll()
        if returncode is None:
            return state
        _CHILDREN.pop(job_id, None)
    elif _pid_alive(state.pid):
        return state
    else:
        # Orphaned after Streamlit reload — process gone.
        returncode = 0

    new_status: JobStatus
    if returncode == 0:
        new_status = "succeeded"
    elif returncode is not None and returncode < 0:
        new_status = "killed"
    else:
        new_status = "failed"
    state = AdminJobState(
        job_id=state.job_id,
        name=state.name,
        status=new_status,
        pid=state.pid,
        command=state.command,
        started_at=state.started_at,
        finished_at=_utc_now(),
        returncode=int(returncode if returncode is not None else 1),
        log_path=state.log_path,
        error=state.error,
    )
    _write_state(processed_dir, state)
    logger.info(
        "Admin job %s finished status=%s returncode=%s",
        job_id,
        state.status,
        state.returncode,
    )
    return state


def start_admin_job(
    *,
    name: str,
    command: list[str],
    processed_dir: Path,
    env: dict[str, str] | None = None,
) -> AdminJobState:
    """Spawn a detached subprocess; stream stdout/stderr into a log file."""
    root = jobs_dir(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "job"
    job_id = f"{stamp}_{safe}"
    log_path = root / f"{job_id}.log"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    # Force line-buffered Python logs in the child.
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(  # nosec B603 — argv built in-process, shell=False
        command,
        cwd=str(Path.cwd()),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
        start_new_session=True,
    )
    # Parent closes its copy; child keeps the fd open for streaming.
    log_handle.close()
    state = AdminJobState(
        job_id=job_id,
        name=name,
        status="running",
        pid=int(proc.pid),
        command=list(command),
        started_at=_utc_now(),
        finished_at=None,
        returncode=None,
        log_path=str(log_path.resolve()),
        error=None,
    )
    _write_state(processed_dir, state)
    _set_active(processed_dir, job_id)
    _CHILDREN[job_id] = proc
    logger.info("Started admin job %s pid=%s → %s", job_id, proc.pid, log_path)
    return state


_CHILDREN: dict[str, subprocess.Popen[str]] = {}
"""In-process Popen handles for ``poll()`` while the Streamlit worker lives."""


def stop_admin_job(processed_dir: Path, job_id: str) -> AdminJobState:
    """Terminate a running job (SIGTERM then SIGKILL)."""
    state = load_admin_job(processed_dir, job_id)
    if state is None:
        raise FileNotFoundError(f"admin job not found: {job_id}")
    if state.status != "running":
        return state
    proc = _CHILDREN.get(job_id)
    try:
        if proc is not None:
            proc.terminate()
        else:
            os.killpg(state.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        logger.warning("SIGTERM failed for job %s: %s", job_id, exc)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            break
        if proc is None and not _pid_alive(state.pid):
            break
        time.sleep(0.1)
    still_up = (proc is not None and proc.poll() is None) or (
        proc is None and _pid_alive(state.pid)
    )
    if still_up:
        try:
            if proc is not None:
                proc.kill()
            else:
                os.killpg(state.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            logger.warning("SIGKILL failed for job %s: %s", job_id, exc)
    rc = proc.poll() if proc is not None else -9
    _CHILDREN.pop(job_id, None)
    state = AdminJobState(
        job_id=state.job_id,
        name=state.name,
        status="killed",
        pid=state.pid,
        command=state.command,
        started_at=state.started_at,
        finished_at=_utc_now(),
        returncode=int(rc) if rc is not None else -9,
        log_path=state.log_path,
        error="stopped by Admin",
    )
    _write_state(processed_dir, state)
    return state


def detect_log_level(line: str) -> str:
    """Best-effort level token from a log line (default INFO)."""
    match = _LEVEL_RE.search(line)
    if match is None:
        return "INFO"
    level = match.group(1).upper()
    if level == "WARN":
        return "WARNING"
    return level


def read_log_tail(path: Path, max_lines: int = 200) -> str:
    """Return the last ``max_lines`` of a UTF-8 log file."""
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read log %s: %s", path, exc)
        return ""
    lines = raw.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def format_log_lines_markdown(lines: list[str]) -> str:
    """Streamlit GFM with level colour shortcodes (no custom CSS)."""
    if not lines:
        return "_No log output yet._"
    out: list[str] = []
    for line in lines:
        text = line.replace("`", "'")
        level = detect_log_level(line)
        if level == "ERROR" or level == "CRITICAL":
            out.append(f":red[**{level}**] `{text}`")
        elif level == "WARNING":
            out.append(f":orange[**{level}**] `{text}`")
        elif level == "DEBUG":
            out.append(f":gray[**{level}**] `{text}`")
        else:
            out.append(f":blue[**{level}**] `{text}`")
    return "\n\n".join(out)


def job_elapsed_s(state: AdminJobState, now: datetime | None = None) -> float:
    """Wall seconds since job start (or until finish)."""
    try:
        started = datetime.fromisoformat(state.started_at)
    except ValueError:
        return 0.0
    if state.finished_at:
        try:
            ended = datetime.fromisoformat(state.finished_at)
        except ValueError:
            ended = now or datetime.now(tz=UTC)
    else:
        ended = now or datetime.now(tz=UTC)
    return max(0.0, (ended - started).total_seconds())
