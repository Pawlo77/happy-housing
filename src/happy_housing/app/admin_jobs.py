"""Background Admin jobs with streamed log files (Ops scrape / pipeline)."""

import contextlib
import json
import logging
import os
import re
import signal
import subprocess  # nosec B404 — Admin/worker spawn fixed local argv
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["running", "succeeded", "failed", "killed"]
"""Lifecycle values persisted in Admin job JSON."""

_LEVEL_RE = re.compile(
    r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b",
    re.IGNORECASE,
)
"""Match stdlib logging level tokens in a line."""

_ACTIVE_NAME: str = "active.json"
"""Pointer file naming the current Ops job id."""

_SIGSEGV_HINT: str = (
    "Child died SIGSEGV (rc=-11) — Streamlit cannot fork on this macOS host. "
    "Use `make app` (starts Ops job worker) or run "
    "`uv run python -m happy_housing.pipeline` in Terminal."
)
"""User-facing hint when the job child never starts."""


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
    """Child process id (0 while queued for the Ops worker)."""
    command: list[str]
    """Argv for the job payload."""
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


def cursor_agent_sandbox_active() -> bool:
    """True when this process inherits Cursor agent seatbelt markers."""
    return bool(os.environ.get("CURSOR_AGENT")) or bool(
        os.environ.get("__CURSOR_SANDBOX_ENV_RESTORE")
    )


def jobs_dir(processed_dir: Path) -> Path:
    """Directory for Admin job JSON + log files."""
    return processed_dir / "admin_jobs"


def _state_path(processed_dir: Path, job_id: str) -> Path:
    return jobs_dir(processed_dir) / f"{job_id}.json"


def _exit_path(processed_dir: Path, job_id: str) -> Path:
    return jobs_dir(processed_dir) / f"{job_id}.exit"


def _runner_path(processed_dir: Path, job_id: str) -> Path:
    return jobs_dir(processed_dir) / f"{job_id}.py"


def _request_path(processed_dir: Path, job_id: str) -> Path:
    return jobs_dir(processed_dir) / f"{job_id}.request"


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


def _read_exit_code(processed_dir: Path, job_id: str) -> int | None:
    path = _exit_path(processed_dir, job_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text.splitlines()[0].strip())
    except ValueError:
        return None


def _error_for_returncode(returncode: int) -> str | None:
    if returncode == -11:
        return _SIGSEGV_HINT
    if returncode < 0:
        return f"Child killed by signal {-returncode}"
    return None


def _reap_spawned_pid(pid: int) -> int | None:
    """Non-blocking waitpid for children we spawned via posix_spawn."""
    if pid <= 0 or pid not in _SPAWNED_PIDS:
        return None
    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        _SPAWNED_PIDS.discard(pid)
        return None
    if wpid == 0:
        return None
    _SPAWNED_PIDS.discard(pid)
    return int(os.waitstatus_to_exitcode(status))


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
            exit_code = _read_exit_code(processed_dir, job_id)
            if exit_code is not None:
                returncode = exit_code
                _CHILDREN.pop(job_id, None)
            else:
                return state
        else:
            _CHILDREN.pop(job_id, None)
            exit_code = _read_exit_code(processed_dir, job_id)
            if exit_code is not None:
                returncode = exit_code
    else:
        exit_code = _read_exit_code(processed_dir, job_id)
        if exit_code is not None:
            returncode = exit_code
            _reap_spawned_pid(state.pid)
        else:
            reaped = _reap_spawned_pid(state.pid)
            if reaped is not None:
                returncode = reaped
            elif state.pid <= 0:
                # Queued for Ops worker — keep waiting.
                return state
            elif _pid_alive(state.pid):
                return state
            elif _request_path(processed_dir, job_id).exists():
                # Worker has not claimed the request yet.
                return state
            else:
                returncode = -11

    new_status: JobStatus
    if returncode == 0:
        new_status = "succeeded"
    elif returncode is not None and returncode < 0:
        new_status = "killed"
    else:
        new_status = "failed"
    error = state.error or _error_for_returncode(int(returncode))
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
        error=error,
    )
    _write_state(processed_dir, state)
    logger.info(
        "Admin job %s finished status=%s returncode=%s",
        job_id,
        state.status,
        state.returncode,
    )
    return state


def _write_runner_script(
    runner: Path,
    command: list[str],
    cwd: Path,
    log_path: Path,
    exit_path: Path,
    env: dict[str, str],
) -> None:
    """Write a Python wrapper that tees logs and records the exit code."""
    payload = {
        "cwd": str(cwd.resolve()),
        "log_path": str(log_path.resolve()),
        "exit_path": str(exit_path.resolve()),
        "command": [str(c) for c in command],
        "env": {str(k): str(v) for k, v in env.items()},
    }
    body = (
        '"""Auto-generated Admin Ops job runner."""\n'
        "import os\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        f"PAYLOAD = {payload!r}\n\n"
        "def main() -> None:\n"
        '    cwd = Path(PAYLOAD["cwd"])\n'
        '    log_path = Path(PAYLOAD["log_path"])\n'
        '    exit_path = Path(PAYLOAD["exit_path"])\n'
        '    command = list(PAYLOAD["command"])\n'
        "    env = os.environ.copy()\n"
        '    env.update({str(k): str(v) for k, v in PAYLOAD["env"].items()})\n'
        "    os.chdir(cwd)\n"
        "    log_path.parent.mkdir(parents=True, exist_ok=True)\n"
        '    with log_path.open("a", encoding="utf-8", buffering=1) as log:\n'
        '        log.write(f"Admin job runner pid={os.getpid()}\\n")\n'
        '        log.write(f"Admin job cwd={cwd}\\n")\n'
        '        log.write(f"Admin job cmd={command!r}\\n")\n'
        "        log.flush()\n"
        "        completed = subprocess.run(\n"
        "            command,\n"
        "            stdout=log,\n"
        "            stderr=subprocess.STDOUT,\n"
        "            env=env,\n"
        "            check=False,\n"
        "        )\n"
        "        rc = int(completed.returncode)\n"
        '        log.write(f"Admin job exit rc={rc}\\n")\n'
        '    exit_path.write_text(f"{rc}\\n", encoding="utf-8")\n'
        "    raise SystemExit(rc)\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
    runner.write_text(body, encoding="utf-8")


def _spawn_via_posix_spawn(launcher: list[str], env: dict[str, str]) -> int:
    """Start ``launcher`` with ``posix_spawnp`` (safe from multithreaded parents)."""
    if not hasattr(os, "posix_spawnp"):
        raise OSError("posix_spawnp unavailable")
    exe = str(launcher[0])
    argv = [str(a) for a in launcher]
    clean_env = {str(k): str(v) for k, v in env.items()}
    try:
        pid = int(os.posix_spawnp(exe, argv, clean_env, setsid=True))
    except TypeError:
        pid = int(os.posix_spawnp(exe, argv, clean_env))
    except OSError:
        raise
    _SPAWNED_PIDS.add(pid)
    return pid


def _launcher_for_runner(command: list[str], runner: Path) -> list[str]:
    python_bin = sys.executable
    if command and Path(str(command[0])).name.startswith("python"):
        python_bin = str(command[0])
    return [python_bin, str(runner)]


def _build_runner_env(
    child_env: dict[str, str],
) -> dict[str, str]:
    runner_env = {
        "PYTHONUNBUFFERED": child_env.get("PYTHONUNBUFFERED", "1"),
        "HAPPY_HOUSING_LOG_LEVEL": child_env.get("HAPPY_HOUSING_LOG_LEVEL", "INFO"),
        "PATH": child_env.get("PATH", os.environ.get("PATH", "")),
        "HOME": child_env.get("HOME", str(Path.home())),
        "TMPDIR": child_env.get("TMPDIR", ""),
        "VIRTUAL_ENV": child_env.get("VIRTUAL_ENV", ""),
        "LANG": child_env.get("LANG", "en_US.UTF-8"),
        "PYTHONPATH": child_env.get("PYTHONPATH", ""),
    }
    for key in (
        "DOCKER_HOST",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ):
        if child_env.get(key):
            runner_env[key] = child_env[key]
    return runner_env


def run_queued_admin_job(processed_dir: Path, job_id: str) -> bool:
    """Execute a queued job from the Ops worker (never from Streamlit).

    Returns:
        True when a new child was started.
    """
    state = load_admin_job(processed_dir, job_id)
    if state is None or state.status != "running":
        return False
    if _read_exit_code(processed_dir, job_id) is not None:
        return False
    if state.pid > 0 and _pid_alive(state.pid):
        return False
    runner = _runner_path(processed_dir, job_id).resolve()
    if not runner.exists():
        logger.warning("Missing runner for job %s", job_id)
        return False
    launcher = _launcher_for_runner(state.command, runner)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        pid = _spawn_via_posix_spawn(launcher, env)
    except OSError:
        proc = subprocess.Popen(  # nosec B603 — runner script, shell=False
            launcher,
            cwd=str(Path.cwd()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            env=env,
        )
        pid = int(proc.pid)
        _CHILDREN[job_id] = proc
    state = AdminJobState(
        job_id=state.job_id,
        name=state.name,
        status="running",
        pid=pid,
        command=state.command,
        started_at=state.started_at,
        finished_at=None,
        returncode=None,
        log_path=state.log_path,
        error=None,
    )
    _write_state(processed_dir, state)
    logger.info("Worker spawned admin job %s pid=%s", job_id, pid)
    return True


def start_admin_job(
    name: str,
    command: list[str],
    processed_dir: Path,
    env: dict[str, str] | None = None,
) -> AdminJobState:
    """Queue/spawn a detached job; stream stdout/stderr into a log file.

    Prefer ``posix_spawn`` (fork-safe). Always write a ``.request`` so the
    Ops worker can claim the job when Streamlit cannot spawn.
    """
    root = jobs_dir(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "job"
    job_id = f"{stamp}_{safe}"
    log_path = (root / f"{job_id}.log").resolve()
    log_path.write_text("", encoding="utf-8")
    exit_path = _exit_path(processed_dir, job_id).resolve()
    if exit_path.exists():
        exit_path.unlink()
    runner = _runner_path(processed_dir, job_id).resolve()
    child_env = {k: str(v) for k, v in os.environ.items()}
    if env:
        child_env.update({k: str(v) for k, v in env.items()})
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.pop("CURSOR_AGENT", None)
    child_env.pop("__CURSOR_SANDBOX_ENV_RESTORE", None)
    cwd = Path.cwd().resolve()
    runner_env = _build_runner_env(child_env)
    _write_runner_script(
        runner,
        command=list(command),
        cwd=cwd,
        log_path=log_path,
        exit_path=exit_path,
        env=runner_env,
    )
    launcher = _launcher_for_runner(list(command), runner)
    request = _request_path(processed_dir, job_id)
    request.write_text("queued\n", encoding="utf-8")

    pid = 0
    spawn_error: str | None = None
    proc: subprocess.Popen[str] | None = None
    try:
        pid = _spawn_via_posix_spawn(launcher, {**child_env, **runner_env})
        request.unlink(missing_ok=True)
        logger.info(
            "Started admin job %s via posix_spawn pid=%s → %s",
            job_id,
            pid,
            log_path,
        )
    except OSError as exc:
        spawn_error = (
            f"direct spawn failed ({exc}); queued for Ops worker "
            f"(start app with `make app`)"
        )
        logger.warning("%s", spawn_error)

    state = AdminJobState(
        job_id=job_id,
        name=name,
        status="running",
        pid=pid,
        command=list(command),
        started_at=_utc_now(),
        finished_at=None,
        returncode=None,
        log_path=str(log_path),
        error=spawn_error,
    )
    _write_state(processed_dir, state)
    _set_active(processed_dir, job_id)
    if proc is not None:
        _CHILDREN[job_id] = proc
    logger.info("Started admin job %s pid=%s → %s", job_id, pid, log_path)
    return state


_CHILDREN: dict[str, subprocess.Popen[str]] = {}
"""In-process Popen handles for ``poll()`` while the Streamlit worker lives."""

_SPAWNED_PIDS: set[int] = set()
"""Pids created with ``posix_spawn`` that still need ``waitpid`` reaping."""


def stop_admin_job(processed_dir: Path, job_id: str) -> AdminJobState:
    """Terminate a running job (SIGTERM then SIGKILL)."""
    state = load_admin_job(processed_dir, job_id)
    if state is None:
        raise FileNotFoundError(f"admin job not found: {job_id}")
    if state.status != "running":
        return state
    _request_path(processed_dir, job_id).unlink(missing_ok=True)
    proc = _CHILDREN.get(job_id)
    if state.pid > 0:
        try:
            if proc is not None:
                proc.terminate()
            else:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(state.pid, signal.SIGTERM)
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.kill(state.pid, signal.SIGTERM)
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
                    with contextlib.suppress(
                        ProcessLookupError, PermissionError, OSError
                    ):
                        os.killpg(state.pid, signal.SIGKILL)
                    with contextlib.suppress(
                        ProcessLookupError, PermissionError, OSError
                    ):
                        os.kill(state.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError) as exc:
                logger.warning("SIGKILL failed for job %s: %s", job_id, exc)
    rc = proc.poll() if proc is not None else -9
    exit_code = _read_exit_code(processed_dir, job_id)
    if exit_code is not None:
        rc = exit_code
    _CHILDREN.pop(job_id, None)
    _SPAWNED_PIDS.discard(state.pid)
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
