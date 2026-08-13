"""Tests for Admin background jobs and log parsing."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from happy_housing.app.admin_jobs import (
    AdminJobState,
    cursor_agent_sandbox_active,
    detect_log_level,
    format_log_lines_markdown,
    jobs_dir,
    load_active_job,
    read_log_tail,
    refresh_admin_job,
    start_admin_job,
    stop_admin_job,
)


def test_detect_log_level() -> None:
    assert detect_log_level("2026-08-13 21:00:00 INFO [hh] hi") == "INFO"
    assert detect_log_level("WARNING [x] bad") == "WARNING"
    assert detect_log_level("oops ERROR boom") == "ERROR"
    assert detect_log_level("plain text") == "INFO"


def test_format_log_lines_markdown_colors() -> None:
    text = format_log_lines_markdown([
        "2026-08-13 21:00:00 INFO [a] ok",
        "2026-08-13 21:00:01 WARNING [a] slow",
        "2026-08-13 21:00:02 ERROR [a] fail",
    ])
    assert ":blue[" in text or "**INFO**" in text
    assert ":orange[" in text or "**WARNING**" in text
    assert ":red[" in text or "**ERROR**" in text


def test_start_and_refresh_job_writes_log(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    job = start_admin_job(
        name="echo",
        command=[sys.executable, "-c", "print('hello-job'); print('done')"],
        processed_dir=root,
    )
    assert job.status == "running"
    assert job.job_id
    assert Path(job.log_path).exists()
    active = load_active_job(root)
    assert active is not None
    assert active.job_id == job.job_id

    deadline = time.time() + 10.0
    refreshed = job
    while time.time() < deadline:
        refreshed = refresh_admin_job(root, job.job_id)
        if refreshed.status != "running":
            break
        time.sleep(0.05)
    assert refreshed.status == "succeeded"
    assert refreshed.returncode == 0
    exit_path = jobs_dir(root) / f"{job.job_id}.exit"
    assert exit_path.exists()
    assert exit_path.read_text(encoding="utf-8").strip() == "0"
    tail = read_log_tail(Path(refreshed.log_path), max_lines=50)
    assert "hello-job" in tail


def test_refresh_reads_exit_sidecar_without_popen(tmp_path: Path) -> None:
    """After Streamlit reload, exit sidecar still reports the real returncode."""
    root = tmp_path / "processed"
    job = start_admin_job(
        name="fail",
        command=[sys.executable, "-c", "raise SystemExit(7)"],
        processed_dir=root,
    )
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if (jobs_dir(root) / f"{job.job_id}.exit").exists():
            break
        time.sleep(0.05)
    # Drop in-process handle (simulates Streamlit worker reload).
    from happy_housing.app import admin_jobs as mod

    mod._CHILDREN.pop(job.job_id, None)
    refreshed = refresh_admin_job(root, job.job_id)
    assert refreshed.status == "failed"
    assert refreshed.returncode == 7


def test_stop_admin_job(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    job = start_admin_job(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        processed_dir=root,
    )
    assert job.status == "running"
    stopped = stop_admin_job(root, job.job_id)
    assert stopped.status in {"failed", "killed", "succeeded"}
    refreshed = refresh_admin_job(root, job.job_id)
    assert refreshed.status != "running"


def test_admin_job_state_roundtrip(tmp_path: Path) -> None:
    state = AdminJobState(
        job_id="j1",
        name="pipe",
        status="running",
        pid=1,
        command=["echo"],
        started_at="2026-08-13T20:00:00+00:00",
        finished_at=None,
        returncode=None,
        log_path=str(tmp_path / "j1.log"),
        error=None,
    )
    path = jobs_dir(tmp_path) / "j1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    loaded = AdminJobState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.job_id == "j1"
    assert loaded.name == "pipe"


def test_cursor_agent_sandbox_active_reads_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("CURSOR_AGENT", raising=False)
    monkeypatch.delenv("__CURSOR_SANDBOX_ENV_RESTORE", raising=False)
    assert cursor_agent_sandbox_active() is False
    monkeypatch.setenv("CURSOR_AGENT", "1")
    assert cursor_agent_sandbox_active() is True


def test_start_admin_job_uses_osascript_escape_on_darwin(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    fake = MagicMock(returncode=0, stdout="4242\n", stderr="")

    with (
        patch(
            "happy_housing.app.admin_jobs.cursor_agent_sandbox_active",
            return_value=True,
        ),
        patch("happy_housing.app.admin_jobs.sys.platform", "darwin"),
        patch(
            "happy_housing.app.admin_jobs.subprocess.run",
            return_value=fake,
        ) as run_mock,
        patch("happy_housing.app.admin_jobs.subprocess.Popen") as popen_mock,
    ):
        job = start_admin_job(
            name="pipe",
            command=[sys.executable, "-c", "print(1)"],
            processed_dir=root,
        )
    assert job.pid == 4242
    popen_mock.assert_not_called()
    assert run_mock.called
    argv = run_mock.call_args.args[0]
    assert argv[0] == "osascript"
    assert job.error is None
