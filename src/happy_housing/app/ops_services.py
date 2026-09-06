"""Preflight checks for Docker / OSRM / OTP used by Admin Ops."""

import http.client
import json
import logging
import os
import shutil
import socket
import subprocess  # nosec B404 — fixed `docker info` probe, no shell
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_S: float = 3.0
"""Short timeout for Ops health probes."""

_DOCKER_SOCK_TIMEOUT_S: float = 3.0
"""Unix-socket timeout for Docker Engine API."""


@dataclass(frozen=True, slots=True)
class ServiceCheck:
    """One Ops service health row."""

    key: str
    """Stable id (``docker``, ``osrm``, ``otp``)."""
    label: str
    """UI label."""
    ok: bool
    """Reachable / healthy."""
    detail: str
    """Short status text."""
    latency_ms: float | None
    """Round-trip ms when measured."""
    required_for: tuple[str, ...]
    """Which Admin actions need this service."""


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a Unix domain socket (Docker Engine API)."""

    def __init__(self, sock_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._sock_path = sock_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._sock_path)
        self.sock = sock


def _docker_socket_candidates() -> list[Path]:
    """Ordered Docker Engine socket paths (Desktop + Linux defaults)."""
    paths: list[Path] = []
    host = (os.environ.get("DOCKER_HOST") or "").strip()
    if host.startswith("unix://"):
        paths.append(Path(host.removeprefix("unix://")))
    paths.extend((
        Path.home() / ".docker" / "run" / "docker.sock",
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
    ))
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _probe_docker_engine_socket() -> tuple[str, float] | None:
    """Return ``(version, latency_ms)`` if Engine API answers on a local sock.

    Prefer the socket over the ``docker`` CLI: CLI can SIGSEGV / deny under
    sandboxed Streamlit parents while the daemon (and HTTP-published
    containers) stay healthy.
    """
    t0 = time.perf_counter()
    for sock_path in _docker_socket_candidates():
        if not sock_path.exists():
            continue
        try:
            conn = _UnixHTTPConnection(str(sock_path), timeout=_DOCKER_SOCK_TIMEOUT_S)
            try:
                conn.request("GET", "/version")
                response = conn.getresponse()
                body = response.read()
            finally:
                conn.close()
            if response.status != 200:
                continue
            payload = json.loads(body.decode("utf-8", errors="replace") or "{}")
            version = str(payload.get("Version") or "").strip() or "ok"
            ms = (time.perf_counter() - t0) * 1000.0
            return version, ms
        except (OSError, TimeoutError, json.JSONDecodeError, http.client.HTTPException):
            logger.debug("Docker sock probe failed for %s", sock_path, exc_info=True)
            continue
    return None


def check_docker_engine() -> ServiceCheck:
    """Probe Docker Engine (unix socket first, then CLI fallbacks)."""
    sock = _probe_docker_engine_socket()
    if sock is not None:
        version, ms = sock
        return ServiceCheck(
            key="docker",
            label="Docker engine",
            ok=True,
            detail=f"server {version}",
            latency_ms=ms,
            required_for=("pipeline", "osrm", "otp"),
        )

    if shutil.which("docker") is None:
        return ServiceCheck(
            key="docker",
            label="Docker engine",
            ok=False,
            detail="docker sock unreachable and docker binary not on PATH",
            latency_ms=None,
            required_for=("pipeline", "osrm", "otp"),
        )
    probes: tuple[tuple[str, ...], ...] = (
        ("docker", "info", "--format", "{{.ServerVersion}}"),
        ("docker", "version", "--format", "{{.Server.Version}}"),
        ("docker", "ps", "-q"),
    )
    t0 = time.perf_counter()
    last_detail = "docker probe failed"
    try:
        for argv in probes:
            try:
                completed = subprocess.run(  # nosec B603 B607 — fixed argv, no shell
                    list(argv),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                )
            except subprocess.TimeoutExpired:
                last_detail = (
                    "docker probe timed out (8s) — is Docker Desktop starting?"
                )
                continue
            except OSError as exc:
                last_detail = f"docker spawn failed: {exc}"
                continue
            out = (completed.stdout or "").strip()
            err = (completed.stderr or "").strip()
            if completed.returncode == 0:
                # ``ps -q`` may be empty when no containers; still means daemon up.
                if argv[1] == "ps":
                    detail = "ps ok"
                else:
                    label = out.splitlines()[0] if out else "ok"
                    detail = f"server {label}"
                ms = (time.perf_counter() - t0) * 1000.0
                return ServiceCheck(
                    key="docker",
                    label="Docker engine",
                    ok=True,
                    detail=detail,
                    latency_ms=ms,
                    required_for=("pipeline", "osrm", "otp"),
                )
            snippet = (err or out).strip()
            if completed.returncode < 0:
                last_detail = (
                    f"docker {' '.join(argv[1:])} killed by signal "
                    f"{-completed.returncode} — Engine sock also unreachable"
                )
            elif snippet:
                last_detail = snippet[:160]
            else:
                last_detail = (
                    f"docker {' '.join(argv[1:])} exit rc={completed.returncode} "
                    "(no output) — start Docker Desktop / check `docker context`"
                )
    except Exception as exc:
        last_detail = f"docker probe error: {exc}"
    ms = (time.perf_counter() - t0) * 1000.0
    return ServiceCheck(
        key="docker",
        label="Docker engine",
        ok=False,
        detail=last_detail[:200],
        latency_ms=ms,
        required_for=("pipeline", "osrm", "otp"),
    )


def check_osrm(
    base_url: str,
    *,
    key: str = "osrm",
    label: str = "OSRM car",
    profile: str = "driving",
) -> ServiceCheck:
    """HTTP probe OSRM with a tiny Warsaw route request."""
    root = str(base_url or "").rstrip("/")
    profile_key = str(profile or "driving").strip() or "driving"
    url = f"{root}/route/v1/{profile_key}/21.012,52.230;21.020,52.235"
    params: dict[str, Any] = {"overview": "false"}
    t0 = time.perf_counter()
    try:
        response = requests.get(url, params=params, timeout=_HTTP_TIMEOUT_S)
        ms = (time.perf_counter() - t0) * 1000.0
        payload = response.json() if response.ok else {}
        code = str(payload.get("code") or "")
        ok = response.ok and code.upper() in {"", "OK"}
        detail = f"HTTP {response.status_code}" + (f" code={code}" if code else "")
        return ServiceCheck(
            key=key,
            label=label,
            ok=ok,
            detail=(
                detail
                if ok
                else f"{detail} — make osrm-up (car :5001 / foot :5002 / bike :5003)"
            ),
            latency_ms=ms,
            required_for=("pipeline", "commute"),
        )
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000.0
        logger.debug("OSRM health failed (%s): %s", key, exc)
        return ServiceCheck(
            key=key,
            label=label,
            ok=False,
            detail=f"{exc} — need container on {root}",
            latency_ms=ms,
            required_for=("pipeline", "commute"),
        )


def check_otp(base_url: str) -> ServiceCheck:
    """HTTP probe OTP serve port (optional transit enrich)."""
    root = str(base_url or "").rstrip("/")
    # OTP 2 may answer on / or /otp; try a cheap GET.
    candidates = (f"{root}/otp", f"{root}/", f"{root}/otp/actuators/health")
    t0 = time.perf_counter()
    last_err = "unreachable"
    for url in candidates:
        try:
            response = requests.get(url, timeout=_HTTP_TIMEOUT_S)
            ms = (time.perf_counter() - t0) * 1000.0
            if response.status_code < 500:
                return ServiceCheck(
                    key="otp",
                    label="OTP transit",
                    ok=True,
                    detail=f"HTTP {response.status_code} via {url}",
                    latency_ms=ms,
                    required_for=("pipeline",),
                )
            last_err = f"HTTP {response.status_code}"
        except Exception as exc:
            last_err = str(exc)
            continue
    ms = (time.perf_counter() - t0) * 1000.0
    return ServiceCheck(
        key="otp",
        label="OTP transit",
        ok=False,
        detail=f"{last_err} — optional; compose profile otp",
        latency_ms=ms,
        required_for=("pipeline",),
    )


def check_pipeline_services(
    osrm_url: str,
    otp_url: str,
    osrm_foot_url: str | None = None,
    osrm_bike_url: str | None = None,
) -> list[ServiceCheck]:
    """Docker + OSRM car/foot/bike + OTP checks for the Ops panel."""
    foot = str(osrm_foot_url or "http://127.0.0.1:5002")
    bike = str(osrm_bike_url or "http://127.0.0.1:5003")
    return [
        check_docker_engine(),
        check_osrm(osrm_url, key="osrm", label="OSRM car", profile="driving"),
        check_osrm(foot, key="osrm_foot", label="OSRM foot", profile="foot"),
        check_osrm(bike, key="osrm_bike", label="OSRM bike", profile="bike"),
        check_otp(otp_url),
    ]
