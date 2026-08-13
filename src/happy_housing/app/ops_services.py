"""Preflight checks for Docker / OSRM / OTP used by Admin Ops."""

import logging
import shutil
import subprocess  # nosec B404 — fixed `docker info` probe, no shell
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_S: float = 3.0
"""Short timeout for Ops health probes."""


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


def check_docker_engine() -> ServiceCheck:
    """Probe local Docker CLI (info → version → ps fallbacks)."""
    if shutil.which("docker") is None:
        return ServiceCheck(
            key="docker",
            label="Docker engine",
            ok=False,
            detail="docker binary not found on PATH",
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
                    label = "ps ok"
                    detail = label
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
            if snippet:
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


def check_osrm(base_url: str) -> ServiceCheck:
    """HTTP probe OSRM with a tiny Warsaw route request."""
    root = str(base_url or "").rstrip("/")
    url = f"{root}/route/v1/driving/21.012,52.230;21.020,52.235"
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
            key="osrm",
            label="OSRM routing",
            ok=ok,
            detail=detail if ok else f"{detail} — start docker/docker-compose.osrm.yml",
            latency_ms=ms,
            required_for=("pipeline", "commute"),
        )
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000.0
        logger.debug("OSRM health failed: %s", exc)
        return ServiceCheck(
            key="osrm",
            label="OSRM routing",
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
) -> list[ServiceCheck]:
    """Docker + OSRM + OTP checks for the Ops panel."""
    return [
        check_docker_engine(),
        check_osrm(osrm_url),
        check_otp(otp_url),
    ]
