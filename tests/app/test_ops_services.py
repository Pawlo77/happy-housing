"""Tests for Ops routing / Docker service health checks."""

from unittest.mock import MagicMock, patch

from happy_housing.app.ops_services import (
    ServiceCheck,
    check_docker_engine,
    check_osrm,
    check_otp,
    check_pipeline_services,
)


def test_check_osrm_ok() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"code": "Ok"}
    with patch("happy_housing.app.ops_services.requests.get", return_value=resp):
        result = check_osrm("http://127.0.0.1:5001")
    assert result.ok is True
    assert result.key == "osrm"
    assert result.latency_ms is not None


def test_check_osrm_down() -> None:
    with patch(
        "happy_housing.app.ops_services.requests.get",
        side_effect=ConnectionError("refused"),
    ):
        result = check_osrm("http://127.0.0.1:5001")
    assert result.ok is False
    assert "refused" in result.detail.lower() or "fail" in result.detail.lower()


def test_check_otp_ok() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "OTP"
    with patch("happy_housing.app.ops_services.requests.get", return_value=resp):
        result = check_otp("http://127.0.0.1:8080")
    assert result.ok is True
    assert result.key == "otp"


def test_check_pipeline_services_orders() -> None:
    fake = ServiceCheck(
        key="osrm",
        label="OSRM",
        ok=True,
        detail="ok",
        latency_ms=1.0,
        required_for=("pipeline", "commute"),
    )
    with (
        patch("happy_housing.app.ops_services.check_osrm", return_value=fake),
        patch(
            "happy_housing.app.ops_services.check_otp",
            return_value=ServiceCheck(
                key="otp",
                label="OTP",
                ok=False,
                detail="down",
                latency_ms=None,
                required_for=("pipeline",),
            ),
        ),
        patch(
            "happy_housing.app.ops_services.check_docker_engine",
            return_value=ServiceCheck(
                key="docker",
                label="Docker",
                ok=True,
                detail="ok",
                latency_ms=None,
                required_for=("pipeline",),
            ),
        ),
    ):
        rows = check_pipeline_services(
            osrm_url="http://127.0.0.1:5001",
            otp_url="http://127.0.0.1:8080",
        )
    assert [r.key for r in rows] == ["docker", "osrm", "otp"]


def test_check_docker_engine_missing_binary() -> None:
    with patch(
        "happy_housing.app.ops_services.shutil.which",
        return_value=None,
    ):
        result = check_docker_engine()
    assert result.ok is False
    assert "not found" in result.detail.lower() or "missing" in result.detail.lower()


def test_check_docker_engine_empty_stderr_falls_back_to_version() -> None:
    """Regression: empty stderr must not collapse to bare ``error``."""
    failed = MagicMock(returncode=1, stdout="", stderr="")
    ok_version = MagicMock(returncode=0, stdout="29.6.2\n", stderr="")

    def _run(argv: list[str], **_kwargs: object) -> MagicMock:
        joined = " ".join(argv)
        if "version" in joined:
            return ok_version
        return failed

    with (
        patch(
            "happy_housing.app.ops_services.shutil.which",
            return_value="/bin/docker",
        ),
        patch("happy_housing.app.ops_services.subprocess.run", side_effect=_run),
    ):
        result = check_docker_engine()
    assert result.ok is True
    assert "29.6.2" in result.detail


def test_check_docker_engine_all_probes_fail_message() -> None:
    failed = MagicMock(returncode=1, stdout="", stderr="")
    with (
        patch(
            "happy_housing.app.ops_services.shutil.which",
            return_value="/bin/docker",
        ),
        patch("happy_housing.app.ops_services.subprocess.run", return_value=failed),
    ):
        result = check_docker_engine()
    assert result.ok is False
    assert result.detail.strip().lower() != "error"
    assert "rc=" in result.detail
    assert "Docker Desktop" in result.detail or "docker context" in result.detail
