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
