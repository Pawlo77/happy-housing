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

    def _osrm_side_effect(
        url: str,
        *,
        key: str = "osrm",
        label: str = "OSRM car",
        profile: str = "driving",
    ) -> ServiceCheck:
        del url, label, profile
        return ServiceCheck(
            key=key,
            label=key,
            ok=True,
            detail="ok",
            latency_ms=1.0,
            required_for=("pipeline", "commute"),
        )

    with (
        patch(
            "happy_housing.app.ops_services.check_osrm",
            side_effect=_osrm_side_effect,
        ),
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
    assert [r.key for r in rows] == [
        "docker",
        "osrm",
        "osrm_foot",
        "osrm_bike",
        "otp",
    ]
    del fake


def test_check_docker_engine_missing_binary() -> None:
    with (
        patch(
            "happy_housing.app.ops_services.shutil.which",
            return_value=None,
        ),
        patch(
            "happy_housing.app.ops_services._probe_docker_engine_socket",
            return_value=None,
        ),
    ):
        result = check_docker_engine()
    assert result.ok is False
    assert "not" in result.detail.lower() or "unreachable" in result.detail.lower()


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
        patch(
            "happy_housing.app.ops_services._probe_docker_engine_socket",
            return_value=None,
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
        patch(
            "happy_housing.app.ops_services._probe_docker_engine_socket",
            return_value=None,
        ),
        patch("happy_housing.app.ops_services.subprocess.run", return_value=failed),
    ):
        result = check_docker_engine()
    assert result.ok is False
    assert result.detail.strip().lower() != "error"
    assert "rc=" in result.detail
    assert "Docker Desktop" in result.detail or "docker context" in result.detail


def test_check_docker_engine_socket_ok_when_cli_segfaults() -> None:
    """CLI may SIGSEGV under sandboxed parents; Engine API via sock still works."""
    crashed = MagicMock(returncode=-11, stdout="", stderr="")
    with (
        patch(
            "happy_housing.app.ops_services.shutil.which",
            return_value="/bin/docker",
        ),
        patch(
            "happy_housing.app.ops_services._probe_docker_engine_socket",
            return_value=("29.6.2", 12.5),
        ),
        patch("happy_housing.app.ops_services.subprocess.run", return_value=crashed),
    ):
        result = check_docker_engine()
    assert result.ok is True
    assert "29.6.2" in result.detail
    assert result.latency_ms == 12.5


def test_check_docker_engine_prefers_socket_over_cli() -> None:
    with (
        patch(
            "happy_housing.app.ops_services._probe_docker_engine_socket",
            return_value=("28.0.0", 3.0),
        ),
        patch("happy_housing.app.ops_services.subprocess.run") as run_mock,
    ):
        result = check_docker_engine()
    assert result.ok is True
    assert "28.0.0" in result.detail
    run_mock.assert_not_called()
