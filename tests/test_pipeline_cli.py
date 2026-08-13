"""Tests for pipeline CLI force flag parsing."""

from happy_housing.pipeline import parse_pipeline_cli_args


def test_parse_pipeline_cli_force_flags() -> None:
    assert parse_pipeline_cli_args([]).force_pipeline is False
    assert parse_pipeline_cli_args(["--force"]).force_pipeline is True
    assert parse_pipeline_cli_args(["--force-pipeline"]).force_pipeline is True
    assert parse_pipeline_cli_args(["--force-gis"]).force_gis_refresh is True
