"""Tests for pipeline CLI force flag parsing."""

from happy_housing.pipeline import parse_pipeline_cli_args, resolve_pipeline_force_flags


def test_parse_pipeline_cli_force_flags() -> None:
    assert parse_pipeline_cli_args([]).force_pipeline is False
    assert parse_pipeline_cli_args(["--force"]).force_pipeline is True
    assert parse_pipeline_cli_args(["--force-pipeline"]).force_pipeline is True
    assert parse_pipeline_cli_args(["--force-gis"]).force_gis_refresh is True


def test_resolve_pipeline_force_flags_decouple_steps_from_gis() -> None:
    """Full recompute must not imply GUNB/OSM re-download."""
    steps, remote = resolve_pipeline_force_flags(
        force_pipeline=True,
        settings_force=False,
        force_gis_refresh=False,
    )
    assert steps is True
    assert remote is False

    steps, remote = resolve_pipeline_force_flags(
        force_pipeline=None,
        settings_force=True,
        force_gis_refresh=False,
    )
    assert steps is True
    assert remote is False

    steps, remote = resolve_pipeline_force_flags(
        force_pipeline=False,
        settings_force=True,
        force_gis_refresh=False,
    )
    assert steps is False
    assert remote is False

    steps, remote = resolve_pipeline_force_flags(
        force_pipeline=False,
        settings_force=False,
        force_gis_refresh=True,
    )
    assert steps is False
    assert remote is True

    steps, remote = resolve_pipeline_force_flags(
        force_pipeline=None,
        settings_force=False,
        force_gis_refresh=False,
    )
    assert steps is False
    assert remote is False
