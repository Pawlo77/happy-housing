"""Tests for Admin Pipeline ingest resolve (app_settings vs YAML)."""

from __future__ import annotations

from happy_housing.app.app_settings import AppSettings
from happy_housing.app.pipeline_ingest import (
    resolve_ingest_flags,
    resolve_ingest_limits,
    resolve_ingest_urls,
)
from happy_housing.config import (
    ExperimentConfig,
    IngestLimitsConfig,
    IngestUrlsConfig,
)


def _config(
    *,
    pages: int | None = 50,
    listings: int | None = 1200,
    rcn: int | None = 5000,
    incremental: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        ingest=IngestLimitsConfig(
            otodom_max_pages=pages,
            otodom_max_listings=listings,
            otodom_incremental=incremental,
            rcn_max_rows=rcn,
        ),
        ingest_urls=IngestUrlsConfig(
            gtfs_zip="https://yaml.example/gtfs.zip",
            gios_stations="https://yaml.example/gios",
            rcn_zip="https://yaml.example/rcn.zip",
            gunb_zip="https://yaml.example/gunb.zip",
            crime_panorama_7z="https://yaml.example/crime.7z",
        ),
    )


def test_resolve_limits_uses_yaml_when_overrides_disabled() -> None:
    settings = AppSettings(
        pipeline_overrides_enabled=False,
        otodom_max_pages=3,
        otodom_max_listings=10,
        rcn_max_rows=99,
        otodom_incremental=True,
    )
    limits = resolve_ingest_limits(settings, _config())
    assert limits.otodom_max_pages == 50
    assert limits.otodom_max_listings == 1200
    assert limits.rcn_max_rows == 5000
    assert limits.otodom_incremental is False


def test_resolve_limits_prefers_app_settings_when_enabled() -> None:
    settings = AppSettings(
        pipeline_overrides_enabled=True,
        otodom_max_pages=None,
        otodom_max_listings=100,
        rcn_max_rows=None,
        otodom_incremental=True,
    )
    limits = resolve_ingest_limits(settings, _config())
    assert limits.otodom_max_pages is None
    assert limits.otodom_max_listings == 100
    assert limits.rcn_max_rows is None
    assert limits.otodom_incremental is True


def test_resolve_urls_empty_override_falls_back_to_yaml() -> None:
    settings = AppSettings(
        pipeline_overrides_enabled=True,
        gtfs_zip_url="",
        gios_stations_url="https://admin.example/gios",
        rcn_zip_url="  ",
        gunb_zip_url="https://admin.example/gunb.zip",
        crime_panorama_7z_url="",
    )
    urls = resolve_ingest_urls(settings, _config())
    assert urls.gtfs_zip == "https://yaml.example/gtfs.zip"
    assert urls.gios_stations == "https://admin.example/gios"
    assert urls.rcn_zip == "https://yaml.example/rcn.zip"
    assert urls.gunb_zip == "https://admin.example/gunb.zip"
    assert urls.crime_panorama_7z == "https://yaml.example/crime.7z"


def test_resolve_flags_defaults_and_overrides() -> None:
    yaml_cfg = _config()
    off = resolve_ingest_flags(
        AppSettings(pipeline_overrides_enabled=False),
        yaml_cfg,
        download_remote=True,
    )
    assert off.download_gtfs is True
    assert off.download_crime is True
    assert off.use_osm is True
    assert off.scrape_otodom is True

    custom = resolve_ingest_flags(
        AppSettings(
            pipeline_overrides_enabled=True,
            download_gtfs=False,
            download_gios=True,
            download_rcn=False,
            download_gunb=True,
            download_crime=False,
            use_osm=False,
            scrape_otodom_in_pipeline=False,
        ),
        yaml_cfg,
        download_remote=None,
    )
    assert custom.download_gtfs is False
    assert custom.download_gios is True
    assert custom.download_rcn is False
    assert custom.download_crime is False
    assert custom.use_osm is False
    assert custom.scrape_otodom is False

    shorthand = resolve_ingest_flags(
        AppSettings(pipeline_overrides_enabled=True, download_gtfs=False),
        yaml_cfg,
        download_remote=False,
    )
    assert shorthand.download_gtfs is False
    assert shorthand.download_gios is False
    assert shorthand.download_rcn is False
    assert shorthand.download_gunb is False
    assert shorthand.download_crime is False
