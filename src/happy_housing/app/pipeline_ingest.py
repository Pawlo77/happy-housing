"""Resolve Pipeline ingest limits / URLs / flags from Admin vs YAML."""

from dataclasses import dataclass

from happy_housing.app.app_settings import AppSettings
from happy_housing.config import ExperimentConfig, IngestLimitsConfig, IngestUrlsConfig


@dataclass(frozen=True, slots=True)
class IngestFlags:
    """Which remote sources to refresh during a pipeline run."""

    download_gtfs: bool
    """Refresh GTFS stops zip."""
    download_gios: bool
    """Refresh GIOŚ stations (+ AQ enrichment)."""
    download_rcn: bool
    """Refresh RCN GML zip + extract."""
    download_gunb: bool
    """Refresh GUNB Mazowieckie dump."""
    download_crime: bool
    """Refresh GUS Panorama dzielnic crime rates."""
    use_osm: bool
    """Fetch OSM amenities + shops (else fixtures)."""
    scrape_otodom: bool
    """Scrape Otodom inside the pipeline."""


def resolve_ingest_limits(
    settings: AppSettings,
    config: ExperimentConfig,
) -> IngestLimitsConfig:
    """Prefer Admin Pipeline overrides when enabled; else YAML ingest."""
    if not settings.pipeline_overrides_enabled:
        return config.ingest
    return IngestLimitsConfig(
        otodom_max_pages=settings.otodom_max_pages,
        otodom_max_listings=settings.otodom_max_listings,
        otodom_incremental=bool(settings.otodom_incremental),
        rcn_max_rows=settings.rcn_max_rows,
    )


def resolve_ingest_urls(
    settings: AppSettings,
    config: ExperimentConfig,
) -> IngestUrlsConfig:
    """Prefer non-empty Admin URL overrides when enabled; else YAML."""
    yaml_urls = config.ingest_urls
    if not settings.pipeline_overrides_enabled:
        return yaml_urls

    def _url(override: str, fallback: str) -> str:
        text = str(override or "").strip()
        return text if text else fallback

    return IngestUrlsConfig(
        gtfs_zip=_url(settings.gtfs_zip_url, yaml_urls.gtfs_zip),
        gios_stations=_url(settings.gios_stations_url, yaml_urls.gios_stations),
        rcn_zip=_url(settings.rcn_zip_url, yaml_urls.rcn_zip),
        gunb_zip=_url(settings.gunb_zip_url, yaml_urls.gunb_zip),
        crime_panorama_7z=_url(
            settings.crime_panorama_7z_url,
            yaml_urls.crime_panorama_7z,
        ),
    )


def resolve_ingest_flags(
    settings: AppSettings,
    config: ExperimentConfig,
    *,
    download_remote: bool | None = None,
    use_osm: bool | None = None,
    scrape_otodom: bool | None = None,
) -> IngestFlags:
    """Resolve per-source download flags.

    ``download_remote`` True/False forces all remote sources on/off (CLI
    back-compat). ``None`` uses Admin overrides when enabled, else all on.
    Explicit ``use_osm`` / ``scrape_otodom`` kwargs always win when not None.
    """
    del config  # reserved for future YAML defaults
    if download_remote is True:
        base_gtfs = base_gios = base_rcn = base_gunb = base_crime = True
    elif download_remote is False:
        base_gtfs = base_gios = base_rcn = base_gunb = base_crime = False
    elif settings.pipeline_overrides_enabled:
        base_gtfs = bool(settings.download_gtfs)
        base_gios = bool(settings.download_gios)
        base_rcn = bool(settings.download_rcn)
        base_gunb = bool(settings.download_gunb)
        base_crime = bool(settings.download_crime)
    else:
        base_gtfs = base_gios = base_rcn = base_gunb = base_crime = True

    if use_osm is not None:
        osm = bool(use_osm)
    elif settings.pipeline_overrides_enabled:
        osm = bool(settings.use_osm)
    else:
        osm = True

    if scrape_otodom is not None:
        scrape = bool(scrape_otodom)
    elif settings.pipeline_overrides_enabled:
        scrape = bool(settings.scrape_otodom_in_pipeline)
    else:
        scrape = True

    return IngestFlags(
        download_gtfs=base_gtfs,
        download_gios=base_gios,
        download_rcn=base_rcn,
        download_gunb=base_gunb,
        download_crime=base_crime,
        use_osm=osm,
        scrape_otodom=scrape,
    )
