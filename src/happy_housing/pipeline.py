"""End-to-end pipeline: ingest → GIS enrich → train → score → recommend.

Expensive steps (RCN extract, enrich, train/score, recommender) skip when
input fingerprints match ``pipeline_steps.json`` unless force refresh is set.
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from happy_housing.app.app_settings import (
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.app.attr_extract import enrich_frame_listing_attrs
from happy_housing.app.commute_target import (
    mark_commute_features_applied,
    resolve_commute_target,
)
from happy_housing.app.detail_backfill_jobs import maybe_schedule_detail_backfill
from happy_housing.app.listing_attrs import promote_listing_flags
from happy_housing.app.llm_metrics import llm_metrics_path
from happy_housing.app.metadata import (
    DEFAULT_SOURCES,
    RunMetadata,
    write_run_metadata,
)
from happy_housing.app.methodology_docs import format_methodology_note
from happy_housing.app.pipeline_ingest import (
    resolve_ingest_flags,
    resolve_ingest_limits,
    resolve_ingest_urls,
)
from happy_housing.app.preferences import load_preferences, preferences_path
from happy_housing.app.recommender import (
    impressions_path,
    retrain_recommender,
)
from happy_housing.app.recommender import (
    metrics_path as recommender_metrics_path,
)
from happy_housing.config import load_experiment_config
from happy_housing.gis import (
    enrich_listings,
    fetch_osm_amenities,
    fetch_osm_shops,
    load_amenity_points,
    load_shop_points,
)
from happy_housing.gis.amenity_catalog import coerce_amenity_weights
from happy_housing.gis.bike_paths import fetch_bike_paths, load_bike_path_sample
from happy_housing.gis.gios_aq_field import (
    attach_gios_aq_field_features,
    fetch_gios_aq_field_rasters,
)
from happy_housing.gis.location_fix import snap_listings_out_of_parks
from happy_housing.gis.routing import attach_active_mode_commutes
from happy_housing.gis.street_frontage import attach_street_frontage_hints
from happy_housing.gis.traffic import fetch_traffic_road_sample
from happy_housing.gis.warsaw_noise import (
    attach_warsaw_noise_features,
    fetch_warsaw_noise_rasters,
)
from happy_housing.ingest import (
    build_gtfs_stop_hourly,
    download_file,
    download_gtfs_stops,
    fetch_gios_stations,
    load_listings_json,
)
from happy_housing.ingest.crime import (
    attach_crime_features,
    download_panorama_crime,
    load_crime_by_district,
)
from happy_housing.ingest.fx import apply_fx_to_listings_frame, fx_cache_path
from happy_housing.ingest.gios_aq import (
    AIR_QUALITY_COLUMNS,
    enrich_stations_with_air_quality,
)
from happy_housing.ingest.gunb import download_gunb_mazowieckie, gunb_points_gdf
from happy_housing.ingest.listing_common import (
    LISTING_ACTIVE_COLUMN,
    listing_is_active,
    prepare_listings_for_model,
)
from happy_housing.ingest.listing_dedup import dedupe_cross_source_listings
from happy_housing.ingest.market import ensure_market_primary
from happy_housing.ingest.nol import scrape_nol_warsaw
from happy_housing.ingest.olx import scrape_olx_warsaw
from happy_housing.ingest.otodom import scrape_otodom_warsaw
from happy_housing.ingest.proxy_pool import configure_http_proxies
from happy_housing.ingest.rcn import extract_rcn_transactions
from happy_housing.ingest.warsaw_map import (
    fetch_warsaw_city_pois,
    fetch_warsaw_parks,
    fetch_warsaw_schools,
)
from happy_housing.models.metrics import coerce_opt_metric
from happy_housing.models.neighborhood import attach_neighborhood_rcn_features
from happy_housing.models.train import (
    FEATURE_COLUMNS,
    TrainSettings,
    oof_predict_price_per_m2,
    predict_price_per_m2,
    run_training_with_mlflow,
)
from happy_housing.pipeline_cache import (
    StepManifest,
    file_token,
    fingerprint,
    payload_token,
    pipeline_steps_path,
)
from happy_housing.pipeline_progress import log_step
from happy_housing.storage import write_geoparquet
from happy_housing.vision.photo_features import attach_photo_features

logger = logging.getLogger(__name__)


def _listings_fingerprint(frame: pd.DataFrame) -> str:
    """Cheap content token for listing universe changes."""
    if frame.empty or "listing_id" not in frame.columns:
        return "empty"
    ids = frame["listing_id"].astype(str)
    if "price_pln" in frame.columns:
        price = pd.to_numeric(frame["price_pln"], errors="coerce").fillna(0.0)
    else:
        price = pd.Series(0.0, index=frame.index)
    token = int(pd.util.hash_pandas_object(ids).sum()) ^ round(float(price.sum()))
    return f"n={len(frame)};t={token}"


def _load_prior_run_metrics(
    path: Path,
) -> tuple[dict[str, float], list[str]]:
    """Read model metrics / feature list from a prior ``run_metadata.json``."""
    if not path.exists():
        return {}, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, []
    if not isinstance(payload, dict):
        return {}, []
    metrics_raw = payload.get("model_metrics") or {}
    metrics: dict[str, float] = {}
    if isinstance(metrics_raw, dict):
        for key, value in metrics_raw.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    features_raw = payload.get("feature_columns") or []
    features = [str(x) for x in features_raw] if isinstance(features_raw, list) else []
    return metrics, features


def _rcn_as_listings(rcn: pd.DataFrame) -> pd.DataFrame:
    """Map RCN transactions into the listing schema for joint training."""
    if rcn.empty:
        return pd.DataFrame()
    frame = rcn.dropna(subset=["lat", "lon", "price_pln"]).copy()
    frame = frame[frame["price_pln"] > 50_000]
    if "area_m2" not in frame.columns:
        frame["area_m2"] = pd.NA
    frame = frame[frame["area_m2"].notna() & (frame["area_m2"] > 15)]
    frame = frame[frame["area_m2"] < 250]
    frame["price_per_m2"] = frame["price_pln"] / frame["area_m2"]
    # Drop absurd zł/m² outliers.
    frame = frame[(frame["price_per_m2"] > 4_000) & (frame["price_per_m2"] < 50_000)]
    out = pd.DataFrame({
        "listing_id": [f"RCN-{i}" for i in range(len(frame))],
        "lat": frame["lat"].to_numpy(),
        "lon": frame["lon"].to_numpy(),
        "price_pln": frame["price_pln"].to_numpy(),
        "area_m2": frame["area_m2"].to_numpy(),
        "rooms": 2,
        "year_built": 1995,
        "district": frame
        .get("district", pd.Series(["rcn"] * len(frame)))
        .fillna("rcn")
        .astype(str)
        .str.casefold()
        .to_numpy(),
        "listed_at": frame
        .get("listed_at", pd.Series(["2020-01-01"] * len(frame)))
        .fillna("2020-01-01")
        .astype(str)
        .to_numpy(),
        "price_per_m2": frame["price_per_m2"].to_numpy(),
        "source": "rcn",
    })
    for col in (
        "kind",
        "rodzaj_rynku",
        "rodzaj_transakcji",
        "rodzaj_prawa",
        "adres_dzialki",
        "sposob_uzytkowania",
        "dodatkowe_informacje",
        "transaction_id",
    ):
        if col in frame.columns:
            out[col] = frame[col].to_numpy()
    out = ensure_market_primary(out)
    return out.reset_index(drop=True)


def _reload_fx_listings_parquet(path: Path, processed: Path) -> pd.DataFrame:
    """Re-read a listings parquet, refresh FX columns, and persist."""
    frame = apply_fx_to_listings_frame(
        pd.read_parquet(path),
        fx_cache_path(processed),
    )
    frame.to_parquet(path, index=False)
    return frame


def _geocoded_otodom(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with both coordinates (empty when none)."""
    if frame.empty or not {"lat", "lon"} <= set(frame.columns):
        return frame.iloc[0:0].copy()
    return frame.dropna(subset=["lat", "lon"])


def _inactive_otodom_asks(frame: pd.DataFrame) -> pd.DataFrame:
    """Closed Otodom search rows kept for historical ask-price context."""
    if frame.empty or LISTING_ACTIVE_COLUMN not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = ~frame[LISTING_ACTIVE_COLUMN].map(listing_is_active)
    return frame.loc[mask].copy()


def _split_listings_for_enrich(
    listings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows with coordinates from passthrough historical asks without pins."""
    if listings.empty or not {"lat", "lon"} <= set(listings.columns):
        return listings.copy(), listings.iloc[0:0].copy()
    has_coords = listings["lat"].notna() & listings["lon"].notna()
    return listings.loc[has_coords].copy(), listings.loc[~has_coords].copy()


def _ingest_buyer_source_into_parts(
    parts: list[pd.DataFrame],
    frame: pd.DataFrame,
    processed: Path,
    source: str,
    *,
    log_search_only_pending: bool = False,
) -> None:
    """Append geocoded + closed search-source rows and maybe queue detail backfill."""
    maybe_schedule_detail_backfill(processed)
    geocoded = _geocoded_otodom(frame)
    closed = _inactive_otodom_asks(frame)
    to_add = pd.concat([geocoded, closed], ignore_index=True, sort=False)
    if to_add.empty:
        if log_search_only_pending and not frame.empty:
            logger.info(
                "%s scrape produced no geocoded rows yet "
                "(search-only or detail backfill pending)",
                source,
            )
        return
    tagged = to_add.copy()
    tagged["source"] = source
    parts.append(tagged)
    if not closed.empty:
        logger.info(
            "Including %s closed %s asks for historical stats (map hidden)",
            len(closed),
            source,
        )


def _append_sourced_listings(
    parts: list[pd.DataFrame],
    frame: pd.DataFrame,
    source: str,
) -> None:
    """Tag and append a non-empty listings frame."""
    if frame.empty:
        return
    tagged = frame.copy()
    tagged["source"] = source
    parts.append(tagged)


def run_pipeline(
    config_path: Path | str = Path("configs/experiment.yaml"),
    *,
    download_remote: bool | None = None,
    use_osm: bool | None = None,
    scrape_otodom: bool | None = None,
    scrape_olx: bool | None = None,
    scrape_nol: bool | None = None,
    otodom_max_pages: int | None = None,
    otodom_max_listings: int | None = None,
    olx_max_listings: int | None = None,
    nol_max_listings: int | None = None,
    rcn_max_rows: int | None = None,
    force_gis_refresh: bool = False,
    force_pipeline: bool | None = None,
) -> dict[str, Path]:
    """Run the full local pipeline and return key output paths.

    ``download_remote`` True/False forces all remote sources on/off; ``None``
    uses Admin Pipeline overrides when saved, else downloads everything.
    Explicit listing / RCN kwargs override resolved caps.
    ``force_gis_refresh`` ignores OSM / Warsaw map caches and re-downloads.
    ``force_pipeline`` (or Admin ``pipeline_force_refresh``) ignores step
    fingerprints and re-exports GTFS/GIOŚ/RCN plus enrich/train/score/recommend.
    """
    from happy_housing.logging_config import configure_logging

    started = time.perf_counter()
    config = load_experiment_config(config_path)
    configure_logging(config.logging.level, force=False)
    processed = config.data.processed_path()
    app_settings = load_app_settings(settings_path(processed))
    configure_http_proxies(enabled=bool(app_settings.http_free_proxies_enabled))
    force = bool(
        force_pipeline
        if force_pipeline is not None
        else app_settings.pipeline_force_refresh
    ) or bool(force_gis_refresh)
    manifest = StepManifest.load(pipeline_steps_path(processed))
    skipped_steps: list[str] = []
    limits = resolve_ingest_limits(app_settings, config)
    urls = resolve_ingest_urls(app_settings, config)
    flags = resolve_ingest_flags(
        app_settings,
        config,
        download_remote=download_remote,
        use_osm=use_osm,
        scrape_otodom=scrape_otodom,
        scrape_olx=scrape_olx,
        scrape_nol=scrape_nol,
    )
    logger.info(
        "Pipeline start (gtfs=%s gios=%s rcn=%s gunb=%s crime=%s osm=%s "
        "otodom=%s olx=%s nol=%s force=%s)",
        flags.download_gtfs,
        flags.download_gios,
        flags.download_rcn,
        flags.download_gunb,
        flags.download_crime,
        flags.use_osm,
        flags.scrape_otodom,
        flags.scrape_olx,
        flags.scrape_nol,
        force,
    )
    page_cap = (
        otodom_max_pages if otodom_max_pages is not None else limits.otodom_max_pages
    )
    listing_cap = (
        otodom_max_listings
        if otodom_max_listings is not None
        else limits.otodom_max_listings
    )
    olx_listing_cap = (
        olx_max_listings if olx_max_listings is not None else limits.olx_max_listings
    )
    nol_listing_cap = (
        nol_max_listings if nol_max_listings is not None else limits.nol_max_listings
    )
    rcn_cap = rcn_max_rows if rcn_max_rows is not None else limits.rcn_max_rows
    raw = config.data.raw_path()
    fixtures = config.data.fixtures_path()
    cache = config.data.cache_path()
    for path in (raw, processed, cache):
        path.mkdir(parents=True, exist_ok=True)

    otodom_parquet = processed / "otodom_listings.parquet"
    listings_parts: list[pd.DataFrame] = []

    # Always keep fixtures as a tiny offline baseline.
    fixture_listings = load_listings_json(fixtures / "listings.json")
    fixture_listings["source"] = "fixture"
    listings_parts.append(fixture_listings)

    if flags.scrape_otodom:
        try:
            with log_step(logger, "otodom_scrape"):
                otodom = scrape_otodom_warsaw(
                    otodom_parquet,
                    max_pages=page_cap,
                    max_listings=listing_cap,
                    incremental=limits.otodom_incremental,
                    fetch_details=limits.otodom_fetch_details,
                )
            _ingest_buyer_source_into_parts(
                listings_parts,
                otodom,
                processed,
                "otodom",
                log_search_only_pending=True,
            )
        except Exception as exc:
            logger.warning("Otodom scrape failed: %s", exc)
            if otodom_parquet.exists():
                otodom = _reload_fx_listings_parquet(otodom_parquet, processed)
                _ingest_buyer_source_into_parts(
                    listings_parts, otodom, processed, "otodom"
                )
    elif otodom_parquet.exists():
        otodom = _reload_fx_listings_parquet(otodom_parquet, processed)
        _ingest_buyer_source_into_parts(listings_parts, otodom, processed, "otodom")

    olx_parquet = processed / "olx_listings.parquet"
    if flags.scrape_olx:
        try:
            with log_step(logger, "olx_scrape"):
                olx = scrape_olx_warsaw(
                    olx_parquet,
                    max_listings=olx_listing_cap,
                    incremental=limits.olx_incremental,
                    include_rent=limits.olx_include_rent,
                )
            _append_sourced_listings(listings_parts, olx, "olx")
        except Exception as exc:
            logger.warning("OLX scrape failed: %s", exc)
            if olx_parquet.exists():
                olx = _reload_fx_listings_parquet(olx_parquet, processed)
                _append_sourced_listings(listings_parts, olx, "olx")
    elif olx_parquet.exists():
        olx = _reload_fx_listings_parquet(olx_parquet, processed)
        _append_sourced_listings(listings_parts, olx, "olx")

    nol_parquet = processed / "nol_listings.parquet"
    if flags.scrape_nol:
        try:
            with log_step(logger, "nol_scrape"):
                nol = scrape_nol_warsaw(
                    nol_parquet,
                    max_pages=limits.nol_max_pages,
                    max_listings=nol_listing_cap,
                    incremental=limits.nol_incremental,
                    include_rent=limits.nol_include_rent,
                    fetch_details=limits.nol_fetch_details,
                )
            _ingest_buyer_source_into_parts(
                listings_parts,
                nol,
                processed,
                "nol",
                log_search_only_pending=True,
            )
        except Exception as exc:
            logger.warning("NOL scrape failed: %s", exc)
            if nol_parquet.exists():
                nol = _reload_fx_listings_parquet(nol_parquet, processed)
                _ingest_buyer_source_into_parts(listings_parts, nol, processed, "nol")
    elif nol_parquet.exists():
        nol = _reload_fx_listings_parquet(nol_parquet, processed)
        _ingest_buyer_source_into_parts(listings_parts, nol, processed, "nol")

    gtfs_parquet = processed / "gtfs_stops.parquet"
    gios_parquet = processed / "gios_stations.parquet"
    rcn_parquet = processed / "rcn_transactions.parquet"
    gunb_parquet = processed / "gunb_mazowieckie.parquet"
    crime_parquet = processed / "crime_by_district.parquet"

    def _ensure_empty(path: Path, cols: list[str]) -> None:
        if not path.exists():
            pd.DataFrame(columns=cols).to_parquet(path, index=False)

    if flags.download_gtfs:
        download_gtfs_stops(
            urls.gtfs_zip, raw, gtfs_parquet, force=force_gis_refresh or force
        )
    else:
        _ensure_empty(gtfs_parquet, ["stop_id", "stop_lat", "stop_lon"])

    if flags.download_gios:
        fetch_gios_stations(
            urls.gios_stations,
            gios_parquet,
            force=force_gis_refresh or force,
        )
    else:
        _ensure_empty(gios_parquet, ["station_id", "lat", "lon", "name", "city"])

    if flags.download_rcn:
        zip_path = download_file(
            urls.rcn_zip,
            raw / "rcn_warsaw.zip",
            timeout_s=1800.0,
            force=bool(force_gis_refresh or force),
        )
        rcn_fp = fingerprint(file_token(zip_path), f"max_rows={rcn_cap}")
        if manifest.should_skip(
            "extract_rcn", rcn_fp, [rcn_parquet], force=force_gis_refresh or force
        ):
            logger.info("Skip RCN extract (unchanged zip / max_rows)")
            skipped_steps.append("extract_rcn")
        else:
            try:
                extract_rcn_transactions(zip_path, rcn_parquet, max_rows=rcn_cap)
                manifest.mark_and_save("extract_rcn", rcn_fp)
            except Exception as exc:
                logger.warning("RCN extract failed: %s", exc)
                pd.DataFrame(columns=["lon", "lat", "price_pln", "area_m2"]).to_parquet(
                    rcn_parquet, index=False
                )
    else:
        _ensure_empty(rcn_parquet, ["lon", "lat", "price_pln", "area_m2"])

    if flags.download_gunb:
        download_gunb_mazowieckie(
            raw,
            gunb_parquet,
            url=urls.gunb_zip,
            fixture_path=fixtures / "gunb_points.json",
            refresh_wfs_cache=bool(force_gis_refresh or force),
            wfs_full_history=bool(app_settings.gunb_wfs_full_history),
        )
    else:
        _ensure_empty(gunb_parquet, ["lon", "lat"])

    if flags.download_crime:
        download_panorama_crime(
            urls.crime_panorama_7z,
            raw,
            crime_parquet,
            fixture_path=fixtures / "crime_by_district.json",
            force=force_gis_refresh,
        )
    else:
        load_crime_by_district(
            crime_parquet,
            fixture_path=fixtures / "crime_by_district.json",
        )
    crime_by_district = pd.read_parquet(crime_parquet)

    rcn = pd.read_parquet(rcn_parquet)
    rcn_listings = _rcn_as_listings(rcn)
    if not rcn_listings.empty:
        listings_parts.append(rcn_listings)

    listings = pd.concat(listings_parts, ignore_index=True, sort=False)
    listings = listings.drop_duplicates(subset=["listing_id"], keep="last")
    listings = dedupe_cross_source_listings(listings)
    listings = prepare_listings_for_model(listings)
    listings = ensure_market_primary(listings)
    logger.info(
        "Training/scoring universe size=%s sources=%s",
        len(listings),
        listings.get("source", pd.Series(dtype=str)).value_counts().to_dict(),
    )

    gtfs_stops = pd.read_parquet(gtfs_parquet)
    hourly_path = processed / "gtfs_stop_hourly.parquet"
    gtfs_zip = raw / "gtfs_warsaw.zip"
    if not hourly_path.exists() and gtfs_zip.exists():
        build_gtfs_stop_hourly(gtfs_zip, hourly_path)
    gios_stations = pd.read_parquet(gios_parquet)
    aq_missing = any(col not in gios_stations.columns for col in AIR_QUALITY_COLUMNS)
    if flags.download_gios or aq_missing:
        gios_stations = enrich_stations_with_air_quality(
            gios_stations,
            fixture_path=fixtures / "gios_air_quality.json",
        )
        gios_stations.to_parquet(gios_parquet, index=False)
    gunb = gunb_points_gdf(pd.read_parquet(gunb_parquet))

    if flags.use_osm:
        amenities = fetch_osm_amenities(
            config.bbox.as_tuple(),
            cache,
            fixture_path=fixtures / "amenities.json",
            force_refresh=force_gis_refresh,
        )
        shops = fetch_osm_shops(
            config.bbox.as_tuple(),
            cache,
            fixture_path=fixtures / "shops.json",
            force_refresh=force_gis_refresh,
        )
    else:
        amenities = load_amenity_points(fixtures / "amenities.json")
        shops_cache = cache / "osm_shops.geojson"
        if shops_cache.exists():
            shops = gpd.read_file(shops_cache)
        else:
            shops = load_shop_points(fixtures / "shops.json")

    # Independent HTTP / WFS fetches in parallel (not OSMnx — shared settings).
    logger.info("STEP start warsaw_gis")
    _gis_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_parks = pool.submit(
            fetch_warsaw_parks,
            cache,
            fixture_path=fixtures / "parks.geojson",
            force_refresh=force_gis_refresh,
            bbox=config.bbox.as_tuple(),
        )
        fut_schools = pool.submit(
            fetch_warsaw_schools,
            cache,
            fixture_path=fixtures / "schools.geojson",
            force_refresh=force_gis_refresh,
        )
        fut_pois = pool.submit(
            fetch_warsaw_city_pois,
            cache,
            fixture_path=fixtures / "city_pois.geojson",
            force_refresh=force_gis_refresh,
            bbox=config.bbox.as_tuple(),
        )
        parks = fut_parks.result()
        schools = fut_schools.result()
        city_pois = fut_pois.result()
    logger.info(
        "STEP done warsaw_gis in %.1fs (parks=%s schools=%s pois=%s)",
        time.perf_counter() - _gis_t0,
        len(parks),
        len(schools),
        len(city_pois),
    )  # OSMnx road / bike fetches stay serial (global ox.settings).
    traffic_sample = fetch_traffic_road_sample(
        cache,
        bbox=config.bbox.as_tuple(),
        fixture_path=fixtures / "traffic_sample.json",
        force_refresh=force_gis_refresh,
    )
    bike_paths = fetch_bike_paths(
        cache,
        bbox=config.bbox.as_tuple(),
        fixture_path=fixtures / "bike_paths.geojson",
        force_refresh=force_gis_refresh,
    )
    bike_path_sample = load_bike_path_sample(cache)

    def _zone_fixture(name: str) -> gpd.GeoDataFrame:
        path = fixtures / name
        if path.exists():
            return gpd.read_file(path)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    military = _zone_fixture("military_areas.geojson")
    rail_areas = _zone_fixture("rail_areas.geojson")
    airport_areas = _zone_fixture("airport_areas.geojson")

    commute = resolve_commute_target(app_settings, config)
    logger.info(
        "Commute target %s (%.5f, %.5f)", commute.label, commute.lon, commute.lat
    )

    enriched_path = processed / "listings_enriched.parquet"
    scored_path = processed / "listings_scored.parquet"
    model_path = processed / "lightgbm_price_per_m2.txt"
    metadata_path = processed / "run_metadata.json"

    enrich_settings = {
        "snap": bool(app_settings.snap_listings_out_of_parks),
        "bike_r": float(app_settings.bike_path_coverage_radius_m),
        "traffic_r": float(app_settings.traffic_neighbourhood_radius_m),
        "aq": bool(app_settings.aq_field_enabled),
        "aq_power": float(app_settings.aq_idw_power),
        "extract": bool(app_settings.extract_attrs_enabled),
        "extract_workers": int(app_settings.extract_lm_workers),
        "photo": True,
        "photo_n": int(app_settings.photo_cv_max_photos),
        "photo_model": str(app_settings.photo_cv_model_id),
        "nbhd": bool(app_settings.model_use_neighborhood),
        "nbhd_r": float(app_settings.model_nbhd_radius_m),
        "otp": bool(config.routing.use_otp_transit),
        "osrm": str(config.routing.osrm_url),
        "require_osrm": bool(config.routing.require_osrm),
        "n_features": len(FEATURE_COLUMNS),
    }
    enrich_fp = fingerprint(
        _listings_fingerprint(listings),
        file_token(otodom_parquet),
        file_token(rcn_parquet),
        file_token(gtfs_parquet),
        file_token(gios_parquet),
        file_token(gunb_parquet),
        file_token(crime_parquet),
        f"commute={commute.lon:.6f},{commute.lat:.6f}",
        payload_token(enrich_settings),
    )

    if manifest.should_skip("enrich", enrich_fp, [enriched_path], force=force):
        logger.info("Skip enrich (inputs unchanged)")
        skipped_steps.append("enrich")
        train_frame = pd.read_parquet(enriched_path)
        train_frame = train_frame.drop(columns=["geometry"], errors="ignore")
    else:
        logger.info("STEP start enrich")
        _enrich_t0 = time.perf_counter()
        to_enrich, passthrough = _split_listings_for_enrich(listings)
        if app_settings.snap_listings_out_of_parks:
            to_enrich = snap_listings_out_of_parks(
                to_enrich,
                parks=parks,
                cache_dir=cache,
                bbox=config.bbox.as_tuple(),
                force_refresh_buildings=force_gis_refresh,
            )

        to_enrich = attach_street_frontage_hints(to_enrich)

        road_lden_path = fixtures / "road_lden.geojson"
        rail_lden_path = fixtures / "rail_lden.geojson"
        if to_enrich.empty:
            enriched = gpd.GeoDataFrame(to_enrich, geometry=[], crs="EPSG:4326")
        else:
            enriched = enrich_listings(
                to_enrich,
                projected_crs=config.bbox.projected_crs,
                amenities=amenities,
                gios_stations=gios_stations,
                gtfs_stops=gtfs_stops,
                commute_lon=commute.lon,
                commute_lat=commute.lat,
                gunb_points=gunb,
                parks=parks,
                schools=schools,
                shops=shops,
                city_pois=city_pois,
                bike_paths=bike_paths,
                bike_path_coverage_radius_m=float(
                    app_settings.bike_path_coverage_radius_m
                ),
                bike_path_sample=bike_path_sample
                if not bike_path_sample.empty
                else None,
                rail_areas=rail_areas,
                military=military,
                airport_areas=airport_areas,
                traffic_sample=traffic_sample,
                traffic_neighbourhood_radius_m=float(
                    app_settings.traffic_neighbourhood_radius_m
                ),
                aircraft_lden_path=fixtures / "chopin_aircraft_lden.geojson",
                aircraft_lnight_path=fixtures / "chopin_aircraft_lnight.geojson",
                road_lden_path=road_lden_path if road_lden_path.exists() else None,
                rail_lden_path=rail_lden_path if rail_lden_path.exists() else None,
            )
        noise_rasters = fetch_warsaw_noise_rasters(
            cache,
            bbox=config.bbox.as_tuple(),
            force_refresh=force_gis_refresh,
        )
        if noise_rasters and not enriched.empty:
            enriched = attach_warsaw_noise_features(enriched, noise_rasters)
        if (
            app_settings.aq_field_enabled
            and not gios_stations.empty
            and not enriched.empty
        ):
            aq_rasters = fetch_gios_aq_field_rasters(
                cache,
                gios_stations,
                bbox_wgs84=config.bbox.as_tuple(),
                power=float(app_settings.aq_idw_power),
                force_refresh=force_gis_refresh,
            )
            if aq_rasters:
                enriched = attach_gios_aq_field_features(enriched, aq_rasters)
        train_frame = prepare_listings_for_model(
            pd.DataFrame(enriched.drop(columns=["geometry"], errors="ignore"))
        )
        if not passthrough.empty:
            train_frame = pd.concat(
                [
                    train_frame,
                    prepare_listings_for_model(passthrough.reset_index(drop=True)),
                ],
                ignore_index=True,
                sort=False,
            )
        train_frame = promote_listing_flags(train_frame)
        train_frame = attach_crime_features(train_frame, crime_by_district)
        if app_settings.extract_attrs_enabled:
            cap = int(app_settings.extract_max_listings) or None
            train_frame = enrich_frame_listing_attrs(
                train_frame,
                model=app_settings.extract_lm_model,
                max_chars=int(app_settings.extract_max_chars),
                max_tokens=int(app_settings.extract_max_tokens),
                use_llm=True,
                metrics_path=llm_metrics_path(processed),
                max_llm_calls=cap,
                lm_workers=int(app_settings.extract_lm_workers),
            )
        train_frame = attach_active_mode_commutes(
            train_frame,
            dest_lon=commute.lon,
            dest_lat=commute.lat,
            osrm_url=config.routing.osrm_url,
            require_osrm=bool(config.routing.require_osrm),
            use_otp_transit=bool(config.routing.use_otp_transit),
            otp_url=config.routing.otp_url,
        )
        train_frame = attach_photo_features(
            train_frame,
            max_photos=int(app_settings.photo_cv_max_photos),
            model_id=str(app_settings.photo_cv_model_id),
        )

        def _write_enriched_geoparquet(frame: pd.DataFrame) -> None:
            """Persist attribute frame with Hilbert-sorted GeoParquet geometry."""
            geo = enriched.copy()
            for col in frame.columns:
                geo[col] = frame[col].to_numpy()
            write_geoparquet(geo, enriched_path)

        _write_enriched_geoparquet(train_frame)

        rcn_for_model = (
            pd.read_parquet(rcn_parquet) if rcn_parquet.exists() else pd.DataFrame()
        )
        if app_settings.model_use_neighborhood and not rcn_for_model.empty:
            train_frame = attach_neighborhood_rcn_features(
                train_frame,
                rcn_for_model,
                radius_m=float(app_settings.model_nbhd_radius_m),
            )
            _write_enriched_geoparquet(train_frame)
        manifest.mark_and_save("enrich", enrich_fp)
        logger.info(
            "STEP done enrich in %.1fs (%.2f min)",
            time.perf_counter() - _enrich_t0,
            (time.perf_counter() - _enrich_t0) / 60.0,
        )

    rcn_for_model = (
        pd.read_parquet(rcn_parquet) if rcn_parquet.exists() else pd.DataFrame()
    )
    train_settings = TrainSettings(
        opt_metric=coerce_opt_metric(app_settings.model_opt_metric),
        val_size=float(app_settings.model_val_size),
        test_size=float(config.model.test_size),
        hpo_trials=int(app_settings.model_hpo_trials),
        use_neighborhood=bool(app_settings.model_use_neighborhood),
        use_engineered=bool(app_settings.model_use_engineered),
        random_state=int(config.model.random_state),
        train_buyer_asks_only=bool(app_settings.train_buyer_asks_only),
        log_target=bool(app_settings.model_log_target),
        district_calibration=bool(app_settings.model_district_calibration),
        early_stopping_rounds=int(app_settings.model_early_stopping_rounds),
    )
    train_score_fp = fingerprint(
        file_token(enriched_path),
        payload_token({
            "opt": train_settings.opt_metric,
            "val": train_settings.val_size,
            "test": train_settings.test_size,
            "hpo": train_settings.hpo_trials,
            "nbhd": train_settings.use_neighborhood,
            "eng": train_settings.use_engineered,
            "early": train_settings.early_stopping_rounds,
            "seed": train_settings.random_state,
            "buyer_asks": train_settings.train_buyer_asks_only,
            "log": train_settings.log_target,
            "district_cal": train_settings.district_calibration,
            "n_features": len(FEATURE_COLUMNS),
        }),
    )
    model_metrics: dict[str, float]
    feature_names: list[str]
    if manifest.should_skip(
        "train_score",
        train_score_fp,
        [model_path, scored_path],
        force=force,
    ):
        logger.info("Skip train+score (enriched + settings unchanged)")
        skipped_steps.append("train_score")
        train_frame = pd.read_parquet(scored_path)
        model_metrics, feature_names = _load_prior_run_metrics(metadata_path)
        if not feature_names:
            feature_names = list(FEATURE_COLUMNS)
    else:
        logger.info("STEP start train_score")
        _train_t0 = time.perf_counter()
        result = run_training_with_mlflow(
            train_frame,
            config,
            model_path,
            train_settings=train_settings,
            rcn=rcn_for_model if app_settings.model_use_neighborhood else None,
        )
        try:
            train_frame["model_price_per_m2"] = oof_predict_price_per_m2(
                train_frame,
                config.model,
                train_settings,
                n_splits=5,
                params=result.best_params or None,
            )
        except Exception as exc:
            logger.warning("OOF scoring failed (%s); falling back to in-sample", exc)
            train_frame["model_price_per_m2"] = predict_price_per_m2(
                result.model,
                train_frame,
                use_neighborhood=bool(app_settings.model_use_neighborhood),
                log_target=result.log_target,
                district_offsets=result.district_offsets,
                use_engineered=result.use_engineered,
                district_ppm_stats=result.district_ppm_stats,
                district_global_median=result.district_global_median,
                district_global_mean=result.district_global_mean,
            )
        train_frame["delta_price_per_m2"] = (
            train_frame["price_per_m2"] - train_frame["model_price_per_m2"]
        )
        train_frame.to_parquet(scored_path, index=False)
        model_metrics = dict(result.metrics)
        feature_names = list(result.feature_names)
        manifest.mark_and_save("train_score", train_score_fp)
        logger.info(
            "STEP done train_score in %.1fs (%.2f min)",
            time.perf_counter() - _train_t0,
            (time.perf_counter() - _train_t0) / 60.0,
        )

    prefs = load_preferences(preferences_path(processed))
    rec_settings = {
        "top_n": int(app_settings.recommender_top_n),
        "alpha": float(app_settings.recommender_demotion_alpha),
        "w_bargain": float(app_settings.recommender_w_bargain),
        "w_commute": float(app_settings.recommender_w_commute),
        "w_transit": float(app_settings.recommender_w_transit),
        "w_overprice": float(app_settings.recommender_w_overprice),
        "otodom_pool": bool(app_settings.recommender_otodom_pool_default),
        "amenity": coerce_amenity_weights(app_settings.recommender_amenity_weights),
        "prefs_enabled": bool(prefs.enabled),
    }
    rec_fp = fingerprint(
        file_token(scored_path),
        file_token(impressions_path(processed)),
        payload_token(rec_settings),
    )
    rec_metrics_file = recommender_metrics_path(processed)
    if manifest.should_skip("recommender", rec_fp, [rec_metrics_file], force=force):
        logger.info("Skip recommender (scored + settings unchanged)")
        skipped_steps.append("recommender")
    else:
        logger.info("STEP start recommender")
        _rec_t0 = time.perf_counter()
        if "model_price_per_m2" not in train_frame.columns:
            train_frame = pd.read_parquet(scored_path)
        retrain_recommender(
            train_frame,
            processed_dir=processed,
            top_n=int(app_settings.recommender_top_n),
            demotion_alpha=float(app_settings.recommender_demotion_alpha),
            prefs=prefs,
            amenity_weights=coerce_amenity_weights(
                app_settings.recommender_amenity_weights
            ),
            w_bargain=float(app_settings.recommender_w_bargain),
            w_commute=float(app_settings.recommender_w_commute),
            w_transit=float(app_settings.recommender_w_transit),
            w_overprice=float(app_settings.recommender_w_overprice),
            otodom_pool_default=bool(app_settings.recommender_otodom_pool_default),
            commute_mode=app_settings.commute_display_mode,
        )
        manifest.mark_and_save("recommender", rec_fp)
        logger.info(
            "STEP done recommender in %.1fs (%.2f min)",
            time.perf_counter() - _rec_t0,
            (time.perf_counter() - _rec_t0) / 60.0,
        )

    if manifest.steps:
        manifest.save()
    if skipped_steps:
        logger.info("Incremental skips: %s", ", ".join(skipped_steps))

    mark_commute_features_applied(app_settings, commute)
    if app_settings.commute_target_lon is None:
        app_settings.commute_target_lon = float(commute.lon)
        app_settings.commute_target_lat = float(commute.lat)
        app_settings.commute_target_label = commute.label
    # Clear one-shot force flag after a forced run so next make pipeline is incremental.
    if app_settings.pipeline_force_refresh and force_pipeline is not False:
        app_settings.pipeline_force_refresh = False
    save_app_settings(settings_path(processed), app_settings)

    duration_s = time.perf_counter() - started
    n_by_source = (
        train_frame["source"].astype(str).value_counts().to_dict()
        if "source" in train_frame.columns
        else {}
    )
    write_run_metadata(
        metadata_path,
        RunMetadata(
            refreshed_at=datetime.now(tz=UTC).isoformat(),
            duration_s=float(duration_s),
            n_offers=len(train_frame),
            n_by_source={str(k): int(v) for k, v in n_by_source.items()},
            model_metrics=model_metrics,
            model_path=str(model_path),
            feature_columns=feature_names,
            sources=list(DEFAULT_SOURCES),
            methodology=format_methodology_note(commute.label),
        ),
    )

    return {
        "enriched": enriched_path,
        "scored": scored_path,
        "model": model_path,
        "gtfs": gtfs_parquet,
        "gios": gios_parquet,
        "rcn": rcn_parquet,
        "otodom": otodom_parquet,
        "gunb": gunb_parquet,
        "metadata": metadata_path,
        "recommender": rec_metrics_file,
        "steps": pipeline_steps_path(processed),
    }


def parse_pipeline_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``python -m happy_housing.pipeline`` flags."""
    parser = argparse.ArgumentParser(
        prog="happy_housing.pipeline",
        description=(
            "Run ingest → enrich → train/score → recommend. "
            "Incremental by default; use --force to recompute."
        ),
    )
    parser.add_argument(
        "--force",
        "--force-pipeline",
        dest="force_pipeline",
        action="store_true",
        help=(
            "Ignore step fingerprints and re-run GTFS/GIOŚ/RCN export, "
            "enrich, train/score, and recommender."
        ),
    )
    parser.add_argument(
        "--force-gis",
        dest="force_gis_refresh",
        action="store_true",
        help="Re-download OSM / Warsaw map GIS caches.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Experiment YAML path (default: configs/experiment.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry for ``python -m happy_housing.pipeline``."""
    from happy_housing.logging_config import configure_logging

    args = parse_pipeline_cli_args(argv)
    config = load_experiment_config(args.config)
    configure_logging(config.logging.level, force=True)
    logger.info(
        "Starting pipeline (log level=%s force_pipeline=%s force_gis=%s)",
        config.logging.level,
        args.force_pipeline,
        args.force_gis_refresh,
    )
    paths = run_pipeline(
        args.config,
        force_pipeline=True if args.force_pipeline else None,
        force_gis_refresh=bool(args.force_gis_refresh),
    )
    for name, path in paths.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
