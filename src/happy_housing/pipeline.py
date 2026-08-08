"""End-to-end pipeline: Otodom/RCN ingest → GIS → store → train → score."""

from __future__ import annotations

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
from happy_housing.app.listing_attrs import promote_listing_flags
from happy_housing.app.llm_metrics import llm_metrics_path
from happy_housing.app.metadata import (
    DEFAULT_METHODOLOGY,
    DEFAULT_SOURCES,
    RunMetadata,
    write_run_metadata,
)
from happy_housing.app.pipeline_ingest import (
    resolve_ingest_flags,
    resolve_ingest_limits,
    resolve_ingest_urls,
)
from happy_housing.config import load_experiment_config
from happy_housing.gis import (
    enrich_listings,
    fetch_osm_amenities,
    fetch_osm_shops,
    load_amenity_points,
    load_shop_points,
)
from happy_housing.gis.bike_paths import fetch_bike_paths, load_bike_path_sample
from happy_housing.gis.gios_aq_field import (
    attach_gios_aq_field_features,
    fetch_gios_aq_field_rasters,
)
from happy_housing.gis.location_fix import snap_listings_out_of_parks
from happy_housing.gis.routing import (
    attach_commute_minutes,
    attach_transit_commute_minutes,
)
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
from happy_housing.ingest.gios_aq import (
    AIR_QUALITY_COLUMNS,
    enrich_stations_with_air_quality,
)
from happy_housing.ingest.gunb import download_gunb_mazowieckie, gunb_points_gdf
from happy_housing.ingest.market import ensure_market_primary
from happy_housing.ingest.otodom import scrape_otodom_warsaw
from happy_housing.ingest.rcn import extract_rcn_transactions
from happy_housing.ingest.warsaw_map import (
    fetch_warsaw_city_pois,
    fetch_warsaw_parks,
    fetch_warsaw_schools,
)
from happy_housing.models.metrics import coerce_opt_metric
from happy_housing.models.neighborhood import attach_neighborhood_rcn_features
from happy_housing.models.train import (
    TrainSettings,
    oof_predict_price_per_m2,
    predict_price_per_m2,
    run_training_with_mlflow,
)
from happy_housing.storage import write_geoparquet

logger = logging.getLogger(__name__)


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


def run_pipeline(
    config_path: Path | str = Path("configs/experiment.yaml"),
    *,
    download_remote: bool | None = None,
    use_osm: bool | None = None,
    scrape_otodom: bool | None = None,
    otodom_max_pages: int | None = None,
    otodom_max_listings: int | None = None,
    rcn_max_rows: int | None = None,
    force_gis_refresh: bool = False,
) -> dict[str, Path]:
    """Run the full local pipeline and return key output paths.

    ``download_remote`` True/False forces all remote sources on/off; ``None``
    uses Admin Pipeline overrides when saved, else downloads everything.
    Explicit ``otodom_max_*`` / ``rcn_max_rows`` kwargs override resolved caps.
    ``force_gis_refresh`` ignores OSM / Warsaw map caches and re-downloads.
    """
    from happy_housing.logging_config import configure_logging

    started = time.perf_counter()
    config = load_experiment_config(config_path)
    configure_logging(config.logging.level, force=False)
    processed = config.data.processed_path()
    app_settings = load_app_settings(settings_path(processed))
    limits = resolve_ingest_limits(app_settings, config)
    urls = resolve_ingest_urls(app_settings, config)
    flags = resolve_ingest_flags(
        app_settings,
        config,
        download_remote=download_remote,
        use_osm=use_osm,
        scrape_otodom=scrape_otodom,
    )
    logger.info(
        "Pipeline start (gtfs=%s gios=%s rcn=%s gunb=%s crime=%s osm=%s otodom=%s)",
        flags.download_gtfs,
        flags.download_gios,
        flags.download_rcn,
        flags.download_gunb,
        flags.download_crime,
        flags.use_osm,
        flags.scrape_otodom,
    )
    page_cap = (
        otodom_max_pages if otodom_max_pages is not None else limits.otodom_max_pages
    )
    listing_cap = (
        otodom_max_listings
        if otodom_max_listings is not None
        else limits.otodom_max_listings
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
            otodom = scrape_otodom_warsaw(
                otodom_parquet,
                max_pages=page_cap,
                max_listings=listing_cap,
                incremental=limits.otodom_incremental,
            )
            otodom["source"] = "otodom"
            listings_parts.append(otodom)
        except Exception as exc:
            logger.warning("Otodom scrape failed: %s", exc)
            if otodom_parquet.exists():
                otodom = pd.read_parquet(otodom_parquet)
                otodom["source"] = "otodom"
                listings_parts.append(otodom)
    elif otodom_parquet.exists():
        otodom = pd.read_parquet(otodom_parquet)
        otodom["source"] = "otodom"
        listings_parts.append(otodom)

    gtfs_parquet = processed / "gtfs_stops.parquet"
    gios_parquet = processed / "gios_stations.parquet"
    rcn_parquet = processed / "rcn_transactions.parquet"
    gunb_parquet = processed / "gunb_mazowieckie.parquet"
    crime_parquet = processed / "crime_by_district.parquet"

    def _ensure_empty(path: Path, cols: list[str]) -> None:
        if not path.exists():
            pd.DataFrame(columns=cols).to_parquet(path, index=False)

    if flags.download_gtfs:
        download_gtfs_stops(urls.gtfs_zip, raw, gtfs_parquet)
    else:
        _ensure_empty(gtfs_parquet, ["stop_id", "stop_lat", "stop_lon"])

    if flags.download_gios:
        fetch_gios_stations(urls.gios_stations, gios_parquet)
    else:
        _ensure_empty(gios_parquet, ["station_id", "lat", "lon", "name", "city"])

    if flags.download_rcn:
        zip_path = download_file(
            urls.rcn_zip,
            raw / "rcn_warsaw.zip",
            timeout_s=1800.0,
        )
        try:
            extract_rcn_transactions(zip_path, rcn_parquet, max_rows=rcn_cap)
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
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_parks = pool.submit(
            fetch_warsaw_parks,
            cache,
            fixture_path=fixtures / "parks.geojson",
            force_refresh=force_gis_refresh,
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
    # OSMnx road / bike fetches stay serial (global ox.settings).
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

    if app_settings.snap_listings_out_of_parks:
        listings = snap_listings_out_of_parks(
            listings,
            parks=parks,
            cache_dir=cache,
            bbox=config.bbox.as_tuple(),
            force_refresh_buildings=force_gis_refresh,
        )

    commute = resolve_commute_target(app_settings, config)
    logger.info(
        "Commute target %s (%.5f, %.5f)", commute.label, commute.lon, commute.lat
    )

    road_lden_path = fixtures / "road_lden.geojson"
    rail_lden_path = fixtures / "rail_lden.geojson"
    enriched = enrich_listings(
        listings,
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
        bike_path_coverage_radius_m=float(app_settings.bike_path_coverage_radius_m),
        bike_path_sample=bike_path_sample if not bike_path_sample.empty else None,
        rail_areas=rail_areas,
        military=military,
        airport_areas=airport_areas,
        traffic_sample=traffic_sample,
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
    if noise_rasters:
        enriched = attach_warsaw_noise_features(enriched, noise_rasters)
    if app_settings.aq_field_enabled and not gios_stations.empty:
        aq_rasters = fetch_gios_aq_field_rasters(
            cache,
            gios_stations,
            bbox_wgs84=config.bbox.as_tuple(),
            power=float(app_settings.aq_idw_power),
            force_refresh=force_gis_refresh,
        )
        if aq_rasters:
            enriched = attach_gios_aq_field_features(enriched, aq_rasters)
    train_frame = pd.DataFrame(enriched.drop(columns=["geometry"], errors="ignore"))
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
        )
    train_frame = attach_commute_minutes(
        train_frame,
        dest_lon=commute.lon,
        dest_lat=commute.lat,
        osrm_url=config.routing.osrm_url,
    )
    if config.routing.use_otp_transit:
        train_frame = attach_transit_commute_minutes(
            train_frame,
            dest_lon=commute.lon,
            dest_lat=commute.lat,
            otp_url=config.routing.otp_url,
        )

    enriched_path = processed / "listings_enriched.parquet"

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

    train_settings = TrainSettings(
        opt_metric=coerce_opt_metric(app_settings.model_opt_metric),
        val_size=float(app_settings.model_val_size),
        test_size=float(config.model.test_size),
        hpo_trials=int(app_settings.model_hpo_trials),
        use_neighborhood=bool(app_settings.model_use_neighborhood),
        random_state=int(config.model.random_state),
        train_otodom_labels_only=bool(app_settings.train_otodom_labels_only),
    )
    model_path = processed / "lightgbm_price_per_m2.txt"
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
        )
    train_frame["delta_price_per_m2"] = (
        train_frame["price_per_m2"] - train_frame["model_price_per_m2"]
    )
    scored_path = processed / "listings_scored.parquet"
    train_frame.to_parquet(scored_path, index=False)

    mark_commute_features_applied(app_settings, commute)
    if app_settings.commute_target_lon is None:
        app_settings.commute_target_lon = float(commute.lon)
        app_settings.commute_target_lat = float(commute.lat)
        app_settings.commute_target_label = commute.label
    save_app_settings(settings_path(processed), app_settings)

    duration_s = time.perf_counter() - started
    n_by_source = (
        train_frame["source"].astype(str).value_counts().to_dict()
        if "source" in train_frame.columns
        else {}
    )
    write_run_metadata(
        processed / "run_metadata.json",
        RunMetadata(
            refreshed_at=datetime.now(tz=UTC).isoformat(),
            duration_s=float(duration_s),
            n_offers=len(train_frame),
            n_by_source={str(k): int(v) for k, v in n_by_source.items()},
            model_metrics=dict(result.metrics),
            model_path=str(model_path),
            feature_columns=list(result.feature_names),
            sources=list(DEFAULT_SOURCES),
            methodology=DEFAULT_METHODOLOGY,
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
        "metadata": processed / "run_metadata.json",
    }


def main() -> None:
    """CLI entry for ``python -m happy_housing.pipeline``."""
    from happy_housing.logging_config import configure_logging

    config = load_experiment_config(Path("configs/experiment.yaml"))
    configure_logging(config.logging.level, force=True)
    logger.info("Starting pipeline (log level=%s)", config.logging.level)
    paths = run_pipeline()
    for name, path in paths.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
