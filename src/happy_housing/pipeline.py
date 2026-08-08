"""End-to-end pipeline: Otodom/RCN ingest → GIS → store → train → score."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from happy_housing.app.app_settings import load_app_settings, settings_path
from happy_housing.app.metadata import (
    DEFAULT_METHODOLOGY,
    DEFAULT_SOURCES,
    RunMetadata,
    write_run_metadata,
)
from happy_housing.config import load_experiment_config
from happy_housing.gis import (
    enrich_listings,
    fetch_osm_amenities,
    fetch_osm_shops,
    load_amenity_points,
)
from happy_housing.gis.routing import attach_commute_minutes
from happy_housing.ingest import (
    build_gtfs_stop_hourly,
    download_file,
    download_gtfs_stops,
    fetch_gios_stations,
    load_listings_json,
)
from happy_housing.ingest.gunb import download_gunb_mazowieckie, gunb_points_gdf
from happy_housing.ingest.otodom import scrape_otodom_warsaw
from happy_housing.ingest.rcn import extract_rcn_transactions
from happy_housing.models.metrics import coerce_opt_metric
from happy_housing.models.neighborhood import attach_neighborhood_rcn_features
from happy_housing.models.train import (
    TrainSettings,
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
    return out.reset_index(drop=True)


def run_pipeline(
    config_path: Path | str = Path("configs/experiment.yaml"),
    *,
    download_remote: bool = True,
    use_osm: bool = True,
    scrape_otodom: bool = True,
    otodom_max_pages: int | None = None,
    otodom_max_listings: int | None = None,
    rcn_max_rows: int | None = None,
) -> dict[str, Path]:
    """Run the full local pipeline and return key output paths."""
    from happy_housing.logging_config import configure_logging

    started = time.perf_counter()
    config = load_experiment_config(config_path)
    configure_logging(config.logging.level, force=False)
    logger.info(
        "Pipeline start (download_remote=%s use_osm=%s scrape_otodom=%s)",
        download_remote,
        use_osm,
        scrape_otodom,
    )
    limits = config.ingest
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
    processed = config.data.processed_path()
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

    if scrape_otodom:
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

    if download_remote:
        download_gtfs_stops(config.ingest_urls.gtfs_zip, raw, gtfs_parquet)
        fetch_gios_stations(config.ingest_urls.gios_stations, gios_parquet)
        zip_path = download_file(
            config.ingest_urls.rcn_zip,
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
        download_gunb_mazowieckie(
            raw,
            gunb_parquet,
            url=config.ingest_urls.gunb_zip,
            fixture_path=fixtures / "gunb_points.json",
        )
    else:
        for path, cols in (
            (gtfs_parquet, ["stop_id", "stop_lat", "stop_lon"]),
            (gios_parquet, ["station_id", "lat", "lon", "name", "city"]),
            (rcn_parquet, ["lon", "lat", "price_pln", "area_m2"]),
            (gunb_parquet, ["lon", "lat"]),
        ):
            if not path.exists():
                pd.DataFrame(columns=cols).to_parquet(path, index=False)

    rcn = pd.read_parquet(rcn_parquet)
    rcn_listings = _rcn_as_listings(rcn)
    if not rcn_listings.empty:
        listings_parts.append(rcn_listings)

    listings = pd.concat(listings_parts, ignore_index=True, sort=False)
    listings = listings.drop_duplicates(subset=["listing_id"], keep="last")
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
    gunb = gunb_points_gdf(pd.read_parquet(gunb_parquet))

    if use_osm:
        amenities = fetch_osm_amenities(
            config.bbox.as_tuple(),
            cache,
            fixture_path=fixtures / "amenities.json",
        )
        fetch_osm_shops(
            config.bbox.as_tuple(),
            cache,
            fixture_path=fixtures / "shops.json",
        )
    else:
        amenities = load_amenity_points(fixtures / "amenities.json")

    enriched = enrich_listings(
        listings,
        projected_crs=config.bbox.projected_crs,
        amenities=amenities,
        gios_stations=gios_stations,
        gtfs_stops=gtfs_stops,
        commute_lon=config.commute_target_lon,
        commute_lat=config.commute_target_lat,
        gunb_points=gunb,
        aircraft_lden_path=fixtures / "chopin_aircraft_lden.geojson",
        aircraft_lnight_path=fixtures / "chopin_aircraft_lnight.geojson",
    )
    train_frame = pd.DataFrame(enriched.drop(columns=["geometry"], errors="ignore"))
    train_frame = attach_commute_minutes(
        train_frame,
        dest_lon=config.commute_target_lon,
        dest_lat=config.commute_target_lat,
    )

    enriched_path = processed / "listings_enriched.parquet"
    write_geoparquet(enriched, enriched_path)
    # Persist commute column onto enriched parquet companion table.
    train_frame.to_parquet(enriched_path, index=False)

    app_settings = load_app_settings(settings_path(processed))
    rcn_for_model = (
        pd.read_parquet(rcn_parquet) if rcn_parquet.exists() else pd.DataFrame()
    )
    if app_settings.model_use_neighborhood and not rcn_for_model.empty:
        train_frame = attach_neighborhood_rcn_features(
            train_frame,
            rcn_for_model,
            radius_m=float(app_settings.model_nbhd_radius_m),
        )
        train_frame.to_parquet(enriched_path, index=False)

    train_settings = TrainSettings(
        opt_metric=coerce_opt_metric(app_settings.model_opt_metric),
        val_size=float(app_settings.model_val_size),
        test_size=float(config.model.test_size),
        hpo_trials=int(app_settings.model_hpo_trials),
        use_neighborhood=bool(app_settings.model_use_neighborhood),
        random_state=int(config.model.random_state),
    )
    model_path = processed / "lightgbm_price_per_m2.txt"
    result = run_training_with_mlflow(
        train_frame,
        config,
        model_path,
        train_settings=train_settings,
        rcn=rcn_for_model if app_settings.model_use_neighborhood else None,
    )
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
