"""Tests for Admin Freshness report (sources + derived products)."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from happy_housing.app.freshness import (
    build_freshness_report,
    derived_to_frame,
    sources_to_frame,
)
from happy_housing.app.metadata import DataSource


def _catalog_otodom_gios() -> tuple[DataSource, ...]:
    return (
        DataSource(
            name="Otodom",
            url="https://www.otodom.pl/",
            role="listings",
            key="otodom",
            unit="listings",
        ),
        DataSource(
            name="GIOŚ",
            url="https://powietrze.gios.gov.pl/",
            role="air",
            key="gios",
            unit="stations",
        ),
    )


def _touch(path: Path, when: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.write_text("{}", encoding="utf-8")
    os.utime(path, (when.timestamp(), when.timestamp()))


def test_source_status_missing_fixture_ok_empty(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    fixtures = tmp_path / "fixtures"
    processed.mkdir()
    cache.mkdir()
    fixtures.mkdir()

    pd.DataFrame({"id": [1, 2]}).to_parquet(
        processed / "otodom_listings.parquet", index=False
    )
    fixtures.joinpath("gios_air_quality.json").write_text("[]", encoding="utf-8")
    # Empty gios preferred would win if present — use only fixture for fixture status.
    catalog = (
        *_catalog_otodom_gios(),
        DataSource(
            name="Empty parks",
            url="https://example.com/",
            role="parks",
            key="warsaw_parks",
            unit="polygons",
        ),
        DataSource(
            name="Gone schools",
            url="https://example.com/",
            role="schools",
            key="warsaw_schools",
            unit="schools",
        ),
    )
    cache.joinpath("warsaw_parks.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )

    report = build_freshness_report(
        processed_dir=processed,
        cache_dir=cache,
        fixtures_dir=fixtures,
        catalog=catalog,
        source_warn_h=168.0,
        derived_warn_h=168.0,
    )
    by_key = {row.key: row for row in report.sources}
    assert by_key["otodom"].status == "ok"
    assert by_key["otodom"].origin == "processed"
    assert by_key["otodom"].row_count == 2
    assert by_key["gios"].status == "fixture"
    assert by_key["gios"].origin == "fixtures"
    assert by_key["warsaw_parks"].status == "empty"
    assert by_key["warsaw_schools"].status == "missing"
    assert report.n_sources_ok == 1
    assert report.n_sources_fixture == 1
    assert report.n_sources_empty == 1
    assert report.n_sources_missing == 1


def test_derived_missing_ok_stale_vs_upstream(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    fixtures = tmp_path / "fixtures"
    processed.mkdir()
    cache.mkdir()
    fixtures.mkdir()

    old = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    new = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)
    now = datetime(2026, 8, 13, 18, 0, 0, tzinfo=UTC)

    pd.DataFrame({"id": [1]}).to_parquet(
        processed / "otodom_listings.parquet", index=False
    )
    _touch(processed / "otodom_listings.parquet", new)

    scored = processed / "listings_scored.parquet"
    pd.DataFrame({"listing_id": ["a"]}).to_parquet(scored, index=False)
    _touch(scored, old)

    model = processed / "lightgbm_price_per_m2.txt"
    model.write_text("tree", encoding="utf-8")
    _touch(model, new)

    report = build_freshness_report(
        processed_dir=processed,
        cache_dir=cache,
        fixtures_dir=fixtures,
        catalog=_catalog_otodom_gios()[:1],
        now=now,
        source_warn_h=168.0,
        derived_warn_h=168.0,
    )
    by_key = {row.key: row for row in report.derived}
    assert by_key["listings_scored"].status == "stale"
    assert by_key["price_model"].status == "ok"
    assert by_key["recommender_metrics"].status == "missing"
    assert by_key["run_metadata"].status == "missing"
    assert report.n_derived_stale >= 1
    assert report.n_derived_missing >= 1
    assert report.n_derived_ok >= 1


def test_derived_stale_by_age_threshold(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    fixtures = tmp_path / "fixtures"
    processed.mkdir()
    cache.mkdir()
    fixtures.mkdir()

    # Source older than derived, but derived age exceeds warn hours.
    src_t = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)
    der_t = datetime(2026, 7, 2, 0, 0, 0, tzinfo=UTC)
    now = datetime(2026, 8, 13, 0, 0, 0, tzinfo=UTC)

    pd.DataFrame({"id": [1]}).to_parquet(
        processed / "otodom_listings.parquet", index=False
    )
    _touch(processed / "otodom_listings.parquet", src_t)
    scored = processed / "listings_scored.parquet"
    pd.DataFrame({"listing_id": ["a"]}).to_parquet(scored, index=False)
    _touch(scored, der_t)

    report = build_freshness_report(
        processed_dir=processed,
        cache_dir=cache,
        fixtures_dir=fixtures,
        catalog=_catalog_otodom_gios()[:1],
        now=now,
        source_warn_h=24.0,
        derived_warn_h=24.0,
    )
    scored_row = next(r for r in report.derived if r.key == "listings_scored")
    assert scored_row.status == "stale"


def test_frames_have_expected_columns(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    fixtures = tmp_path / "fixtures"
    for d in (processed, cache, fixtures):
        d.mkdir()
    report = build_freshness_report(
        processed_dir=processed,
        cache_dir=cache,
        fixtures_dir=fixtures,
        catalog=_catalog_otodom_gios()[:1],
        source_warn_h=168.0,
        derived_warn_h=168.0,
    )
    src = sources_to_frame(report.sources)
    der = derived_to_frame(report.derived)
    assert "status" in src.columns
    assert "fetched_at" in src.columns
    assert "age" in src.columns
    assert "status" in der.columns
    assert "key" in der.columns
