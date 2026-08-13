"""Tests for Home Data-sources inventory (counts + fetch times)."""

import json
from pathlib import Path

import pandas as pd

from happy_housing.app.metadata import DataSource
from happy_housing.app.source_inventory import (
    build_source_inventory,
    count_json_records,
    count_parquet_rows,
    format_fetched_at,
    format_row_count,
)


def test_count_parquet_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path, index=False)
    assert count_parquet_rows(path) == 3


def test_count_json_feature_collection(tmp_path: Path) -> None:
    path = tmp_path / "pts.geojson"
    path.write_text(
        json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": None, "properties": {}},
                {"type": "Feature", "geometry": None, "properties": {}},
            ],
        }),
        encoding="utf-8",
    )
    assert count_json_records(path) == 2


def test_format_helpers() -> None:
    assert format_row_count(1020, "listings") == "1,020 listings"
    assert format_row_count(None, "listings") == "— listings"
    assert format_fetched_at(None) == "—"
    from datetime import UTC, datetime

    stamp = 1_754_587_949.0
    assert format_fetched_at(stamp) == datetime.fromtimestamp(stamp, tz=UTC).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def test_build_source_inventory_reads_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    fixtures = tmp_path / "fixtures"
    processed.mkdir()
    cache.mkdir()
    fixtures.mkdir()
    pd.DataFrame({"id": [1, 2]}).to_parquet(
        processed / "otodom_listings.parquet", index=False
    )
    catalog = (
        DataSource(
            name="Otodom",
            url="https://www.otodom.pl/",
            role="Asks",
            key="otodom",
            unit="listings",
        ),
        DataSource(
            name="Missing",
            url="https://example.com/",
            role="Gone",
            key="gios",
            unit="stations",
        ),
    )
    rows = build_source_inventory(
        catalog,
        processed_dir=processed,
        cache_dir=cache,
        fixtures_dir=fixtures,
    )
    assert len(rows) == 2
    assert rows[0].rows_label == "2 listings"
    assert rows[0].fetched_label != "—"
    assert rows[0].origin_label == "processed"
    assert rows[1].rows_label == "— stations"
    assert rows[1].fetched_label == "—"
    assert rows[1].origin_label == "missing"
