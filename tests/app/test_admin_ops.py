"""Tests for Admin Ops freshness helpers."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from happy_housing.app.admin_ops import (
    age_hours_label,
    build_ops_freshness_frame,
    clear_pipeline_step_cache,
    inventory_to_frame,
)
from happy_housing.app.metadata import DataSource
from happy_housing.app.source_inventory import SourceInventoryRow


def test_age_hours_label_formats() -> None:
    now = datetime(2026, 8, 13, 21, 0, 0, tzinfo=UTC)
    assert age_hours_label("—", now=now) == "—"
    stamp = "2026-08-13 20:30:00"
    assert age_hours_label(stamp, now=now) == "30 min"
    stamp2 = "2026-08-12 21:00:00"
    assert age_hours_label(stamp2, now=now) == "24.0 h"


def test_inventory_to_frame_columns() -> None:
    rows = [
        SourceInventoryRow(
            name="Otodom",
            url="https://www.otodom.pl/",
            role="listings",
            rows_label="10 listings",
            fetched_label="2026-08-13 10:00:00",
            origin_label="processed",
        )
    ]
    frame = inventory_to_frame(rows)
    assert list(frame.columns) == [
        "source",
        "rows",
        "fetched_at",
        "age",
        "origin",
        "role",
        "url",
    ]
    assert frame.iloc[0]["source"] == "Otodom"


def test_build_ops_freshness_frame_includes_scored(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    fixtures = tmp_path / "fixtures"
    processed.mkdir()
    cache.mkdir()
    fixtures.mkdir()
    scored = processed / "listings_scored.parquet"
    pd.DataFrame({"listing_id": ["a"]}).to_parquet(scored, index=False)
    catalog = (
        DataSource(
            name="Otodom (public listing pages)",
            url="https://www.otodom.pl/",
            role="listings",
            key="otodom",
            unit="listings",
        ),
    )
    frame = build_ops_freshness_frame(
        processed_dir=processed,
        cache_dir=cache,
        fixtures_dir=fixtures,
        catalog=catalog,
    )
    names = set(frame["source"].astype(str))
    assert "Otodom (public listing pages)" in names
    assert "Scored listings (model output)" in names


def test_clear_pipeline_step_cache(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    steps = processed / "pipeline_steps.json"
    steps.write_text("{}", encoding="utf-8")
    assert clear_pipeline_step_cache(processed) is True
    assert not steps.exists()
    assert clear_pipeline_step_cache(processed) is False
