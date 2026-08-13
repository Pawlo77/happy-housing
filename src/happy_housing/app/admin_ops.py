"""Ops freshness table helpers for Admin."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from happy_housing.app.io_cache import file_mtime
from happy_housing.app.metadata import DEFAULT_SOURCE_CATALOG, DataSource
from happy_housing.app.source_inventory import (
    SourceInventoryRow,
    build_source_inventory,
    format_fetched_at,
)
from happy_housing.pipeline_cache import pipeline_steps_path


def age_hours_label(fetched_label: str, *, now: datetime | None = None) -> str:
    """Human age from ``format_fetched_at`` UTC stamp, or em dash."""
    text = str(fetched_label or "").strip()
    if not text or text == "—":
        return "—"
    try:
        fetched = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return "—"
    current = now or datetime.now(tz=UTC)
    hours = max(0.0, (current - fetched).total_seconds() / 3600.0)
    if hours < 1.0:
        return f"{hours * 60:.0f} min"
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


def inventory_to_frame(rows: list[SourceInventoryRow]) -> pd.DataFrame:
    """Turn inventory rows into an Admin Ops dataframe."""
    if not rows:
        return pd.DataFrame(
            columns=["source", "rows", "fetched_at", "age", "origin", "role"]
        )
    records = [
        {
            "source": row.name,
            "rows": row.rows_label,
            "fetched_at": row.fetched_label,
            "age": age_hours_label(row.fetched_label),
            "origin": row.origin_label,
            "role": row.role,
            "url": row.url,
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records)


def clear_pipeline_step_cache(processed_dir: Path) -> bool:
    """Delete ``pipeline_steps.json`` so the next run ignores step fingerprints."""
    path = pipeline_steps_path(processed_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def build_ops_freshness_frame(
    *,
    processed_dir: Path,
    cache_dir: Path,
    fixtures_dir: Path,
    catalog: tuple[DataSource, ...] | list[DataSource] | None = None,
) -> pd.DataFrame:
    """Full Admin Ops freshness table (catalog sources + scored/pipeline)."""
    sources = catalog if catalog is not None else DEFAULT_SOURCE_CATALOG
    rows = build_source_inventory(
        sources,
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        fixtures_dir=fixtures_dir,
    )
    frame = inventory_to_frame(rows)

    extras: list[dict[str, str]] = []
    scored = processed_dir / "listings_scored.parquet"
    if scored.exists():
        mtime = file_mtime(scored)
        fetched = format_fetched_at(mtime if mtime >= 0 else None)
        extras.append({
            "source": "Scored listings (model output)",
            "rows": "— offers",
            "fetched_at": fetched,
            "age": age_hours_label(fetched),
            "origin": "processed",
            "role": "Explore / Stats / Home universe after enrich + score.",
            "url": "",
        })
    steps = pipeline_steps_path(processed_dir)
    if steps.exists():
        mtime = file_mtime(steps)
        fetched = format_fetched_at(mtime if mtime >= 0 else None)
        extras.append({
            "source": "Pipeline step cache",
            "rows": "—",
            "fetched_at": fetched,
            "age": age_hours_label(fetched),
            "origin": "processed",
            "role": "Fingerprint skip file — clear to force enrich/train/score.",
            "url": "",
        })
    if not extras:
        return frame
    extra_frame = pd.DataFrame.from_records(extras)
    if frame.empty:
        return extra_frame
    return pd.concat([frame, extra_frame], ignore_index=True)
