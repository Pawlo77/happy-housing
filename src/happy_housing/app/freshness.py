"""Local-disk freshness report for Admin Freshness (sources + derived)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from happy_housing.app.admin_ops import age_hours_label
from happy_housing.app.io_cache import file_mtime
from happy_housing.app.metadata import DEFAULT_SOURCE_CATALOG, DataSource
from happy_housing.app.source_inventory import (
    format_fetched_at,
    format_row_count,
    resolve_source_artifact,
)
from happy_housing.pipeline_cache import pipeline_steps_path

SourceStatus = Literal["ok", "empty", "fixture", "missing"]
"""Local presence class for a catalog source artifact."""

DerivedStatus = Literal["ok", "stale", "missing"]
"""Derived product freshness relative to upstream + age warn."""


@dataclass(frozen=True, slots=True)
class SourceFreshnessRow:
    """One catalog source with download status and age."""

    key: str
    """Catalog key."""
    name: str
    """Display name."""
    status: SourceStatus
    """``ok`` / ``empty`` / ``fixture`` / ``missing``."""
    origin: str
    """Winning root label."""
    rows_label: str
    """Human row count."""
    row_count: int | None
    """Numeric count when known."""
    fetched_label: str
    """UTC mtime stamp or em dash."""
    age: str
    """Human age label."""
    mtime: float | None
    """Raw mtime for upstream clock."""
    age_warn: bool
    """True when age exceeds source warn hours."""
    role: str
    """Catalog role blurb."""
    url: str
    """Publisher URL."""


@dataclass(frozen=True, slots=True)
class DerivedFreshnessRow:
    """One derived pipeline product (score / model / metrics)."""

    key: str
    """Stable id (``listings_scored``, ``price_model``, …)."""
    name: str
    """Display name."""
    status: DerivedStatus
    """``ok`` / ``stale`` / ``missing``."""
    required: bool
    """Whether required for overall app freshness."""
    fetched_label: str
    """UTC mtime stamp or em dash."""
    age: str
    """Human age label."""
    detail: str
    """Short reason (stale vs upstream, age warn, …)."""


@dataclass(frozen=True, slots=True)
class FreshnessReport:
    """Full Admin Freshness snapshot."""

    sources: list[SourceFreshnessRow]
    """Per-catalog source rows."""
    derived: list[DerivedFreshnessRow]
    """Derived product rows."""
    upstream_mtime: float | None
    """Max mtime among ``ok`` sources."""
    n_sources_ok: int
    """Count of ``ok`` sources."""
    n_sources_empty: int
    """Count of ``empty`` sources."""
    n_sources_fixture: int
    """Count of fixture-only sources."""
    n_sources_missing: int
    """Count of missing sources."""
    n_derived_ok: int
    """Count of required+informational derived ``ok``."""
    n_derived_stale: int
    """Count of derived ``stale``."""
    n_derived_missing: int
    """Count of derived ``missing``."""


@dataclass(frozen=True, slots=True)
class _DerivedSpec:
    """One derived artifact to probe under ``processed``."""

    key: str
    name: str
    relative: str
    required: bool


_DERIVED_SPECS: tuple[_DerivedSpec, ...] = (
    _DerivedSpec(
        "listings_scored",
        "Scored listings",
        "listings_scored.parquet",
        True,
    ),
    _DerivedSpec(
        "price_model",
        "Price model (LightGBM)",
        "lightgbm_price_per_m2.txt",
        True,
    ),
    _DerivedSpec(
        "recommender_metrics",
        "Recommender metrics",
        "recommender_metrics.json",
        True,
    ),
    _DerivedSpec(
        "run_metadata",
        "Run metadata",
        "run_metadata.json",
        True,
    ),
    _DerivedSpec(
        "pipeline_steps",
        "Pipeline step cache",
        "pipeline_steps.json",
        False,
    ),
)
"""Derived products checked on Freshness (``pipeline_steps`` informational)."""


def _age_hours(fetched_label: str, now: datetime) -> float | None:
    text = str(fetched_label or "").strip()
    if not text or text == "—":
        return None
    try:
        fetched = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0.0, (now - fetched).total_seconds() / 3600.0)


def _source_status(
    origin: str,
    row_count: int | None,
) -> SourceStatus:
    if origin == "missing":
        return "missing"
    if origin == "fixtures":
        return "fixture"
    if row_count == 0:
        return "empty"
    return "ok"


def _classify_derived(
    path: Path,
    upstream_mtime: float | None,
    now: datetime,
    derived_warn_h: float,
) -> tuple[DerivedStatus, str, str, str]:
    """Return status, fetched_label, age, detail."""
    if not path.exists():
        return "missing", "—", "—", "File absent"
    mtime = file_mtime(path)
    if mtime < 0:
        return "missing", "—", "—", "File unreadable"
    fetched = format_fetched_at(mtime)
    age = age_hours_label(fetched, now=now)
    hours = _age_hours(fetched, now)
    reasons: list[str] = []
    if upstream_mtime is not None and mtime + 1e-6 < upstream_mtime:
        reasons.append("older than newest ok source")
    if hours is not None and hours > float(derived_warn_h):
        reasons.append(f"age > {derived_warn_h:g} h warn")
    if reasons:
        return "stale", fetched, age, "; ".join(reasons)
    return "ok", fetched, age, "Up to date"


def build_freshness_report(
    processed_dir: Path,
    cache_dir: Path,
    fixtures_dir: Path,
    catalog: tuple[DataSource, ...] | list[DataSource] | None = None,
    now: datetime | None = None,
    source_warn_h: float = 168.0,
    derived_warn_h: float = 168.0,
) -> FreshnessReport:
    """Build local-disk freshness for catalog sources and derived products."""
    current = now or datetime.now(tz=UTC)
    sources_catalog = catalog if catalog is not None else DEFAULT_SOURCE_CATALOG
    source_rows: list[SourceFreshnessRow] = []
    for src in sources_catalog:
        resolved = resolve_source_artifact(
            src.key,
            processed_dir=processed_dir,
            cache_dir=cache_dir,
            fixtures_dir=fixtures_dir,
        )
        src_status = _source_status(resolved.origin, resolved.row_count)
        fetched = format_fetched_at(resolved.mtime)
        hours = _age_hours(fetched, current)
        source_rows.append(
            SourceFreshnessRow(
                key=src.key,
                name=src.name,
                status=src_status,
                origin=resolved.origin,
                rows_label=format_row_count(resolved.row_count, src.unit),
                row_count=resolved.row_count,
                fetched_label=fetched,
                age=age_hours_label(fetched, now=current),
                mtime=resolved.mtime,
                age_warn=bool(hours is not None and hours > float(source_warn_h)),
                role=src.role,
                url=src.url,
            )
        )

    upstream_candidates = [
        row.mtime for row in source_rows if row.status == "ok" and row.mtime is not None
    ]
    upstream_mtime = max(upstream_candidates) if upstream_candidates else None

    derived_rows: list[DerivedFreshnessRow] = []
    for spec in _DERIVED_SPECS:
        if spec.key == "pipeline_steps":
            path = pipeline_steps_path(processed_dir)
        else:
            path = processed_dir / spec.relative
        der_status, fetched, age, detail = _classify_derived(
            path,
            upstream_mtime=upstream_mtime,
            now=current,
            derived_warn_h=derived_warn_h,
        )
        if spec.key == "price_model" and path.exists():
            meta = path.with_suffix(path.suffix + ".meta.json")
            if not meta.exists():
                detail = (
                    f"{detail}; meta sidecar missing"
                    if detail != "File absent"
                    else "Model present; meta sidecar missing"
                )
        derived_rows.append(
            DerivedFreshnessRow(
                key=spec.key,
                name=spec.name,
                status=der_status,
                required=spec.required,
                fetched_label=fetched,
                age=age,
                detail=detail,
            )
        )

    return FreshnessReport(
        sources=source_rows,
        derived=derived_rows,
        upstream_mtime=upstream_mtime,
        n_sources_ok=sum(1 for r in source_rows if r.status == "ok"),
        n_sources_empty=sum(1 for r in source_rows if r.status == "empty"),
        n_sources_fixture=sum(1 for r in source_rows if r.status == "fixture"),
        n_sources_missing=sum(1 for r in source_rows if r.status == "missing"),
        n_derived_ok=sum(1 for r in derived_rows if r.status == "ok"),
        n_derived_stale=sum(1 for r in derived_rows if r.status == "stale"),
        n_derived_missing=sum(1 for r in derived_rows if r.status == "missing"),
    )


def sources_to_frame(rows: list[SourceFreshnessRow]) -> pd.DataFrame:
    """Admin table for catalog sources."""
    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "status",
                "rows",
                "fetched_at",
                "age",
                "age_warn",
                "origin",
                "role",
            ]
        )
    return pd.DataFrame.from_records([
        {
            "source": row.name,
            "status": row.status,
            "rows": row.rows_label,
            "fetched_at": row.fetched_label,
            "age": row.age,
            "age_warn": row.age_warn,
            "origin": row.origin,
            "role": row.role,
        }
        for row in rows
    ])


def derived_to_frame(rows: list[DerivedFreshnessRow]) -> pd.DataFrame:
    """Admin table for derived products."""
    if not rows:
        return pd.DataFrame(
            columns=[
                "key",
                "product",
                "status",
                "required",
                "fetched_at",
                "age",
                "detail",
            ]
        )
    return pd.DataFrame.from_records([
        {
            "key": row.key,
            "product": row.name,
            "status": row.status,
            "required": row.required,
            "fetched_at": row.fetched_label,
            "age": row.age,
            "detail": row.detail,
        }
        for row in rows
    ])
