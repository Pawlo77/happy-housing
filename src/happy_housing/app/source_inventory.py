"""Inventory helpers for Home Data-sources cards (counts + mtimes)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from happy_housing.app.io_cache import file_mtime, json_record_count, parquet_num_rows
from happy_housing.app.metadata import DataSource

RootName = Literal["processed", "cache", "fixtures"]


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Relative artifact under a known data root."""

    root: RootName
    """Which data root holds the file."""
    name: str
    """File name under that root."""


@dataclass(frozen=True, slots=True)
class SourceInventoryRow:
    """One Data-sources card with live disk stats."""

    name: str
    url: str
    role: str
    rows_label: str
    fetched_label: str
    origin_label: str


_SOURCE_ARTIFACTS: dict[str, tuple[ArtifactRef, ...]] = {
    "otodom": (ArtifactRef("processed", "otodom_listings.parquet"),),
    "rcn": (ArtifactRef("processed", "rcn_transactions.parquet"),),
    "osm": (
        ArtifactRef("cache", "osm_amenities.geojson"),
        ArtifactRef("fixtures", "amenities.json"),
    ),
    "warsaw_parks": (
        ArtifactRef("cache", "warsaw_parks.geojson"),
        ArtifactRef("fixtures", "parks.geojson"),
    ),
    "gtfs": (ArtifactRef("processed", "gtfs_stops.parquet"),),
    "gios": (
        ArtifactRef("processed", "gios_stations.parquet"),
        ArtifactRef("fixtures", "gios_air_quality.json"),
    ),
    "gunb": (
        ArtifactRef("processed", "gunb_mazowieckie.parquet"),
        ArtifactRef("fixtures", "gunb_points.json"),
    ),
    "oou": (
        ArtifactRef("fixtures", "chopin_aircraft_lden.geojson"),
        ArtifactRef("fixtures", "chopin_aircraft_lnight.geojson"),
    ),
}
"""Preferred artifact paths per catalog ``key`` (first existing wins)."""


def count_parquet_rows(path: Path) -> int | None:
    """Return parquet row count from footer metadata (no full scan)."""
    if not path.exists():
        return None
    return parquet_num_rows(str(path), file_mtime(path))


def count_json_records(path: Path) -> int | None:
    """Count GeoJSON features or JSON array / object rows."""
    if not path.exists():
        return None
    return json_record_count(str(path), file_mtime(path))


def format_row_count(n: int | None, unit: str) -> str:
    """Human-readable count with unit (``— unit`` when unknown)."""
    if n is None:
        return f"— {unit}"
    return f"{n:,} {unit}"


def format_fetched_at(mtime: float | None) -> str:
    """UTC timestamp from filesystem mtime, or em dash."""
    if mtime is None:
        return "—"
    return datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_roots(
    processed_dir: Path,
    cache_dir: Path,
    fixtures_dir: Path,
) -> dict[RootName, Path]:
    return {
        "processed": processed_dir,
        "cache": cache_dir,
        "fixtures": fixtures_dir,
    }


def _inspect_path(path: Path) -> tuple[int | None, float | None]:
    suffix = path.suffix.lower()
    mtime = path.stat().st_mtime
    if suffix == ".parquet":
        return count_parquet_rows(path), mtime
    if suffix in {".json", ".geojson"}:
        return count_json_records(path), mtime
    return None, mtime


def build_source_inventory(
    catalog: tuple[DataSource, ...] | list[DataSource],
    *,
    processed_dir: Path,
    cache_dir: Path,
    fixtures_dir: Path,
) -> list[SourceInventoryRow]:
    """Build equal-structure inventory rows for Data-sources cards."""
    roots = _resolve_roots(processed_dir, cache_dir, fixtures_dir)
    rows: list[SourceInventoryRow] = []
    for src in catalog:
        refs = _SOURCE_ARTIFACTS.get(src.key, ())
        chosen: Path | None = None
        origin = "missing"
        for ref in refs:
            candidate = roots[ref.root] / ref.name
            if candidate.exists():
                chosen = candidate
                origin = ref.root
                break
        n: int | None = None
        mtime: float | None = None
        if chosen is not None:
            n, mtime = _inspect_path(chosen)
        rows.append(
            SourceInventoryRow(
                name=src.name,
                url=src.url,
                role=src.role,
                rows_label=format_row_count(n, src.unit),
                fetched_label=format_fetched_at(mtime),
                origin_label=origin,
            )
        )
    return rows
