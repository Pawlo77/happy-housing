"""Background detail-page backfill for listing sources that split search vs detail."""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final

import pandas as pd

from happy_housing.ingest.nol import (
    backfill_nol_details,
    row_needs_nol_detail_backfill,
)
from happy_housing.ingest.otodom import (
    backfill_otodom_details,
    row_needs_detail_backfill,
)

logger = logging.getLogger(__name__)

DetailSourceName = str
"""Registered detail-backfill source key (``otodom``, ``nol``, …)."""

_OTODOM_PARQUET: Final[str] = "otodom_listings.parquet"
"""Otodom listings artifact under ``data/processed``."""

_NOL_PARQUET: Final[str] = "nol_listings.parquet"
"""Nieruchomości Online listings artifact under ``data/processed``."""

_PendingPred = Callable[[dict[str, Any]], bool]
"""Row predicate for pending detail backfill."""


def otodom_listings_path(processed_dir: Path) -> Path:
    """Path to the Otodom listings parquet."""
    return processed_dir / _OTODOM_PARQUET


def nol_listings_path(processed_dir: Path) -> Path:
    """Path to the Nieruchomości Online listings parquet."""
    return processed_dir / _NOL_PARQUET


def _iter_pending_rows(path: Path, needs: _PendingPred) -> Iterator[dict[str, Any]]:
    """Yield stored rows that still need a detail fetch."""
    if not path.exists():
        return
    frame = pd.read_parquet(path)
    for row in frame.to_dict(orient="records"):
        if needs(row):
            yield row


def count_pending_detail_backfill(processed_dir: Path) -> dict[DetailSourceName, int]:
    """Count listings still missing detail fields per registered source."""
    return {
        "otodom": sum(
            1
            for _ in _iter_pending_rows(
                otodom_listings_path(processed_dir),
                row_needs_detail_backfill,
            )
        ),
        "nol": sum(
            1
            for _ in _iter_pending_rows(
                nol_listings_path(processed_dir),
                row_needs_nol_detail_backfill,
            )
        ),
    }


def pending_detail_backfill_rows(
    parquet_path: Path,
    *,
    source: DetailSourceName = "otodom",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` pending rows for ``source``."""
    if source == "otodom":
        needs: _PendingPred = row_needs_detail_backfill
    elif source == "nol":
        needs = row_needs_nol_detail_backfill
    else:
        return []
    pending: list[dict[str, Any]] = []
    for row in _iter_pending_rows(parquet_path, needs):
        pending.append(row)
        if limit is not None and len(pending) >= limit:
            break
    return pending


def _run_source_batch(
    *,
    source: DetailSourceName,
    before: int,
    run: Callable[[], None],
    after_count: Callable[[], int],
) -> dict[str, int]:
    """Run one source backfill when pending rows exist."""
    if before <= 0:
        logger.info("Detail backfill batch: %s pending=0", source)
        return {
            f"{source}_pending_before": 0,
            f"{source}_pending_after": 0,
            f"{source}_updated": 0,
        }
    run()
    after = after_count()
    logger.info("Detail backfill batch: %s pending %s→%s", source, before, after)
    return {
        f"{source}_pending_before": before,
        f"{source}_pending_after": after,
        f"{source}_updated": max(0, before - after),
    }


def run_detail_backfill_batch(
    processed_dir: Path,
    *,
    batch_size: int = 100,
    detail_workers: int = 1,
    sleep_s: float = 1.0,
) -> dict[str, int]:
    """Run one detail-backfill batch for all registered sources."""
    counts = count_pending_detail_backfill(processed_dir)
    stats: dict[str, int] = {}
    stats.update(
        _run_source_batch(
            source="otodom",
            before=counts.get("otodom", 0),
            run=lambda: backfill_otodom_details(
                otodom_listings_path(processed_dir),
                max_listings=batch_size,
                detail_workers=detail_workers,
                sleep_s=sleep_s,
            ),
            after_count=lambda: count_pending_detail_backfill(processed_dir).get(
                "otodom", 0
            ),
        )
    )
    stats.update(
        _run_source_batch(
            source="nol",
            before=counts.get("nol", 0),
            run=lambda: backfill_nol_details(
                nol_listings_path(processed_dir),
                max_listings=batch_size,
                detail_workers=detail_workers,
                sleep_s=sleep_s,
            ),
            after_count=lambda: count_pending_detail_backfill(processed_dir).get(
                "nol", 0
            ),
        )
    )
    return stats
