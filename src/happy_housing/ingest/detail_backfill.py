"""Background detail-page backfill for listing sources that split search vs detail."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pandas as pd

from happy_housing.ingest.otodom import (
    backfill_otodom_details,
    row_needs_detail_backfill,
)

logger = logging.getLogger(__name__)

DetailSourceName = str
"""Registered detail-backfill source key (``otodom``, …)."""

_OTODOM_PARQUET: Final[str] = "otodom_listings.parquet"
"""Otodom listings artifact under ``data/processed``."""


def otodom_listings_path(processed_dir: Path) -> Path:
    """Path to the Otodom listings parquet."""
    return processed_dir / _OTODOM_PARQUET


def _iter_otodom_pending_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield Otodom rows still missing detail fields."""
    if not path.exists():
        return
    frame = pd.read_parquet(path)
    for row in frame.to_dict(orient="records"):
        if row_needs_detail_backfill(row):
            yield row


def count_pending_detail_backfill(processed_dir: Path) -> dict[DetailSourceName, int]:
    """Count listings still missing detail fields per registered source."""
    otodom_path = otodom_listings_path(processed_dir)
    return {"otodom": sum(1 for _ in _iter_otodom_pending_rows(otodom_path))}


def pending_detail_backfill_rows(
    parquet_path: Path,
    *,
    source: DetailSourceName = "otodom",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` pending rows for ``source``."""
    if source != "otodom":
        return []
    pending: list[dict[str, Any]] = []
    for row in _iter_otodom_pending_rows(parquet_path):
        pending.append(row)
        if limit is not None and len(pending) >= limit:
            break
    return pending


def run_detail_backfill_batch(
    processed_dir: Path,
    *,
    batch_size: int = 100,
    detail_workers: int = 1,
    sleep_s: float = 1.0,
) -> dict[str, int]:
    """Run one detail-backfill batch for all registered sources."""
    stats: dict[str, int] = {}
    before = count_pending_detail_backfill(processed_dir).get("otodom", 0)
    if before <= 0:
        logger.info("Detail backfill batch: otodom pending=0")
        stats["otodom_pending_before"] = 0
        stats["otodom_pending_after"] = 0
        stats["otodom_updated"] = 0
        return stats
    otodom_path = otodom_listings_path(processed_dir)
    backfill_otodom_details(
        otodom_path,
        max_listings=batch_size,
        detail_workers=detail_workers,
        sleep_s=sleep_s,
    )
    after = count_pending_detail_backfill(processed_dir).get("otodom", 0)
    stats["otodom_pending_before"] = before
    stats["otodom_pending_after"] = after
    stats["otodom_updated"] = max(0, before - after)
    logger.info(
        "Detail backfill batch: otodom pending %s→%s",
        before,
        after,
    )
    return stats
