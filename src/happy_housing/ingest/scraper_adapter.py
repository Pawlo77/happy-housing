"""Optional listing-scraper adapter seam (paid scrapers stay out of core)."""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from happy_housing.ingest.otodom import scrape_otodom_warsaw

logger = logging.getLogger(__name__)

ScrapeFn = Callable[..., pd.DataFrame]
"""Callable matching ``scrape_otodom_warsaw`` kwargs."""


def default_scrape_fn() -> ScrapeFn:
    """Return the built-in Otodom HTML client (no paid API)."""
    return scrape_otodom_warsaw


def scrape_with_adapter(
    output_parquet: Path,
    *,
    scrape_fn: ScrapeFn | None = None,
    max_pages: int | None = None,
    max_listings: int | None = None,
    incremental: bool = True,
) -> pd.DataFrame:
    """Run a scrape function behind a narrow adapter boundary.

    Swap ``scrape_fn`` for a ScrapingAnt (or other) client without changing
    pipeline call sites. Default is the free Otodom HTML scraper.
    """
    fn = scrape_fn or default_scrape_fn()
    logger.info("Scraping listings via adapter %s", getattr(fn, "__name__", fn))
    return fn(
        output_parquet,
        max_pages=max_pages,
        max_listings=max_listings,
        incremental=incremental,
    )
