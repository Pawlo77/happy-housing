"""Tests for optional scraper adapter seam."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from happy_housing.ingest.scraper_adapter import scrape_with_adapter


def test_scrape_with_adapter_uses_injected_fn(tmp_path: Path) -> None:
    out = tmp_path / "listings.parquet"

    def fake_scrape(
        path: Path,
        *,
        max_pages: int | None = None,
        max_listings: int | None = None,
        incremental: bool = True,
    ) -> pd.DataFrame:
        del max_pages, max_listings, incremental
        frame = pd.DataFrame({"listing_id": ["X"], "price_pln": [1.0]})
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    frame = scrape_with_adapter(out, scrape_fn=fake_scrape, max_pages=1)
    assert out.exists()
    assert list(frame["listing_id"]) == ["X"]
