"""Tests for pipeline run metadata persistence."""

from __future__ import annotations

from pathlib import Path

from happy_housing.app.metadata import (
    RunMetadata,
    load_run_metadata,
    write_run_metadata,
)


def test_write_and_load_run_metadata(tmp_path: Path) -> None:
    """Round-trip metadata JSON used by the Home page."""
    path = tmp_path / "run_metadata.json"
    meta = RunMetadata(
        refreshed_at="2026-08-08T10:00:00+00:00",
        duration_s=12.5,
        n_offers=100,
        n_by_source={"otodom": 60, "rcn": 40},
        model_metrics={"mae": 3100.0, "r2": 0.62, "n_train": 80.0, "n_test": 20.0},
        model_path="lightgbm_price_per_m2.txt",
        feature_columns=["area_m2", "rooms"],
        sources=["Otodom", "UM Warszawa RCN", "OSM"],
        methodology=(
            "LightGBM regresses zł/m² from livability distances and property traits."
        ),
    )
    write_run_metadata(path, meta)
    loaded = load_run_metadata(path)
    assert loaded is not None
    assert loaded.n_offers == 100
    assert loaded.model_metrics["r2"] == 0.62
    assert loaded.n_by_source["otodom"] == 60
    assert "LightGBM" in loaded.methodology


def test_load_missing_metadata_returns_none(tmp_path: Path) -> None:
    """Missing file yields None instead of raising."""
    assert load_run_metadata(tmp_path / "missing.json") is None
