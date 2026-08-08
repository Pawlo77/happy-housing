"""Tests for LightGBM training and SHAP shapes."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from happy_housing.config import ModelConfig
from happy_housing.gis import enrich_listings
from happy_housing.ingest import load_listings_json
from happy_housing.models.train import (
    FEATURE_COLUMNS,
    shap_explanation_for_row,
    train_price_model,
)


def _enriched_frame(tmp_path: Path) -> pd.DataFrame:
    listings = load_listings_json(Path("data/fixtures/listings.json"))
    amenities = gpd.GeoDataFrame(
        {"amenity": ["hospital", "fire_station", "police"]},
        geometry=gpd.points_from_xy(
            [21.0, 20.98, 21.02],
            [52.22, 52.21, 52.23],
        ),
        crs="EPSG:4326",
    )
    gios = pd.DataFrame({
        "station_id": [1],
        "lat": [52.23],
        "lon": [21.01],
        "name": ["Test"],
        "city": ["Warszawa"],
    })
    stops = pd.DataFrame({"stop_id": ["A"], "stop_lat": [52.23], "stop_lon": [21.00]})
    enriched = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        oou_path=tmp_path / "oou.geojson",
        amenities=amenities,
        gios_stations=gios,
        gtfs_stops=stops,
        commute_lon=20.983,
        commute_lat=52.23,
    )
    return pd.DataFrame(enriched.drop(columns=["geometry"], errors="ignore"))


def test_train_and_shap(tmp_path: Path) -> None:
    """Model trains on enriched fixtures and SHAP returns feature-length values."""
    frame = _enriched_frame(tmp_path)
    result = train_price_model(
        frame,
        ModelConfig(n_estimators=50, num_leaves=15, test_size=0.3),
    )
    assert result.metrics["n_train"] > 0
    assert result.metrics["mae"] >= 0.0
    explanation = shap_explanation_for_row(result.model, frame, row_index=0)
    assert len(explanation.values) == len(FEATURE_COLUMNS)
