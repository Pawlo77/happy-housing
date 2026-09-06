"""Tests for photo feature aggregation."""

import json

import pandas as pd

from happy_housing.vision.photo_features import (
    PHOTO_FEATURE_COLUMNS,
    HeuristicPhotoEncoder,
    aggregate_photo_features,
    attach_photo_features,
)


def test_aggregate_photo_features_has_expected_keys() -> None:
    feats = aggregate_photo_features(
        image_urls=["https://example.com/a.jpg", "https://example.com/b.jpg"],
        floor_plan_urls=["https://example.com/plan.png"],
        encoder=HeuristicPhotoEncoder(),
        max_photos=4,
    )
    assert set(PHOTO_FEATURE_COLUMNS) <= set(feats)
    assert feats["n_photos_scored"] == 2.0
    assert feats["has_floor_plan"] == 1.0


def test_attach_photo_features_frame() -> None:
    frame = pd.DataFrame([
        {
            "listing_id": "OTO-1",
            "image_urls_json": json.dumps(["https://x/a.jpg"]),
            "floor_plan_urls_json": "[]",
        }
    ])
    out = attach_photo_features(frame, encoder=HeuristicPhotoEncoder())
    assert out.iloc[0]["n_photos_scored"] == 1.0
    assert "photo_quality_mean" in out.columns


def test_attach_photo_features_skips_already_scored_rows() -> None:
    frame = pd.DataFrame([
        {
            "listing_id": "old",
            "image_urls_json": json.dumps(["https://x/old.jpg"]),
            "photo_quality_mean": 0.4,
            "n_photos_scored": 3.0,
        },
        {
            "listing_id": "new",
            "image_urls_json": json.dumps(["https://x/new.jpg"]),
            "floor_plan_urls_json": "[]",
        },
    ])
    out = attach_photo_features(frame, encoder=HeuristicPhotoEncoder())
    assert float(out.iloc[0]["photo_quality_mean"]) == 0.4
    assert float(out.iloc[0]["n_photos_scored"]) == 3.0
    assert pd.notna(out.iloc[1]["photo_quality_mean"])
    assert float(out.iloc[1]["n_photos_scored"]) == 1.0
