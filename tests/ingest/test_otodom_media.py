"""Otodom multi-photo URL extraction and media backfill triggers."""

from __future__ import annotations

import json

import pytest

from happy_housing.ingest.otodom import (
    card_needs_detail_refresh,
    extract_image_urls,
    flatten_otodom_ad,
    listing_needs_media_backfill,
    listing_photo_count,
)


def test_extract_image_urls_keeps_gallery_order() -> None:
    urls = extract_image_urls({
        "images": [
            {"medium": "https://cdn.example/a.jpg"},
            {"large": "https://cdn.example/b.jpg"},
            {"medium": "https://cdn.example/a.jpg"},
            {"small": "https://cdn.example/c.jpg"},
        ]
    })
    assert urls == [
        "https://cdn.example/a.jpg",
        "https://cdn.example/b.jpg",
        "https://cdn.example/c.jpg",
    ]


def test_flatten_otodom_ad_persists_image_urls_json() -> None:
    ad = {
        "location": {"coordinates": {"latitude": 52.2, "longitude": 21.0}},
        "images": [
            {"medium": "https://cdn.example/1.jpg"},
            {"medium": "https://cdn.example/2.jpg"},
        ],
        "floorPlans": [{"medium": "https://cdn.example/plan.png"}],
    }
    flat = flatten_otodom_ad(ad)
    assert flat["image_url"] == "https://cdn.example/1.jpg"
    assert json.loads(flat["image_urls_json"]) == [
        "https://cdn.example/1.jpg",
        "https://cdn.example/2.jpg",
    ]
    assert json.loads(flat["floor_plan_urls_json"]) == ["https://cdn.example/plan.png"]


def test_listing_photo_count_prefers_urls_json() -> None:
    row = {
        "image_url": "https://cdn.example/cover.jpg",
        "image_urls_json": json.dumps([
            "https://cdn.example/1.jpg",
            "https://cdn.example/2.jpg",
        ]),
    }
    assert listing_photo_count(row) == 2
    assert listing_photo_count({"image_url": "https://cdn.example/x.jpg"}) == 1
    assert listing_photo_count({}) == 0


def test_listing_needs_media_backfill_when_gallery_json_absent() -> None:
    assert listing_needs_media_backfill({"image_url": "https://x"}) is True
    assert listing_needs_media_backfill({"image_urls_json": None}) is True
    assert listing_needs_media_backfill({"image_urls_json": ""}) is True
    assert (
        listing_needs_media_backfill({
            "image_urls_json": json.dumps(["https://a", "https://b"]),
        })
        is False
    )
    # Single-URL JSON still means detail was already resolved.
    assert (
        listing_needs_media_backfill({
            "image_urls_json": json.dumps(["https://only"]),
        })
        is False
    )


def test_backfill_skips_marking_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403/network fails must stay pending; 410 may close with empty gallery."""
    import pandas as pd
    import requests

    from happy_housing.ingest.otodom import backfill_otodom_media

    frame = pd.DataFrame([
        {
            "listing_id": "A",
            "slug": "gone-listing",
            "image_url": "https://cdn.example/a.jpg",
        },
        {
            "listing_id": "B",
            "slug": "blocked-listing",
            "image_url": "https://cdn.example/b.jpg",
        },
        {
            "listing_id": "C",
            "slug": "ok-listing",
            "image_url": "https://cdn.example/c.jpg",
        },
    ])

    def fake_detail(session: object, slug: str, **kwargs: object) -> dict[str, object]:
        _ = session, kwargs
        if slug == "gone-listing":
            resp = requests.Response()
            resp.status_code = 410
            raise requests.HTTPError("Gone", response=resp)
        if slug == "blocked-listing":
            resp = requests.Response()
            resp.status_code = 403
            raise requests.HTTPError("Forbidden", response=resp)
        return {
            "image_url": "https://cdn.example/c2.jpg",
            "image_urls_json": json.dumps([
                "https://cdn.example/c2.jpg",
                "https://cdn.example/c3.jpg",
            ]),
            "floor_plan_urls_json": "[]",
            "description": "",
        }

    monkeypatch.setattr("happy_housing.ingest.otodom.fetch_listing_detail", fake_detail)
    monkeypatch.setattr("happy_housing.ingest.otodom.time.sleep", lambda _s: None)
    out = backfill_otodom_media(frame, detail_workers=1, sleep_s=0.0)
    assert out.loc[0, "image_urls_json"] == "[]"  # 410 → closed
    assert listing_needs_media_backfill(out.loc[1].to_dict()) is True  # 403 pending
    assert json.loads(str(out.loc[2, "image_urls_json"])) == [
        "https://cdn.example/c2.jpg",
        "https://cdn.example/c3.jpg",
    ]


def test_card_needs_detail_refresh_when_prior_lacks_gallery_json() -> None:
    card = {"listing_id": "OTO-1", "price_pln": 100.0}
    prior = {
        "OTO-1": {
            "listing_id": "OTO-1",
            "lat": 52.2,
            "lon": 21.0,
            "price_pln": 100.0,
            "image_url": "https://cdn.example/cover.jpg",
        }
    }
    assert card_needs_detail_refresh(card, prior) is True
    prior["OTO-1"]["image_urls_json"] = json.dumps([
        "https://cdn.example/1.jpg",
        "https://cdn.example/2.jpg",
    ])
    assert card_needs_detail_refresh(card, prior) is False
