"""Tests for Offer media deferral helpers."""

import pandas as pd

from happy_housing.app.page_explore import (
    offer_needs_live_media_fetch,
)


def test_offer_needs_live_media_only_when_cover_and_desc_missing() -> None:
    bare = pd.Series({"listing_id": "a", "slug": "x", "description": ""})
    assert offer_needs_live_media_fetch(bare) is True

    cover = pd.Series({
        "listing_id": "a",
        "slug": "x",
        "description": "",
        "image_url": "https://example.com/p.jpg",
    })
    assert offer_needs_live_media_fetch(cover) is False

    desc = pd.Series({
        "listing_id": "a",
        "slug": "x",
        "description": "Nice flat",
        "image_urls_json": "[]",
    })
    assert offer_needs_live_media_fetch(desc) is False


def test_offer_needs_live_media_false_when_gallery_already_rich() -> None:
    rich = pd.Series({
        "listing_id": "a",
        "slug": "x",
        "description": "Nice",
        "image_urls_json": '["https://a","https://b"]',
        "floor_plan_urls_json": '["https://plan"]',
    })
    assert offer_needs_live_media_fetch(rich) is False
