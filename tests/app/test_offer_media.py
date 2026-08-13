"""Tests for safe listing HTML and Otodom media helpers."""

import json

import pandas as pd

from happy_housing.app.offer_display import (
    floor_plan_urls_from_row,
    image_urls_from_row,
    sanitize_listing_html,
)
from happy_housing.ingest.otodom import (
    extract_floor_plan_urls,
    extract_image_urls,
    flatten_otodom_ad,
)


def test_sanitize_listing_html_keeps_basic_markup_strips_script() -> None:
    raw = (
        "<p>Hello <b>world</b></p><script>alert(1)</script>"
        '<a href="https://example.com" onclick="x">link</a>'
        '<img src="https://evil.test/x.png" onerror="x">'
    )
    clean = sanitize_listing_html(raw)
    assert "<b>world</b>" in clean
    assert "Hello" in clean
    assert "<script" not in clean.lower()
    assert "onclick" not in clean.lower()
    assert "<img" not in clean.lower()
    assert 'href="https://example.com"' in clean


def test_sanitize_listing_html_plain_text_passthrough() -> None:
    assert "Just text" in sanitize_listing_html("Just text")


def test_extract_image_and_floor_plan_urls() -> None:
    ad = {
        "images": [
            {"medium": "https://cdn.test/a.jpg", "large": "https://cdn.test/a-l.jpg"},
            {"medium": "https://cdn.test/b.jpg"},
        ],
        "floorPlans": ["https://cdn.test/plan.png"],
    }
    assert extract_image_urls(ad) == [
        "https://cdn.test/a.jpg",
        "https://cdn.test/b.jpg",
    ]
    assert extract_floor_plan_urls(ad) == ["https://cdn.test/plan.png"]
    flat = flatten_otodom_ad({
        **ad,
        "location": {"coordinates": {"latitude": 52.2, "longitude": 21.0}},
        "description": "<p>ok</p>",
    })
    assert flat["image_url"] == "https://cdn.test/a.jpg"
    assert json.loads(flat["image_urls_json"]) == [
        "https://cdn.test/a.jpg",
        "https://cdn.test/b.jpg",
    ]
    assert json.loads(flat["floor_plan_urls_json"]) == ["https://cdn.test/plan.png"]


def test_image_urls_from_row_prefers_json_then_single() -> None:
    row = pd.Series({
        "image_urls_json": '["https://a","https://b"]',
        "image_url": "https://a",
    })
    assert image_urls_from_row(row) == ["https://a", "https://b"]
    row2 = pd.Series({"image_url": "https://only"})
    assert image_urls_from_row(row2) == ["https://only"]
    assert floor_plan_urls_from_row(
        pd.Series({"floor_plan_urls_json": '["https://p"]'})
    ) == ["https://p"]
