"""Cross-portal listing deduplication tests."""

import pandas as pd

from happy_housing.ingest.listing_dedup import dedupe_cross_source_listings


def _row(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "listing_id": "X-1",
        "source": "otodom",
        "lat": 52.2299,
        "lon": 21.0123,
        "price_pln": 850_000.0,
        "area_m2": 52.0,
        "rooms": 2,
        "district": "Mokotów",
        "listed_at": "2026-08-01",
        "image_url": None,
        "floor_no": 3,
    }
    base.update(kwargs)
    return base


def test_dedupe_same_geo_price_area_rooms_keeps_otodom() -> None:
    """Otodom wins over OLX when geo + economics match."""
    frame = pd.DataFrame([
        _row(
            listing_id="OTO-1",
            source="otodom",
            title="Mokotów 52 m²",
        ),
        _row(
            listing_id="OLX-9",
            source="olx",
            title="OLX copy",
            lat=52.22991,
            lon=21.01231,
        ),
    ])
    out = dedupe_cross_source_listings(frame)
    assert len(out) == 1
    assert out.iloc[0]["listing_id"] == "OTO-1"


def test_dedupe_keeps_nol_over_olx() -> None:
    """NOL wins over OLX when geo + economics match."""
    frame = pd.DataFrame([
        _row(
            listing_id="NOL-1",
            source="nol",
            title="NOL Mokotów",
        ),
        _row(
            listing_id="OLX-9",
            source="olx",
            title="OLX copy",
            lat=52.22991,
            lon=21.01231,
        ),
    ])
    out = dedupe_cross_source_listings(frame)
    assert len(out) == 1
    assert out.iloc[0]["listing_id"] == "NOL-1"


def test_dedupe_same_image_url_without_coords() -> None:
    """Shared cover photo links search-only Otodom to geocoded OLX."""
    image = "https://ireland.apollo.olxcdn.com/v1/files/abc/image.jpg"
    frame = pd.DataFrame([
        _row(
            listing_id="OTO-2",
            source="otodom",
            lat=None,
            lon=None,
            image_url=image,
        ),
        _row(
            listing_id="OLX-2",
            source="olx",
            image_url=f"{image}?rule=legacy",
        ),
    ])
    out = dedupe_cross_source_listings(frame)
    assert len(out) == 1
    assert out.iloc[0]["listing_id"] == "OTO-2"


def test_dedupe_leaves_rcn_and_distinct_buyer_listings() -> None:
    """RCN rows and genuinely different asks stay in the universe."""
    frame = pd.DataFrame([
        _row(listing_id="OTO-10", source="otodom", price_pln=900_000.0, area_m2=55.0),
        _row(
            listing_id="OLX-10",
            source="olx",
            price_pln=720_000.0,
            area_m2=48.0,
            lat=52.24,
            lon=21.01,
        ),
        {
            "listing_id": "RCN-1",
            "source": "rcn",
            "lat": 52.2299,
            "lon": 21.0123,
            "price_pln": 800_000.0,
            "area_m2": 52.0,
            "rooms": 2,
            "district": "mokotow",
            "listed_at": "2020-01-01",
        },
    ])
    out = dedupe_cross_source_listings(frame)
    assert len(out) == 3
    assert set(out["listing_id"]) == {"OTO-10", "OLX-10", "RCN-1"}


def test_dedupe_same_source_rows_unchanged() -> None:
    """Within-source duplicates are handled upstream by listing_id."""
    frame = pd.DataFrame([
        _row(listing_id="OLX-1", source="olx", price_pln=600_000.0),
        _row(
            listing_id="OLX-2",
            source="olx",
            price_pln=600_000.0,
            lat=52.2299,
            lon=21.0123,
        ),
    ])
    out = dedupe_cross_source_listings(frame)
    assert len(out) == 2
