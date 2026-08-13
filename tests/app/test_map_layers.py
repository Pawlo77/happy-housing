"""Tests for Explore map layer data prep and hour filters."""

import pandas as pd

from happy_housing.app.map_layers import (
    LayerFlags,
    diurnal_traffic_weight,
    prepare_listings_for_map,
    prepare_shops_for_map,
    prepare_transit_for_hour,
    traffic_weights_for_hour,
)


def test_diurnal_traffic_weight_peaks_commute_hours() -> None:
    """Rush hours outweigh deep night."""
    assert diurnal_traffic_weight(8) > diurnal_traffic_weight(3)
    assert diurnal_traffic_weight(17) > diurnal_traffic_weight(1)


def test_prepare_transit_for_hour_sizes_by_departures() -> None:
    """Stops active at the selected hour keep departure counts for radius."""
    stops = pd.DataFrame({
        "stop_id": ["a", "b", "c"],
        "stop_name": ["Alpha", "Beta", "Gamma"],
        "stop_lat": [52.2, 52.3, 52.25],
        "stop_lon": [21.0, 21.1, 21.05],
    })
    hourly = pd.DataFrame({
        "stop_id": ["a", "a", "b"],
        "hour": [8, 9, 8],
        "n_departures": [40, 10, 5],
    })
    out = prepare_transit_for_hour(stops, hourly, hour=8, min_departures=1)
    assert set(out["stop_id"]) == {"a", "b"}
    assert float(out.loc[out["stop_id"] == "a", "n_departures"].iloc[0]) == 40
    assert "tooltip_name" in out.columns
    assert "tooltip_detail" in out.columns


def test_traffic_weights_scale_with_hour() -> None:
    """Traffic proxy points scale base weight by diurnal curve."""
    sample = pd.DataFrame({
        "lon": [21.0, 21.1],
        "lat": [52.2, 52.3],
        "base_weight": [1.0, 2.0],
        "name": ["road-a", "road-b"],
    })
    night = traffic_weights_for_hour(sample, hour=3)
    peak = traffic_weights_for_hour(sample, hour=8)
    assert peak["weight"].sum() > night["weight"].sum()
    assert list(peak.columns) == ["lon", "lat", "weight"]


def test_prepare_listings_and_shops_expose_tooltip_fields() -> None:
    """Map payloads expose hover fields without tiny-dot-only markers."""
    listings = prepare_listings_for_map(
        pd.DataFrame({
            "listing_id": ["x"],
            "district": ["wola"],
            "source": ["otodom"],
            "lat": [52.2],
            "lon": [21.0],
            "delta_price_per_m2": [-100.0],
            "price_per_m2": [15000.0],
            "model_price_per_m2": [15100.0],
            "price_pln": [750000.0],
            "area_m2": [50.0],
            "rooms": [2.0],
            "year_built": [2005.0],
            "commute_min": [18.0],
            "transit_commute_min": [32.0],
        })
    )
    assert listings.iloc[0]["tooltip_name"] == "x"
    assert listings.iloc[0]["listing_id"] == "x"
    assert "radius" in listings.columns
    detail = str(listings.iloc[0]["tooltip_detail"])
    assert "wola" in detail
    assert "<br/>" in detail
    assert "Transit 32 min" in detail
    assert "zł/m²" in detail

    car = prepare_listings_for_map(
        pd.DataFrame({
            "listing_id": ["x"],
            "district": ["wola"],
            "source": ["otodom"],
            "lat": [52.2],
            "lon": [21.0],
            "delta_price_per_m2": [-100.0],
            "price_per_m2": [15000.0],
            "model_price_per_m2": [15100.0],
            "price_pln": [750000.0],
            "area_m2": [50.0],
            "rooms": [2.0],
            "year_built": [2005.0],
            "commute_min": [18.0],
            "transit_commute_min": [32.0],
        }),
        commute_mode="car",
    )
    assert "Car 18 min" in str(car.iloc[0]["tooltip_detail"])

    shops = prepare_shops_for_map(
        pd.DataFrame({
            "name": ["Biedronka"],
            "shop": ["supermarket"],
            "lon": [21.0],
            "lat": [52.2],
        })
    )
    assert "supermarket" in shops.iloc[0]["tooltip_detail"]


def test_prepare_listings_boosts_recommended_radius() -> None:
    """Recommended offer pins are larger and labelled on the map."""
    frame = pd.DataFrame({
        "listing_id": ["std", "rec"],
        "district": ["wola", "wola"],
        "source": ["otodom", "otodom"],
        "lat": [52.2, 52.21],
        "lon": [21.0, 21.01],
        "delta_price_per_m2": [-100.0, -100.0],
        "price_per_m2": [15000.0, 15000.0],
        "model_price_per_m2": [15100.0, 15100.0],
        "price_pln": [750000.0, 750000.0],
        "area_m2": [50.0, 50.0],
        "rooms": [2.0, 2.0],
        "year_built": [2005.0, 2005.0],
        "commute_min": [18.0, 18.0],
    })
    out = prepare_listings_for_map(frame, recommended_ids=frozenset({"rec"}))
    std = out.loc[out["listing_id"] == "std"].iloc[0]
    rec = out.loc[out["listing_id"] == "rec"].iloc[0]
    assert float(rec["radius"]) > float(std["radius"])
    assert bool(rec["is_recommended"]) is True
    assert bool(std["is_recommended"]) is False
    assert "Recommended" in str(rec["tooltip_detail"])


def test_prepare_listings_drops_missing_coordinates() -> None:
    listings = prepare_listings_for_map(
        pd.DataFrame({
            "listing_id": ["ok", "bad"],
            "district": ["wola", "wola"],
            "lat": [52.2, float("nan")],
            "lon": [21.0, 21.1],
            "delta_price_per_m2": [-100.0, 50.0],
            "price_per_m2": [15000.0, 16000.0],
            "area_m2": [50.0, 55.0],
        })
    )
    assert list(listings["tooltip_name"]) == ["ok"]


def test_layer_flags_default_offers_on() -> None:
    """Offers stay visible by default; dense overlays start off."""
    flags = LayerFlags()
    assert flags.offers is True
    assert flags.city_boundary is False
    assert flags.district_boundaries is False
    assert flags.noise_overlay == "off"
    assert flags.military is False
    assert flags.rail_areas is False
    assert flags.airport is False
    assert flags.transit_intensity is False
    assert flags.transit_show_metro is False
    assert flags.shop_show_grocery is False
    assert flags.traffic is False
    assert flags.any_transit_network() is False
    assert flags.any_shops() is False
