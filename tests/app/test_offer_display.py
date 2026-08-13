"""Tests for Explore Offer display helpers."""

import pandas as pd

from happy_housing.app.offer_display import (
    delta_band,
    delta_hex,
    flag_yes_no,
    format_meters,
    livability_rows,
    offer_fact_rows,
    round_meters,
)


def test_round_and_format_meters() -> None:
    assert round_meters(1234.6) == 1235
    assert format_meters(1234.6) == "1,235 m"
    assert format_meters(None) == "—"
    assert format_meters(float("nan")) == "—"


def test_flag_yes_no_handles_nan() -> None:
    assert flag_yes_no(1) == "Yes"
    assert flag_yes_no(0) == "No"
    assert flag_yes_no(float("nan")) == "No"
    assert flag_yes_no(None) == "No"


def test_delta_band_colours() -> None:
    assert delta_band(-800.0) == "green"
    assert delta_band(100.0) == "yellow"
    assert delta_band(900.0) == "red"
    assert delta_hex(-800.0).startswith("#")


def test_livability_rows_round_distances() -> None:
    row = pd.Series({
        "aircraft_lden_db": 60.0,
        "aircraft_lnight_db": 0.0,
        "in_oou": 1,
        "commute_min": 12.4,
        "transit_commute_min": 22.0,
        "bike_commute_min": 18.0,
        "walk_commute_min": 55.0,
        "dist_transit_stop_m": 371.9,
        "dist_hospital_m": 8248.5,
        "dist_fire_station_m": 1881.2,
        "dist_police_m": 2177.9,
        "dist_air_station_m": 1608.4,
        "dist_gunb_m": 9516.5,
        "dist_daszynskiego_m": 9206.0,
    })
    frame = livability_rows(row)
    transit = frame.loc[frame["signal"] == "Nearest transit stop", "value"].iloc[0]
    assert transit == "372 m"
    assert "Yes" in frame["value"].astype(str).tolist()
    lden = frame.loc[frame["signal"] == "Aircraft Lden (dB)", "value"].iloc[0]
    assert lden == "≥ 60 dB"
    primary = frame.loc[
        frame["signal"].str.startswith("Public transport to"), "value"
    ].iloc[0]
    assert primary == "22 min"


def test_offer_fact_rows_survive_nan_flags() -> None:
    row = pd.Series({
        "listing_id": "OTO-1",
        "has_balcony": float("nan"),
        "has_garage": float("nan"),
        "has_lift": 1,
        "is_private_owner": float("nan"),
        "area_m2": 40.0,
    })
    frame = offer_fact_rows(row)
    assert not frame.empty
    balcony = frame.loc[frame["field"] == "Balcony", "value"].iloc[0]
    assert balcony == "No"


def test_offer_fact_rows_value_column_is_arrow_safe_strings() -> None:
    """Mixed numpy ints / floats / None must not stay as object for st.dataframe."""
    import pyarrow as pa

    row = pd.Series({
        "listing_id": "OTO-1",
        "title": "Flat",
        "rooms": pd.Series([2], dtype="int64").iloc[0],
        "floor_no": pd.Series([3], dtype="int64").iloc[0],
        "area_m2": 39.0,
        "price_pln": 884_500.0,
        "lat": 52.24,
        "lon": 21.01,
        "market": None,
        "has_lift": 1,
    })
    frame = offer_fact_rows(row)
    assert frame["value"].map(type).eq(str).all()
    # Round-trip through Arrow — the failure mode reported by Streamlit.
    pa.Table.from_pandas(frame)
