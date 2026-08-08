"""Explore page: sidebar filters + Map / Offer / Board tabs."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from happy_housing.app.data import (
    load_amenities,
    load_oou,
    load_scored,
    load_shap_bundle,
)
from happy_housing.app.filters import FilterState, apply_filters
from happy_housing.app.maps import (
    amenity_layers,
    focus_marker_layer,
    listings_scatter_layer,
    oou_polygon_layer,
    render_deck,
)
from happy_housing.models.train import build_feature_matrix

_BOARD_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "title",
    "district",
    "source",
    "area_m2",
    "rooms",
    "price_pln",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "commute_min",
    "in_oou",
    "listed_at",
    "url",
)
"""Column order for the Explore board table when present."""

_LIVABILITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("in_oou", "OOU / aircraft zone"),
    ("commute_min", "Commute to Daszyńskiego (min)"),
    ("dist_transit_stop_m", "Nearest transit stop (m)"),
    ("dist_hospital_m", "Nearest hospital (m)"),
    ("dist_fire_station_m", "Nearest fire station (m)"),
    ("dist_police_m", "Nearest police (m)"),
    ("dist_air_station_m", "Nearest air station (m)"),
    ("dist_gunb_m", "Nearest GUNB permit (m)"),
    ("dist_daszynskiego_m", "Distance to Daszyńskiego (m)"),
)
"""Offer livability rows: (column, label)."""


def _sidebar_filters(frame: pd.DataFrame) -> FilterState:
    """Render Explore filters in the sidebar and return state."""
    st.sidebar.header("Filters")
    districts = sorted(
        frame["district"].dropna().astype(str).unique().tolist()
        if "district" in frame.columns
        else []
    )
    sources = sorted(
        frame["source"].dropna().astype(str).unique().tolist()
        if "source" in frame.columns
        else []
    )
    months = sorted([
        m
        for m in frame
        .get("listed_month", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if m not in {"NaT", "unknown"}
    ])
    ppm_min = int(frame["price_per_m2"].min()) if not frame.empty else 0
    ppm_max = int(frame["price_per_m2"].max()) + 1 if not frame.empty else 1

    district = st.sidebar.selectbox("District", ["(all)", *districts])
    source = st.sidebar.selectbox("Source", ["(all)", *sources])
    oou = st.sidebar.selectbox(
        "OOU / aircraft zone",
        ["(all)", "inside", "outside"],
        help="Chopin restricted-use zone — noise/livability proxy, not measured dB.",
    )
    max_ppm = st.sidebar.slider(
        "Max listing zł/m²",
        min_value=ppm_min,
        max_value=max(ppm_max, ppm_min + 1),
        value=max(ppm_max, ppm_min + 1),
    )
    month = "(all)"
    if months:
        month = st.sidebar.select_slider(
            "Listed month",
            options=["(all)", *months],
            value="(all)",
        )
    return FilterState(
        district=district,
        source=source,
        oou=oou,
        max_price_per_m2=float(max_ppm),
        month=month,
    )


def _tab_map(filtered: pd.DataFrame, oou: gpd.GeoDataFrame | None) -> None:
    """Market map of filtered offers + OOU overlay."""
    st.caption(
        "Colour = listing - model zł/m² (red overpriced, green underpriced). "
        "OOU polygon = aircraft restricted-use proxy."
    )
    if filtered.empty:
        st.warning("No offers match filters.")
        return
    layers = [
        *oou_polygon_layer(oou),
        listings_scatter_layer(filtered),
    ]
    render_deck(
        layers,
        latitude=float(filtered["lat"].mean()),
        longitude=float(filtered["lon"].mean()),
        zoom=10.5,
    )


def _livability_table(row: pd.Series) -> pd.DataFrame:
    """Human-readable livability / amenity distances for one offer."""
    rows: list[tuple[str, object]] = []
    for col, label in _LIVABILITY_FIELDS:
        if col == "in_oou":
            rows.append((label, "Yes" if int(row.get(col) or 0) == 1 else "No"))
        else:
            rows.append((label, row.get(col)))
    return pd.DataFrame(rows, columns=["signal", "value"])


def _tab_offer(
    filtered: pd.DataFrame,
    oou: gpd.GeoDataFrame | None,
    amenities: gpd.GeoDataFrame,
) -> None:
    """Single-offer deep dive: facts, link, SHAP, local map context."""
    if filtered.empty:
        st.warning("No offers match filters.")
        return
    labels = filtered["listing_id"].astype(str).tolist()
    if "title" in filtered.columns:
        options = {
            f"{lid} — {title}" if isinstance(title, str) and title else lid: lid
            for lid, title in zip(
                filtered["listing_id"],
                filtered["title"],
                strict=True,
            )
        }
        choice = st.selectbox("Offer", list(options.keys()))
        selected_id = options[choice]
    else:
        selected_id = st.selectbox("Offer", labels)

    row = filtered.loc[filtered["listing_id"] == selected_id].iloc[0]
    left, right = st.columns([1, 2])
    with left:
        if isinstance(row.get("image_url"), str) and row["image_url"]:
            st.image(row["image_url"], use_container_width=True)
        if isinstance(row.get("url"), str) and row["url"]:
            st.link_button("Open source listing", str(row["url"]))
        st.write({
            "listing_id": row.get("listing_id"),
            "title": row.get("title"),
            "district": row.get("district"),
            "source": row.get("source"),
            "listed_at": row.get("listed_at"),
            "rooms": row.get("rooms"),
            "area_m2": row.get("area_m2"),
            "year_built": row.get("year_built"),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
        })
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Listing zł/m²", f"{float(row['price_per_m2']):.0f}")
        m2.metric("Model zł/m²", f"{float(row['model_price_per_m2']):.0f}")
        m3.metric(
            "Delta (listing - model)",
            f"{float(row['delta_price_per_m2']):.0f}",
        )
        st.subheader("Livability & amenity context")
        st.caption(
            "Distances are metres to nearest known point. "
            "OOU is a zone flag (sound/aircraft risk proxy), not a dB meter."
        )
        st.dataframe(_livability_table(row), hide_index=True, width="stretch")

    st.subheader("Local map")
    lat, lon = float(row["lat"]), float(row["lon"])
    nearby = filtered.head(200)
    layers = [
        *oou_polygon_layer(oou),
        *amenity_layers(amenities),
        listings_scatter_layer(nearby),
        focus_marker_layer(lat, lon),
    ]
    render_deck(layers, latitude=lat, longitude=lon, zoom=13.5)

    st.subheader("Why this price? (SHAP)")
    bundle = load_shap_bundle()
    if bundle is not None:
        try:
            _booster, explainer = bundle
            one = filtered.loc[filtered["listing_id"] == selected_id]
            features = build_feature_matrix(one)
            explanation = explainer(features)
            fig, _ax = plt.subplots(figsize=(8, 4.5))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"SHAP unavailable: {exc}")
    else:
        st.info("Model file missing — retrain from Admin.")


def _tab_board(filtered: pd.DataFrame) -> None:
    """Tabular offer board for the filtered set."""
    cols = [c for c in _BOARD_COLUMNS if c in filtered.columns]
    st.caption(f"{len(filtered)} offers after filters")
    st.dataframe(filtered[cols], width="stretch", hide_index=True)


def render_explore() -> None:
    """Render Explore with sidebar filters and three tabs."""
    st.title("Explore")
    try:
        frame = load_scored()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    state = _sidebar_filters(frame)
    filtered = apply_filters(frame, state)
    oou = load_oou()
    amenities = load_amenities()

    map_tab, offer_tab, board_tab = st.tabs(["Map", "Offer", "Board"])
    with map_tab:
        _tab_map(filtered, oou)
    with offer_tab:
        _tab_offer(filtered, oou, amenities)
    with board_tab:
        _tab_board(filtered)
