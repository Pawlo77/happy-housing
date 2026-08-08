"""Explore page: sidebar filters + Map / Offer / Board tabs."""

from __future__ import annotations

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from happy_housing.app.data import (
    load_amenities,
    load_oou,
    load_scored,
    model_path,
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


def _tab_map(filtered: pd.DataFrame) -> None:
    """Market map of filtered offers + OOU overlay."""
    st.caption(
        "Colour = listing - model zł/m² (red overpriced, green underpriced). "
        "OOU polygon = aircraft restricted-use proxy."
    )
    if filtered.empty:
        st.warning("No offers match filters.")
        return
    layers = [
        *oou_polygon_layer(load_oou()),
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
    rows = [
        ("OOU / aircraft zone", "Yes" if int(row.get("in_oou") or 0) == 1 else "No"),
        ("Commute to Daszyńskiego (min)", row.get("commute_min")),
        ("Nearest transit stop (m)", row.get("dist_transit_stop_m")),
        ("Nearest hospital (m)", row.get("dist_hospital_m")),
        ("Nearest fire station (m)", row.get("dist_fire_station_m")),
        ("Nearest police (m)", row.get("dist_police_m")),
        ("Nearest air station (m)", row.get("dist_air_station_m")),
        ("Nearest GUNB permit (m)", row.get("dist_gunb_m")),
        ("Distance to Daszyńskiego (m)", row.get("dist_daszynskiego_m")),
    ]
    return pd.DataFrame(rows, columns=["signal", "value"])


def _tab_offer(filtered: pd.DataFrame) -> None:
    """Single-offer deep dive: facts, link, SHAP, local map context."""
    if filtered.empty:
        st.warning("No offers match filters.")
        return
    labels = filtered["listing_id"].astype(str).tolist()
    # Prefer titles in the selector when available.
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
    nearby = filtered.copy()
    layers = [
        *oou_polygon_layer(load_oou()),
        *amenity_layers(load_amenities()),
        listings_scatter_layer(nearby.head(200)),
        focus_marker_layer(lat, lon),
    ]
    render_deck(layers, latitude=lat, longitude=lon, zoom=13.5)

    st.subheader("Why this price? (SHAP)")
    path = model_path()
    if path.exists():
        try:
            booster = lgb.Booster(model_file=str(path))
            one = filtered.loc[filtered["listing_id"] == selected_id]
            features = build_feature_matrix(one)
            explainer = shap.TreeExplainer(booster)
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
    cols = [
        c
        for c in (
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
        if c in filtered.columns
    ]
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

    map_tab, offer_tab, board_tab = st.tabs(["Map", "Offer", "Board"])
    with map_tab:
        _tab_map(filtered)
    with offer_tab:
        _tab_offer(filtered)
    with board_tab:
        _tab_board(filtered)
