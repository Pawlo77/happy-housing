"""Streamlit GUI for offer cards, SHAP, time slider, and map layers."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import shap
import streamlit as st

from happy_housing.config import load_experiment_config
from happy_housing.models.train import build_feature_matrix
from happy_housing.storage import connect_duckdb, query_listings

TOOLTIPS: dict[str, str] = {
    "listings": (
        "Market listings from Otodom plus open transaction context so asking "
        "prices can be judged against location reality."
    ),
    "model": (
        "LightGBM estimates a fair zł/m² from livability and property features "
        "so you can spot overpriced or underpriced offers."
    ),
    "shap": (
        "SHAP shows how much each factor (transit, OOU noise, fire station "
        "distance, commute, …) adds or subtracts from the fair price."
    ),
    "map": (
        "Map listings like a portal board: colour = listing-model delta, "
        "size ~ area; filter by month to replay market timing."
    ),
}


def _default_scored_path() -> Path:
    """Resolve the default scored listings parquet path from config."""
    config = load_experiment_config(Path("configs/experiment.yaml"))
    return config.data.processed_path() / "listings_scored.parquet"


def load_scored(path: Path) -> pd.DataFrame:
    """Load scored listings or raise a clear UI error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: uv run python -m happy_housing.pipeline"
        )
    return pd.read_parquet(path)


def _month_key(series: pd.Series) -> pd.Series:
    """Parse listing dates to YYYY-MM strings for the time slider."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None).dt.strftime("%Y-%m")


def render_app(scored_path: Path | None = None) -> None:
    """Render the happy-housing Streamlit dashboard."""
    st.set_page_config(page_title="happy-housing", layout="wide")
    st.title("happy-housing")
    st.caption(
        "Warsaw housing decision aid — fair price, livability context, "
        "explainable maps from free open data."
    )

    path = scored_path or _default_scored_path()
    try:
        frame = load_scored(path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    frame = frame.copy()
    if "listed_at" in frame.columns:
        frame["listed_month"] = _month_key(frame["listed_at"])
    else:
        frame["listed_month"] = "unknown"

    conn = connect_duckdb()
    districts = sorted([
        d for d in frame["district"].dropna().astype(str).unique().tolist()
    ])
    sources = sorted(frame.get("source", pd.Series(["unknown"])).dropna().unique())
    months = sorted([m for m in frame["listed_month"].dropna().unique() if m != "NaT"])

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        district = st.selectbox(
            "District",
            options=["(all)", *districts],
            help=TOOLTIPS["listings"],
        )
    with col_b:
        max_ppm = st.slider(
            "Max listing zł/m²",
            min_value=int(frame["price_per_m2"].min()),
            max_value=int(frame["price_per_m2"].max()) + 1,
            value=int(frame["price_per_m2"].max()) + 1,
            help=TOOLTIPS["model"],
        )
    with col_c:
        oou_filter = st.selectbox(
            "OOU filter",
            options=["(all)", "inside", "outside"],
            help="Aircraft restricted-use zone (Chopin OOU).",
        )
    with col_d:
        source = st.selectbox(
            "Source",
            options=["(all)", *sources],
            help="Otodom asks vs RCN transactions vs fixtures.",
        )

    month = "(all)"
    if months:
        month = st.select_slider(
            "Market month (time playback)",
            options=["(all)", *months],
            value="(all)",
            help=TOOLTIPS["map"],
        )

    in_oou = None
    if oou_filter == "inside":
        in_oou = 1
    elif oou_filter == "outside":
        in_oou = 0
    filtered = query_listings(
        conn,
        path,
        max_price_per_m2=float(max_ppm),
        district=None if district == "(all)" else district,
        in_oou=in_oou,
    )
    if filtered.empty:
        filtered = frame.head(0)
    else:
        extra_cols = [
            c
            for c in ("source", "image_url", "url", "listed_month", "title")
            if c in frame.columns
        ]
        if extra_cols:
            extras = frame[["listing_id", *extra_cols]].drop_duplicates(
                subset=["listing_id"]
            )
            drop_overlap = [c for c in extra_cols if c in filtered.columns]
            filtered = filtered.drop(columns=drop_overlap, errors="ignore")
            filtered = filtered.merge(extras, on="listing_id", how="left")

    if source != "(all)" and "source" in filtered.columns:
        filtered = filtered[filtered["source"] == source]
    if month != "(all)" and "listed_month" in filtered.columns:
        filtered = filtered[filtered["listed_month"] == month]

    st.subheader("Offer board")
    board_cols = [
        c
        for c in (
            "listing_id",
            "title",
            "district",
            "source",
            "area_m2",
            "price_pln",
            "price_per_m2",
            "model_price_per_m2",
            "delta_price_per_m2",
            "commute_min",
            "in_oou",
        )
        if c in filtered.columns
    ]
    st.dataframe(filtered[board_cols], width="stretch")
    st.caption(f"{len(filtered)} offers after filters · {TOOLTIPS['listings']}")

    if filtered.empty:
        st.warning("No listings match filters.")
        return

    listing_ids = filtered["listing_id"].tolist()
    selected_id = st.selectbox(
        "Offer card",
        options=listing_ids,
        help=TOOLTIPS["shap"],
    )
    row = filtered.loc[filtered["listing_id"] == selected_id].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        if isinstance(row.get("image_url"), str) and row["image_url"]:
            st.image(row["image_url"], use_container_width=True)
        if isinstance(row.get("url"), str) and row["url"]:
            st.link_button("Open listing", row["url"])
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Listing zł/m²", f"{row['price_per_m2']:.0f}")
        m2.metric("Model zł/m²", f"{row['model_price_per_m2']:.0f}")
        m3.metric(
            "Delta (listing - model)",
            f"{row['delta_price_per_m2']:.0f}",
            help=TOOLTIPS["model"],
        )
        st.write({
            "district": row.get("district"),
            "source": row.get("source"),
            "commute_min": row.get("commute_min"),
            "in_oou": row.get("in_oou"),
        })

    st.subheader("Why this price? (SHAP)")
    model_path = path.parent / "lightgbm_price_per_m2.txt"
    if model_path.exists():
        try:
            booster = lgb.Booster(model_file=str(model_path))
            one = filtered.loc[filtered["listing_id"] == selected_id]
            features = build_feature_matrix(one)
            explainer = shap.TreeExplainer(booster)
            explanation = explainer(features)
            fig, _ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"SHAP plot unavailable: {exc}")
    else:
        st.info("Model file missing — run the pipeline to enable SHAP.")

    st.subheader("Map")
    st.caption(TOOLTIPS["map"])
    map_df = filtered.copy()
    map_df["lat"] = map_df["lat"].astype(float)
    map_df["lon"] = map_df["lon"].astype(float)
    map_df["color_r"] = map_df["delta_price_per_m2"].apply(
        lambda d: 220 if d > 0 else 30
    )
    map_df["color_g"] = map_df["delta_price_per_m2"].apply(
        lambda d: 40 if d > 0 else 170
    )
    map_df["radius"] = (map_df["area_m2"].fillna(50).astype(float) * 2.5).clip(60, 250)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="[color_r, color_g, 70, 190]",
        pickable=True,
    )
    view = pdk.ViewState(
        latitude=float(map_df["lat"].mean()),
        longitude=float(map_df["lon"].mean()),
        zoom=10.5,
    )
    try:
        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                layers=[layer],
                initial_view_state=view,
                tooltip={
                    "html": "<b>{listing_id}</b><br/>{district}<br/>"
                    "Δ {delta_price_per_m2} zł/m²<br/>{price_pln} PLN",
                    "style": {"backgroundColor": "#111", "color": "white"},
                },
            )
        )
    except Exception as exc:
        st.warning(f"pydeck unavailable ({exc}); falling back to st.map")
        st.map(map_df[["lat", "lon"]])

    st.markdown(
        "Data attribution: Otodom (public pages) · OpenStreetMap · "
        "UM Warszawa RCN · ZTM/mkuran GTFS · GIOŚ · GUNB"
    )


def main() -> None:
    """Streamlit entrypoint."""
    render_app()


if __name__ == "__main__":
    main()
