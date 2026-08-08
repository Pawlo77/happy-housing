"""Home page: sources, methodology, refresh stats, model performance."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from happy_housing.app.data import load_metadata, load_scored, scored_path
from happy_housing.app.metadata import DEFAULT_METHODOLOGY, DEFAULT_SOURCES


def render_home() -> None:
    """Render the metadata / overview Home page."""
    st.title("happy-housing")
    st.caption(
        "Warsaw real-estate decision aid — free open data, fair zł/m², "
        "explainable maps."
    )

    meta = load_metadata()
    try:
        frame = load_scored()
        n_live = len(frame)
        by_source = (
            frame["source"].astype(str).value_counts().to_dict()
            if "source" in frame.columns
            else {}
        )
    except FileNotFoundError as exc:
        st.warning(str(exc))
        n_live = 0
        by_source = {}

    refreshed = meta.refreshed_at if meta else None
    if not refreshed and scored_path().exists():
        mtime = scored_path().stat().st_mtime
        refreshed = datetime.fromtimestamp(mtime, tz=UTC).isoformat()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Offers (scored)", f"{meta.n_offers if meta else n_live}")
    c2.metric(
        "Last refresh (UTC)",
        (refreshed or "—")[:19].replace("T", " ") if refreshed else "—",
    )
    c3.metric(
        "Pipeline duration",
        f"{meta.duration_s:.1f}s" if meta else "—",
    )
    r2 = (meta.model_metrics.get("r2") if meta else None) if meta else None
    c4.metric("Model R² (hold-out)", f"{r2:.3f}" if r2 is not None else "—")

    st.subheader("Model performance")
    if meta and meta.model_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE (zł/m²)", f"{meta.model_metrics.get('mae', float('nan')):.0f}")
        m2.metric("R²", f"{meta.model_metrics.get('r2', float('nan')):.3f}")
        m3.metric("Train rows", f"{int(meta.model_metrics.get('n_train', 0))}")
        m4.metric("Test rows", f"{int(meta.model_metrics.get('n_test', 0))}")
        if meta.feature_columns:
            st.caption("Features: " + ", ".join(meta.feature_columns))
    else:
        st.info("No training metrics yet — run the pipeline from Admin.")

    st.subheader("Offers by source")
    counts = meta.n_by_source if meta and meta.n_by_source else by_source
    if counts:
        st.dataframe(
            {"source": list(counts.keys()), "count": list(counts.values())},
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No scored offers loaded.")

    st.subheader("Data sources")
    for src in meta.sources if meta and meta.sources else DEFAULT_SOURCES:
        st.markdown(f"- {src}")

    st.subheader("Modeling methodology")
    st.write(meta.methodology if meta and meta.methodology else DEFAULT_METHODOLOGY)
