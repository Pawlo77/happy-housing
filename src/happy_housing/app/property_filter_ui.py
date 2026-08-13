"""Shared Streamlit widgets for property-attribute filters."""

from typing import Any

import streamlit as st

_YES_NO: tuple[str, ...] = ("(all)", "yes", "no")
"""Tri-state amenity filter choices."""


def render_property_attr_filters(key_prefix: str) -> dict[str, Any]:
    """Render property/amenity filter controls; return kwargs for filter state."""
    st.markdown("##### Property")
    object_type = st.selectbox(
        "Object type",
        options=["(all)", "flat", "house"],
        key=f"{key_prefix}_object_type",
        help="Flat vs house (from Otodom estate / LM extract).",
    )
    ownership = st.selectbox(
        "Ownership",
        options=["(all)", "full", "usufruct", "other"],
        key=f"{key_prefix}_ownership",
        help="full = KW / pełna własność · usufruct = użytkowanie wieczyste.",
    )
    c1, c2 = st.columns(2)
    with c1:
        floor_min_raw = st.number_input(
            "Min floor",
            min_value=-1,
            max_value=50,
            value=-1,
            key=f"{key_prefix}_floor_min",
            help="-1 = no minimum (cellar allowed).",
        )
        year_built_min_raw = st.number_input(
            "Min year built",
            min_value=0,
            max_value=2100,
            value=0,
            key=f"{key_prefix}_year_built_min",
            help="0 = no filter.",
        )
        year_renovated_min_raw = st.number_input(
            "Min year renovated",
            min_value=0,
            max_value=2100,
            value=0,
            key=f"{key_prefix}_year_renovated_min",
            help="From description extract when known. 0 = no filter.",
        )
    with c2:
        floor_max_raw = st.number_input(
            "Max floor",
            min_value=-1,
            max_value=99,
            value=99,
            key=f"{key_prefix}_floor_max",
            help="99 = no maximum (includes attic sentinel).",
        )

    st.markdown("##### Amenities")
    has_garage = st.selectbox("Garage", _YES_NO, key=f"{key_prefix}_garage")
    has_outdoor_parking = st.selectbox(
        "Outdoor parking", _YES_NO, key=f"{key_prefix}_outdoor_parking"
    )
    has_basement = st.selectbox("Basement", _YES_NO, key=f"{key_prefix}_basement")
    has_balcony = st.selectbox("Balcony", _YES_NO, key=f"{key_prefix}_balcony")
    has_garden = st.selectbox("Garden", _YES_NO, key=f"{key_prefix}_garden")
    has_terrace = st.selectbox("Terrace", _YES_NO, key=f"{key_prefix}_terrace")
    has_lift = st.selectbox("Lift", _YES_NO, key=f"{key_prefix}_lift")
    is_attic = st.selectbox("Attic", _YES_NO, key=f"{key_prefix}_attic")
    needs_renovation = st.selectbox(
        "Needs renovation", _YES_NO, key=f"{key_prefix}_renovation"
    )
    plumbing_replaced = st.selectbox(
        "Plumbing replaced",
        _YES_NO,
        key=f"{key_prefix}_plumbing",
        help="From description LM extract when available.",
    )
    wiring_replaced = st.selectbox(
        "Wiring replaced",
        _YES_NO,
        key=f"{key_prefix}_wiring",
        help="From description LM extract when available.",
    )

    floor_min = None if int(floor_min_raw) < 0 else int(floor_min_raw)
    floor_max = None if int(floor_max_raw) >= 99 else int(floor_max_raw)
    year_built_min = None if int(year_built_min_raw) <= 0 else int(year_built_min_raw)
    year_renovated_min = (
        None if int(year_renovated_min_raw) <= 0 else int(year_renovated_min_raw)
    )
    return {
        "object_type": str(object_type),
        "ownership": str(ownership),
        "has_garage": str(has_garage),
        "has_outdoor_parking": str(has_outdoor_parking),
        "has_basement": str(has_basement),
        "has_balcony": str(has_balcony),
        "has_garden": str(has_garden),
        "has_terrace": str(has_terrace),
        "has_lift": str(has_lift),
        "is_attic": str(is_attic),
        "needs_renovation": str(needs_renovation),
        "plumbing_replaced": str(plumbing_replaced),
        "wiring_replaced": str(wiring_replaced),
        "floor_min": floor_min,
        "floor_max": floor_max,
        "year_built_min": year_built_min,
        "year_renovated_min": year_renovated_min,
    }
