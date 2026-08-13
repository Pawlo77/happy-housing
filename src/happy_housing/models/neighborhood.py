"""RCN neighborhood spatial features (self-excluded comps for fair-value priors).

When RCN deeds are also training labels, comps at the same coordinates or with
the same ``listing_id`` are excluded so neighborhood aggregates do not leak
the target. Otodom asks never match RCN ids, so their comps stay intact.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius for haversine BallTree queries (metres)."""

_SELF_EXCLUDE_M = 5.0
"""Exclude RCN comps within this distance of the listing (self / duplicate)."""

NEIGHBORHOOD_FEATURE_COLUMNS: tuple[str, ...] = (
    "nbhd_rcn_median_ppm",
    "nbhd_rcn_mean_ppm",
    "nbhd_rcn_n",
    "nbhd_rcn_nearest_m",
    "nbhd_rcn_idw_ppm",
)
"""Feature columns produced by :func:`attach_neighborhood_rcn_features`."""


def _rad_coords(frame: pd.DataFrame) -> np.ndarray:
    """Lat/lon degrees → radians array shaped ``(n, 2)`` for BallTree haversine."""
    lat = pd.to_numeric(frame["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(frame["lon"], errors="coerce").to_numpy(dtype=float)
    return np.radians(np.column_stack([lat, lon]))


def _valid_rcn(rcn: pd.DataFrame) -> pd.DataFrame:
    """Filter RCN rows with usable coordinates and zł/m²."""
    if rcn.empty:
        return rcn
    work = rcn.copy()
    if "price_per_m2" not in work.columns:
        if {"price_pln", "area_m2"} <= set(work.columns):
            work["price_per_m2"] = pd.to_numeric(
                work["price_pln"], errors="coerce"
            ) / pd.to_numeric(work["area_m2"], errors="coerce")
        else:
            return pd.DataFrame(columns=[*list(work.columns), "price_per_m2"])
    work["price_per_m2"] = pd.to_numeric(work["price_per_m2"], errors="coerce")
    work = work.dropna(subset=["lat", "lon", "price_per_m2"])
    work = work[(work["price_per_m2"] > 4_000) & (work["price_per_m2"] < 50_000)]
    return work.reset_index(drop=True)


def attach_neighborhood_rcn_features(
    listings: pd.DataFrame,
    rcn: pd.DataFrame,
    *,
    radius_m: float = 800.0,
    min_neighbors: int = 3,
) -> pd.DataFrame:
    """Attach local RCN zł/m² aggregates to each listing row.

    Excludes comps at the same coordinates (≤5 m) and comps whose
    ``listing_id`` matches the query row when both have ids — avoids
    RCN-as-label self-inclusion when deeds are also training rows.

    Args:
        listings: Rows with ``lat`` / ``lon``.
        rcn: Historical transactions with ``lat`` / ``lon`` / price.
        radius_m: Search radius in metres.
        min_neighbors: Soft floor logged for diagnostics (still uses whatever
            falls inside the radius when fewer comps exist).

    Returns:
        Copy of ``listings`` with neighborhood feature columns.
    """
    del min_neighbors  # reserved for future filtering; radius is primary gate
    out = listings.copy()
    for col in NEIGHBORHOOD_FEATURE_COLUMNS:
        out[col] = np.nan

    comps = _valid_rcn(rcn)
    if comps.empty or out.empty:
        logger.warning("Neighborhood features skipped — empty RCN or listings")
        return out
    if not {"lat", "lon"} <= set(out.columns):
        return out

    listing_coords = _rad_coords(out)
    rcn_coords = _rad_coords(comps)
    valid_listing = np.isfinite(listing_coords).all(axis=1)
    valid_rcn = np.isfinite(rcn_coords).all(axis=1)
    if not valid_rcn.any():
        return out

    tree = BallTree(rcn_coords[valid_rcn], metric="haversine")
    radius_rad = float(radius_m) / _EARTH_RADIUS_M
    ppm = comps.loc[valid_rcn, "price_per_m2"].to_numpy(dtype=float)
    rcn_ids = (
        comps.loc[valid_rcn, "listing_id"].astype(str).to_numpy()
        if "listing_id" in comps.columns
        else None
    )
    listing_ids = (
        out["listing_id"].astype(str).to_numpy()
        if "listing_id" in out.columns
        else None
    )

    medians = np.full(len(out), np.nan)
    means = np.full(len(out), np.nan)
    counts = np.zeros(len(out), dtype=float)
    nearest = np.full(len(out), np.nan)
    idw = np.full(len(out), np.nan)

    query_idx = np.flatnonzero(valid_listing)
    if len(query_idx) == 0:
        return out

    ind_list, dist_list = tree.query_radius(
        listing_coords[valid_listing],
        r=radius_rad,
        return_distance=True,
        sort_results=True,
    )
    for local_i, (neighbors, dists_rad) in enumerate(
        zip(ind_list, dist_list, strict=True)
    ):
        row_i = int(query_idx[local_i])
        if len(neighbors) == 0:
            d_rad, idx = tree.query(listing_coords[[row_i]], k=1)
            neighbors = np.asarray(idx[0], dtype=int)
            dists_m = np.asarray(d_rad[0], dtype=float) * _EARTH_RADIUS_M
        else:
            neighbors = np.asarray(neighbors, dtype=int)
            dists_m = np.asarray(dists_rad, dtype=float) * _EARTH_RADIUS_M

        keep = dists_m > _SELF_EXCLUDE_M
        if listing_ids is not None and rcn_ids is not None:
            keep &= rcn_ids[neighbors] != listing_ids[row_i]
        neighbors = neighbors[keep]
        dists_m = dists_m[keep]
        if len(neighbors) == 0:
            continue

        values = ppm[neighbors]
        counts[row_i] = float(len(values))
        medians[row_i] = float(np.median(values))
        means[row_i] = float(np.mean(values))
        nearest[row_i] = float(dists_m[0])
        weights = 1.0 / np.clip(dists_m, 25.0, None)
        idw[row_i] = float(np.average(values, weights=weights))

    out["nbhd_rcn_median_ppm"] = medians
    out["nbhd_rcn_mean_ppm"] = means
    out["nbhd_rcn_n"] = counts
    out["nbhd_rcn_nearest_m"] = nearest
    out["nbhd_rcn_idw_ppm"] = idw
    logger.info(
        "Neighborhood RCN features: median coverage %.0f%% (radius=%sm)",
        100.0 * np.isfinite(medians).mean(),
        radius_m,
    )
    return out


def idw_neighborhood_baseline(
    listings: pd.DataFrame,
    rcn: pd.DataFrame,
    *,
    radius_m: float = 800.0,
) -> np.ndarray:
    """Inverse-distance-weighted RCN zł/m² baseline (second model for MLflow)."""
    enriched = attach_neighborhood_rcn_features(
        listings,
        rcn,
        radius_m=radius_m,
    )
    return pd.to_numeric(enriched["nbhd_rcn_idw_ppm"], errors="coerce").to_numpy(
        dtype=float
    )
