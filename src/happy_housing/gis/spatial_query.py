"""Fast Euclidean BallTree helpers for neighbourhood GIS aggregates."""

import numpy as np
from sklearn.neighbors import BallTree


def balltree_nearest_distances(
    query_xy: np.ndarray,
    data_xy: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour Euclidean distances (metres in projected CRS)."""
    dist, _ = balltree_nearest(query_xy, data_xy)
    return dist


def balltree_nearest(
    query_xy: np.ndarray,
    data_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour distances and data indices (projected metres)."""
    if query_xy.size == 0:
        empty = np.array([], dtype=float)
        return empty, np.array([], dtype=int)
    if data_xy.size == 0:
        n = int(query_xy.shape[0])
        return (
            np.full(n, np.nan, dtype=float),
            np.full(n, -1, dtype=int),
        )
    tree = BallTree(data_xy, metric="euclidean")
    dist, idx = tree.query(query_xy, k=1)
    return (
        np.asarray(dist[:, 0], dtype=float),
        np.asarray(idx[:, 0], dtype=int),
    )


def balltree_radius_weight_sums(
    query_xy: np.ndarray,
    data_xy: np.ndarray,
    weights: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Sum ``weights`` of data points within ``radius_m`` of each query point."""
    n = int(query_xy.shape[0])
    if n == 0:
        return np.array([], dtype=float)
    out = np.zeros(n, dtype=float)
    if data_xy.size == 0:
        return out
    tree = BallTree(data_xy, metric="euclidean")
    neighbours = tree.query_radius(query_xy, r=float(radius_m))
    w = np.asarray(weights, dtype=float)
    for i, idxs in enumerate(neighbours):
        if len(idxs):
            out[i] = float(w[idxs].sum())
    return out
