"""Offer photo carousel helpers (index wrap + session keys)."""

from __future__ import annotations


def photo_index_key(listing_id: object) -> str:
    """Session-state key for the active photo index of a listing."""
    return f"offer_photo_idx_{listing_id}"


def wrap_photo_index(index: int, n_photos: int, delta: int) -> int:
    """Return a wrapped photo index after adding ``delta``.

    Args:
        index: Current zero-based index.
        n_photos: Number of photos (must be positive).
        delta: Steps to move (negative = previous).

    Returns:
        Wrapped index in ``[0, n_photos)``.
    """
    if n_photos <= 0:
        return 0
    return (int(index) + int(delta)) % int(n_photos)


def clamp_photo_index(index: int, n_photos: int) -> int:
    """Clamp ``index`` into ``[0, n_photos)`` (or 0 when empty)."""
    if n_photos <= 0:
        return 0
    if index < 0:
        return 0
    if index >= n_photos:
        return n_photos - 1
    return index
