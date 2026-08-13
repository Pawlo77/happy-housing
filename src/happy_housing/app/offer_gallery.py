"""Offer photo carousel helpers (index wrap + session keys)."""


def photo_index_key(listing_id: object) -> str:
    """Session-state key for the active photo index of a listing."""
    return f"offer_photo_idx_{listing_id}"


def photo_select_key(listing_id: object) -> str:
    """Session-state key for the photo selectbox widget."""
    return f"offer_photo_sel_{listing_id}"


def photo_nav_sync_key(listing_id: object) -> str:
    """One-shot flag: align selectbox label from index after arrow nav."""
    return f"offer_photo_nav_{listing_id}"


def photo_fullscreen_key(listing_id: object) -> str:
    """Session-state flag: keep the full-size photo dialog open across reruns."""
    return f"offer_photo_fs_open_{listing_id}"


def should_keep_photo_dialog_open(*, open_flag: bool, open_clicked: bool) -> bool:
    """True when the lightbox must be invoked this run.

    Arrow buttons call ``st.rerun()``; the open button is then False, so the
    dialog only stays open if a persistent session flag is still set.
    """
    return bool(open_flag or open_clicked)


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


def resolve_photo_select_label(
    labels: list[str],
    *,
    idx: int,
    current_label: str | None,
    sync_from_index: bool,
) -> str:
    """Pick selectbox label to set *before* the widget is instantiated.

    Arrow buttons must not write the selectbox key after it exists. They set
    ``sync_from_index`` for the next run instead.
    """
    if not labels:
        return ""
    safe_idx = clamp_photo_index(idx, len(labels))
    if sync_from_index:
        return labels[safe_idx]
    if current_label in labels:
        return str(current_label)
    return labels[safe_idx]
