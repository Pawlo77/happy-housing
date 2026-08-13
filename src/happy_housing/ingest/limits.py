"""Ingest limit helpers — ``None`` means download everything available."""


def resolve_ingest_cap(value: int | None) -> int | None:
    """Normalize a config cap; ``None`` / non-positive → unlimited."""
    if value is None:
        return None
    if int(value) <= 0:
        return None
    return int(value)


def reached_cap(count: int, cap: int | None) -> bool:
    """Return True when ``count`` has hit a finite ``cap``."""
    return cap is not None and count >= cap
