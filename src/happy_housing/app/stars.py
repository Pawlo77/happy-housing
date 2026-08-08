"""Persistent starred offers for Explore Map / Board / Offer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def stars_path(processed: Path) -> Path:
    """JSON file storing starred ``listing_id`` values."""
    return processed / "starred_offers.json"


def load_stars(path: Path) -> frozenset[str]:
    """Load starred listing ids; missing/invalid file → empty set."""
    if not path.exists():
        return frozenset()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load stars %s: %s", path, exc)
        return frozenset()
    if not isinstance(payload, list):
        return frozenset()
    return frozenset(str(item) for item in payload)


def save_stars(path: Path, starred: frozenset[str] | set[str]) -> None:
    """Persist starred ids as a sorted JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(str(item) for item in starred)
    path.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def is_starred(starred: frozenset[str] | set[str], listing_id: str) -> bool:
    """Whether ``listing_id`` is in the starred set."""
    return str(listing_id) in starred


def toggle_star(path: Path, listing_id: str) -> bool:
    """Flip star for ``listing_id``; return ``True`` when now starred."""
    current = set(load_stars(path))
    key = str(listing_id)
    if key in current:
        current.remove(key)
        save_stars(path, current)
        return False
    current.add(key)
    save_stars(path, current)
    return True


def star_marker(listing_id: str, starred: frozenset[str] | set[str]) -> str:
    """Board/map marker: filled star when starred, else empty."""
    return "★" if is_starred(starred, listing_id) else "☆"


def clear_stars(path: Path) -> None:
    """Delete the starred-offers file (empty set)."""
    if path.exists():
        path.unlink()
        logger.info("Cleared starred offers at %s", path)
    else:
        save_stars(path, frozenset())
