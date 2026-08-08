"""Starred-offer snapshots and change news for the Explore News tab."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

WATCH_FIELDS: tuple[str, ...] = (
    "title",
    "price_pln",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "area_m2",
    "rooms",
    "district",
    "commute_min",
    "in_oou",
    "url",
)
"""Listing fields compared between starred snapshots and the live frame."""

_FIELD_LABELS: dict[str, str] = {
    "title": "Title",
    "price_pln": "Price (PLN)",
    "price_per_m2": "zł/m²",
    "model_price_per_m2": "Model zł/m²",
    "delta_price_per_m2": "Delta zł/m²",
    "area_m2": "Area m²",
    "rooms": "Rooms",
    "district": "District",
    "commute_min": "Commute min",
    "in_oou": "OOU",
    "url": "URL",
}


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One field that moved between snapshot and live listing."""

    field: str
    """Column name."""
    old: Any
    """Previous snapshot value."""
    new: Any
    """Current listing value."""


@dataclass(frozen=True, slots=True)
class StarNewsItem:
    """One news entry about a starred offer."""

    id: str
    """Stable event id."""
    listing_id: str
    """Offer id."""
    title: str
    """Display title at detection time."""
    kind: str
    """``changed`` or ``removed``."""
    detected_at: str
    """ISO-8601 UTC timestamp."""
    changes: tuple[FieldChange, ...] = field(default_factory=tuple)
    """Field-level diffs (empty when removed)."""


def snapshots_path(processed: Path) -> Path:
    """JSON map of last-seen starred listing fields."""
    return processed / "starred_snapshots.json"


def news_path(processed: Path) -> Path:
    """JSON list of starred-offer news events (newest last on disk)."""
    return processed / "starred_news.json"


def load_snapshots(path: Path) -> dict[str, dict[str, Any]]:
    """Load listing_id → field snapshot map."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load snapshots %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[str(key)] = dict(value)
    return out


def save_snapshots(path: Path, snapshots: dict[str, dict[str, Any]]) -> None:
    """Persist starred listing snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshots, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_news(path: Path) -> list[StarNewsItem]:
    """Load news events; missing/invalid → empty list."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load news %s: %s", path, exc)
        return []
    if not isinstance(payload, list):
        return []
    items: list[StarNewsItem] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        changes_raw = raw.get("changes") or []
        changes: list[FieldChange] = []
        if isinstance(changes_raw, list):
            for ch in changes_raw:
                if isinstance(ch, dict) and "field" in ch:
                    changes.append(
                        FieldChange(
                            field=str(ch["field"]),
                            old=ch.get("old"),
                            new=ch.get("new"),
                        )
                    )
        items.append(
            StarNewsItem(
                id=str(raw.get("id", "")),
                listing_id=str(raw.get("listing_id", "")),
                title=str(raw.get("title", "")),
                kind=str(raw.get("kind", "changed")),
                detected_at=str(raw.get("detected_at", "")),
                changes=tuple(changes),
            )
        )
    return items


def save_news(path: Path, items: list[StarNewsItem]) -> None:
    """Persist news events."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for item in items:
        payload = asdict(item)
        payload["changes"] = [asdict(c) for c in item.changes]
        serializable.append(payload)
    path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def clear_star_news_memory(processed: Path) -> None:
    """Delete starred-news feed and snapshots."""
    for path in (news_path(processed), snapshots_path(processed)):
        if path.exists():
            path.unlink()
            logger.info("Removed %s", path)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        return float(value) if isinstance(value, float) else int(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text not in {"nan", "None", "<NA>"} else None


def snapshot_from_row(row: pd.Series) -> dict[str, Any]:
    """Extract watched fields from a scored listing row."""
    return {name: _coerce_scalar(row.get(name)) for name in WATCH_FIELDS}


def _values_differ(old: Any, new: Any) -> bool:
    if old is None and new is None:
        return False
    if old is None or new is None:
        return True
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return abs(float(old) - float(new)) >= 0.5
    return str(old) != str(new)


def diff_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[FieldChange, ...]:
    """Return field changes between two snapshots."""
    changes: list[FieldChange] = []
    for name in WATCH_FIELDS:
        old = before.get(name)
        new = after.get(name)
        if _values_differ(old, new):
            changes.append(FieldChange(field=name, old=old, new=new))
    return tuple(changes)


def format_value(field: str, value: Any) -> str:
    """Human-readable value for news display."""
    if value is None:
        return "—"
    if field in {
        "price_pln",
        "price_per_m2",
        "model_price_per_m2",
        "delta_price_per_m2",
        "area_m2",
        "commute_min",
    }:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if field == "price_pln":
            return f"{number:,.0f}"
        if field in {"area_m2", "commute_min"}:
            return f"{number:,.1f}"
        return f"{number:,.0f}"
    if field == "in_oou":
        try:
            return "inside" if int(value) == 1 else "outside"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def format_change_markdown(field: str, old: Any, new: Any) -> str:
    """Markdown line with strikethrough old value and bold new value."""
    label = _FIELD_LABELS.get(field, field)
    old_s = format_value(field, old)
    new_s = format_value(field, new)
    return f"**{label}:** ~~{old_s}~~ → **{new_s}**"


def sync_starred_news(
    starred_ids: frozenset[str] | set[str],
    frame: pd.DataFrame,
    *,
    snapshots_path: Path,
    news_path: Path,
    history_cap: int = 200,
) -> list[StarNewsItem]:
    """Compare live starred rows to snapshots; append news; refresh snapshots.

    First sighting of a starred id only baselines a snapshot (no news).
    Returns newly appended news items (newest last).
    """
    snapshots = load_snapshots(snapshots_path)
    existing_news = load_news(news_path)
    new_items: list[StarNewsItem] = []
    live_ids = set()
    if not frame.empty and "listing_id" in frame.columns:
        live = frame.copy()
        live["_lid"] = live["listing_id"].astype(str)
        live_ids = set(live["_lid"].tolist())
    else:
        live = frame

    starred = {str(x) for x in starred_ids}
    # Drop snapshots for unstarred offers.
    snapshots = {k: v for k, v in snapshots.items() if k in starred}

    detected_at = _now_iso()
    for listing_id in sorted(starred):
        if listing_id not in live_ids:
            prev = snapshots.get(listing_id)
            if prev is not None:
                title = str(prev.get("title") or listing_id)
                new_items.append(
                    StarNewsItem(
                        id=str(uuid.uuid4()),
                        listing_id=listing_id,
                        title=title,
                        kind="removed",
                        detected_at=detected_at,
                        changes=(),
                    )
                )
                del snapshots[listing_id]
            continue
        row = live.loc[live["_lid"] == listing_id].iloc[0]
        current = snapshot_from_row(row)
        title = str(current.get("title") or listing_id)
        prev = snapshots.get(listing_id)
        if prev is None:
            snapshots[listing_id] = current
            continue
        changes = diff_snapshots(prev, current)
        if changes:
            new_items.append(
                StarNewsItem(
                    id=str(uuid.uuid4()),
                    listing_id=listing_id,
                    title=title,
                    kind="changed",
                    detected_at=detected_at,
                    changes=changes,
                )
            )
        snapshots[listing_id] = current

    if new_items:
        existing_news.extend(new_items)
        # Cap history to keep the file small.
        cap = max(10, int(history_cap))
        existing_news = existing_news[-cap:]
        save_news(news_path, existing_news)
    save_snapshots(snapshots_path, snapshots)
    return new_items
