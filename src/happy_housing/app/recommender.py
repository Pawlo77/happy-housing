"""Bargain/proximity recommender with impression memory."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from happy_housing.app.io_cache import read_json
from happy_housing.app.preferences import (
    OOU_PREFER_OUTSIDE,
    UserPreferences,
    apply_preferences,
)

logger = logging.getLogger(__name__)


def impressions_path(processed: Path) -> Path:
    """JSON map of recommendation impressions per listing."""
    return processed / "offer_impressions.json"


def metrics_path(processed: Path) -> Path:
    """Latest recommender metrics JSON for Home / Admin."""
    return processed / "recommender_metrics.json"


def load_impressions(path: Path) -> dict[str, dict[str, Any]]:
    """Load impression records; missing/invalid → empty."""
    if not path.exists():
        return {}
    try:
        payload = read_json(str(path), path.stat().st_mtime)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Failed to load impressions %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[str(key)] = dict(value)
    return out


def save_impressions(path: Path, impressions: dict[str, dict[str, Any]]) -> None:
    """Persist impression map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(impressions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def record_impressions(path: Path, listing_ids: list[str]) -> None:
    """Bump impression counts for listings shown as recommendations."""
    current = load_impressions(path)
    now = datetime.now(tz=UTC).isoformat()
    for listing_id in listing_ids:
        key = str(listing_id)
        entry = current.get(key) or {
            "count": 0,
            "last_seen_at": now,
            "opened": False,
        }
        entry["count"] = int(entry.get("count") or 0) + 1
        entry["last_seen_at"] = now
        current[key] = entry
    save_impressions(path, current)


def mark_opened(path: Path, listing_id: str) -> None:
    """Mark a listing as opened from a recommendation / Offer."""
    current = load_impressions(path)
    key = str(listing_id)
    entry = current.get(key) or {
        "count": 0,
        "last_seen_at": datetime.now(tz=UTC).isoformat(),
        "opened": False,
    }
    entry["opened"] = True
    current[key] = entry
    save_impressions(path, current)


def clear_recommendation_memory(processed: Path) -> None:
    """Delete impression memory and recommender metrics."""
    for path in (impressions_path(processed), metrics_path(processed)):
        if path.exists():
            path.unlink()
            logger.info("Removed %s", path)


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def score_offers(
    frame: pd.DataFrame,
    *,
    demotion_alpha: float,
    impressions: dict[str, dict[str, Any]],
    prefs: UserPreferences | None = None,
) -> pd.DataFrame:
    """Rank offers by bargain + proximity (+ soft prefs) with seen-demotion.

    When ``prefs.enabled``, hard-filters first via ``apply_preferences``.
    """
    work = frame
    if prefs is not None and prefs.enabled:
        work = apply_preferences(frame, prefs)
    if work.empty or "listing_id" not in work.columns:
        return pd.DataFrame(columns=["listing_id", "confidence"])
    work = work.copy()
    delta = _col(work, "delta_price_per_m2")
    commute = _col(work, "commute_min")
    transit = _col(work, "dist_transit_stop_m")
    bargain = ((-delta).clip(lower=0.0) / 2000.0).clip(upper=1.0).fillna(0.0)
    commute_cap = (
        float(prefs.max_commute_min)
        if prefs is not None and prefs.max_commute_min
        else 60.0
    )
    transit_cap = (
        float(prefs.max_dist_transit_m)
        if prefs is not None and prefs.max_dist_transit_m
        else 1000.0
    )
    commute_score = (1.0 - (commute.fillna(commute_cap) / commute_cap)).clip(0.0, 1.0)
    transit_score = (1.0 - (transit.fillna(transit_cap) / transit_cap)).clip(0.0, 1.0)
    noise_bonus = pd.Series(0.0, index=work.index, dtype=float)
    if prefs is not None and prefs.oou == OOU_PREFER_OUTSIDE:
        noise_bonus = (1.0 - _col(work, "in_oou").fillna(0.0).clip(0.0, 1.0)) * 0.15
    base = (
        0.50 * bargain + 0.25 * commute_score + 0.20 * transit_score + noise_bonus
    ).clip(0.0, 1.0)
    counts = (
        work["listing_id"]
        .astype(str)
        .map(lambda lid: float((impressions.get(lid) or {}).get("count") or 0))
    )
    opened = (
        work["listing_id"]
        .astype(str)
        .map(lambda lid: 1.0 if (impressions.get(lid) or {}).get("opened") else 0.0)
    )
    demoted = base / (1.0 + float(demotion_alpha) * counts + 0.5 * opened)
    out = pd.DataFrame({
        "listing_id": work["listing_id"].astype(str),
        "confidence": demoted.astype(float),
    })
    return out.sort_values("confidence", ascending=False).reset_index(drop=True)


def retrain_recommender(
    frame: pd.DataFrame,
    *,
    processed_dir: Path,
    top_n: int,
    demotion_alpha: float,
    prefs: UserPreferences | None = None,
) -> dict[str, float]:
    """Rescore the universe, persist metrics, return summary stats."""
    impressions = load_impressions(impressions_path(processed_dir))
    ranked = score_offers(
        frame,
        demotion_alpha=demotion_alpha,
        impressions=impressions,
        prefs=prefs,
    )
    top = ranked.head(max(1, int(top_n)))
    unseen = 0
    for lid in top["listing_id"].tolist():
        if int((impressions.get(str(lid)) or {}).get("count") or 0) == 0:
            unseen += 1
    metrics = {
        "n_candidates": float(len(ranked)),
        "n_recommended": float(len(top)),
        "mean_confidence": float(top["confidence"].mean()) if not top.empty else 0.0,
        "share_unseen": float(unseen / len(top)) if not top.empty else 0.0,
        "prefs_enabled": 1.0 if (prefs is not None and prefs.enabled) else 0.0,
        "retrained_at": datetime.now(tz=UTC).timestamp(),
    }
    path = metrics_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Retrained recommender → %s", metrics)
    return metrics


def top_recommendations(
    frame: pd.DataFrame,
    *,
    processed_dir: Path,
    top_n: int,
    demotion_alpha: float,
    prefs: UserPreferences | None = None,
    record: bool = True,
) -> pd.DataFrame:
    """Return top-N ranked listing rows; optionally record impressions + metrics."""
    impressions = load_impressions(impressions_path(processed_dir))
    ranked = score_offers(
        frame,
        demotion_alpha=demotion_alpha,
        impressions=impressions,
        prefs=prefs,
    )
    top = ranked.head(max(0, int(top_n)))
    if top.empty:
        return frame.iloc[0:0].copy()
    if record:
        record_impressions(impressions_path(processed_dir), top["listing_id"].tolist())
        retrain_recommender(
            frame,
            processed_dir=processed_dir,
            top_n=top_n,
            demotion_alpha=demotion_alpha,
            prefs=prefs,
        )
    ids = top["listing_id"].tolist()
    conf_map = dict(zip(top["listing_id"], top["confidence"], strict=True))
    subset = frame.loc[frame["listing_id"].astype(str).isin(ids)].copy()
    subset["confidence"] = subset["listing_id"].astype(str).map(conf_map)
    return subset.sort_values("confidence", ascending=False).reset_index(drop=True)


def load_recommender_metrics(processed: Path) -> dict[str, float]:
    """Load last recommender metrics dict."""
    path = metrics_path(processed)
    if not path.exists():
        return {}
    try:
        payload = read_json(str(path), path.stat().st_mtime)
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out
