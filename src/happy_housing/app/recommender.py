"""Bargain/proximity recommender with impression memory."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from happy_housing.app.commute_modes import commute_column_for_mode
from happy_housing.app.io_cache import read_json
from happy_housing.app.preferences import (
    OOU_PREFER_OUTSIDE,
    UserPreferences,
    apply_preferences,
)
from happy_housing.gis.amenity_catalog import (
    RECOMMENDER_DIST_CAP_M,
    amenity_keys,
    dist_column,
)

logger = logging.getLogger(__name__)

_BUYER_SOURCES: frozenset[str] = frozenset({"otodom", "fixture"})
"""Default candidate sources when prefs do not set sources."""


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


def mark_skipped(path: Path, listing_id: str, *, count_bump: int = 5) -> None:
    """Hard-demote a recommended listing so the next refresh replaces it."""
    current = load_impressions(path)
    key = str(listing_id)
    now = datetime.now(tz=UTC).isoformat()
    entry = current.get(key) or {
        "count": 0,
        "last_seen_at": now,
        "opened": False,
        "skipped": False,
    }
    entry["count"] = int(entry.get("count") or 0) + max(1, int(count_bump))
    entry["last_seen_at"] = now
    entry["opened"] = True
    entry["skipped"] = True
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


def normalize_recommender_weights(
    *,
    w_bargain: float,
    w_commute: float,
    w_transit: float,
) -> tuple[float, float, float]:
    """Normalize positive blend weights to sum to 1 (keep zeros if all zero)."""
    weights = np.array(
        [
            max(0.0, float(w_bargain)),
            max(0.0, float(w_commute)),
            max(0.0, float(w_transit)),
        ],
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 0:
        return 0.50, 0.25, 0.20
    weights = weights / total
    return float(weights[0]), float(weights[1]), float(weights[2])


def _soft_pref_bonus(work: pd.DataFrame, prefs: UserPreferences) -> pd.Series:
    """Small additive fit for rooms/area/year toward preferred ranges."""
    bonus = pd.Series(0.0, index=work.index, dtype=float)
    if prefs.min_rooms is not None and "rooms" in work.columns:
        rooms = _col(work, "rooms")
        bonus = bonus + (rooms >= float(prefs.min_rooms)).astype(float) * 0.03
    if prefs.min_area_m2 is not None and "area_m2" in work.columns:
        area = _col(work, "area_m2")
        bonus = bonus + (area >= float(prefs.min_area_m2)).astype(float) * 0.03
    if prefs.max_price_pln is not None and "price_pln" in work.columns:
        price = _col(work, "price_pln")
        bonus = bonus + (price <= float(prefs.max_price_pln)).astype(float) * 0.02
    return bonus


def score_offers(
    frame: pd.DataFrame,
    *,
    demotion_alpha: float,
    impressions: dict[str, dict[str, Any]],
    prefs: UserPreferences | None = None,
    amenity_weights: dict[str, float] | None = None,
    w_bargain: float = 0.50,
    w_commute: float = 0.25,
    w_transit: float = 0.20,
    w_overprice: float = 0.15,
    otodom_pool_default: bool = True,
    commute_mode: object | None = None,
    delta_scale_zl: float = 2000.0,
    dist_cap_m: float = RECOMMENDER_DIST_CAP_M,
    opened_penalty: float = 0.5,
    skipped_penalty: float = 3.0,
) -> pd.DataFrame:
    """Rank offers by bargain + proximity (+ soft prefs / amenity weights).

    When ``prefs.enabled``, hard-filters first via ``apply_preferences``.
    Non-zero ``amenity_weights`` nudge score by proximity to each type
    (positive = prefer near, negative = prefer far), capped at
    ``dist_cap_m``. ``commute_mode`` selects which door-to-door
    minutes column feeds the commute term (default: public transport).
    """
    work = frame
    if prefs is not None and prefs.enabled:
        work = apply_preferences(frame, prefs, commute_mode=commute_mode)
    elif otodom_pool_default and "source" in frame.columns:
        sources = frame["source"].astype(str).str.lower()
        work = frame.loc[sources.isin(_BUYER_SOURCES)].copy()
    if work.empty or "listing_id" not in work.columns:
        return pd.DataFrame(
            columns=[
                "listing_id",
                "confidence",
                "rec_bargain",
                "rec_commute",
                "rec_transit",
                "rec_overprice",
                "rec_base",
            ]
        )
    work = work.copy()
    wb, wc, wt = normalize_recommender_weights(
        w_bargain=w_bargain, w_commute=w_commute, w_transit=w_transit
    )
    wo = max(0.0, float(w_overprice))
    scale = max(100.0, float(delta_scale_zl))
    delta = _col(work, "delta_price_per_m2")
    commute = _col(work, commute_column_for_mode(commute_mode))
    transit = _col(work, "dist_transit_stop_m")
    bargain = ((-delta).clip(lower=0.0) / scale).clip(upper=1.0).fillna(0.0)
    overprice = (delta.clip(lower=0.0) / scale).clip(upper=1.0).fillna(0.0)
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
    amenity_bonus = pd.Series(0.0, index=work.index, dtype=float)
    weights = amenity_weights or {}
    cap = max(100.0, float(dist_cap_m))
    for key in amenity_keys():
        weight = float(weights.get(key, 0.0) or 0.0)
        if abs(weight) < 1e-9:
            continue
        dist = _col(work, dist_column(key))
        prox = (1.0 - (dist.fillna(cap) / cap)).clip(0.0, 1.0)
        amenity_bonus = amenity_bonus + (0.20 * weight * prox)
    soft = (
        _soft_pref_bonus(work, prefs)
        if prefs is not None and prefs.enabled
        else pd.Series(0.0, index=work.index, dtype=float)
    )
    base = (
        wb * bargain
        + wc * commute_score
        + wt * transit_score
        - wo * overprice
        + noise_bonus
        + amenity_bonus
        + soft
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
    skipped = (
        work["listing_id"]
        .astype(str)
        .map(lambda lid: 1.0 if (impressions.get(lid) or {}).get("skipped") else 0.0)
    )
    demoted = base / (
        1.0
        + float(demotion_alpha) * counts
        + float(opened_penalty) * opened
        + float(skipped_penalty) * skipped
    )
    out = pd.DataFrame({
        "listing_id": work["listing_id"].astype(str),
        "confidence": demoted.astype(float),
        "rec_bargain": bargain.astype(float),
        "rec_commute": commute_score.astype(float),
        "rec_transit": transit_score.astype(float),
        "rec_overprice": overprice.astype(float),
        "rec_base": base.astype(float),
    })
    return out.sort_values("confidence", ascending=False).reset_index(drop=True)


def retrain_recommender(
    frame: pd.DataFrame,
    *,
    processed_dir: Path,
    top_n: int,
    demotion_alpha: float,
    prefs: UserPreferences | None = None,
    amenity_weights: dict[str, float] | None = None,
    w_bargain: float = 0.50,
    w_commute: float = 0.25,
    w_transit: float = 0.20,
    w_overprice: float = 0.15,
    otodom_pool_default: bool = True,
    commute_mode: object | None = None,
    delta_scale_zl: float = 2000.0,
    dist_cap_m: float = RECOMMENDER_DIST_CAP_M,
    opened_penalty: float = 0.5,
    skipped_penalty: float = 3.0,
) -> dict[str, float]:
    """Rescore the universe, persist metrics, return summary stats."""
    logger.info(
        "Retrain recommender: n=%s top_n=%s demotion_alpha=%.3f commute_mode=%s",
        len(frame),
        int(top_n),
        float(demotion_alpha),
        commute_mode,
    )
    impressions = load_impressions(impressions_path(processed_dir))
    ranked = score_offers(
        frame,
        demotion_alpha=demotion_alpha,
        impressions=impressions,
        prefs=prefs,
        amenity_weights=amenity_weights,
        w_bargain=w_bargain,
        w_commute=w_commute,
        w_transit=w_transit,
        w_overprice=w_overprice,
        otodom_pool_default=otodom_pool_default,
        commute_mode=commute_mode,
        delta_scale_zl=delta_scale_zl,
        dist_cap_m=dist_cap_m,
        opened_penalty=opened_penalty,
        skipped_penalty=skipped_penalty,
    )
    top = ranked.head(max(1, int(top_n)))
    unseen = 0.0
    if not top.empty:
        unseen = float(
            top["listing_id"]
            .map(lambda lid: 0.0 if impressions.get(str(lid)) else 1.0)
            .mean()
        )
    metrics = {
        "n_candidates": float(len(ranked)),
        "n_recommended": float(len(top)),
        "mean_confidence": float(top["confidence"].mean()) if not top.empty else 0.0,
        "share_unseen": unseen,
        "prefs_enabled": 1.0 if prefs is not None and prefs.enabled else 0.0,
        "refreshed_at": datetime.now(tz=UTC).timestamp(),
    }
    path = metrics_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Retrain recommender done: recommended=%s/%s mean_confidence=%.3f → %s",
        int(metrics["n_recommended"]),
        int(metrics["n_candidates"]),
        float(metrics["mean_confidence"]),
        path,
    )
    return metrics


def load_recommender_metrics(processed: Path) -> dict[str, float]:
    """Load latest recommender metrics; missing → empty."""
    path = metrics_path(processed)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load recommender metrics %s: %s", path, exc)
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


def top_recommendations(
    frame: pd.DataFrame,
    *,
    processed_dir: Path,
    top_n: int,
    demotion_alpha: float,
    prefs: UserPreferences | None = None,
    amenity_weights: dict[str, float] | None = None,
    record: bool = True,
    w_bargain: float = 0.50,
    w_commute: float = 0.25,
    w_transit: float = 0.20,
    w_overprice: float = 0.15,
    otodom_pool_default: bool = True,
    exclude_ids: frozenset[str] | set[str] | None = None,
    commute_mode: object | None = None,
    delta_scale_zl: float = 2000.0,
    dist_cap_m: float = RECOMMENDER_DIST_CAP_M,
    opened_penalty: float = 0.5,
    skipped_penalty: float = 3.0,
) -> pd.DataFrame:
    """Return top-N ranked listing rows; optionally record impressions + metrics."""
    impressions = load_impressions(impressions_path(processed_dir))
    ranked = score_offers(
        frame,
        demotion_alpha=demotion_alpha,
        impressions=impressions,
        prefs=prefs,
        amenity_weights=amenity_weights,
        w_bargain=w_bargain,
        w_commute=w_commute,
        w_transit=w_transit,
        w_overprice=w_overprice,
        otodom_pool_default=otodom_pool_default,
        commute_mode=commute_mode,
        delta_scale_zl=delta_scale_zl,
        dist_cap_m=dist_cap_m,
        opened_penalty=opened_penalty,
        skipped_penalty=skipped_penalty,
    )
    if exclude_ids:
        blocked = {str(x) for x in exclude_ids}
        ranked = ranked.loc[
            ~ranked["listing_id"].astype(str).isin(blocked)
        ].reset_index(drop=True)
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
            amenity_weights=amenity_weights,
            w_bargain=w_bargain,
            w_commute=w_commute,
            w_transit=w_transit,
            w_overprice=w_overprice,
            otodom_pool_default=otodom_pool_default,
            commute_mode=commute_mode,
            delta_scale_zl=delta_scale_zl,
            dist_cap_m=dist_cap_m,
            opened_penalty=opened_penalty,
            skipped_penalty=skipped_penalty,
        )
    ids = top["listing_id"].tolist()
    conf_map = dict(zip(top["listing_id"], top["confidence"], strict=True))
    subset = frame.loc[frame["listing_id"].astype(str).isin(ids)].copy()
    subset["confidence"] = subset["listing_id"].astype(str).map(conf_map)
    return subset.sort_values("confidence", ascending=False).reset_index(drop=True)
