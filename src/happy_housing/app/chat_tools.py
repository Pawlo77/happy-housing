"""Tool implementations for the Explore LMS Assistant."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests

from happy_housing.app.ai_review import load_review, reviews_db_path
from happy_housing.app.app_settings import AppSettings
from happy_housing.app.preferences import UserPreferences, save_preferences
from happy_housing.app.recommender import top_recommendations
from happy_housing.app.star_news import load_news, news_path
from happy_housing.app.stars import load_stars, stars_path

logger = logging.getLogger(__name__)

_SLIM_KEYS: tuple[str, ...] = (
    "listing_id",
    "title",
    "district",
    "source",
    "price_pln",
    "price_per_m2",
    "model_price_per_m2",
    "delta_price_per_m2",
    "area_m2",
    "rooms",
    "year_built",
    "commute_min",
    "dist_transit_stop_m",
    "dist_metro_m",
    "dist_park_m",
    "dist_bike_path_m",
    "bike_path_coverage_m",
    "dist_rail_area_m",
    "dist_military_m",
    "dist_airport_area_m",
    "dist_traffic_road_m",
    "traffic_weight_nearby",
    "in_oou",
    "aircraft_lden_db",
    "has_balcony",
    "has_garage",
    "has_lift",
    "url",
    "lat",
    "lon",
)
"""Columns safe to expose to the LMS in tool results."""

_DDG_HTML = "https://html.duckduckgo.com/html/"
"""DuckDuckGo HTML search endpoint (no API key)."""


@dataclass(slots=True)
class ChatToolContext:
    """Mutable session context shared by all Assistant tools."""

    scored: pd.DataFrame
    """Current scored listings universe."""
    processed: Path
    """Processed data directory."""
    prefs: UserPreferences
    """Explore preferences (source of truth for chat)."""
    prefs_path: Path | None = None
    """Where to persist preference updates."""
    active_listing_id: str | None = None
    """Offer currently open in Explore, if any."""
    settings_top_n: int = 15
    """Default recommendation count."""
    demotion_alpha: float = 1.0
    """Recommender demotion strength."""
    amenity_weights: dict[str, float] | None = None
    """Per-amenity recommender weights from Admin."""
    settings: AppSettings | None = None
    """App settings for reviews / model id."""
    pending_open_listing_id: str | None = None
    """Set by ``open_offer`` for the UI to navigate after the turn."""
    last_tool_notes: list[str] = field(default_factory=list)
    """Short traces for debugging / meta."""


def slim_offer_dict(row: pd.Series) -> dict[str, Any]:
    """Compact offer facts for LMS tool results."""
    out: dict[str, Any] = {}
    for key in _SLIM_KEYS:
        if key not in row.index:
            continue
        value = row[key]
        if value is None or (isinstance(value, float) and value != value):
            continue
        if hasattr(value, "item"):
            with contextlib.suppress(ValueError, AttributeError):
                value = value.item()
        if isinstance(value, (pd.Timestamp,)):
            value = str(value)
        out[key] = value
    return out


def offer_summary_row(ctx: ChatToolContext, listing_id: str) -> dict[str, Any] | None:
    """Return slim facts for one listing, or None."""
    frame = ctx.scored
    if frame.empty or "listing_id" not in frame.columns:
        return None
    match = frame.loc[frame["listing_id"].astype(str) == str(listing_id)]
    if match.empty:
        return None
    return slim_offer_dict(match.iloc[0])


def search_offers(
    ctx: ChatToolContext,
    *,
    district: str | None = None,
    max_price_pln: float | None = None,
    min_area_m2: float | None = None,
    min_rooms: int | None = None,
    max_commute_min: float | None = None,
    underpriced_only: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Filter scored offers and return slim rows (capped)."""
    work = ctx.scored
    if work.empty:
        return []
    mask = pd.Series(True, index=work.index)
    if district:
        needle = str(district).strip().casefold()
        if "district" in work.columns:
            mask &= (
                work["district"]
                .astype(str)
                .str.casefold()
                .str.contains(re.escape(needle), na=False)
            )
    if max_price_pln is not None and "price_pln" in work.columns:
        mask &= pd.to_numeric(work["price_pln"], errors="coerce") <= float(
            max_price_pln
        )
    if min_area_m2 is not None and "area_m2" in work.columns:
        mask &= pd.to_numeric(work["area_m2"], errors="coerce") >= float(min_area_m2)
    if min_rooms is not None and "rooms" in work.columns:
        mask &= pd.to_numeric(work["rooms"], errors="coerce") >= int(min_rooms)
    if max_commute_min is not None and "commute_min" in work.columns:
        mask &= pd.to_numeric(work["commute_min"], errors="coerce") <= float(
            max_commute_min
        )
    if underpriced_only and "delta_price_per_m2" in work.columns:
        mask &= pd.to_numeric(work["delta_price_per_m2"], errors="coerce") < 0
    subset = work.loc[mask]
    if "delta_price_per_m2" in subset.columns:
        subset = subset.sort_values("delta_price_per_m2", ascending=True)
    cap = max(1, min(int(limit), 25))
    return [slim_offer_dict(row) for _, row in subset.head(cap).iterrows()]


def recommend_offers(
    ctx: ChatToolContext, top_n: int | None = None
) -> list[dict[str, Any]]:
    """Top recommendations using Explore prefs (no impression write)."""
    n = int(top_n) if top_n is not None else int(ctx.settings_top_n)
    top = top_recommendations(
        ctx.scored,
        processed_dir=ctx.processed,
        top_n=max(1, min(n, 30)),
        demotion_alpha=float(ctx.demotion_alpha),
        prefs=ctx.prefs,
        amenity_weights=ctx.amenity_weights,
        record=False,
    )
    return [slim_offer_dict(row) for _, row in top.iterrows()]


def get_preferences(ctx: ChatToolContext) -> dict[str, Any]:
    """Serialize Explore preferences for the model."""
    from dataclasses import asdict

    return asdict(ctx.prefs)


def update_preferences(
    ctx: ChatToolContext,
    *,
    notes: str | None = None,
    enabled: bool | None = None,
    max_price_pln: float | None = None,
    min_area_m2: float | None = None,
    min_rooms: int | None = None,
    max_commute_min: float | None = None,
    districts: str | None = None,
) -> dict[str, Any]:
    """Patch Explore preferences and persist when a path is set."""
    prefs = ctx.prefs
    if enabled is not None:
        prefs.enabled = bool(enabled)
    if notes is not None:
        prefs.nl_text = str(notes)
    if max_price_pln is not None:
        prefs.max_price_pln = float(max_price_pln)
    if min_area_m2 is not None:
        prefs.min_area_m2 = float(min_area_m2)
    if min_rooms is not None:
        prefs.min_rooms = int(min_rooms)
    if max_commute_min is not None:
        prefs.max_commute_min = float(max_commute_min)
    if districts is not None:
        parts = [p.strip() for p in str(districts).split(",") if p.strip()]
        prefs.districts = parts
    if ctx.prefs_path is not None:
        save_preferences(ctx.prefs_path, prefs)
    ctx.last_tool_notes.append("preferences_updated")
    return get_preferences(ctx)


def list_starred(ctx: ChatToolContext) -> list[dict[str, Any]]:
    """Starred listing ids with slim facts when available."""
    stars = load_stars(stars_path(ctx.processed))
    out: list[dict[str, Any]] = []
    for listing_id in stars:
        slim = offer_summary_row(ctx, listing_id)
        out.append(slim or {"listing_id": listing_id})
    return out


def list_star_news(ctx: ChatToolContext, limit: int = 10) -> list[dict[str, Any]]:
    """Recent starred-offer news events."""
    items = load_news(news_path(ctx.processed))
    cap = max(1, min(int(limit), 40))
    rows: list[dict[str, Any]] = []
    for item in items[-cap:][::-1]:
        change_bits = [
            f"{c.field}: {c.old} → {c.new}" for c in (item.changes or ())[:6]
        ]
        rows.append({
            "listing_id": item.listing_id,
            "kind": item.kind,
            "title": item.title,
            "detected_at": item.detected_at,
            "changes": change_bits,
        })
    return rows


def get_cached_review(ctx: ChatToolContext, listing_id: str) -> dict[str, Any]:
    """Return cached AI review text if present."""
    model = (ctx.settings.lm_model if ctx.settings else None) or "gemma-4-31b-it-mlx"
    review = load_review(reviews_db_path(ctx.processed), listing_id, model=model)
    if review is None:
        return {"listing_id": listing_id, "found": False}
    return {
        "listing_id": listing_id,
        "found": True,
        "model": review.model,
        "review_text": review.review_text[:4000],
        "created_at": review.created_at,
    }


def get_active_offer(ctx: ChatToolContext) -> dict[str, Any]:
    """Facts for the currently open Explore offer."""
    if not ctx.active_listing_id:
        return {"active": False, "message": "No offer open in Explore."}
    slim = offer_summary_row(ctx, ctx.active_listing_id)
    if slim is None:
        return {
            "active": True,
            "listing_id": ctx.active_listing_id,
            "message": "Active id not in scored universe.",
        }
    return {"active": True, **slim}


def open_offer(ctx: ChatToolContext, listing_id: str) -> dict[str, Any]:
    """Queue Explore navigation to ``listing_id`` after this turn."""
    lid = str(listing_id).strip()
    if not lid:
        return {"ok": False, "error": "listing_id required"}
    ctx.pending_open_listing_id = lid
    ctx.last_tool_notes.append(f"open_offer:{lid}")
    slim = offer_summary_row(ctx, lid)
    return {"ok": True, "listing_id": lid, "offer": slim}


def web_search_area(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    """Search the web (DuckDuckGo HTML) for area opinions / news."""
    needle = str(query or "").strip()
    if not needle:
        return []
    cap = max(1, min(int(limit), 8))
    try:
        response = requests.post(
            _DDG_HTML,
            data={"q": needle},
            headers={
                "User-Agent": "HappyHousing/1.0 (local Warsaw housing app)",
            },
            timeout=12.0,
        )
        response.raise_for_status()
        html = response.text
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return [{"title": "search_error", "url": "", "snippet": str(exc)}]
    # Lightweight parse — result links in DDG HTML.
    pattern = re.compile(
        r'uddg=([^&"]+).*?class="result__a"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        re.IGNORECASE | re.DOTALL,
    )
    hits: list[dict[str, str]] = []
    for match in pattern.finditer(html):
        from urllib.parse import unquote

        url = unquote(match.group(1))
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        if title and url:
            hits.append({
                "title": title[:200],
                "url": url[:500],
                "snippet": snippet[:400],
            })
        if len(hits) >= cap:
            break
    if not hits:
        # Fallback: return the search URL for the model to cite.
        hits.append({
            "title": f"Search: {needle}",
            "url": f"https://duckduckgo.com/?q={quote_plus(needle)}",
            "snippet": "No structured hits parsed; open the search URL.",
        })
    return hits


def tool_result_json(payload: object) -> str:
    """Serialize tool output for LMS."""
    return json.dumps(payload, ensure_ascii=False, default=str)
