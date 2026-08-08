"""Persist and generate LM Studio AI reviews for Explore offers."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from happy_housing.app.app_settings import (
    DEFAULT_AI_REVIEW_SYSTEM_PROMPT,
    DEFAULT_LM_MODEL,
    AppSettings,
)

logger = logging.getLogger(__name__)

DEFAULT_AI_MODEL = DEFAULT_LM_MODEL
"""Default LM Studio model id for offer reviews."""

DEFAULT_SYSTEM_PROMPT = DEFAULT_AI_REVIEW_SYSTEM_PROMPT
"""Default system prompt for offer AI reviews."""
_CORE_FACT_KEYS: tuple[tuple[str, str], ...] = (
    ("listing_id", "Listing"),
    ("title", "Title"),
    ("district", "District"),
    ("source", "Source"),
    ("url", "URL"),
    ("listed_at", "Listed at"),
    ("market", "Market"),
    ("advertiser_type", "Advertiser"),
    ("price_pln", "Asking price (PLN)"),
    ("area_m2", "Area (m²)"),
    ("rooms", "Rooms"),
    ("year_built", "Year built"),
    ("floor_no", "Floor"),
    ("building_floors_num", "Building floors"),
    ("building_type", "Building type"),
    ("building_material", "Building material"),
    ("building_ownership", "Ownership"),
    ("construction_status", "Construction status"),
    ("windows_type", "Windows"),
    ("heating", "Heating"),
    ("rent", "Rent (PLN/mo)"),
    ("has_balcony", "Balcony"),
    ("has_terrace", "Terrace"),
    ("has_garden", "Garden"),
    ("has_garage", "Garage / parking"),
    ("has_lift", "Lift"),
    ("is_private_owner", "Private owner"),
    ("lat", "Latitude"),
    ("lon", "Longitude"),
)
"""Human labels for core listing fact fields."""

_MODEL_KEYS: tuple[tuple[str, str], ...] = (
    ("price_per_m2", "Listing zł/m²"),
    ("model_price_per_m2", "Model zł/m²"),
    ("delta_price_per_m2", "Delta (listing - model)"),
)
"""Model valuation fields."""

_PROXIMITY_KEYS: tuple[tuple[str, str], ...] = (
    ("aircraft_lden_db", "Aircraft Lden (dB, EEA END)"),
    ("aircraft_lnight_db", "Aircraft Lnight (dB, EEA END)"),
    ("in_oou", "Aircraft Lden ≥ 55 dB"),
    ("commute_min", "Commute min to Daszyńskiego"),
    ("dist_daszynskiego_m", "Distance to Daszyńskiego (m)"),
    ("dist_transit_stop_m", "Nearest transit stop (m)"),
    ("dist_metro_m", "Nearest metro (m)"),
    ("dist_hospital_m", "Nearest hospital (m)"),
    ("dist_fire_station_m", "Nearest fire station (m)"),
    ("dist_police_m", "Nearest police (m)"),
    ("dist_air_station_m", "Nearest air-quality station (m)"),
    ("dist_gunb_m", "Nearest GUNB permit (m)"),
)
"""Livability / proximity fields stored on scored offers."""

_NEIGHBORHOOD_KEYS: tuple[tuple[str, str], ...] = (
    ("nbhd_rcn_median_ppm", "Local RCN median zł/m²"),
    ("nbhd_rcn_mean_ppm", "Local RCN mean zł/m²"),
    ("nbhd_rcn_idw_ppm", "Local RCN IDW zł/m²"),
    ("nbhd_rcn_n", "Local RCN comparable count"),
    ("nbhd_rcn_nearest_m", "Nearest RCN sale (m)"),
)
"""Neighborhood historical sale comps from RCN."""

_RAW_JSON_KEYS: tuple[str, ...] = (
    "characteristics_json",
    "additional_information_json",
)
"""Raw Otodom JSON blobs when present."""

_SKIP_REMAINING: frozenset[str] = frozenset({
    "geometry",
    "image_url",
    "description",
    *{k for k, _ in _CORE_FACT_KEYS},
    *{k for k, _ in _MODEL_KEYS},
    *{k for k, _ in _PROXIMITY_KEYS},
    *{k for k, _ in _NEIGHBORHOOD_KEYS},
    *_RAW_JSON_KEYS,
})
"""Columns already covered by named sections (or noisy binaries)."""


@dataclass(frozen=True, slots=True)
class OfferAiReview:
    """Cached AI review for one listing."""

    listing_id: str
    """Offer identifier."""
    model: str
    """LM Studio model id used to generate the text."""
    review_text: str
    """Generated markdown/plain review body."""
    created_at: str
    """ISO-8601 UTC timestamp when the review was stored."""
    context_hash: str = ""
    """Hash of prompt settings + buyer prefs used for this review."""


def reviews_db_path(processed: Path) -> Path:
    """DuckDB file that stores offer AI reviews."""
    return processed / "offer_ai_reviews.duckdb"


def clear_ai_reviews(path: Path) -> None:
    """Delete the AI reviews DuckDB file."""
    if path.exists():
        path.unlink()
        logger.info("Cleared AI reviews DB at %s", path)


def review_context_hash(settings: AppSettings) -> str:
    """Stable hash of prompt/preference settings that affect review text."""
    payload = {
        "system_prompt": settings.ai_review_system_prompt,
        "include_preferences": settings.ai_review_include_preferences,
        "include_core_facts": settings.ai_review_include_core_facts,
        "include_model_valuation": settings.ai_review_include_model_valuation,
        "include_proximity": settings.ai_review_include_proximity,
        "include_neighborhood_prices": settings.ai_review_include_neighborhood_prices,
        "include_description": settings.ai_review_include_description,
        "include_raw_characteristics": settings.ai_review_include_raw_characteristics,
        "include_all_remaining": settings.ai_review_include_all_remaining,
        "max_description_chars": settings.ai_review_max_description_chars,
        "max_field_chars": settings.ai_review_max_field_chars,
        "temperature": settings.ai_review_temperature,
        "max_tokens": settings.ai_review_max_tokens,
        "buyer_pref_notes": settings.buyer_pref_notes,
        "buyer_max_price_pln": settings.buyer_max_price_pln,
        "buyer_max_price_per_m2": settings.buyer_max_price_per_m2,
        "buyer_min_area_m2": settings.buyer_min_area_m2,
        "buyer_rooms_min": settings.buyer_rooms_min,
        "buyer_rooms_max": settings.buyer_rooms_max,
        "buyer_preferred_districts": settings.buyer_preferred_districts,
        "buyer_max_commute_min": settings.buyer_max_commute_min,
        "buyer_avoid_oou": settings.buyer_avoid_oou,
        "buyer_must_haves": settings.buyer_must_haves,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS offer_ai_reviews (
            listing_id VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            review_text VARCHAR NOT NULL,
            created_at VARCHAR NOT NULL,
            context_hash VARCHAR NOT NULL DEFAULT '',
            PRIMARY KEY (listing_id, model)
        )
        """
    )
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info('offer_ai_reviews')").fetchall()
    }
    if "context_hash" not in cols:
        conn.execute(
            "ALTER TABLE offer_ai_reviews ADD COLUMN context_hash VARCHAR DEFAULT ''"
        )
    return conn


def load_review(
    path: Path,
    listing_id: str,
    model: str = DEFAULT_AI_MODEL,
    context_hash: str | None = None,
) -> OfferAiReview | None:
    """Return a cached review for ``listing_id`` + ``model``, if any.

    When ``context_hash`` is set, only return a row with the same hash so that
    preference / prompt setting changes invalidate stale cache entries.
    """
    if not path.exists():
        return None
    conn = _connect(path)
    try:
        frame = conn.execute(
            """
            SELECT listing_id, model, review_text, created_at, context_hash
            FROM offer_ai_reviews
            WHERE listing_id = ? AND model = ?
            """,
            [listing_id, model],
        ).df()
    finally:
        conn.close()
    if frame.empty:
        return None
    row = frame.iloc[0]
    stored_hash = str(row.get("context_hash") or "")
    if context_hash is not None and stored_hash != context_hash:
        return None
    return OfferAiReview(
        listing_id=str(row["listing_id"]),
        model=str(row["model"]),
        review_text=str(row["review_text"]),
        created_at=str(row["created_at"]),
        context_hash=stored_hash,
    )


def save_review(path: Path, review: OfferAiReview) -> None:
    """Upsert an offer AI review into DuckDB."""
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO offer_ai_reviews
                (listing_id, model, review_text, created_at, context_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                review.listing_id,
                review.model,
                review.review_text,
                review.created_at,
                review.context_hash,
            ],
        )
    finally:
        conn.close()
    logger.info("Saved AI review for %s (%s)", review.listing_id, review.model)


def _fmt_value(value: object, *, max_chars: int) -> str | None:
    """Format a scalar/series cell for the prompt; skip empties."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, pd.Series):
        return None
    if isinstance(value, (bytes, bytearray)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "<na>"}:
        return None
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _append_labeled(
    parts: list[str],
    row: pd.Series,
    keys: tuple[tuple[str, str], ...],
    *,
    max_chars: int,
) -> None:
    for key, label in keys:
        if key not in row.index:
            continue
        formatted = _fmt_value(row.get(key), max_chars=max_chars)
        if formatted is None:
            continue
        parts.append(f"{label}: {formatted}")


def _buyer_preferences_block(settings: AppSettings) -> list[str]:
    lines: list[str] = []
    if settings.buyer_pref_notes.strip():
        lines.append(f"Notes: {settings.buyer_pref_notes.strip()}")
    if settings.buyer_max_price_pln is not None:
        lines.append(f"Max asking price (PLN): {settings.buyer_max_price_pln:g}")
    if settings.buyer_max_price_per_m2 is not None:
        lines.append(f"Max zł/m²: {settings.buyer_max_price_per_m2:g}")
    if settings.buyer_min_area_m2 is not None:
        lines.append(f"Min area (m²): {settings.buyer_min_area_m2:g}")
    if settings.buyer_rooms_min is not None:
        lines.append(f"Min rooms: {settings.buyer_rooms_min}")
    if settings.buyer_rooms_max is not None:
        lines.append(f"Max rooms: {settings.buyer_rooms_max}")
    if settings.buyer_preferred_districts.strip():
        lines.append(
            f"Preferred districts: {settings.buyer_preferred_districts.strip()}"
        )
    if settings.buyer_max_commute_min is not None:
        lines.append(f"Max commute (min): {settings.buyer_max_commute_min:g}")
    if settings.buyer_avoid_oou:
        lines.append("Avoid aircraft Lden ≥ 55 dB: yes")
    if settings.buyer_must_haves.strip():
        lines.append(f"Must-haves: {settings.buyer_must_haves.strip()}")
    return lines


def build_review_prompt(
    row: pd.Series,
    settings: AppSettings | None = None,
) -> str:
    """Compose the user prompt from prefs, offer facts, and stored signals."""
    cfg = settings or AppSettings()
    max_field = int(cfg.ai_review_max_field_chars)
    sections: list[str] = []

    if cfg.ai_review_include_preferences:
        pref_lines = _buyer_preferences_block(cfg)
        if pref_lines:
            sections.append("## Buyer preferences\n" + "\n".join(pref_lines))
        else:
            sections.append(
                "## Buyer preferences\n(none configured — review for a general buyer)"
            )

    if cfg.ai_review_include_core_facts:
        body: list[str] = []
        _append_labeled(body, row, _CORE_FACT_KEYS, max_chars=max_field)
        if body:
            sections.append("## Offer facts\n" + "\n".join(body))

    if cfg.ai_review_include_model_valuation:
        body = []
        _append_labeled(body, row, _MODEL_KEYS, max_chars=max_field)
        if body:
            sections.append("## Model valuation\n" + "\n".join(body))

    if cfg.ai_review_include_proximity:
        body = []
        _append_labeled(body, row, _PROXIMITY_KEYS, max_chars=max_field)
        if body:
            sections.append(
                "## Proximity & livability\n"
                "(distances in metres; aircraft Lden/Lnight are EEA END "
                "contour band lower edges in dB; traffic intensity is a "
                "map-layer diurnal proxy and is not stored per offer)\n"
                + "\n".join(body)
            )

    if cfg.ai_review_include_neighborhood_prices:
        body = []
        _append_labeled(body, row, _NEIGHBORHOOD_KEYS, max_chars=max_field)
        if body:
            sections.append(
                "## Neighborhood price history (RCN comps)\n" + "\n".join(body)
            )

    if cfg.ai_review_include_description:
        formatted = _fmt_value(
            row.get("description"),
            max_chars=int(cfg.ai_review_max_description_chars),
        )
        if formatted is not None:
            sections.append(f"## Source description\n{formatted}")

    if cfg.ai_review_include_raw_characteristics:
        raw_lines: list[str] = []
        for key in _RAW_JSON_KEYS:
            formatted = _fmt_value(row.get(key), max_chars=max_field)
            if formatted is not None:
                raw_lines.append(f"{key}: {formatted}")
        if raw_lines:
            sections.append("## Raw listing characteristics\n" + "\n".join(raw_lines))

    if cfg.ai_review_include_all_remaining:
        leftover: list[str] = []
        for key in row.index:
            name = str(key)
            if name in _SKIP_REMAINING:
                continue
            formatted = _fmt_value(row.get(key), max_chars=max_field)
            if formatted is None:
                continue
            leftover.append(f"{name}: {formatted}")
        if leftover:
            sections.append("## Other stored fields\n" + "\n".join(sorted(leftover)))

    return "\n\n".join(sections)


def resolve_system_prompt(settings: AppSettings | None = None) -> str:
    """Return the configured system prompt, falling back to the default."""
    cfg = settings or AppSettings()
    text = (cfg.ai_review_system_prompt or "").strip()
    return text or DEFAULT_SYSTEM_PROMPT


def generate_review_text(
    row: pd.Series,
    *,
    model: str = DEFAULT_AI_MODEL,
    settings: AppSettings | None = None,
) -> str:
    """Call LM Studio and return the assistant review text."""
    import lmstudio as lms

    cfg = settings or AppSettings()
    chat = lms.Chat(resolve_system_prompt(cfg))
    chat.add_user_message(build_review_prompt(row, settings=cfg))
    llm = lms.llm(model)
    result = llm.respond(
        chat,
        config={
            "temperature": float(cfg.ai_review_temperature),
            "maxTokens": int(cfg.ai_review_max_tokens),
        },
    )
    text = getattr(result, "content", None) or str(result)
    return str(text).strip()


def get_or_create_review(
    path: Path,
    row: pd.Series,
    *,
    model: str = DEFAULT_AI_MODEL,
    force: bool = False,
    settings: AppSettings | None = None,
) -> OfferAiReview:
    """Load a cached review or generate + persist a new one."""
    cfg = settings or AppSettings()
    listing_id = str(row.get("listing_id") or "")
    if not listing_id:
        raise ValueError("listing_id required for AI review")
    ctx = review_context_hash(cfg)
    if not force:
        cached = load_review(path, listing_id, model=model, context_hash=ctx)
        if cached is not None:
            return cached
    text = generate_review_text(row, model=model, settings=cfg)
    review = OfferAiReview(
        listing_id=listing_id,
        model=model,
        review_text=text,
        created_at=datetime.now(tz=UTC).isoformat(),
        context_hash=ctx,
    )
    save_review(path, review)
    return review


def review_to_dict(review: OfferAiReview) -> dict[str, Any]:
    """Serialize a review for Streamlit / tests."""
    return asdict(review)
