"""Persist and generate LM Studio AI reviews for Explore offers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_AI_MODEL = "gemma-4-31b-it-mlx"
"""Default LM Studio model id for offer reviews."""

_SYSTEM_PROMPT = (
    "You are a Warsaw residential real-estate analyst. "
    "Write a concise buyer-facing review (120-220 words) covering: "
    "(1) price fairness vs the model, (2) livability / commute / amenities, "
    "(3) notable risks or upsides, (4) a clear buy / watch / skip lean. "
    "Be concrete; use the numbers given; do not invent missing facts."
)
"""System prompt for offer AI reviews."""


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


def reviews_db_path(processed: Path) -> Path:
    """DuckDB file that stores offer AI reviews."""
    return processed / "offer_ai_reviews.duckdb"


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
            PRIMARY KEY (listing_id, model)
        )
        """
    )
    return conn


def load_review(
    path: Path, listing_id: str, model: str = DEFAULT_AI_MODEL
) -> OfferAiReview | None:
    """Return a cached review for ``listing_id`` + ``model``, if any."""
    if not path.exists():
        return None
    conn = _connect(path)
    try:
        frame = conn.execute(
            """
            SELECT listing_id, model, review_text, created_at
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
    return OfferAiReview(
        listing_id=str(row["listing_id"]),
        model=str(row["model"]),
        review_text=str(row["review_text"]),
        created_at=str(row["created_at"]),
    )


def save_review(path: Path, review: OfferAiReview) -> None:
    """Upsert an offer AI review into DuckDB."""
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO offer_ai_reviews
                (listing_id, model, review_text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                review.listing_id,
                review.model,
                review.review_text,
                review.created_at,
            ],
        )
    finally:
        conn.close()
    logger.info("Saved AI review for %s (%s)", review.listing_id, review.model)


def build_review_prompt(row: pd.Series) -> str:
    """Compose the user prompt from offer facts and livability numbers."""
    parts = [
        f"Listing: {row.get('listing_id')}",
        f"Title: {row.get('title') or '—'}",
        f"District: {row.get('district') or '—'}",
        f"Source: {row.get('source') or '—'}",
        f"Asking price: {row.get('price_pln')} PLN",
        f"Area: {row.get('area_m2')} m²",
        f"Rooms: {row.get('rooms')}",
        f"Year built: {row.get('year_built')}",
        f"Listing zł/m²: {row.get('price_per_m2')}",
        f"Model zł/m²: {row.get('model_price_per_m2')}",
        f"Delta (listing - model): {row.get('delta_price_per_m2')}",
        f"OOU / aircraft zone: {row.get('in_oou')}",
        f"Commute min to Daszyńskiego: {row.get('commute_min')}",
        f"Transit stop m: {row.get('dist_transit_stop_m')}",
        f"Hospital m: {row.get('dist_hospital_m')}",
        f"Fire station m: {row.get('dist_fire_station_m')}",
        f"Police m: {row.get('dist_police_m')}",
        f"Air station m: {row.get('dist_air_station_m')}",
        f"GUNB m: {row.get('dist_gunb_m')}",
    ]
    description = row.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(f"Source description:\n{description.strip()[:2500]}")
    return "\n".join(parts)


def generate_review_text(
    row: pd.Series,
    *,
    model: str = DEFAULT_AI_MODEL,
) -> str:
    """Call LM Studio and return the assistant review text."""
    import lmstudio as lms

    chat = lms.Chat(_SYSTEM_PROMPT)
    chat.add_user_message(build_review_prompt(row))
    llm = lms.llm(model)
    result = llm.respond(chat, config={"temperature": 0.4, "maxTokens": 500})
    text = getattr(result, "content", None) or str(result)
    return str(text).strip()


def get_or_create_review(
    path: Path,
    row: pd.Series,
    *,
    model: str = DEFAULT_AI_MODEL,
    force: bool = False,
) -> OfferAiReview:
    """Load a cached review or generate + persist a new one."""
    listing_id = str(row.get("listing_id") or "")
    if not listing_id:
        raise ValueError("listing_id required for AI review")
    if not force:
        cached = load_review(path, listing_id, model=model)
        if cached is not None:
            return cached
    text = generate_review_text(row, model=model)
    review = OfferAiReview(
        listing_id=listing_id,
        model=model,
        review_text=text,
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    save_review(path, review)
    return review


def review_to_dict(review: OfferAiReview) -> dict[str, Any]:
    """Serialize a review for Streamlit / tests."""
    return {
        "listing_id": review.listing_id,
        "model": review.model,
        "review_text": review.review_text,
        "created_at": review.created_at,
    }
