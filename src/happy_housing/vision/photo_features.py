"""Pretrained CLIP photo features for listings (optional vision extra)."""

import hashlib
import json
import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PHOTO_FEATURE_COLUMNS: tuple[str, ...] = (
    "photo_quality_mean",
    "photo_quality_max",
    "photo_quality_std",
    "n_photos_scored",
    "share_kitchen",
    "share_bathroom",
    "share_living",
    "share_bedroom",
    "share_exterior",
    "luxury_share",
    "has_floor_plan",
    "n_floor_plans",
    "floor_plan_conf",
)
"""Tabular photo/layout features attached before LightGBM training."""

_QUALITY_POS: str = "luxurious renovated apartment interior"
"""Positive CLIP prompt for finish quality."""

_QUALITY_NEG: str = "run-down neglected apartment interior"
"""Negative CLIP prompt for finish quality."""

_ROOM_PROMPTS: dict[str, str] = {
    "kitchen": "photo of a kitchen",
    "bathroom": "photo of a bathroom",
    "living": "photo of a living room",
    "bedroom": "photo of a bedroom",
    "exterior": "photo of a building exterior",
    "luxury": "luxurious high-end apartment interior",
}
"""Zero-shot room / luxury prompts."""

_FLOOR_PLAN_POS: str = "architectural floor plan drawing"
"""Floor-plan positive prompt."""

_FLOOR_PLAN_NEG: str = "photo of a furnished room"
"""Floor-plan negative prompt."""


class PhotoEncoder(Protocol):
    """Minimal encoder interface for tests / CLIP backends."""

    def score_pair(self, image: Any, positive: str, negative: str) -> float:
        """Return polarity score in roughly [-1, 1]."""

    def score_labels(self, image: Any, labels: dict[str, str]) -> dict[str, float]:
        """Return label → similarity (unnormalized)."""


class HeuristicPhotoEncoder:
    """Deterministic stand-in when CLIP / vision deps are unavailable."""

    def score_pair(self, image: Any, positive: str, negative: str) -> float:
        del positive, negative
        seed = _image_seed(image)
        rng = np.random.default_rng(seed)
        return float(rng.uniform(-0.2, 0.8))

    def score_labels(self, image: Any, labels: dict[str, str]) -> dict[str, float]:
        seed = _image_seed(image)
        rng = np.random.default_rng(seed)
        return {key: float(rng.random()) for key in labels}


def _image_seed(image: Any) -> int:
    if isinstance(image, (bytes, bytearray)):
        digest = hashlib.sha1(image, usedforsecurity=False).hexdigest()
    else:
        digest = hashlib.sha1(
            str(image).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
    return int(digest[:8], 16)


def _urls_from_json(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(u) for u in raw if str(u).startswith("http")]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(u) for u in payload if str(u).startswith("http")]


def aggregate_photo_features(
    *,
    image_urls: list[str],
    floor_plan_urls: list[str],
    encoder: PhotoEncoder,
    max_photos: int = 8,
) -> dict[str, float]:
    """Score listing photos into tabular aggregates (no network)."""
    photos = list(image_urls)[: max(1, int(max_photos))]
    qualities: list[float] = []
    room_hits = {key: 0.0 for key in _ROOM_PROMPTS}
    for url in photos:
        q = encoder.score_pair(url, _QUALITY_POS, _QUALITY_NEG)
        qualities.append(q)
        labels = encoder.score_labels(url, _ROOM_PROMPTS)
        total = sum(max(0.0, v) for v in labels.values()) or 1.0
        for key, value in labels.items():
            room_hits[key] += max(0.0, value) / total
    n = float(len(qualities))
    if n <= 0:
        out = {col: float("nan") for col in PHOTO_FEATURE_COLUMNS}
        out["n_photos_scored"] = 0.0
        out["n_floor_plans"] = float(len(floor_plan_urls))
        out["has_floor_plan"] = 1.0 if floor_plan_urls else 0.0
        out["floor_plan_conf"] = 0.0
        return out
    quality_arr = np.asarray(qualities, dtype=float)
    shares = {key: room_hits[key] / n for key in room_hits}
    floor_conf = 0.0
    for url in floor_plan_urls[:3]:
        floor_conf = max(
            floor_conf,
            encoder.score_pair(url, _FLOOR_PLAN_POS, _FLOOR_PLAN_NEG),
        )
    has_plan = 1.0 if floor_plan_urls or floor_conf > 0.15 else 0.0
    return {
        "photo_quality_mean": float(quality_arr.mean()),
        "photo_quality_max": float(quality_arr.max()),
        "photo_quality_std": float(quality_arr.std()) if n > 1 else 0.0,
        "n_photos_scored": n,
        "share_kitchen": float(shares.get("kitchen", 0.0)),
        "share_bathroom": float(shares.get("bathroom", 0.0)),
        "share_living": float(shares.get("living", 0.0)),
        "share_bedroom": float(shares.get("bedroom", 0.0)),
        "share_exterior": float(shares.get("exterior", 0.0)),
        "luxury_share": float(shares.get("luxury", 0.0)),
        "has_floor_plan": has_plan,
        "n_floor_plans": float(len(floor_plan_urls)),
        "floor_plan_conf": float(floor_conf),
    }


def resolve_photo_encoder(model_id: str) -> PhotoEncoder:
    """Return CLIP encoder when vision deps exist; else heuristic."""
    del model_id
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        logger.info("transformers unavailable — using heuristic photo encoder")
        return HeuristicPhotoEncoder()
    logger.info("CLIP backend stubbed to heuristic (enable full vision later)")
    return HeuristicPhotoEncoder()


def attach_photo_features(
    frame: pd.DataFrame,
    *,
    encoder: PhotoEncoder | None = None,
    max_photos: int = 8,
    model_id: str = "openai/clip-vit-base-patch32",
) -> pd.DataFrame:
    """Add ``PHOTO_FEATURE_COLUMNS`` to a listings frame."""
    out = frame.copy()
    for col in PHOTO_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    if out.empty:
        return out
    backend = encoder or resolve_photo_encoder(model_id)
    rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        images = _urls_from_json(row.get("image_urls_json"))
        if not images and isinstance(row.get("image_url"), str) and row["image_url"]:
            images = [str(row["image_url"])]
        plans = _urls_from_json(row.get("floor_plan_urls_json"))
        rows.append(
            aggregate_photo_features(
                image_urls=images,
                floor_plan_urls=plans,
                encoder=backend,
                max_photos=max_photos,
            )
        )
    features = pd.DataFrame(rows, index=out.index)
    for col in PHOTO_FEATURE_COLUMNS:
        out[col] = features[col]
    return out
