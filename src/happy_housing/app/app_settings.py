"""Persisted Admin / app behaviour settings (not experiment YAML)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from happy_housing.app.io_cache import read_json
from happy_housing.models.metrics import coerce_opt_metric

logger = logging.getLogger(__name__)

DEFAULT_LM_MODEL = "gemma-4-31b-it-mlx"
"""Default LM Studio model id."""

DEFAULT_AI_REVIEW_SYSTEM_PROMPT = (
    "You are a Warsaw residential real-estate analyst. "
    "Write a concise buyer-facing review (120-280 words) covering: "
    "(1) fit vs the buyer's stated preferences, "
    "(2) price fairness vs the model and local historical comps, "
    "(3) livability / commute / amenities / noise proxies, "
    "(4) notable risks or upsides, "
    "(5) a clear buy / watch / skip lean. "
    "Be concrete; use the numbers given; do not invent missing facts."
)
"""Default LM Studio system prompt for offer reviews."""


@dataclass(slots=True)
class AppSettings:
    """User-tunable app behaviour stored under processed data."""

    lm_model: str = DEFAULT_LM_MODEL
    """LM Studio model id for offer AI reviews / NL tools."""
    ai_review_on_offer_open: bool = True
    """Auto-request AI review when opening Offer → AI review."""
    ai_review_system_prompt: str = DEFAULT_AI_REVIEW_SYSTEM_PROMPT
    """System prompt sent to LM Studio for offer reviews."""
    ai_review_include_preferences: bool = True
    """Include buyer preference block in the review user prompt."""
    ai_review_include_core_facts: bool = True
    """Include core listing facts (price, rooms, building, …)."""
    ai_review_include_model_valuation: bool = True
    """Include listing/model/delta zł/m²."""
    ai_review_include_proximity: bool = True
    """Include transit/metro/hospital/… distances and OOU."""
    ai_review_include_neighborhood_prices: bool = True
    """Include local RCN historical sale comps."""
    ai_review_include_description: bool = True
    """Include the source listing description."""
    ai_review_include_raw_characteristics: bool = True
    """Include raw Otodom characteristics JSON when present."""
    ai_review_include_all_remaining: bool = True
    """Append every other non-empty stored column not covered above."""
    ai_review_max_description_chars: int = 4000
    """Truncate listing description in the prompt after this many chars."""
    ai_review_max_field_chars: int = 1500
    """Truncate individual non-description fields in the prompt."""
    ai_review_temperature: float = 0.4
    """LM Studio sampling temperature for reviews."""
    ai_review_max_tokens: int = 800
    """LM Studio maxTokens for reviews."""
    buyer_pref_notes: str = ""
    """Free-text buyer preferences for AI reviews."""
    buyer_max_price_pln: float | None = None
    """Optional max asking price (PLN); None = unset."""
    buyer_max_price_per_m2: float | None = None
    """Optional max listing zł/m²; None = unset."""
    buyer_min_area_m2: float | None = None
    """Optional minimum area (m²); None = unset."""
    buyer_rooms_min: int | None = None
    """Optional minimum rooms; None = unset."""
    buyer_rooms_max: int | None = None
    """Optional maximum rooms; None = unset."""
    buyer_preferred_districts: str = ""
    """Comma-separated preferred districts."""
    buyer_max_commute_min: float | None = None
    """Optional max commute minutes; None = unset."""
    buyer_avoid_oou: bool = False
    """Prefer offers outside the OOU / noise-proxy zone."""
    buyer_must_haves: str = ""
    """Free-text must-haves (balcony, garage, quiet street, …)."""
    news_history_cap: int = 200
    """Max starred-news events retained on disk."""
    recommender_top_n: int = 15
    """How many promising offers to surface after retrain."""
    recommender_demotion_alpha: float = 1.0
    """Soft-demotion strength for already-seen recommendations."""
    stats_min_district_offers: int = 5
    """Minimum offers before a district appears on Stats scoreboards."""
    stats_top_n: int = 15
    """How many hunt / watch offers to list on Stats → Value."""
    log_level: str = "INFO"
    """Package log level for the Streamlit app (DEBUG/INFO/WARNING/ERROR)."""
    model_opt_metric: str = "mae"
    """Validation metric for LightGBM HPO (mae/rmse/MAPE/medae/r2)."""
    model_val_size: float = 0.15
    """Fraction of data used as HPO validation (test held out separately)."""
    model_hpo_trials: int = 12
    """Random hyperparameter trials (1 disables search)."""
    model_use_neighborhood: bool = True
    """Attach RCN neighborhood comps and log IDW baseline in MLflow."""
    model_nbhd_radius_m: float = 800.0
    """Search radius (metres) for local RCN comparable sales."""


def settings_path(processed: Path) -> Path:
    """JSON path for app settings."""
    return processed / "app_settings.json"


def _opt_positive_float(value: object) -> float | None:
    """Parse an optional positive float; blank/≤0 → None."""
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:
        return None
    return number


def _opt_positive_int(value: object) -> int | None:
    """Parse an optional positive int; blank/≤0 → None."""
    if value is None or value == "":
        return None
    try:
        number = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def load_app_settings(path: Path) -> AppSettings:
    """Load settings; missing/invalid → defaults."""
    defaults = AppSettings()
    if not path.exists():
        return defaults
    try:
        payload_obj = read_json(str(path), path.stat().st_mtime)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load app settings %s: %s", path, exc)
        return defaults
    if not isinstance(payload_obj, dict):
        return defaults
    payload = payload_obj
    prompt = str(
        payload.get("ai_review_system_prompt", defaults.ai_review_system_prompt) or ""
    ).strip()
    return AppSettings(
        lm_model=str(payload.get("lm_model") or defaults.lm_model).strip()
        or defaults.lm_model,
        ai_review_on_offer_open=bool(
            payload.get("ai_review_on_offer_open", defaults.ai_review_on_offer_open)
        ),
        ai_review_system_prompt=prompt or defaults.ai_review_system_prompt,
        ai_review_include_preferences=bool(
            payload.get(
                "ai_review_include_preferences",
                defaults.ai_review_include_preferences,
            )
        ),
        ai_review_include_core_facts=bool(
            payload.get(
                "ai_review_include_core_facts",
                defaults.ai_review_include_core_facts,
            )
        ),
        ai_review_include_model_valuation=bool(
            payload.get(
                "ai_review_include_model_valuation",
                defaults.ai_review_include_model_valuation,
            )
        ),
        ai_review_include_proximity=bool(
            payload.get(
                "ai_review_include_proximity",
                defaults.ai_review_include_proximity,
            )
        ),
        ai_review_include_neighborhood_prices=bool(
            payload.get(
                "ai_review_include_neighborhood_prices",
                defaults.ai_review_include_neighborhood_prices,
            )
        ),
        ai_review_include_description=bool(
            payload.get(
                "ai_review_include_description",
                defaults.ai_review_include_description,
            )
        ),
        ai_review_include_raw_characteristics=bool(
            payload.get(
                "ai_review_include_raw_characteristics",
                defaults.ai_review_include_raw_characteristics,
            )
        ),
        ai_review_include_all_remaining=bool(
            payload.get(
                "ai_review_include_all_remaining",
                defaults.ai_review_include_all_remaining,
            )
        ),
        ai_review_max_description_chars=max(
            200,
            int(
                payload.get(
                    "ai_review_max_description_chars",
                    defaults.ai_review_max_description_chars,
                )
            ),
        ),
        ai_review_max_field_chars=max(
            100,
            int(
                payload.get(
                    "ai_review_max_field_chars",
                    defaults.ai_review_max_field_chars,
                )
            ),
        ),
        ai_review_temperature=min(
            1.5,
            max(
                0.0,
                float(
                    payload.get(
                        "ai_review_temperature",
                        defaults.ai_review_temperature,
                    )
                ),
            ),
        ),
        ai_review_max_tokens=max(
            64,
            int(payload.get("ai_review_max_tokens", defaults.ai_review_max_tokens)),
        ),
        buyer_pref_notes=str(
            payload.get("buyer_pref_notes", defaults.buyer_pref_notes) or ""
        ),
        buyer_max_price_pln=_opt_positive_float(
            payload.get("buyer_max_price_pln", defaults.buyer_max_price_pln)
        ),
        buyer_max_price_per_m2=_opt_positive_float(
            payload.get("buyer_max_price_per_m2", defaults.buyer_max_price_per_m2)
        ),
        buyer_min_area_m2=_opt_positive_float(
            payload.get("buyer_min_area_m2", defaults.buyer_min_area_m2)
        ),
        buyer_rooms_min=_opt_positive_int(
            payload.get("buyer_rooms_min", defaults.buyer_rooms_min)
        ),
        buyer_rooms_max=_opt_positive_int(
            payload.get("buyer_rooms_max", defaults.buyer_rooms_max)
        ),
        buyer_preferred_districts=str(
            payload.get("buyer_preferred_districts", defaults.buyer_preferred_districts)
            or ""
        ),
        buyer_max_commute_min=_opt_positive_float(
            payload.get("buyer_max_commute_min", defaults.buyer_max_commute_min)
        ),
        buyer_avoid_oou=bool(payload.get("buyer_avoid_oou", defaults.buyer_avoid_oou)),
        buyer_must_haves=str(
            payload.get("buyer_must_haves", defaults.buyer_must_haves) or ""
        ),
        news_history_cap=max(
            10, int(payload.get("news_history_cap", defaults.news_history_cap))
        ),
        recommender_top_n=max(
            1, int(payload.get("recommender_top_n", defaults.recommender_top_n))
        ),
        recommender_demotion_alpha=max(
            0.0,
            float(
                payload.get(
                    "recommender_demotion_alpha",
                    defaults.recommender_demotion_alpha,
                )
            ),
        ),
        stats_min_district_offers=max(
            1,
            int(
                payload.get(
                    "stats_min_district_offers",
                    defaults.stats_min_district_offers,
                )
            ),
        ),
        stats_top_n=max(1, int(payload.get("stats_top_n", defaults.stats_top_n))),
        log_level=_coerce_log_level(
            payload.get("log_level", defaults.log_level), defaults.log_level
        ),
        model_opt_metric=coerce_opt_metric(
            payload.get("model_opt_metric", defaults.model_opt_metric),
            defaults.model_opt_metric,
        ),
        model_val_size=min(
            0.35,
            max(0.05, float(payload.get("model_val_size", defaults.model_val_size))),
        ),
        model_hpo_trials=max(
            1, int(payload.get("model_hpo_trials", defaults.model_hpo_trials))
        ),
        model_use_neighborhood=bool(
            payload.get("model_use_neighborhood", defaults.model_use_neighborhood)
        ),
        model_nbhd_radius_m=max(
            100.0,
            float(payload.get("model_nbhd_radius_m", defaults.model_nbhd_radius_m)),
        ),
    )


def _coerce_log_level(value: object, fallback: str) -> str:
    """Normalize a persisted log level to an uppercase logging name."""
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    name = str(value or fallback).strip().upper()
    return name if name in allowed else fallback


def save_app_settings(path: Path, settings: AppSettings) -> None:
    """Persist app settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote app settings to %s", path)
