"""Persisted Admin / app behaviour settings (not experiment YAML)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from happy_housing.models.metrics import coerce_opt_metric

logger = logging.getLogger(__name__)

DEFAULT_LM_MODEL = "gemma-4-31b-it-mlx"
"""Default LM Studio model id."""


@dataclass(slots=True)
class AppSettings:
    """User-tunable app behaviour stored under processed data."""

    lm_model: str = DEFAULT_LM_MODEL
    """LM Studio model id for offer AI reviews / NL tools."""
    ai_review_on_offer_open: bool = True
    """Auto-request AI review when opening an Offer overview."""
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


def load_app_settings(path: Path) -> AppSettings:
    """Load settings; missing/invalid → defaults."""
    defaults = AppSettings()
    if not path.exists():
        return defaults
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load app settings %s: %s", path, exc)
        return defaults
    if not isinstance(payload, dict):
        return defaults
    return AppSettings(
        lm_model=str(payload.get("lm_model") or defaults.lm_model).strip()
        or defaults.lm_model,
        ai_review_on_offer_open=bool(
            payload.get("ai_review_on_offer_open", defaults.ai_review_on_offer_open)
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
