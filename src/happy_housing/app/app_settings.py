"""Persisted Admin / app behaviour settings (not experiment YAML)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from happy_housing.app.io_cache import read_json
from happy_housing.gis.amenity_catalog import coerce_amenity_weights
from happy_housing.models.metrics import coerce_opt_metric

logger = logging.getLogger(__name__)

DEFAULT_LM_MODEL = "gemma-4-31b-it-mlx"
"""Default LM Studio model id for reviews / prefs."""

DEFAULT_EXTRACT_LM_MODEL = "gemma-3-4b-it"
"""Default lightweight LM Studio model for listing-attr extraction."""

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
    extract_lm_model: str = DEFAULT_EXTRACT_LM_MODEL
    """Lightweight LM Studio model for description attribute extraction."""
    extract_attrs_enabled: bool = True
    """Run short LM extract during pipeline / Admin backfill."""
    extract_max_chars: int = 900
    """Max description characters sent to the extract model."""
    extract_max_tokens: int = 48
    """Max tokens for extract responses (CSV line only)."""
    extract_max_listings: int = 200
    """Cap LM extract calls per pipeline/Admin run (0 = unlimited)."""
    extract_lm_workers: int = 2
    """Parallel LMS attr-extract workers (same model; keep low for LM Studio)."""
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
    """Include transit/metro/hospital/… distances and aircraft noise."""
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
    """Prefer offers outside aircraft Lden ≥55 dB END contours."""
    buyer_must_haves: str = ""
    """Free-text must-haves (balcony, garage, quiet street, …)."""
    news_history_cap: int = 200
    """Max starred-news events retained on disk."""
    filter_presets_max: int = 30
    """Max named Explore map filter presets retained on disk."""
    map_view_presets_max: int = 30
    """Max named Explore map-view (layer) presets retained on disk."""
    recommender_top_n: int = 15
    """How many promising offers to surface after retrain."""
    recommender_demotion_alpha: float = 1.0
    """Soft-demotion strength for already-seen recommendations."""
    recommender_w_bargain: float = 0.50
    """Weight for underpricing bargain term."""
    recommender_w_commute: float = 0.25
    """Weight for commute proximity term."""
    recommender_w_transit: float = 0.20
    """Weight for transit-stop proximity term."""
    recommender_w_overprice: float = 0.15
    """Penalty weight for overpriced asks."""
    recommender_otodom_pool_default: bool = True
    """When prefs sources empty, score Otodom/fixture only."""
    recommender_amenity_weights: dict[str, float] | None = None
    """Per-amenity score weights (-1..+1); None -> catalog defaults (all 0)."""
    stats_min_district_offers: int = 5
    """Minimum offers before a district appears on Stats scoreboards."""
    stats_top_n: int = 15
    """How many hunt / watch offers to list on Stats → Market."""
    stats_shap_sample_n: int = 300
    """Max listings sampled for global mean |SHAP| on Stats → Model."""
    stats_xai_top_n: int = 20
    """How many features to show on importance / SHAP charts."""
    log_level: str = "INFO"
    """Package log level for the Streamlit app (DEBUG/INFO/WARNING/ERROR)."""
    model_opt_metric: str = "mape"
    """Validation metric for LightGBM HPO (mae/rmse/MAPE/medae/r2)."""
    model_val_size: float = 0.15
    """Fraction of data used as HPO validation (test held out separately)."""
    model_hpo_trials: int = 80
    """Random hyperparameter trials (1 disables search)."""
    model_use_neighborhood: bool = True
    """Attach RCN neighborhood comps and log IDW baseline in MLflow."""
    model_use_engineered: bool = True
    """Interaction / log / district train-fold engineered features."""
    model_nbhd_radius_m: float = 800.0
    """Search radius (metres) for local RCN comparable sales."""
    model_log_target: bool = True
    """Train on log1p(zł/m²); report metrics in zł/m²."""
    model_district_calibration: bool = True
    """Apply district median residual offsets after fit."""
    model_early_stopping_rounds: int = 80
    """LightGBM early-stopping patience during HPO validation."""
    photo_cv_enabled: bool = True
    """Attach pretrained CLIP/heuristic photo features during pipeline enrich."""
    photo_cv_max_photos: int = 8
    """Max listing photos to score with CLIP."""
    photo_cv_model_id: str = "openai/clip-vit-base-patch32"
    """Hugging Face CLIP model id for photo features."""
    commute_target_lon: float | None = None
    """Admin override longitude; None → experiment.yaml."""
    commute_target_lat: float | None = None
    """Admin override latitude; None → experiment.yaml."""
    commute_target_label: str = ""
    """Human label for the commute destination (OSM place name)."""
    commute_features_lon: float | None = None
    """Lon last baked into scored dist/commute features."""
    commute_features_lat: float | None = None
    """Lat last baked into scored dist/commute features."""
    pipeline_overrides_enabled: bool = False
    """When True, Admin Pipeline ingest fields override experiment.yaml."""
    pipeline_force_refresh: bool = False
    """When True, ignore step fingerprints and re-run enrich/train/score."""
    otodom_max_pages: int | None = None
    """Otodom search-page cap; None = unlimited (when overrides enabled)."""
    otodom_max_listings: int | None = None
    """Otodom listing cap; None = unlimited (when overrides enabled)."""
    otodom_incremental: bool = True
    """Merge Otodom scrape into existing parquet."""
    rcn_max_rows: int | None = None
    """RCN extract row cap; None = full GML (when overrides enabled)."""
    scrape_otodom_in_pipeline: bool = True
    """Scrape Otodom when running the full pipeline."""
    use_osm: bool = True
    """Fetch OSM amenities + shops during pipeline."""
    snap_listings_out_of_parks: bool = True
    """Snap park/road/misplaced listing pins to nearest OSM building."""
    bike_path_coverage_radius_m: float = 500.0
    """Neighbourhood radius (m) for ``bike_path_coverage_m`` feature."""
    traffic_neighbourhood_radius_m: float = 300.0
    """Neighbourhood radius (m) for traffic weight sums around a listing."""
    download_gtfs: bool = True
    """Re-download GTFS during pipeline."""
    download_gios: bool = True
    """Re-download GIOŚ stations during pipeline."""
    aq_field_enabled: bool = True
    """Build IDW air-quality field from GIOŚ stations (model + map overlay)."""
    aq_idw_power: float = 2.0
    """Inverse-distance power for the GIOŚ AQ field."""
    download_rcn: bool = True
    """Re-download RCN during pipeline."""
    download_gunb: bool = True
    """Re-download GUNB during pipeline."""
    gunb_wfs_full_history: bool = False
    """Fetch all GUNB WFS year layers incl. pozostale (slow; default recent 3y)."""
    download_crime: bool = True
    """Re-download GUS Panorama dzielnic crime rates during pipeline."""
    gtfs_zip_url: str = ""
    """Override GTFS zip URL; empty → YAML."""
    gios_stations_url: str = ""
    """Override GIOŚ stations URL; empty → YAML."""
    rcn_zip_url: str = ""
    """Override RCN zip URL; empty → YAML."""
    gunb_zip_url: str = ""
    """Override GUNB zip URL; empty → YAML."""
    crime_panorama_7z_url: str = ""
    """Override Panorama crime .7z URL; empty → YAML."""
    train_otodom_labels_only: bool = True
    """Train LightGBM on Otodom/fixture asks; RCN used as comps only."""
    chat_enabled: bool = True
    """Show Explore Assistant popover and Chat tab."""
    chat_system_prompt: str = ""
    """Override Assistant system prompt; empty → built-in default."""
    chat_temperature: float = 0.4
    """LM Studio temperature for Assistant turns."""
    chat_max_tokens: int = 900
    """LM Studio maxTokens for Assistant turns."""
    chat_history_max_messages: int = 24
    """Max non-system history messages kept in the LMS context."""
    chat_max_tool_rounds: int = 6
    """Max LMS act prediction rounds (tool loops) per user turn."""


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
        extract_lm_model=str(
            payload.get("extract_lm_model") or defaults.extract_lm_model
        ).strip()
        or defaults.extract_lm_model,
        extract_attrs_enabled=bool(
            payload.get("extract_attrs_enabled", defaults.extract_attrs_enabled)
        ),
        extract_max_chars=max(
            200,
            int(payload.get("extract_max_chars", defaults.extract_max_chars)),
        ),
        extract_max_tokens=max(
            16,
            int(payload.get("extract_max_tokens", defaults.extract_max_tokens)),
        ),
        extract_max_listings=max(
            0,
            int(payload.get("extract_max_listings", defaults.extract_max_listings)),
        ),
        extract_lm_workers=max(
            1,
            min(
                8,
                int(payload.get("extract_lm_workers", defaults.extract_lm_workers)),
            ),
        ),
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
        filter_presets_max=max(
            1, int(payload.get("filter_presets_max", defaults.filter_presets_max))
        ),
        map_view_presets_max=max(
            1,
            int(payload.get("map_view_presets_max", defaults.map_view_presets_max)),
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
        recommender_w_bargain=max(
            0.0,
            float(payload.get("recommender_w_bargain", defaults.recommender_w_bargain)),
        ),
        recommender_w_commute=max(
            0.0,
            float(payload.get("recommender_w_commute", defaults.recommender_w_commute)),
        ),
        recommender_w_transit=max(
            0.0,
            float(payload.get("recommender_w_transit", defaults.recommender_w_transit)),
        ),
        recommender_w_overprice=max(
            0.0,
            float(
                payload.get("recommender_w_overprice", defaults.recommender_w_overprice)
            ),
        ),
        recommender_otodom_pool_default=bool(
            payload.get(
                "recommender_otodom_pool_default",
                defaults.recommender_otodom_pool_default,
            )
        ),
        recommender_amenity_weights=coerce_amenity_weights(
            payload.get(
                "recommender_amenity_weights",
                defaults.recommender_amenity_weights,
            )
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
        stats_shap_sample_n=max(
            20,
            min(
                2000,
                int(payload.get("stats_shap_sample_n", defaults.stats_shap_sample_n)),
            ),
        ),
        stats_xai_top_n=max(
            5,
            min(80, int(payload.get("stats_xai_top_n", defaults.stats_xai_top_n))),
        ),
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
        model_use_engineered=bool(
            payload.get("model_use_engineered", defaults.model_use_engineered)
        ),
        model_nbhd_radius_m=max(
            100.0,
            float(payload.get("model_nbhd_radius_m", defaults.model_nbhd_radius_m)),
        ),
        model_log_target=bool(
            payload.get("model_log_target", defaults.model_log_target)
        ),
        model_district_calibration=bool(
            payload.get(
                "model_district_calibration",
                defaults.model_district_calibration,
            )
        ),
        model_early_stopping_rounds=max(
            0,
            int(
                payload.get(
                    "model_early_stopping_rounds",
                    defaults.model_early_stopping_rounds,
                )
            ),
        ),
        photo_cv_enabled=bool(
            payload.get("photo_cv_enabled", defaults.photo_cv_enabled)
        ),
        photo_cv_max_photos=max(
            1,
            min(
                32,
                int(payload.get("photo_cv_max_photos", defaults.photo_cv_max_photos)),
            ),
        ),
        photo_cv_model_id=str(
            payload.get("photo_cv_model_id", defaults.photo_cv_model_id) or ""
        ).strip()
        or defaults.photo_cv_model_id,
        commute_target_lon=_opt_lon_lat(payload.get("commute_target_lon")),
        commute_target_lat=_opt_lon_lat(payload.get("commute_target_lat")),
        commute_target_label=str(
            payload.get("commute_target_label", defaults.commute_target_label) or ""
        ).strip(),
        commute_features_lon=_opt_lon_lat(payload.get("commute_features_lon")),
        commute_features_lat=_opt_lon_lat(payload.get("commute_features_lat")),
        pipeline_overrides_enabled=bool(
            payload.get(
                "pipeline_overrides_enabled",
                defaults.pipeline_overrides_enabled,
            )
        ),
        pipeline_force_refresh=bool(
            payload.get(
                "pipeline_force_refresh",
                defaults.pipeline_force_refresh,
            )
        ),
        otodom_max_pages=_opt_positive_int(payload.get("otodom_max_pages")),
        otodom_max_listings=_opt_positive_int(payload.get("otodom_max_listings")),
        otodom_incremental=bool(
            payload.get("otodom_incremental", defaults.otodom_incremental)
        ),
        rcn_max_rows=_opt_positive_int(payload.get("rcn_max_rows")),
        scrape_otodom_in_pipeline=bool(
            payload.get(
                "scrape_otodom_in_pipeline",
                defaults.scrape_otodom_in_pipeline,
            )
        ),
        use_osm=bool(payload.get("use_osm", defaults.use_osm)),
        snap_listings_out_of_parks=bool(
            payload.get(
                "snap_listings_out_of_parks",
                defaults.snap_listings_out_of_parks,
            )
        ),
        bike_path_coverage_radius_m=max(
            50.0,
            float(
                payload.get(
                    "bike_path_coverage_radius_m",
                    defaults.bike_path_coverage_radius_m,
                )
            ),
        ),
        traffic_neighbourhood_radius_m=max(
            50.0,
            float(
                payload.get(
                    "traffic_neighbourhood_radius_m",
                    defaults.traffic_neighbourhood_radius_m,
                )
            ),
        ),
        download_gtfs=bool(payload.get("download_gtfs", defaults.download_gtfs)),
        download_gios=bool(payload.get("download_gios", defaults.download_gios)),
        aq_field_enabled=bool(
            payload.get("aq_field_enabled", defaults.aq_field_enabled)
        ),
        aq_idw_power=min(
            4.0,
            max(0.5, float(payload.get("aq_idw_power", defaults.aq_idw_power))),
        ),
        download_rcn=bool(payload.get("download_rcn", defaults.download_rcn)),
        download_gunb=bool(payload.get("download_gunb", defaults.download_gunb)),
        gunb_wfs_full_history=bool(
            payload.get("gunb_wfs_full_history", defaults.gunb_wfs_full_history)
        ),
        download_crime=bool(payload.get("download_crime", defaults.download_crime)),
        gtfs_zip_url=str(payload.get("gtfs_zip_url", defaults.gtfs_zip_url) or ""),
        gios_stations_url=str(
            payload.get("gios_stations_url", defaults.gios_stations_url) or ""
        ),
        rcn_zip_url=str(payload.get("rcn_zip_url", defaults.rcn_zip_url) or ""),
        gunb_zip_url=str(payload.get("gunb_zip_url", defaults.gunb_zip_url) or ""),
        crime_panorama_7z_url=str(
            payload.get(
                "crime_panorama_7z_url",
                defaults.crime_panorama_7z_url,
            )
            or ""
        ),
        train_otodom_labels_only=bool(
            payload.get(
                "train_otodom_labels_only",
                defaults.train_otodom_labels_only,
            )
        ),
        chat_enabled=bool(payload.get("chat_enabled", defaults.chat_enabled)),
        chat_system_prompt=str(
            payload.get("chat_system_prompt", defaults.chat_system_prompt) or ""
        ),
        chat_temperature=min(
            1.5,
            max(
                0.0,
                float(payload.get("chat_temperature", defaults.chat_temperature)),
            ),
        ),
        chat_max_tokens=max(
            64, int(payload.get("chat_max_tokens", defaults.chat_max_tokens))
        ),
        chat_history_max_messages=max(
            4,
            int(
                payload.get(
                    "chat_history_max_messages",
                    defaults.chat_history_max_messages,
                )
            ),
        ),
        chat_max_tool_rounds=max(
            1,
            int(payload.get("chat_max_tool_rounds", defaults.chat_max_tool_rounds)),
        ),
    )


def _opt_lon_lat(value: object) -> float | None:
    """Parse optional longitude/latitude; blank → None."""
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


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
