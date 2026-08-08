"""Tests for persisted Admin app settings."""

from __future__ import annotations

from pathlib import Path

from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)


def test_settings_path(tmp_path: Path) -> None:
    assert settings_path(tmp_path).name == "app_settings.json"


def test_load_missing_returns_defaults(tmp_path: Path) -> None:
    settings = load_app_settings(settings_path(tmp_path))
    assert settings.lm_model == AppSettings().lm_model
    assert settings.recommender_top_n == 15


def test_round_trip_settings(tmp_path: Path) -> None:
    path = settings_path(tmp_path)
    save_app_settings(
        path,
        AppSettings(
            lm_model="other-model",
            recommender_top_n=7,
            ai_review_on_offer_open=False,
            ai_review_system_prompt="Custom review prompt.",
            ai_review_include_proximity=False,
            ai_review_max_description_chars=1200,
            buyer_pref_notes="Prefer Wola",
            buyer_max_price_pln=800_000.0,
            buyer_avoid_oou=True,
            news_history_cap=50,
            recommender_demotion_alpha=2.5,
            stats_min_district_offers=8,
            stats_top_n=20,
            log_level="DEBUG",
            model_opt_metric="rmse",
            model_hpo_trials=8,
            model_use_neighborhood=False,
            model_nbhd_radius_m=600.0,
        ),
    )
    loaded = load_app_settings(path)
    assert loaded.lm_model == "other-model"
    assert loaded.recommender_top_n == 7
    assert loaded.ai_review_on_offer_open is False
    assert loaded.ai_review_system_prompt == "Custom review prompt."
    assert loaded.ai_review_include_proximity is False
    assert loaded.ai_review_max_description_chars == 1200
    assert loaded.buyer_pref_notes == "Prefer Wola"
    assert loaded.buyer_max_price_pln == 800_000.0
    assert loaded.buyer_avoid_oou is True
    assert loaded.news_history_cap == 50
    assert loaded.recommender_demotion_alpha == 2.5
    assert loaded.stats_min_district_offers == 8
    assert loaded.stats_top_n == 20
    assert loaded.log_level == "DEBUG"
    assert loaded.model_opt_metric == "rmse"
    assert loaded.model_hpo_trials == 8
    assert loaded.model_use_neighborhood is False
    assert loaded.model_nbhd_radius_m == 600.0


def test_commute_target_settings_round_trip(tmp_path: Path) -> None:
    path = settings_path(tmp_path)
    save_app_settings(
        path,
        AppSettings(
            commute_target_lon=21.01,
            commute_target_lat=52.25,
            commute_target_label="Plac Bankowy",
            commute_features_lon=20.983,
            commute_features_lat=52.23,
        ),
    )
    loaded = load_app_settings(path)
    assert loaded.commute_target_lon == 21.01
    assert loaded.commute_target_lat == 52.25
    assert loaded.commute_target_label == "Plac Bankowy"
    assert loaded.commute_features_lon == 20.983
