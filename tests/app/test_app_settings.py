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
            news_history_cap=50,
            recommender_demotion_alpha=2.5,
            stats_min_district_offers=8,
            stats_top_n=20,
            log_level="DEBUG",
        ),
    )
    loaded = load_app_settings(path)
    assert loaded.lm_model == "other-model"
    assert loaded.recommender_top_n == 7
    assert loaded.ai_review_on_offer_open is False
    assert loaded.news_history_cap == 50
    assert loaded.recommender_demotion_alpha == 2.5
    assert loaded.stats_min_district_offers == 8
    assert loaded.stats_top_n == 20
    assert loaded.log_level == "DEBUG"
