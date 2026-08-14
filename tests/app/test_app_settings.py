"""Tests for persisted Admin app settings."""

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
    assert settings.olx_include_rent is True
    assert settings.scrape_olx_in_pipeline is True
    assert settings.scrape_nol_in_pipeline is True
    assert settings.scrape_otodom_in_pipeline is True
    assert settings.nol_include_rent is False
    assert settings.nol_incremental is True
    assert settings.nol_max_listings is None
    assert settings.gunb_wfs_full_history is True
    assert settings.download_rcn is True
    assert settings.download_gunb is True
    assert settings.olx_max_listings is None
    assert settings.otodom_max_listings is None
    assert settings.http_free_proxies_enabled is False
    assert settings.offer_comps_max_pins == 200
    assert settings.offer_comps_months == 0
    assert settings.offer_comps_include_metro is True
    assert settings.offer_comps_show_nearby_asks is True
    assert settings.stats_trend_min_n == 8


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
            filter_presets_max=12,
            map_view_presets_max=9,
            recommender_demotion_alpha=2.5,
            stats_min_district_offers=8,
            stats_top_n=20,
            log_level="DEBUG",
            model_opt_metric="rmse",
            model_hpo_trials=8,
            model_use_neighborhood=False,
            model_nbhd_radius_m=600.0,
            offer_comps_max_pins=120,
            offer_comps_months=18,
            offer_comps_include_metro=False,
            offer_comps_show_nearby_asks=False,
            stats_trend_min_n=12,
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
    assert loaded.filter_presets_max == 12
    assert loaded.map_view_presets_max == 9
    assert loaded.recommender_demotion_alpha == 2.5
    assert loaded.stats_min_district_offers == 8
    assert loaded.stats_top_n == 20
    assert loaded.log_level == "DEBUG"
    assert loaded.model_opt_metric == "rmse"
    assert loaded.model_hpo_trials == 8
    assert loaded.model_use_neighborhood is False
    assert loaded.model_nbhd_radius_m == 600.0
    assert loaded.offer_comps_max_pins == 120
    assert loaded.offer_comps_months == 18
    assert loaded.offer_comps_include_metro is False
    assert loaded.offer_comps_show_nearby_asks is False
    assert loaded.stats_trend_min_n == 12


def test_freshness_warn_settings_round_trip(tmp_path: Path) -> None:
    path = settings_path(tmp_path)
    save_app_settings(
        path,
        AppSettings(
            freshness_source_warn_h=72.0,
            freshness_derived_warn_h=48.0,
        ),
    )
    loaded = load_app_settings(path)
    assert loaded.freshness_source_warn_h == 72.0
    assert loaded.freshness_derived_warn_h == 48.0


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
            commute_display_mode="bike",
        ),
    )
    loaded = load_app_settings(path)
    assert loaded.commute_target_lon == 21.01
    assert loaded.commute_target_lat == 52.25
    assert loaded.commute_target_label == "Plac Bankowy"
    assert loaded.commute_features_lon == 20.983
    assert loaded.commute_display_mode == "bike"


def test_commute_display_mode_defaults_to_transit(tmp_path: Path) -> None:
    settings = load_app_settings(settings_path(tmp_path))
    assert settings.commute_display_mode == "transit"


def test_pipeline_ingest_settings_round_trip(tmp_path: Path) -> None:
    path = settings_path(tmp_path)
    save_app_settings(
        path,
        AppSettings(
            pipeline_overrides_enabled=True,
            otodom_max_pages=25,
            otodom_max_listings=None,
            otodom_incremental=False,
            rcn_max_rows=1000,
            download_gtfs=False,
            gtfs_zip_url="https://example.test/gtfs.zip",
            train_buyer_asks_only=False,
        ),
    )
    loaded = load_app_settings(path)
    assert loaded.pipeline_overrides_enabled is True
    assert loaded.otodom_max_pages == 25
    assert loaded.otodom_max_listings is None
    assert loaded.otodom_incremental is False
    assert loaded.rcn_max_rows == 1000
    assert loaded.download_gtfs is False
    assert loaded.gtfs_zip_url == "https://example.test/gtfs.zip"
    assert loaded.train_buyer_asks_only is False
