"""Config loader tests including new data/model sections."""

from pathlib import Path

import pytest

from happy_housing.config import (
    ConfigValidationError,
    ExperimentConfig,
    load_experiment_config,
)


def test_load_default_experiment_yaml() -> None:
    """Default config file parses with expected experiment name and paths."""
    config = load_experiment_config(Path("configs/experiment.yaml"))
    assert config.name == "happy-housing"
    assert config.mlflow.experiment_name == "happy-housing"
    assert config.mlflow.enabled is True
    assert config.data.fixtures == "data/fixtures"
    assert config.model.target == "price_per_m2"
    assert config.bbox.west < config.bbox.east
    assert config.ingest.otodom_max_pages is None
    assert config.ingest.otodom_max_listings is None
    assert config.ingest.otodom_incremental is True
    assert config.ingest.rcn_max_rows is None
    assert "wynik_mazowieckie.zip" in config.ingest_urls.gunb_zip
    assert config.logging.level == "INFO"
    assert config.routing.osrm_url.endswith(":5001")
    assert config.routing.otp_url.endswith(":8080")
    assert config.routing.use_otp_transit is False
    assert config.routing.require_osrm is True


def test_logging_config_from_dict() -> None:
    config = ExperimentConfig.from_dict({
        "name": "logs",
        "logging": {"level": "debug"},
    })
    assert config.logging.level == "DEBUG"


def test_ingest_limits_null_means_unlimited() -> None:
    """YAML null / non-positive ingest caps resolve to unlimited."""
    config = ExperimentConfig.from_dict({
        "name": "caps",
        "ingest": {
            "otodom_max_pages": None,
            "otodom_max_listings": 0,
            "rcn_max_rows": 100,
        },
    })
    assert config.ingest.otodom_max_pages is None
    assert config.ingest.otodom_max_listings is None
    assert config.ingest.rcn_max_rows == 100


def test_experiment_config_requires_name() -> None:
    """Empty experiment names are rejected."""
    with pytest.raises(ConfigValidationError):
        ExperimentConfig.from_dict({"name": "   "})


def test_mlflow_legacy_uri_normalized() -> None:
    """Legacy mlruns URI maps to SQLite backend."""
    config = ExperimentConfig.from_dict({
        "name": "run-a",
        "mlflow": {"tracking_uri": "mlruns"},
    })
    assert config.mlflow.tracking_uri == "sqlite:///mlruns.db"
