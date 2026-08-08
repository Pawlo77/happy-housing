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
