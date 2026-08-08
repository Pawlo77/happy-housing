"""Lock project Streamlit TOML defaults used by local `make app`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def _load_streamlit_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_streamlit_config_enables_run_on_save() -> None:
    config = _load_streamlit_config()
    assert config["server"]["runOnSave"] is True


def test_streamlit_config_binds_local_headless_defaults() -> None:
    config = _load_streamlit_config()
    server = config["server"]
    assert server["headless"] is True
    assert server["address"] == "127.0.0.1"
    assert server["port"] == 8501
    browser = config["browser"]
    assert browser["gatherUsageStats"] is False
    assert browser["serverAddress"] == "127.0.0.1"
    assert browser["serverPort"] == 8501
