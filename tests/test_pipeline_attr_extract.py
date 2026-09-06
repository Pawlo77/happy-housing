"""Pipeline listing-attr extract runs whenever the setting is on."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from happy_housing.app.app_settings import AppSettings
from happy_housing.pipeline import apply_listing_attr_extract


def test_apply_listing_attr_extract_skips_when_disabled() -> None:
    frame = pd.DataFrame({"listing_id": ["a"], "description": ["ok"]})
    with patch("happy_housing.pipeline.enrich_frame_listing_attrs") as mock:
        out, changed = apply_listing_attr_extract(
            frame,
            AppSettings(extract_attrs_enabled=False),
            Path("."),
        )
    mock.assert_not_called()
    assert changed is False
    assert out is frame


def test_apply_listing_attr_extract_runs_when_enabled() -> None:
    frame = pd.DataFrame({
        "listing_id": ["a"],
        "description": ["renovated 2019"],
        "attr_extract_fp": [pd.NA],
    })
    filled = frame.copy()
    filled["attr_extract_fp"] = "abc"
    filled["year_renovated"] = 2019
    with patch(
        "happy_housing.pipeline.enrich_frame_listing_attrs",
        return_value=filled,
    ) as mock:
        out, changed = apply_listing_attr_extract(
            frame,
            AppSettings(extract_attrs_enabled=True, extract_max_listings=0),
            Path("."),
        )
    mock.assert_called_once()
    assert mock.call_args.kwargs["max_llm_calls"] is None
    assert changed is True
    assert int(out["year_renovated"].iloc[0]) == 2019
