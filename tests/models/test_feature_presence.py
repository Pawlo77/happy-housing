"""Hard fail when model feature columns are absent (no silent invent)."""

from __future__ import annotations

import pandas as pd
import pytest

from happy_housing.models.train import (
    _raw_feature_frame,
    check_feature_coverage,
)


def test_check_feature_coverage_raises_on_absent_columns() -> None:
    frame = pd.DataFrame({"area_m2": [50.0, 60.0], "rooms": [2.0, 3.0]})
    with pytest.raises(ValueError, match="absent on frame"):
        check_feature_coverage(
            frame,
            ("area_m2", "rooms", "year_built"),
            min_usable=1,
        )


def test_raw_feature_frame_raises_on_absent_columns() -> None:
    frame = pd.DataFrame({"area_m2": [50.0]})
    with pytest.raises(ValueError, match="absent on frame"):
        _raw_feature_frame(frame, ("area_m2", "rooms"))


def test_check_feature_coverage_ok_when_all_present() -> None:
    from happy_housing.vision.photo_features import PHOTO_FEATURE_COLUMNS

    cols = ("area_m2", "rooms", *PHOTO_FEATURE_COLUMNS[:2])
    frame = pd.DataFrame({c: [1.0, 2.0] for c in cols})
    check_feature_coverage(frame, cols, min_usable=2)
