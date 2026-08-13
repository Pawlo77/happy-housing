"""DataFrame / GeoDataFrame column-join helpers for GIS enrich."""

from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import pandas as pd


def assign_columns(
    frame: pd.DataFrame,
    columns: Mapping[str, Any],
) -> pd.DataFrame:
    """Join many columns in one ``concat`` (avoids fragmented BlockManager).

    Repeated ``frame[col] = …`` on a wide GeoDataFrame triggers pandas
    ``PerformanceWarning`` and slows enrichment. Prefer this helper when
    attaching a block of new feature columns.
    """
    if not columns:
        return frame
    extra = pd.DataFrame(dict(columns), index=frame.index)
    overlap = [c for c in extra.columns if c in frame.columns]
    base = frame.drop(columns=overlap) if overlap else frame
    joined = pd.concat([base, extra], axis=1)
    if isinstance(frame, gpd.GeoDataFrame):
        geom_col = frame.geometry.name
        return gpd.GeoDataFrame(joined, geometry=geom_col, crs=frame.crs)
    return joined
