"""Pure helpers for Home-page market statistics and charts."""

from __future__ import annotations

from typing import Any

import pandas as pd


def market_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate interactive Home stats from a scored listings frame."""
    if frame.empty:
        return {
            "n_offers": 0,
            "by_source": {},
            "by_district": {},
            "median_price_per_m2": float("nan"),
            "median_delta": float("nan"),
            "share_overpriced": float("nan"),
            "share_oou": float("nan"),
        }
    by_source = (
        frame["source"].astype(str).value_counts().to_dict()
        if "source" in frame.columns
        else {}
    )
    by_district = (
        frame["district"].astype(str).value_counts().head(12).to_dict()
        if "district" in frame.columns
        else {}
    )
    ppm = pd.to_numeric(frame.get("price_per_m2"), errors="coerce")
    delta = pd.to_numeric(frame.get("delta_price_per_m2"), errors="coerce")
    in_oou = pd.to_numeric(frame.get("in_oou"), errors="coerce")
    n = len(frame)
    return {
        "n_offers": n,
        "by_source": {str(k): int(v) for k, v in by_source.items()},
        "by_district": {str(k): int(v) for k, v in by_district.items()},
        "median_price_per_m2": (
            float(ppm.median()) if ppm.notna().any() else float("nan")
        ),
        "median_delta": float(delta.median()) if delta.notna().any() else float("nan"),
        "share_overpriced": (
            float((delta > 0).mean()) if delta.notna().any() else float("nan")
        ),
        "share_oou": (
            float((in_oou == 1).mean()) if in_oou.notna().any() else float("nan")
        ),
    }


def counts_frame(
    mapping: dict[str, int],
    *,
    category: str,
    count: str = "count",
) -> pd.DataFrame:
    """Convert a count mapping into a chart-ready dataframe."""
    if not mapping:
        return pd.DataFrame(columns=[category, count])
    return (
        pd
        .DataFrame({
            category: list(mapping.keys()),
            count: list(mapping.values()),
        })
        .sort_values(count, ascending=False)
        .reset_index(drop=True)
    )
