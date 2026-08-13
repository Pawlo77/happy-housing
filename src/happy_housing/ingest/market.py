"""Derive ``market_primary`` from Otodom / RCN market fields."""

from pathlib import Path
from typing import Any

import pandas as pd

# RCN GML: RCW_RodzajRynku — rynekPierwotny=1, rynekWtorny=2
_RCN_PRIMARY = 1
_RCN_SECONDARY = 2


def market_primary_from_rodzaj_rynku(value: Any) -> float:
    """Map RCN ``rodzaj_rynku`` code to 1 (primary), 0 (secondary), or NaN."""
    number = pd.to_numeric(value, errors="coerce")
    try:
        as_float = float(number)
    except (TypeError, ValueError):
        return float("nan")
    if as_float != as_float:
        return float("nan")
    if int(as_float) == _RCN_PRIMARY:
        return 1.0
    if int(as_float) == _RCN_SECONDARY:
        return 0.0
    return float("nan")


def ensure_market_primary(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``market_primary`` exists, filled from ``market`` / ``rodzaj_rynku``."""
    if frame.empty:
        out = frame.copy()
        if "market_primary" not in out.columns:
            out["market_primary"] = pd.Series(dtype="Float64")
        return out

    out = frame.copy()
    if "market_primary" not in out.columns:
        out["market_primary"] = pd.NA
    current = pd.to_numeric(out["market_primary"], errors="coerce")

    if "market" in out.columns:
        market = out["market"].astype(str).str.casefold().str.strip()
        from_market = pd.Series(float("nan"), index=out.index, dtype="float64")
        from_market = from_market.mask(market.eq("primary"), 1.0)
        from_market = from_market.mask(market.eq("secondary"), 0.0)
        current = current.where(current.notna(), from_market)

    if "rodzaj_rynku" in out.columns:
        from_rr = out["rodzaj_rynku"].map(market_primary_from_rodzaj_rynku)
        current = current.where(current.notna(), from_rr)

    out["market_primary"] = current.astype("Float64")
    return out


def backfill_market_primary(
    frame: pd.DataFrame,
    otodom_path: Path | None = None,
) -> pd.DataFrame:
    """Ensure flag, then fill gaps from Otodom parquet by ``listing_id``."""
    out = ensure_market_primary(frame)
    if (
        otodom_path is None
        or not otodom_path.exists()
        or "listing_id" not in out.columns
    ):
        return out
    try:
        oto = pd.read_parquet(otodom_path)
    except (OSError, ValueError):
        return out
    if oto.empty or "listing_id" not in oto.columns:
        return out
    oto = ensure_market_primary(oto)
    patch = oto.loc[:, ["listing_id", "market_primary"]].rename(
        columns={"market_primary": "_market_primary_oto"}
    )
    merged = out.merge(patch, on="listing_id", how="left")
    base = pd.to_numeric(merged["market_primary"], errors="coerce")
    oto_vals = pd.to_numeric(merged["_market_primary_oto"], errors="coerce")
    merged["market_primary"] = base.where(base.notna(), oto_vals).astype("Float64")
    return merged.drop(columns=["_market_primary_oto"])
