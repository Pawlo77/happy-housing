"""Unconventional market insight tables for the Stats page."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def chopin_surcharge(frame: pd.DataFrame) -> dict[str, float]:
    """Median zł/m² inside vs outside OOU — the aircraft-zone premium."""
    ppm = _num(frame, "price_per_m2")
    oou = _num(frame, "in_oou")
    work = pd.DataFrame({"ppm": ppm, "in_oou": oou}).dropna()
    inside = work.loc[work["in_oou"] == 1, "ppm"]
    outside = work.loc[work["in_oou"] != 1, "ppm"]
    med_in = float(inside.median()) if not inside.empty else float("nan")
    med_out = float(outside.median()) if not outside.empty else float("nan")
    surcharge = (
        (med_in / med_out - 1.0) * 100.0
        if med_out and med_out == med_out and med_out > 0 and med_in == med_in
        else float("nan")
    )
    return {
        "n_inside": float(len(inside)),
        "n_outside": float(len(outside)),
        "median_inside": med_in,
        "median_outside": med_out,
        "surcharge_pct": surcharge,
    }


def bargain_density(frame: pd.DataFrame, min_offers: int = 5) -> pd.DataFrame:
    """Share of underpriced offers (delta < 0) by district — where bargains hide."""
    empty = pd.DataFrame(
        columns=["district", "n", "bargain_share", "median_delta", "median_ppm"]
    )
    if frame.empty or "district" not in frame.columns:
        return empty
    delta = _num(frame, "delta_price_per_m2")
    work = pd.DataFrame({
        "district": frame["district"].astype(str),
        "delta": delta,
        "ppm": _num(frame, "price_per_m2"),
        "is_bargain": np.where(delta.isna(), np.nan, (delta < 0).astype(float)),
    })
    grouped = (
        work
        .groupby("district", dropna=True)
        .agg(
            n=("delta", "size"),
            bargain_share=("is_bargain", "mean"),
            median_delta=("delta", "median"),
            median_ppm=("ppm", "median"),
        )
        .reset_index()
    )
    out = grouped.loc[grouped["n"] >= min_offers].copy()
    if out.empty:
        return empty
    return out.sort_values("bargain_share", ascending=False).reset_index(drop=True)


def commute_cost_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by commute-minute bins — minutes that cost thousands."""
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "commute": _num(frame, "commute_min"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["commute_bin", "n", "median_price_per_m2"])
    bins = [0, 8, 12, 16, 20, 25, 35, 60, 120]
    labels = ["0-8", "8-12", "12-16", "16-20", "20-25", "25-35", "35-60", "60+"]
    work["commute_bin"] = pd.cut(
        work["commute"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    grouped = (
        work
        .groupby("commute_bin", observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["n"] > 0].copy()
    grouped["commute_bin"] = grouped["commute_bin"].astype(str)
    return grouped.reset_index(drop=True)


def ask_vs_ink(frame: pd.DataFrame, min_offers: int = 3) -> pd.DataFrame:
    """Otodom ask vs RCN ink medians by district — listing premium over deals."""
    empty = pd.DataFrame(
        columns=[
            "district",
            "ask_median",
            "ink_median",
            "ask_premium_pct",
            "n_ask",
            "n_ink",
        ]
    )
    if frame.empty or "district" not in frame.columns or "source" not in frame.columns:
        return empty
    work = pd.DataFrame({
        "district": frame["district"].astype(str),
        "source": frame["source"].astype(str).str.casefold(),
        "ppm": _num(frame, "price_per_m2"),
    }).dropna(subset=["ppm"])
    ask = (
        work
        .loc[work["source"].eq("otodom")]
        .groupby("district", dropna=True)["ppm"]
        .agg(ask_median="median", n_ask="size")
    )
    ink = (
        work
        .loc[work["source"].eq("rcn")]
        .groupby("district", dropna=True)["ppm"]
        .agg(ink_median="median", n_ink="size")
    )
    out = ask.join(ink, how="inner").reset_index()
    out = out.loc[(out["n_ask"] >= min_offers) & (out["n_ink"] >= min_offers)].copy()
    if out.empty:
        return empty
    out["ask_premium_pct"] = np.where(
        out["ink_median"] > 0,
        (out["ask_median"] / out["ink_median"] - 1.0) * 100.0,
        np.nan,
    )
    return out.sort_values("ask_premium_pct", ascending=False).reset_index(drop=True)


def size_sweet_spot(frame: pd.DataFrame) -> pd.DataFrame:
    """Median residual by area bucket — the small-flat trap / large-flat discount."""
    work = pd.DataFrame({
        "delta": _num(frame, "delta_price_per_m2"),
        "area": _num(frame, "area_m2"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["area_bin", "n", "median_delta"])
    bins = [0, 40, 55, 70, 90, 120, 180, 400]
    labels = [
        "micro <40",
        "compact 40-55",
        "classic 55-70",
        "family 70-90",
        "roomy 90-120",
        "large 120-180",
        "xl 180+",
    ]
    work["area_bin"] = pd.cut(
        work["area"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    return (
        work
        .groupby("area_bin", observed=False)
        .agg(n=("delta", "size"), median_delta=("delta", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(area_bin=lambda d: d["area_bin"].astype(str))
        .reset_index(drop=True)
    )


def room_ladder(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by room count — does an extra room buy efficiency?"""
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "rooms": _num(frame, "rooms"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["rooms", "n", "median_price_per_m2"])
    work["rooms"] = work["rooms"].round().astype(int)
    return (
        work
        .groupby("rooms", observed=True)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .sort_values("rooms")
        .reset_index(drop=True)
    )


def year_paradox(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by build-era — vintage premium vs new-build hype."""
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "year": _num(frame, "year_built"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["era", "n", "median_price_per_m2"])
    bins = [1900, 1945, 1970, 1990, 2005, 2015, 2035]
    labels = [
        "pre-war-ish <1945",
        "1945-1970",
        "1970-1990",
        "1990-2005",
        "2005-2015",
        "2015+",
    ]
    work["era"] = pd.cut(
        work["year"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    return (
        work
        .groupby("era", observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(era=lambda d: d["era"].astype(str))
        .reset_index(drop=True)
    )


def transit_gravity(
    frame: pd.DataFrame, threshold_m: float = 300.0
) -> dict[str, float]:
    """Share of offers within ``threshold_m`` of a transit stop."""
    dist = _num(frame, "dist_transit_stop_m").dropna()
    if dist.empty:
        return {"n": 0.0, "share_within": float("nan"), "threshold_m": threshold_m}
    return {
        "n": float(len(dist)),
        "share_within": float((dist <= threshold_m).mean()),
        "threshold_m": float(threshold_m),
        "median_dist_m": float(dist.median()),
    }


def safety_radius(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by fire-station distance bands — the safety-radius story."""
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "dist": _num(frame, "dist_fire_station_m"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["fire_band", "n", "median_price_per_m2"])
    bins = [0, 400, 800, 1500, 3000, 20000]
    labels = ["<400m", "400-800m", "800-1.5km", "1.5-3km", "3km+"]
    work["fire_band"] = pd.cut(
        work["dist"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    return (
        work
        .groupby("fire_band", observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(fire_band=lambda d: d["fire_band"].astype(str))
        .reset_index(drop=True)
    )


def market_temperature(frame: pd.DataFrame) -> pd.DataFrame:
    """District bubbles: median price vs median delta, sized by offer count."""
    empty = pd.DataFrame(
        columns=["district", "n", "median_ppm", "median_delta", "polarization"]
    )
    if frame.empty or "district" not in frame.columns:
        return empty
    work = pd.DataFrame({
        "district": frame["district"].astype(str),
        "ppm": _num(frame, "price_per_m2"),
        "delta": _num(frame, "delta_price_per_m2"),
    })
    grouped = (
        work
        .groupby("district", dropna=True)
        .agg(
            n=("delta", "size"),
            median_ppm=("ppm", "median"),
            median_delta=("delta", "median"),
            q75=("delta", lambda s: s.quantile(0.75)),
            q25=("delta", lambda s: s.quantile(0.25)),
        )
        .reset_index()
    )
    out = grouped.loc[
        (grouped["n"] >= 3)
        & grouped["median_ppm"].notna()
        & grouped["median_delta"].notna()
    ].copy()
    if out.empty:
        return empty
    out["polarization"] = out["q75"] - out["q25"]
    return (
        out
        .drop(columns=["q75", "q25"])
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def delta_volcano(frame: pd.DataFrame) -> pd.DataFrame:
    """Histogram-ready delta series with over/under labels."""
    delta = _num(frame, "delta_price_per_m2").dropna()
    if delta.empty:
        return pd.DataFrame(columns=["delta", "side"])
    values = delta.to_numpy()
    return pd.DataFrame({
        "delta": values,
        "side": np.where(values >= 0, "overpriced", "underpriced"),
    })
