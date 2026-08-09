"""Business-oriented market analytics helpers for the Stats page."""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from happy_housing.ingest.market import ensure_market_primary


def normalize_district(value: object) -> str:
    """Casefold, strip, and drop diacritics for stable district grouping."""
    text = str(value).strip().casefold()
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def _with_district(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "district" in out.columns:
        out["district"] = out["district"].map(normalize_district)
    return out


def chopin_surcharge(frame: pd.DataFrame) -> dict[str, float]:
    """Median zł/m² inside vs outside Lden ≥55 dB — aircraft-noise premium."""
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
    work = _with_district(frame)
    delta = _num(work, "delta_price_per_m2")
    table = pd.DataFrame({
        "district": work["district"].astype(str),
        "delta": delta,
        "ppm": _num(work, "price_per_m2"),
        "is_bargain": np.where(delta.isna(), np.nan, (delta < 0).astype(float)),
    })
    grouped = (
        table
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


def district_scoreboard(frame: pd.DataFrame, min_offers: int = 5) -> pd.DataFrame:
    """District value board: ask vs model medians, bargain share, volume."""
    empty = pd.DataFrame(
        columns=[
            "district",
            "n",
            "median_ppm",
            "median_model",
            "median_delta",
            "bargain_share",
            "share_overpriced",
        ]
    )
    if frame.empty or "district" not in frame.columns:
        return empty
    work = _with_district(frame)
    table = pd.DataFrame({
        "district": work["district"].astype(str),
        "ppm": _num(work, "price_per_m2"),
        "model": _num(work, "model_price_per_m2"),
        "delta": _num(work, "delta_price_per_m2"),
    })
    table["is_bargain"] = np.where(
        table["delta"].isna(), np.nan, (table["delta"] < 0).astype(float)
    )
    table["is_over"] = np.where(
        table["delta"].isna(), np.nan, (table["delta"] >= 0).astype(float)
    )
    grouped = (
        table
        .groupby("district", dropna=True)
        .agg(
            n=("delta", "size"),
            median_ppm=("ppm", "median"),
            median_model=("model", "median"),
            median_delta=("delta", "median"),
            bargain_share=("is_bargain", "mean"),
            share_overpriced=("is_over", "mean"),
        )
        .reset_index()
    )
    out = grouped.loc[grouped["n"] >= min_offers].copy()
    if out.empty:
        return empty
    return out.sort_values("median_delta", ascending=True).reset_index(drop=True)


def monthly_trends(frame: pd.DataFrame) -> pd.DataFrame:
    """Median listing/model/delta and volume by listed month."""
    empty = pd.DataFrame(
        columns=[
            "listed_month",
            "n",
            "median_price_per_m2",
            "median_model",
            "median_delta",
        ]
    )
    if frame.empty or "listed_month" not in frame.columns:
        return empty
    table = pd.DataFrame({
        "listed_month": frame["listed_month"].astype(str),
        "ppm": _num(frame, "price_per_m2"),
        "model": _num(frame, "model_price_per_m2"),
        "delta": _num(frame, "delta_price_per_m2"),
    })
    table = table.loc[table["listed_month"].ne("unknown")].copy()
    if table.empty:
        return empty
    return (
        table
        .groupby("listed_month", dropna=True)
        .agg(
            n=("ppm", "size"),
            median_price_per_m2=("ppm", "median"),
            median_model=("model", "median"),
            median_delta=("delta", "median"),
        )
        .reset_index()
        .sort_values("listed_month")
        .reset_index(drop=True)
    )


def top_value_offers(
    frame: pd.DataFrame,
    n: int = 15,
    side: str = "underpriced",
) -> pd.DataFrame:
    """Rank offers by residual for a hunt / avoid list."""
    cols = [
        "listing_id",
        "district",
        "source",
        "price_per_m2",
        "model_price_per_m2",
        "delta_price_per_m2",
        "area_m2",
        "rooms",
        "commute_min",
    ]
    empty = pd.DataFrame(columns=cols)
    if frame.empty or "delta_price_per_m2" not in frame.columns:
        return empty
    work = frame.copy()
    work["delta_price_per_m2"] = _num(work, "delta_price_per_m2")
    work = work.dropna(subset=["delta_price_per_m2"])
    if side == "underpriced":
        work = work.loc[work["delta_price_per_m2"] < 0].sort_values(
            "delta_price_per_m2", ascending=True
        )
    else:
        work = work.loc[work["delta_price_per_m2"] >= 0].sort_values(
            "delta_price_per_m2", ascending=False
        )
    if "district" in work.columns:
        work["district"] = work["district"].map(normalize_district)
    keep = [c for c in cols if c in work.columns]
    return work.loc[:, keep].head(max(1, int(n))).reset_index(drop=True)


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


def transit_price_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by distance to nearest GTFS stop."""
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "dist": _num(frame, "dist_transit_stop_m"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["transit_band", "n", "median_price_per_m2"])
    bins = [0, 150, 300, 500, 800, 1500, 50000]
    labels = ["<150m", "150-300m", "300-500m", "500-800m", "800m-1.5km", "1.5km+"]
    work["transit_band"] = pd.cut(
        work["dist"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    return (
        work
        .groupby("transit_band", observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(transit_band=lambda d: d["transit_band"].astype(str))
        .reset_index(drop=True)
    )


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
    work = _with_district(frame)
    table = pd.DataFrame({
        "district": work["district"].astype(str),
        "source": work["source"].astype(str).str.casefold(),
        "ppm": _num(work, "price_per_m2"),
    }).dropna(subset=["ppm"])
    ask = (
        table
        .loc[table["source"].eq("otodom")]
        .groupby("district", dropna=True)["ppm"]
        .agg(ask_median="median", n_ask="size")
    )
    ink = (
        table
        .loc[table["source"].eq("rcn")]
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
    work = _with_district(frame)
    table = pd.DataFrame({
        "district": work["district"].astype(str),
        "ppm": _num(work, "price_per_m2"),
        "delta": _num(work, "delta_price_per_m2"),
    })
    grouped = (
        table
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


def market_kpis(frame: pd.DataFrame) -> dict[str, float]:
    """Headline KPIs for the filtered Stats universe."""
    ppm = _num(frame, "price_per_m2")
    delta = _num(frame, "delta_price_per_m2")
    model = _num(frame, "model_price_per_m2")
    n = float(len(frame))
    share_over = (
        float((delta.dropna() >= 0).mean()) if delta.notna().any() else float("nan")
    )
    return {
        "n_offers": n,
        "median_price_per_m2": float(ppm.median())
        if ppm.notna().any()
        else float("nan"),
        "median_model": float(model.median()) if model.notna().any() else float("nan"),
        "median_delta": float(delta.median()) if delta.notna().any() else float("nan"),
        "share_overpriced": share_over,
    }


def price_band_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Offer volume and residual by listing zł/m² bands (inventory pressure)."""
    empty = pd.DataFrame(columns=["price_band", "n", "share", "median_delta"])
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "delta": _num(frame, "delta_price_per_m2"),
    }).dropna(subset=["ppm"])
    if work.empty:
        return empty
    bins = [0, 12_000, 15_000, 18_000, 22_000, 28_000, 40_000, 200_000]
    labels = [
        "<12k",
        "12-15k",
        "15-18k",
        "18-22k",
        "22-28k",
        "28-40k",
        "40k+",
    ]
    work["price_band"] = pd.cut(
        work["ppm"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    grouped = (
        work
        .groupby("price_band", observed=False)
        .agg(n=("ppm", "size"), median_delta=("delta", "median"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["n"] > 0].copy()
    total = float(grouped["n"].sum()) or 1.0
    grouped["share"] = grouped["n"] / total
    grouped["price_band"] = grouped["price_band"].astype(str)
    return grouped.reset_index(drop=True)


def source_mix_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Ask vs transaction source mix with median zł/m² and delta."""
    empty = pd.DataFrame(columns=["source", "n", "share", "median_ppm", "median_delta"])
    if frame.empty or "source" not in frame.columns:
        return empty
    work = pd.DataFrame({
        "source": frame["source"].astype(str).str.casefold(),
        "ppm": _num(frame, "price_per_m2"),
        "delta": _num(frame, "delta_price_per_m2"),
    })
    grouped = (
        work
        .groupby("source", dropna=True)
        .agg(
            n=("source", "size"),
            median_ppm=("ppm", "median"),
            median_delta=("delta", "median"),
        )
        .reset_index()
    )
    if grouped.empty:
        return empty
    total = float(grouped["n"].sum()) or 1.0
    grouped["share"] = grouped["n"] / total
    return grouped.sort_values("n", ascending=False).reset_index(drop=True)


def rooms_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Supply mix by room count — what the market is stocking."""
    empty = pd.DataFrame(columns=["rooms", "n", "share", "median_ppm"])
    work = pd.DataFrame({
        "rooms": _num(frame, "rooms"),
        "ppm": _num(frame, "price_per_m2"),
    }).dropna(subset=["rooms"])
    if work.empty:
        return empty
    work["rooms"] = work["rooms"].round().astype(int)
    grouped = (
        work
        .groupby("rooms", observed=True)
        .agg(n=("rooms", "size"), median_ppm=("ppm", "median"))
        .reset_index()
    )
    total = float(grouped["n"].sum()) or 1.0
    grouped["share"] = grouped["n"] / total
    return grouped.sort_values("rooms").reset_index(drop=True)


def metro_price_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by metro-station distance (walkability premium)."""
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "dist": _num(frame, "dist_metro_m"),
    }).dropna()
    if work.empty:
        return pd.DataFrame(columns=["metro_band", "n", "median_price_per_m2"])
    bins = [0, 300, 600, 1000, 1500, 3000, 50000]
    labels = ["<300m", "300-600m", "600m-1km", "1-1.5km", "1.5-3km", "3km+"]
    work["metro_band"] = pd.cut(
        work["dist"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    return (
        work
        .groupby("metro_band", observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(metro_band=lambda d: d["metro_band"].astype(str))
        .reset_index(drop=True)
    )


def aircraft_noise_ladder(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by aircraft Lden band (END contour lower edge)."""
    return _db_band_ladder(frame, value_col="aircraft_lden_db", band_col="lden_band")


def noise_total_ladder(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by energy-sum total Lden band (all UM sources)."""
    return _db_band_ladder(frame, value_col="noise_total_lden_db", band_col="lden_band")


def _db_band_ladder(
    frame: pd.DataFrame,
    *,
    value_col: str,
    band_col: str,
) -> pd.DataFrame:
    """Shared Lden-style band ladder for aircraft / total / source columns."""
    empty = pd.DataFrame(columns=[band_col, "n", "median_price_per_m2"])
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "db": _num(frame, value_col),
    }).dropna(subset=["ppm"])
    if work.empty:
        return empty
    work["db"] = work["db"].fillna(0.0)
    bins = [-0.1, 0.1, 45, 50, 55, 60, 65, 70, 100]
    labels = [
        "0 / below",
        "<45",
        "45-50",
        "50-55",
        "55-60",
        "60-65",
        "65-70",
        "70+",
    ]
    work[band_col] = pd.cut(
        work["db"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    return (
        work
        .groupby(band_col, observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(**{band_col: lambda d: d[band_col].astype(str)})
        .reset_index(drop=True)
    )


def high_noise_surcharge(frame: pd.DataFrame) -> dict[str, float]:
    """Median zł/m² inside vs outside total Lden ≥55 dB."""
    ppm = _num(frame, "price_per_m2")
    flag = _num(frame, "in_high_noise")
    work = pd.DataFrame({"ppm": ppm, "flag": flag}).dropna()
    inside = work.loc[work["flag"] == 1, "ppm"]
    outside = work.loc[work["flag"] != 1, "ppm"]
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


_NOISE_SOURCE_COLS: tuple[tuple[str, str], ...] = (
    ("road", "road_lden_db"),
    ("rail", "rail_lden_db"),
    ("tram", "tram_lden_db"),
    ("aircraft", "aircraft_lden_db"),
    ("industry", "industry_lden_db"),
)
"""Display label → listing column for source exposure premiums."""


def noise_source_premiums(
    frame: pd.DataFrame,
    *,
    threshold_db: float = 55.0,
) -> pd.DataFrame:
    """Median zł/m² premium when each noise source is ≥ ``threshold_db``."""
    empty = pd.DataFrame(
        columns=[
            "source",
            "n_exposed",
            "n_quiet",
            "median_ppm_exposed",
            "median_ppm_quiet",
            "premium_pct",
        ]
    )
    ppm = _num(frame, "price_per_m2")
    rows: list[dict[str, float | str]] = []
    for label, col in _NOISE_SOURCE_COLS:
        level = _num(frame, col)
        work = pd.DataFrame({"ppm": ppm, "db": level}).dropna(subset=["ppm"])
        if work.empty:
            continue
        work["db"] = work["db"].fillna(0.0)
        exposed = work.loc[work["db"] >= threshold_db, "ppm"]
        quiet = work.loc[work["db"] < threshold_db, "ppm"]
        med_e = float(exposed.median()) if not exposed.empty else float("nan")
        med_q = float(quiet.median()) if not quiet.empty else float("nan")
        premium = (
            (med_e / med_q - 1.0) * 100.0
            if med_q and med_q == med_q and med_q > 0 and med_e == med_e
            else float("nan")
        )
        rows.append({
            "source": label,
            "n_exposed": float(len(exposed)),
            "n_quiet": float(len(quiet)),
            "median_ppm_exposed": med_e,
            "median_ppm_quiet": med_q,
            "premium_pct": premium,
        })
    if not rows:
        return empty
    return pd.DataFrame(rows)


def distance_price_curve(
    frame: pd.DataFrame,
    *,
    dist_col: str,
    bins: list[float],
    labels: list[str],
    band_col: str,
) -> pd.DataFrame:
    """Median zł/m² by distance band for any metres column."""
    empty = pd.DataFrame(columns=[band_col, "n", "median_price_per_m2"])
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "dist": _num(frame, dist_col),
    }).dropna()
    if work.empty:
        return empty
    work[band_col] = pd.cut(
        work["dist"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return (
        work
        .groupby(band_col, observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .assign(**{band_col: lambda d: d[band_col].astype(str)})
        .reset_index(drop=True)
    )


def highway_class_price_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² and Δ by nearest OSM highway class."""
    empty = pd.DataFrame(
        columns=["nearest_highway", "n", "median_price_per_m2", "median_delta"]
    )
    if "nearest_highway" not in frame.columns:
        return empty
    work = pd.DataFrame({
        "nearest_highway": frame["nearest_highway"].astype(str).fillna("unknown"),
        "ppm": _num(frame, "price_per_m2"),
        "delta": _num(frame, "delta_price_per_m2"),
    }).dropna(subset=["ppm"])
    if work.empty:
        return empty
    return (
        work
        .groupby("nearest_highway", observed=False)
        .agg(
            n=("ppm", "size"),
            median_price_per_m2=("ppm", "median"),
            median_delta=("delta", "median"),
        )
        .reset_index()
        .loc[lambda d: d["n"] > 0]
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def frontage_hint_price_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² / Δ for street-facing vs quiet-side description hints."""
    empty = pd.DataFrame(
        columns=["frontage", "n", "median_price_per_m2", "median_delta"]
    )
    rows: list[dict[str, float | str]] = []
    for label, col in (
        ("street_facing", "street_facing_hint"),
        ("quiet_side", "quiet_side_hint"),
    ):
        if col not in frame.columns:
            continue
        flagged = frame.loc[pd.to_numeric(frame[col], errors="coerce").eq(1)]
        if flagged.empty:
            continue
        ppm = _num(flagged, "price_per_m2").dropna()
        delta = _num(flagged, "delta_price_per_m2")
        if ppm.empty:
            continue
        rows.append({
            "frontage": label,
            "n": float(len(ppm)),
            "median_price_per_m2": float(ppm.median()),
            "median_delta": float(delta.dropna().median())
            if delta.notna().any()
            else float("nan"),
        })
    if not rows:
        return empty
    return pd.DataFrame(rows)


def air_quality_ladder(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² by nearest-station AQI index."""
    empty = pd.DataFrame(columns=["aqi_band", "n", "median_price_per_m2"])
    work = pd.DataFrame({
        "ppm": _num(frame, "price_per_m2"),
        "aqi": _num(frame, "air_aq_index"),
    }).dropna()
    if work.empty:
        return empty
    work["aqi_band"] = work["aqi"].round().astype(int).astype(str)
    return (
        work
        .groupby("aqi_band", observed=False)
        .agg(n=("ppm", "size"), median_price_per_m2=("ppm", "median"))
        .reset_index()
        .sort_values("aqi_band")
        .reset_index(drop=True)
    )


def affordability_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Total asking-price bands with typical area — what money buys."""
    empty = pd.DataFrame(
        columns=["price_band", "n", "share", "median_area_m2", "median_ppm"]
    )
    work = pd.DataFrame({
        "price": _num(frame, "price_pln"),
        "area": _num(frame, "area_m2"),
        "ppm": _num(frame, "price_per_m2"),
    }).dropna(subset=["price"])
    if work.empty:
        return empty
    bins = [
        0,
        500_000,
        750_000,
        1_000_000,
        1_250_000,
        1_500_000,
        2_000_000,
        20_000_000,
    ]
    labels = [
        "<0.5M",
        "0.5-0.75M",
        "0.75-1M",
        "1-1.25M",
        "1.25-1.5M",
        "1.5-2M",
        "2M+",
    ]
    work["price_band"] = pd.cut(
        work["price"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    grouped = (
        work
        .groupby("price_band", observed=False)
        .agg(
            n=("price", "size"),
            median_area_m2=("area", "median"),
            median_ppm=("ppm", "median"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["n"] > 0].copy()
    total = float(grouped["n"].sum()) or 1.0
    grouped["share"] = grouped["n"] / total
    grouped["price_band"] = grouped["price_band"].astype(str)
    return grouped.reset_index(drop=True)


def market_segment_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """Primary vs secondary market share and median zł/m²."""
    empty = pd.DataFrame(columns=["segment", "n", "share", "median_ppm"])
    if frame.empty:
        return empty
    work_frame = frame
    if "market_primary" not in work_frame.columns:
        work_frame = ensure_market_primary(work_frame)
    flag = _num(work_frame, "market_primary")
    known = flag.notna()
    if not bool(known.any()):
        return empty
    work = pd.DataFrame({
        "segment": np.where(flag.loc[known] >= 0.5, "primary", "secondary"),
        "ppm": _num(work_frame, "price_per_m2").loc[known],
    })
    grouped = (
        work
        .groupby("segment", dropna=True)
        .agg(n=("segment", "size"), median_ppm=("ppm", "median"))
        .reset_index()
    )
    if grouped.empty:
        return empty
    total = float(grouped["n"].sum()) or 1.0
    grouped["share"] = grouped["n"] / total
    return grouped.sort_values("segment").reset_index(drop=True)
