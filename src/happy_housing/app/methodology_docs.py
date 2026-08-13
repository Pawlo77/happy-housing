"""Shared methodology markdown + mermaid for Home Model and Assistant."""

from happy_housing.app.commute_target import DEFAULT_COMMUTE_LABEL

PRICE_PIPELINE_MERMAID: str = """```mermaid
flowchart LR
  ingest[Ingest Otodom + RCN + fixtures] --> enrich[GIS enrich]
  enrich --> train[LightGBM train/val/test]
  train --> score[Score zł/m² + Δ]
  score --> shap[SHAP per offer]
  score --> rec[Recommender]
```"""
"""Mermaid: price pipeline stages."""

TRAIN_DATA_MERMAID: str = """```mermaid
flowchart LR
  scored[Scored universe] --> asks[Otodom/fixture asks]
  scored --> rcn[RCN comps only]
  asks --> split[Time or IID split]
  split --> train[Train fold]
  split --> val[Val fold HPO]
  split --> test[Test hold-out]
  train --> fit[Final fit train+val]
  val --> fit
  rcn --> nbhd[nbhd_rcn features]
  nbhd --> fit
```"""
"""Mermaid: label universe vs RCN comps."""

RECOMMENDER_MERMAID: str = """```mermaid
flowchart LR
  prefs[Prefs hard filters] --> pool[Candidate pool]
  pool --> blend[Bargain + commute + transit]
  blend --> over[Overprice penalty]
  over --> soft[Soft pref / amenity nudge]
  soft --> demo[Seen-offer demotion]
  demo --> news[Explore News]
```"""
"""Mermaid: rule recommender path."""

PHOTO_CV_MERMAID: str = """```mermaid
flowchart LR
  urls[image_urls + floor plans] --> cache[Disk image cache]
  cache --> clip[Frozen CLIP]
  clip --> agg[Per-listing aggregates]
  agg --> lgbm[LightGBM features]
```"""
"""Mermaid: pretrained photo CV path."""


def format_methodology_note(commute_label: str | None = None) -> str:
    """Full prose methodology with Admin commute target interpolated."""
    label = (commute_label or DEFAULT_COMMUTE_LABEL).strip() or DEFAULT_COMMUTE_LABEL
    return (
        "LightGBM regresses asking zł/m² from property traits "
        "(area, rooms, year) and livability distances (hospital, fire, police, "
        f"transit, air station, GUNB permits, commute to {label}) plus "
        "an aircraft Lden/Lnight feature from EEA END contours, nearest-station "
        "GIOŚ air quality (AQI + PM10/PM2.5/NO2, not distance alone), and "
        "district police crime rates (per 1,000 residents + detectability). "
        "Training labels are Otodom/fixture asks only when configured; RCN deeds "
        "feed neighbourhood comps, not ask targets. "
        "Optional photo CV (frozen CLIP) adds finish/room-mix features when enabled. "
        "SHAP TreeExplainer explains each offer. "
        "Commute minutes use local OSRM when available, else a 30 km/h distance proxy. "
        "The Explore recommender is a separate rule blend (bargain, commute, transit, "
        "overprice penalty, prefs), not the LightGBM itself."
    )


def methodology_steps(commute_label: str | None = None) -> tuple[str, ...]:
    """Step-wise bullets for Home Model (commute label aware)."""
    label = (commute_label or DEFAULT_COMMUTE_LABEL).strip() or DEFAULT_COMMUTE_LABEL
    return (
        "Ingest Otodom asks and UM Warszawa RCN transactions into one universe.",
        "Enrich each pin with aircraft Lden/Lnight bands and nearest amenity / "
        "transit / air / GUNB distances.",
        f"Attach commute minutes to {label} via local OSRM, or a 30 km/h proxy.",
        "Train LightGBM on Otodom/fixture ask labels (RCN as comps); hold out test.",
        "Score every offer; Explore explains residuals with SHAP TreeExplainer.",
        "Rank News via rule recommender (bargain / commute / transit / overprice).",
    )


def model_tab_methodology_markdown(
    *,
    commute_label: str | None = None,
    photo_cv_enabled: bool = False,
    n_offers: int | None = None,
    n_train: float | None = None,
    n_train_val: float | None = None,
    n_test: float | None = None,
) -> str:
    """Rich methodological markdown for Home → Model (incl. mermaid)."""
    label = (commute_label or DEFAULT_COMMUTE_LABEL).strip() or DEFAULT_COMMUTE_LABEL
    parts: list[str] = [
        "### Price pipeline",
        PRICE_PIPELINE_MERMAID,
        "### Train data roles",
        (
            "Otodom/fixture rows are **ask labels**. RCN rows are **comps** "
            "(`nbhd_rcn_*`) — not dumped into asking-price `y`. "
            "Home “Train rows” is the train fold only; final fit uses train+val."
        ),
    ]
    if n_offers is not None or n_train is not None:
        bits: list[str] = []
        if n_offers is not None:
            bits.append(f"scored universe ≈ {int(n_offers):,}")
        if n_train is not None:
            bits.append(f"train fold {int(n_train):,}")
        if n_train_val is not None:
            bits.append(f"train+val {int(n_train_val):,}")
        if n_test is not None:
            bits.append(f"test {int(n_test):,}")
        parts.append("Live run: " + "; ".join(bits) + ".")
    parts.extend([TRAIN_DATA_MERMAID, "### Recommender", RECOMMENDER_MERMAID])
    parts.append(
        "Blend (Admin-tunable): bargain underpricing, commute to "
        f"**{label}**, transit proximity, overprice penalty, soft prefs, "
        "then demotion for already-seen offers."
    )
    if photo_cv_enabled:
        parts.extend(["### Photo CV (pretrained)", PHOTO_CV_MERMAID])
    else:
        parts.append(
            "### Photo CV\nDisabled in Admin — enable to add CLIP quality / "
            "room-mix features to LightGBM."
        )
    parts.extend(["### Full note", format_methodology_note(label)])
    return "\n\n".join(parts)


def chat_methodology_blurb(commute_label: str | None = None) -> str:
    """Short accurate pipeline summary for the Assistant system prompt."""
    label = (commute_label or DEFAULT_COMMUTE_LABEL).strip() or DEFAULT_COMMUTE_LABEL
    return (
        "Pipeline facts: LightGBM predicts ask zł/m² from Otodom/fixture labels "
        f"(RCN = comps only). Commute target is '{label}' (Admin). "
        "Recommender is a separate rule score (bargain/commute/transit/overprice), "
        "not the price model. For diagrams tell the user to open Home → Model."
    )
