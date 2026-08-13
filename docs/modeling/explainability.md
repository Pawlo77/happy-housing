# Explainability (SHAP & importance)

How happy-housing explains the fair-price LightGBM — per listing and globally.

**Code:** `src/happy_housing/models/train.py` (TreeExplainer helpers),
`src/happy_housing/app/stats_xai.py`, `src/happy_housing/app/data.py`
(`load_shap_bundle`)  
**UI:** Explore Offer → Explain; Stats → Model  
**Admin:** Stats (`stats_shap_sample_n`, `stats_xai_top_n`)

---

## Motivation

Buyers distrust opaque “AI prices.” SHAP attributes the zł/m² prediction to
individual features (metro distance, noise, area, district, …) so a green/red
Δ on the map can be challenged with evidence.

---

## Methodology

### Per-offer (Explore → Offer → Explain)

1. Load booster + feature order from `lightgbm_price.txt` / meta
   (`@st.cache_resource` via `load_shap_bundle`).
2. Build one-row feature vector with the same impute / engineered / nbhd path
   as training.
3. `shap.TreeExplainer` → waterfall / force-style breakdown
   (`shap_explanation_for_row` path in Explore).

### Global (Stats → Model)

1. Sample up to `stats_shap_sample_n` (default **300**) scored listings.
2. LightGBM **gain** feature importance for the fitted booster.
3. Mean \|SHAP\| across the sample for the top `stats_xai_top_n` (default **20**)
   features — session-memoized to avoid recomputing on every Stats rerun.

---

## Data

Uses the same feature columns as the production fair-price model. No separate
training set: explanations are **post-hoc** on the fitted booster.

---

## Performance

| Path | Cost |
|------|------|
| Bundle load | Once per process (`cache_resource`) |
| Single-offer SHAP | Milliseconds–low hundreds ms on CPU for ~160 features |
| Global mean \|SHAP\| | Dominated by sample size × TreeExplainer; memoized in session |

If the booster file is missing, Explain / Stats Model degrade gracefully with
an error caption (run pipeline train first).

---

## Related

- Model training: [fair-price.md](fair-price.md)
- Offer narrative (LLM, not SHAP): [offer-ai.md](offer-ai.md)
