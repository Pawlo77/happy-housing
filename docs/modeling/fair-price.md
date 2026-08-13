# Fair-price model (LightGBM)

Primary supervised model: predict a listing’s fair asking price per square metre
(**zł/m²**). The app surfaces `model_price_per_m2` and
`delta_price_per_m2 = ask − model` (positive ⇒ overpriced vs model).

**Code:** `src/happy_housing/models/train.py`, `feature_eng.py`, `neighborhood.py`,
`metrics.py`, `compare.py`  
**Pipeline:** `src/happy_housing/pipeline.py` (`train` / `score` steps)  
**Artifacts:** `data/processed/lightgbm_price.txt` (+ `.meta.json`), scored parquet  
**Admin:** Price model

---

## Motivation

Otodom asks are noisy: agency markup, stale listings, and location externalities
(noise, air, permits, commute) are not priced consistently. Buyers need a
**local, reproducible** fair-value estimate that:

- Uses only free public / scraped data (no paid valuation APIs).
- Separates **ask labels** from **deed comps** so RCN history does not leak as
  self-labels when neighborhood features are on.
- Remains explainable (SHAP TreeExplainer — see [explainability.md](explainability.md)).

---

## Methodology

### Algorithm

1. Build feature matrix `X` from enriched listings (`build_feature_matrix` /
   `active_feature_columns`).
2. Target `y = price_per_m2` (optionally train on `log1p(y)`, invert for metrics).
3. Split: **test** hold-out (`test_size`, default 0.2) never used for HPO;
   remaining rows → **train** / **val** (`val_size`, default 0.15). Prefer
   time-aware split when `listed_at` is usable; else IID `train_test_split`.
4. **HPO:** `sklearn.model_selection.ParameterSampler` over `_HPO_SPACE`
   (`hpo_trials`, default 80). Trial 0 is always the YAML seed
   (`n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`). Parallelism:
   `min(4, n_candidates, cpu_count)`.
5. Each trial: fit `LGBMRegressor` on train with early stopping on val
   (`early_stopping_rounds`, default 80). Select by Admin `opt_metric`
   (default **mape**; also mae / rmse / medae / r2).
6. Refit winner on **train+val**; optionally fit **district residual offsets**
   (median true−pred per district) and store in meta.
7. Score universe: OOF 5-fold on train labels when possible; full booster for
   unseen rows. Attach `model_price_per_m2`, `delta_price_per_m2`.

### Label universe

| Setting | Labels | RCN role |
|---------|--------|---------|
| `train_otodom_labels_only=True` (default) | `source ∈ {otodom, fixture}` | Features / comps only |
| `False` | All rows with valid zł/m² | Also possible as labels (leak risk with nbhd) |

### Feature groups

Default `FEATURE_COLUMNS` ≈ **164** columns (runtime count from catalog):

| Group | Count (≈) | Module | Contents |
|-------|-----------|--------|----------|
| Core structural + GIS | ~120 | `train._STRUCTURAL_*` + `amenity_catalog.model_distance_feature_columns()` | Area, rooms, year, floors, flags, categoricals, noise/air/crime/traffic/bike/zone distances, amenity `dist_*_m`, commute minutes |
| Neighborhood RCN | 5 | `neighborhood.py` | `nbhd_rcn_{median,mean}_ppm`, `n`, `nearest_m`, `idw_ppm` |
| Photo CV | 13 | `vision/photo_features.py` | Quality / room shares / floor-plan flags |
| Engineered | 26 | `feature_eng.py` | Logs, ratios, interactions, listing calendar, train-fold district ppm stats |

Categoricals for LightGBM: `building_type`, `heating`, `construction_status`,
`building_material`, `district`, `nearest_highway`.

**Neighborhood comps** (`attach_neighborhood_rcn_features`): BallTree haversine,
default radius **800 m**, self-exclude ≤5 m / same `listing_id`. IDW baseline
logged in MLflow for comparison (`idw_neighborhood_baseline`).

**Engineered:**

- Static (target-free): `area_per_room`, `floor_ratio`, `building_age`, ground/top
  flags, log distances/commutes, `rooms_x_area`, `listed_{month,year,dow}`,
  `commute_vs_straight_ratio`, …
- Train-fold only: `district_train_{median,mean}_ppm`, `district_train_n`,
  `nbhd_minus_district_ppm` (fit on train labels → applied with global fallback).

Admin can disable neighborhood and/or engineered blocks.

### Imputation

Numeric medians from the **train** fold only (`train_medians` in meta); applied
at predict time for missing GIS / structural fields.

### District calibration

When enabled, additive zł/m² offsets by district from train+val residuals after
the final fit. Unknown districts use no offset (or global fallback where
engineered stats require it).

---

## Hyperparameter space

Exact discrete grids in `train._HPO_SPACE` (Cartesian size ≈ **2.7×10⁷**;
search is **random**, not exhaustive):

| Parameter | Candidates |
|-----------|------------|
| `n_estimators` | 400, 800, 1200, 2000, 3000 |
| `learning_rate` | 0.005, 0.01, 0.02, 0.03, 0.05, 0.08 |
| `num_leaves` | 31, 63, 127, 255, 511 |
| `max_depth` | −1, 6, 8, 10, 12 |
| `min_child_samples` | 5, 10, 20, 40, 80 |
| `subsample` | 0.6, 0.7, 0.8, 0.9, 1.0 |
| `colsample_bytree` | 0.5 … 1.0 (step 0.1) |
| `reg_lambda` | 0, 0.1, 1, 5, 10 |
| `reg_alpha` | 0, 0.1, 1, 5 |
| `min_split_gain` | 0, 0.01, 0.05, 0.1 |
| `objective` | `regression`, `mae`, `huber` |

**YAML seed** (`configs/experiment.yaml` → `model:`): always evaluated as trial 0.

| Knob | Default | Source |
|------|---------|--------|
| `hpo_trials` | 80 | Admin / `AppSettings` |
| `opt_metric` | `mape` | Admin |
| `val_size` | 0.15 | Admin |
| `test_size` | 0.2 | ExperimentConfig |
| `early_stopping_rounds` | 80 | Admin (0 disables) |
| `log_target` | True | Admin |
| `district_calibration` | True | Admin |
| `random_state` | 42 | ExperimentConfig |

---

## Metrics and expected performance

Computed in zł/m² space (`compute_regression_metrics`):

| Metric | Definition |
|--------|------------|
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `mape` | Mean absolute percentage error (tiny targets clipped) |
| `medae` | Median absolute error |
| `r2` | Coefficient of determination |

Logged keys: `test_*`, `val_*`, `n_train` / `n_val` / `n_test`, optional
`baseline_*` (neighborhood IDW on test).

**Product target** (Admin copy): **≤1% test MAPE**. This is an optimization goal,
not a guaranteed CI fixture. After a full pipeline, read:

- **Home → Model** — last run summary from metadata  
- **MLflow** — `make mlflow` → experiment `happy-housing`

Fixture-scale runs (tiny offline data) produce much worse MAPE; treat them as
smoke tests only.

Optional **CatBoost** hold-out compare (`models/compare.py`) if the package is
installed — not the production path.

---

## Where it is used in the app

| Surface | Role |
|---------|------|
| Pipeline train + score | Writes booster + scored parquet |
| Home → Model | Metrics, methodology mermaid |
| Explore Map / Board | Pin colours and Δ columns from `delta_price_per_m2` |
| Explore Offer | Model zł/m², Δ, Explain (SHAP) |
| Stats → Model | Global importance + SHAP |
| Chat / Offer AI | Numbers passed into prompts / tools |
| Recommender | Bargain / overprice terms from Δ |

---

## Performance (speed / resources)

| Stage | Notes |
|-------|-------|
| HPO | Up to 80 trials × early-stopped trees; capped at 4 worker threads |
| Final fit | Single booster on train+val |
| OOF score | 5-fold; dominant cost when universe is large |
| Inference in app | Load booster once; predict vectorized on filtered frames |
| Pipeline cache | `pipeline_steps.json` fingerprints skip train/score when enriched data + settings unchanged (`pipeline_force_refresh` bypasses) |

Memory: feature matrix is dense float + categoricals; neighborhood BallTree is
built once per attach. Prefer not materializing duplicate full scored copies
in Explore (column projections / caches elsewhere).

---

## Admin knobs (Price model)

- Optimization metric, validation fraction, HPO trials, early stopping  
- Log-target, district calibration, engineered features  
- RCN neighborhood on/off + radius (m)  
- Train Otodom/fixture asks only  
- Photo CV enable / max photos / model id (see [photo-cv.md](photo-cv.md))

Retrain: Admin → **Ops** → run pipeline (or force recompute), not the recommender
button.
