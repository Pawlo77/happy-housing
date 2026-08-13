# Photo CV features

Tabular image / layout features attached before LightGBM training.

**Code:** `src/happy_housing/vision/photo_features.py`  
**Columns:** `PHOTO_FEATURE_COLUMNS` (13)  
**Admin:** Price model (photo CV block)  
**Configured model id:** `openai/clip-vit-base-patch32`

---

## Motivation

Finish quality and room mix influence asks but are absent from Otodom
tabular fields. Zero-shot vision scores turn gallery URLs into regressor
inputs without manual labels.

---

## Methodology (designed)

1. Parse `image_urls_json` / `image_url` and `floor_plan_urls_json`.
2. Score up to `photo_cv_max_photos` (default **8**) images.
3. **Quality:** CLIP-style pair  
   pos *“luxurious renovated apartment interior”* vs  
   neg *“run-down neglected apartment interior”*.
4. **Rooms / luxury:** zero-shot labels  
   kitchen, bathroom, living, bedroom, exterior, luxury.
5. **Floor plans:** pair  
   *“architectural floor plan drawing”* vs *“photo of a furnished room”*.
6. Aggregate → means/max/std, room shares, `has_floor_plan`, …
7. Merge into listing frame for LightGBM.

### Output columns

`photo_quality_{mean,max,std}`, `n_photos_scored`,
`share_{kitchen,bathroom,living,bedroom,exterior}`, `luxury_share`,
`has_floor_plan`, `n_floor_plans`, `floor_plan_conf`.

---

## Current implementation status

`resolve_photo_encoder` **always returns `HeuristicPhotoEncoder`** today:

- If `transformers` is missing → heuristic (logged).
- If present → still stubbed (“enable full vision later”).

Heuristic scores are **deterministic** hash-seeded uniforms — stable for tests,
**not** real CLIP semantics. Treat Admin `photo_cv_model_id` as forward-looking
config until the CLIP backend is wired.

Pipeline still calls `attach_photo_features` so columns exist for the model;
values may be near-noise under the heuristic.

---

## Data

- Listing image / floor-plan URLs from Otodom detail scrape.
- No supervised photo labels in-repo.

---

## Hyperparameters

| Knob | Default |
|------|---------|
| `photo_cv_enabled` | True (Admin flag; see note below) |
| `photo_cv_max_photos` | 8 |
| `photo_cv_model_id` | `openai/clip-vit-base-patch32` |

Note: Admin `photo_cv_enabled` documents intent; verify `pipeline.py` gating if
you need a hard off switch for attach cost.

---

## Metrics

No separate photo-only hold-out. Contribution appears indirectly via LightGBM
gain / SHAP on `photo_*` columns after a real CLIP backend is enabled.

---

## App surfaces

- Features → fair-price model only (not shown as a dedicated UI tab).
- Admin Price model controls.

---

## Performance

| Backend | Cost |
|---------|------|
| Heuristic | Negligible CPU per listing |
| Future CLIP | Dominated by image download + encoder forward; bound by `max_photos` |

Disk image caching may exist alongside Otodom media backfill; prefer incremental
detail scrape so galleries are already local.
