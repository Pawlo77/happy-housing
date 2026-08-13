# Listing attribute extract (LLM)

Short LM Studio calls fill **gaps** in structured listing fields from free-text
descriptions during pipeline enrich or Admin backfill.

**Code:** `src/happy_housing/app/attr_extract.py`  
**Companion (non-LLM):** `listing_attrs.py` (`promote_listing_flags`)  
**Admin:** Extract  
**Default model:** `gemma-3-4b-it` (`extract_lm_model`)

---

## Motivation

Otodom structured fields omit renovation year, wiring/plumbing replacement,
attic flags, etc. that often appear only in Polish descriptions. A tiny local
model returns a fixed CSV line so LightGBM and filters see denser features
without manual tagging.

Structured Otodom values **win** over LLM guesses when both exist.

---

## Methodology

1. Select rows needing extract (missing fields / fingerprint stale).
2. Truncate description to `extract_max_chars` (default **900**).
3. System: emit **one CSV line** in fixed `EXTRACT_FIELDS` order; `N` = unknown.
4. User: description only. Temperature **0.0**, `maxTokens` default **48**.
5. Parse cells → coerce flags / years / ownership; write fingerprint
   `attr_extract_fp` for cache skip.
6. Cap total LMS calls per run (`extract_max_listings`, default **200**);
   workers default **2**.

### Field order (`EXTRACT_FIELDS`)

`object_type`, `floor_no`, `is_attic`, `has_garden`, `has_garage`,
`has_outdoor_parking`, `has_basement`, `has_balcony`, `ownership`,
`needs_renovation`, `year_built`, `year_renovated`, `plumbing_replaced`,
`wiring_replaced`.

---

## Data

- Input: listing `description` (and existing structured columns for merge).
- Labels: none (zero-shot instruction following).
- Offline tests: fixtures + mocked LMS responses.

---

## Hyperparameters

| Knob | Default |
|------|---------|
| `extract_attrs_enabled` | True |
| `extract_lm_model` | `gemma-3-4b-it` |
| `extract_max_chars` | 900 |
| `extract_max_tokens` | 48 |
| `extract_max_listings` | 200 |
| Workers | 2 (`DEFAULT_EXTRACT_LM_WORKERS`) |

---

## Metrics

LLM metrics purpose **`attr_extract`**. Pipeline logs how many rows updated /
skipped via fingerprint.

---

## App surfaces

- **Pipeline enrich** (when enabled)
- Admin → **Extract** backfill / clear

Downstream: promoted flags feed Explore filters and the fair-price feature
matrix when columns align.

---

## Performance

- Intentional **small** prompts (≤900 chars, 48 tokens out) for throughput.
- Fingerprint cache → re-runs skip unchanged descriptions.
- Parallelism kept low (2) to avoid saturating local LMS during full pipeline.

Disable to speed enrich: Admin → Extract → off.
