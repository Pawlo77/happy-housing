# Offer AI review

Local LM Studio generates a buyer-facing markdown review for the active Explore
offer.

**Code:** `src/happy_housing/app/ai_review.py`  
**Store:** DuckDB under processed (`reviews_db_path`) keyed by listing + model +
`context_hash`  
**Admin:** Offer AI  
**Default model:** `gemma-4-31b-it-mlx` (`AppSettings.lm_model`)

---

## Motivation

SHAP lists feature contributions; buyers still want a **narrative** that joins
prefs, model Δ, commute, noise, and comps into buy / watch / skip language.
Running locally avoids sending listing PII / descriptions to cloud APIs.

---

## Methodology

1. Build **system** prompt (`ai_review_system_prompt` or default).
2. Build **user** prompt sections (toggles):

   | Section | Content |
   |---------|---------|
   | Buyer preferences | Explore prefs when enabled |
   | Offer facts | Price, rooms, building, flags, coords, … |
   | Model valuation | Ask / model / Δ zł/m² |
   | Proximity | Transit, metro, hospital, aircraft, … |
   | Neighborhood prices | RCN nbhd aggregates |
   | Description | Truncated listing text |
   | Raw characteristics | Otodom JSON |
   | Remaining columns | Any other non-empty fields |

3. Call LM Studio (`llm.respond`) with `temperature`, `maxTokens`.
4. Cache result; reuse until listing data or settings hash changes.
5. Optional auto-run when Offer opens (`ai_review_on_offer_open`).

### Default system prompt

> You are a Warsaw residential real-estate analyst. Write a concise
> buyer-facing review (120–280 words) covering: (1) fit vs preferences,
> (2) price fairness vs model and local historical comps, (3) livability /
> commute / amenities / noise proxies, (4) risks or upsides, (5) clear
> buy / watch / skip lean. Be concrete; use the numbers given; do not invent
> missing facts.

---

## “Training” data

No fine-tuning in-repo. The model is a **pretrained** local LMS checkpoint.
Prompt grounding data = scored listing row + prefs + optional description.

---

## Hyperparameters / knobs

| Knob | Default |
|------|---------|
| `lm_model` | `gemma-4-31b-it-mlx` |
| `ai_review_temperature` | 0.4 |
| `ai_review_max_tokens` | 800 |
| `ai_review_max_description_chars` | 4000 |
| `ai_review_max_field_chars` | 1500 |
| Section include_* flags | all True |
| `ai_review_on_offer_open` | Admin / settings |

No discrete HPO — quality is prompt + model choice.

---

## Metrics

Logged under LLM metrics purpose **`offer_review`** (Admin → LMS metrics):
call counts, errors, latency aggregates when instrumented.

Quality is subjective; cache hits make repeated Offer opens cheap.

---

## App surfaces

- Explore → Offer → **AI review**
- Chat tool `get_cached_review` (does not force regenerate)

---

## Performance

| Case | Behaviour |
|------|-----------|
| Cache hit | DuckDB read — negligible vs UI |
| Cache miss | Dominated by local LLM (seconds–tens of seconds depending on GPU/MPS) |
| Context size | Truncations bound prompt tokens |

Clear cache: Admin → Offer AI.
