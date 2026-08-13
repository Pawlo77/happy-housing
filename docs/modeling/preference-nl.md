# Preference natural-language translate

Explore **Prefs** can turn free-text buyer notes into structured
`UserPreferences` JSON via LM Studio.

**Code:** `src/happy_housing/app/pref_llm.py`  
**UI:** Explore → Prefs → “Translate with LM Studio”  
**Model:** shared `lm_model` (same family as Offer AI / Chat)

---

## Motivation

Hard filter widgets are precise but slow to fill. Buyers describe needs in
Polish/English prose; a constrained LLM maps that into the same schema the
recommender and Map filters already understand.

---

## Methodology

1. Collect NL notes (+ optional current prefs as context).
2. Prompt model to emit **JSON only** matching preference fields
   (price caps, area, rooms, districts, commute, sources, amenity-ish notes, …).
3. Parse / coerce into `UserPreferences`; persist when user saves.
4. Sampling: temperature **0.2**, maxTokens **~800** (see `pref_llm.py`).

Not a separate Admin section — uses Assistant/Offer AI model id unless code
paths diverge.

---

## Relation to other systems

| System | Link |
|--------|------|
| Recommender | Hard filters + soft bonuses from prefs |
| Offer AI | Optional prefs block in review prompt |
| Chat | `update_preferences` / `get_preferences` tools |

---

## Performance

Single LMS round-trip per translate click. Failures surface as UI errors;
prefs left unchanged.
