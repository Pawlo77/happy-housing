# Chat assistant (tool-calling agent)

Explore **Chat** tab and the **Assistant** popover: a local LM Studio agent with
structured tools over scored listings, prefs, stars, and methodology.

**Code:** `src/happy_housing/app/chat_agent.py`, `chat_tools.py`, `chat_store.py`  
**Store:** DuckDB conversations under processed  
**Admin:** Assistant  
**Shares** `lm_model` with Offer AI by default

---

## Motivation

Buyers ask open questions (“3 rooms under 1.2M near metro, underpriced”) that
filters alone do not answer. A tool-calling agent can search, recommend, open
offers, and cite methodology **without inventing listing ids or prices**.

---

## Methodology

1. Load / create conversation (`chat_store`).
2. System prompt (`chat_system_prompt` or default) + truncated history
   (`chat_history_max_messages`, default 24).
3. Register tools via `build_chat_tools` / LMS `llm.act`.
4. Loop up to `chat_max_tool_rounds` (default **6**).
5. Persist assistant message; optionally navigate UI if `open_offer` fired.

### Default system prompt (summary)

Warsaw flats assistant; use tools for scored data; never invent ids/prices;
LightGBM asks-only labels; recommender is a separate rule blend; prefer Explore
prefs as source of truth; keep answers concise.

### Tools

| Tool | Role |
|------|------|
| `get_active_offer` | Current Offer selection |
| `get_offer` | Lookup by listing id |
| `search_offers` | Filter scored universe |
| `recommend_offers` | Rule recommender top-N (no impression write) |
| `get_preferences` / `update_preferences` | Explore prefs |
| `list_starred` / `list_star_news` | Stars + news feed |
| `get_cached_review` | Offer AI cache |
| `open_offer` | UI navigation |
| `web_search_area` | DuckDuckGo HTML opinions (best-effort) |
| `get_methodology` | Pipeline / model prose |

---

## Data

Inference-only over:

- In-memory / loaded scored frame (`ChatToolContext`)
- Prefs / stars / news / review DuckDB files
- Optional web HTML for area opinions

No training loop in-app.

---

## Hyperparameters

| Knob | Default |
|------|---------|
| `chat_enabled` | True |
| `chat_temperature` | 0.4 |
| `chat_max_tokens` | 900 |
| `chat_history_max_messages` | 24 |
| `chat_max_tool_rounds` | 6 |
| `lm_model` | shared with Offer AI |

---

## Metrics

LLM metrics purpose **`chat`** (and related) in Admin → LMS metrics.

---

## App surfaces

- Explore → **Chat**
- Global **Assistant** popover (when enabled)
- Home About mentions Assistant briefly

---

## Performance

| Factor | Effect |
|--------|--------|
| Tool rounds | Each round may call LMS again — latency multiplies |
| History trim | Caps token growth |
| Fragment / caching | Explore Chat uses Streamlit patterns to avoid full page rebuilds where possible |
| Local hardware | Same LMS server as Offer AI; concurrent review + chat contend for GPU/CPU |

Disable: Admin → Assistant → `chat_enabled` off.
