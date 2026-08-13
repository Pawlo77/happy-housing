"""LM Studio Assistant agent: tool-calling chat over scored housing data."""

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

import lmstudio as lms
from lmstudio import ChatHistoryDataDict

from happy_housing.app.app_settings import AppSettings
from happy_housing.app.chat_store import (
    append_message,
    get_lms_history,
    save_lms_history,
    update_conversation_title,
)
from happy_housing.app.chat_tools import (
    ChatToolContext,
    get_active_offer,
    get_cached_review,
    get_preferences,
    list_star_news,
    list_starred,
    open_offer,
    recommend_offers,
    search_offers,
    update_preferences,
    web_search_area,
)
from happy_housing.app.llm_metrics import PURPOSE_CHAT, record_timed_llm_call
from happy_housing.app.methodology_docs import (
    chat_methodology_blurb,
    model_tab_methodology_markdown,
)

logger = logging.getLogger(__name__)

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are the happy-housing Assistant for Warsaw flats. "
    "Use tools to look up scored listings, preferences, recommendations, "
    "stars, cached reviews, web opinions, and methodology. "
    "Be concrete with zł/m², commute, and deltas from tool results — "
    "never invent listing ids or prices. "
    "Price model: LightGBM on Otodom/fixture ask labels (RCN = comps only). "
    "Recommender: separate rule blend (bargain/commute/transit/overprice). "
    "For diagrams tell the user to open Home → Model. "
    "When the user wants to open an offer, call open_offer. "
    "Prefer Explore preferences as the buyer's source of truth. "
    "Keep answers concise and buyer-facing."
)
"""Default system prompt for the Explore Assistant."""


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    """Outcome of one user → assistant turn."""

    assistant_text: str
    """Final assistant reply text."""
    open_listing_id: str | None
    """If set, UI should navigate to this offer."""
    error: str | None = None
    """Error message when the turn failed."""


def _history_dict(chat: lms.Chat) -> dict[str, Any]:
    history = chat._get_history()
    if isinstance(history, dict):
        return dict(history)
    to_dict = getattr(history, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, dict) else {"messages": []}
    return {"messages": []}


def _text_from_act(chat: lms.Chat) -> str:
    """Best-effort extract last assistant text from chat history."""
    history = _history_dict(chat)
    messages = history.get("messages") or []
    for message in reversed(list(messages)):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or []
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        parts.append(text)
                elif isinstance(block, str) and block.strip():
                    parts.append(block.strip())
        if parts:
            return "\n".join(parts).strip()
    return ""


def build_chat_tools(ctx: ChatToolContext) -> list[lms.ToolFunctionDef]:
    """LM Studio tool definitions bound to ``ctx``."""

    def _get_active_offer() -> str:
        return json.dumps(get_active_offer(ctx), ensure_ascii=False, default=str)

    def _search_offers(
        district: str = "",
        max_price_pln: float = 0.0,
        min_area_m2: float = 0.0,
        min_rooms: int = 0,
        max_commute_min: float = 0.0,
        underpriced_only: bool = False,
        limit: int = 8,
    ) -> str:
        hits = search_offers(
            ctx,
            district=district or None,
            max_price_pln=max_price_pln or None,
            min_area_m2=min_area_m2 or None,
            min_rooms=min_rooms or None,
            max_commute_min=max_commute_min or None,
            underpriced_only=bool(underpriced_only),
            limit=int(limit or 8),
        )
        return json.dumps(hits, ensure_ascii=False, default=str)

    def _recommend_offers(top_n: int = 0) -> str:
        hits = recommend_offers(ctx, top_n=top_n or None)
        return json.dumps(hits, ensure_ascii=False, default=str)

    def _get_preferences() -> str:
        return json.dumps(get_preferences(ctx), ensure_ascii=False, default=str)

    def _update_preferences(
        notes: str = "",
        enabled: bool = False,
        max_price_pln: float = 0.0,
        min_area_m2: float = 0.0,
        min_rooms: int = 0,
        max_commute_min: float = 0.0,
        districts: str = "",
    ) -> str:
        payload = update_preferences(
            ctx,
            notes=notes or None,
            enabled=True if enabled else None,
            max_price_pln=max_price_pln or None,
            min_area_m2=min_area_m2 or None,
            min_rooms=min_rooms or None,
            max_commute_min=max_commute_min or None,
            districts=districts or None,
        )
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _list_starred() -> str:
        return json.dumps(list_starred(ctx), ensure_ascii=False, default=str)

    def _list_star_news(limit: int = 10) -> str:
        return json.dumps(
            list_star_news(ctx, limit=int(limit or 10)),
            ensure_ascii=False,
            default=str,
        )

    def _get_cached_review(listing_id: str) -> str:
        return json.dumps(
            get_cached_review(ctx, listing_id),
            ensure_ascii=False,
            default=str,
        )

    def _open_offer(listing_id: str) -> str:
        return json.dumps(open_offer(ctx, listing_id), ensure_ascii=False, default=str)

    def _web_search_area(query: str, limit: int = 5) -> str:
        return json.dumps(
            web_search_area(query, limit=int(limit or 5)),
            ensure_ascii=False,
            default=str,
        )

    def _get_offer(listing_id: str) -> str:
        from happy_housing.app.chat_tools import offer_summary_row

        slim = offer_summary_row(ctx, listing_id)
        return json.dumps(
            slim or {"found": False, "listing_id": listing_id},
            ensure_ascii=False,
            default=str,
        )

    def _get_methodology() -> str:
        label = getattr(ctx.settings, "commute_target_label", None) or ""
        photo = bool(getattr(ctx.settings, "photo_cv_enabled", False))
        return model_tab_methodology_markdown(
            commute_label=label or None,
            photo_cv_enabled=photo,
        )

    return [
        lms.ToolFunctionDef(
            name="get_active_offer",
            description="Facts for the offer currently open in Explore.",
            parameters={},
            implementation=_get_active_offer,
        ),
        lms.ToolFunctionDef(
            name="get_offer",
            description="Look up one listing by listing_id.",
            parameters={"listing_id": str},
            implementation=_get_offer,
        ),
        lms.ToolFunctionDef(
            name="search_offers",
            description=(
                "Search scored listings. Pass 0 / empty to skip a filter. "
                "underpriced_only keeps negative delta zł/m²."
            ),
            parameters={
                "district": {"type": str, "default": ""},
                "max_price_pln": {"type": float, "default": 0.0},
                "min_area_m2": {"type": float, "default": 0.0},
                "min_rooms": {"type": int, "default": 0},
                "max_commute_min": {"type": float, "default": 0.0},
                "underpriced_only": {"type": bool, "default": False},
                "limit": {"type": int, "default": 8},
            },
            implementation=_search_offers,
        ),
        lms.ToolFunctionDef(
            name="recommend_offers",
            description=(
                "Rank offers with the preference recommender (no impression write)."
            ),
            parameters={"top_n": {"type": int, "default": 0}},
            implementation=_recommend_offers,
        ),
        lms.ToolFunctionDef(
            name="get_preferences",
            description="Return Explore buyer preferences.",
            parameters={},
            implementation=_get_preferences,
        ),
        lms.ToolFunctionDef(
            name="update_preferences",
            description=(
                "Update Explore preferences. Pass enabled=true to apply filters. "
                "districts is a comma-separated list."
            ),
            parameters={
                "notes": {"type": str, "default": ""},
                "enabled": {"type": bool, "default": False},
                "max_price_pln": {"type": float, "default": 0.0},
                "min_area_m2": {"type": float, "default": 0.0},
                "min_rooms": {"type": int, "default": 0},
                "max_commute_min": {"type": float, "default": 0.0},
                "districts": {"type": str, "default": ""},
            },
            implementation=_update_preferences,
        ),
        lms.ToolFunctionDef(
            name="list_starred",
            description="List starred offers.",
            parameters={},
            implementation=_list_starred,
        ),
        lms.ToolFunctionDef(
            name="list_star_news",
            description="Recent news about starred offers.",
            parameters={"limit": {"type": int, "default": 10}},
            implementation=_list_star_news,
        ),
        lms.ToolFunctionDef(
            name="get_cached_review",
            description="Fetch a cached AI review for a listing if available.",
            parameters={"listing_id": str},
            implementation=_get_cached_review,
        ),
        lms.ToolFunctionDef(
            name="open_offer",
            description="Open a listing in the Explore Offer tab.",
            parameters={"listing_id": str},
            implementation=_open_offer,
        ),
        lms.ToolFunctionDef(
            name="web_search_area",
            description="Search the web for opinions / news about an area or street.",
            parameters={
                "query": str,
                "limit": {"type": int, "default": 5},
            },
            implementation=_web_search_area,
        ),
        lms.ToolFunctionDef(
            name="get_methodology",
            description=(
                "Return price-model + recommender methodology markdown "
                "(incl. mermaid). Use when the user asks how scoring works."
            ),
            parameters={},
            implementation=_get_methodology,
        ),
    ]


def run_chat_turn(
    *,
    db_path: Any,
    conversation_id: str,
    user_text: str,
    ctx: ChatToolContext,
    settings: AppSettings,
    metrics_path: Any | None = None,
) -> ChatTurnResult:
    """Append user message, run LMS ``act`` with tools, persist reply + history."""
    from pathlib import Path

    path = Path(db_path)
    text = str(user_text or "").strip()
    if not text:
        return ChatTurnResult(
            assistant_text="", open_listing_id=None, error="Empty message"
        )

    logger.info(
        "Assistant turn start conv=%s chars=%s model=%s",
        conversation_id,
        len(text),
        settings.lm_model,
    )
    append_message(path, conversation_id, role="user", content=text)

    system = (settings.chat_system_prompt or "").strip() or DEFAULT_CHAT_SYSTEM_PROMPT
    system = f"{system}\n\n{chat_methodology_blurb(settings.commute_target_label)}"
    prior = get_lms_history(path, conversation_id)
    if prior and isinstance(prior.get("messages"), list) and prior["messages"]:
        chat = lms.Chat.from_history(cast(ChatHistoryDataDict, prior))
    else:
        chat = lms.Chat(system)
    chat.add_user_message(text)

    # Truncate history for context budget.
    max_msgs = max(4, int(settings.chat_history_max_messages))
    history = _history_dict(chat)
    messages = list(history.get("messages") or [])
    if len(messages) > max_msgs + 1:
        # Keep system (first) + tail.
        system_msgs = [
            m for m in messages if isinstance(m, dict) and m.get("role") == "system"
        ]
        rest = [
            m
            for m in messages
            if not (isinstance(m, dict) and m.get("role") == "system")
        ]
        trimmed = (system_msgs[:1] or []) + rest[-(max_msgs):]
        chat = lms.Chat.from_history(cast(ChatHistoryDataDict, {"messages": trimmed}))

    model = (settings.lm_model or "").strip() or "gemma-4-31b-it-mlx"
    tools = build_chat_tools(ctx)
    input_chars = len(system) + len(text)

    def _run() -> str:
        llm = lms.llm(model)
        llm.act(
            chat,
            tools,
            max_prediction_rounds=int(settings.chat_max_tool_rounds),
            config={
                "temperature": float(settings.chat_temperature),
                "maxTokens": int(settings.chat_max_tokens),
            },
        )
        reply = _text_from_act(chat)
        return reply or "(No assistant text returned.)"

    try:
        if metrics_path is not None:
            assistant_text = record_timed_llm_call(
                Path(metrics_path),
                purpose=PURPOSE_CHAT,
                model=model,
                input_chars=input_chars,
                runner=_run,
            )
        else:
            assistant_text = _run()
    except Exception as exc:
        logger.exception("Assistant turn failed")
        err = str(exc)
        append_message(
            path,
            conversation_id,
            role="assistant",
            content=f"Assistant error: {err}",
            meta={"error": err},
        )
        return ChatTurnResult(
            assistant_text="",
            open_listing_id=ctx.pending_open_listing_id,
            error=err,
        )

    append_message(
        path,
        conversation_id,
        role="assistant",
        content=assistant_text,
        meta={"tools": list(ctx.last_tool_notes)},
    )
    save_lms_history(path, conversation_id, _history_dict(chat))

    # Auto-title from first user message.
    from happy_housing.app.chat_store import list_conversations, load_messages

    msgs = load_messages(path, conversation_id)
    if len([m for m in msgs if m.role == "user"]) == 1:
        title = text[:60] + ("…" if len(text) > 60 else "")
        update_conversation_title(path, conversation_id, title)
    _ = list_conversations  # silence unused if optimized away

    return ChatTurnResult(
        assistant_text=assistant_text,
        open_listing_id=ctx.pending_open_listing_id,
        error=None,
    )
