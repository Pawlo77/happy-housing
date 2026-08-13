"""Explore Assistant UI: popover + Chat tab with conversation load/continue."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from happy_housing.app.app_settings import AppSettings
from happy_housing.app.chat_store import (
    chat_db_path,
    create_conversation,
    delete_conversation,
    list_conversations,
    load_messages,
)
from happy_housing.app.chat_tools import ChatToolContext
from happy_housing.app.explore_nav import open_explore_offer
from happy_housing.app.llm_metrics import llm_metrics_path
from happy_housing.app.preferences import UserPreferences, preferences_path
from happy_housing.gis.amenity_catalog import coerce_amenity_weights

logger = logging.getLogger(__name__)

ACTIVE_CONV_KEY = "assistant_active_conversation_id"
"""Session key for the active Assistant conversation id."""


def _ensure_active_conversation(db: Path) -> str:
    active = st.session_state.get(ACTIVE_CONV_KEY)
    if active:
        ids = {c.id for c in list_conversations(db)}
        if active in ids:
            return str(active)
    conv = create_conversation(db, title="New chat")
    st.session_state[ACTIVE_CONV_KEY] = conv.id
    return conv.id


def _build_ctx(
    *,
    scored: pd.DataFrame,
    processed: Path,
    prefs: UserPreferences,
    settings: AppSettings,
) -> ChatToolContext:
    active = st.session_state.get("explore_offer_id")
    return ChatToolContext(
        scored=scored,
        processed=processed,
        prefs=prefs,
        prefs_path=preferences_path(processed),
        active_listing_id=str(active) if active else None,
        settings_top_n=int(settings.recommender_top_n),
        demotion_alpha=float(settings.recommender_demotion_alpha),
        amenity_weights=coerce_amenity_weights(settings.recommender_amenity_weights),
        settings=settings,
    )


def _render_transcript(db: Path, conversation_id: str) -> None:
    for message in load_messages(db, conversation_id):
        if message.role not in {"user", "assistant", "system"}:
            continue
        with st.chat_message(message.role):
            st.markdown(message.content)


def _handle_prompt(
    *,
    prompt: str,
    db: Path,
    conversation_id: str,
    scored: pd.DataFrame,
    processed: Path,
    prefs: UserPreferences,
    settings: AppSettings,
) -> None:
    from happy_housing.app.chat_agent import run_chat_turn

    ctx = _build_ctx(
        scored=scored,
        processed=processed,
        prefs=prefs,
        settings=settings,
    )
    with st.spinner("Assistant thinking…"):
        logger.info("Assistant UI submitting turn conv=%s", conversation_id)
        result = run_chat_turn(
            db_path=db,
            conversation_id=conversation_id,
            user_text=prompt,
            ctx=ctx,
            settings=settings,
            metrics_path=llm_metrics_path(processed),
        )
    if result.error:
        logger.error("Assistant UI turn error: %s", result.error)
        st.error(result.error)
    if result.open_listing_id:
        open_explore_offer(result.open_listing_id)


def render_assistant_popover(
    *,
    scored: pd.DataFrame,
    processed: Path,
    prefs: UserPreferences,
    settings: AppSettings,
) -> None:
    """Compact always-available Assistant (stock ``st.popover``)."""
    if not settings.chat_enabled:
        return
    db = chat_db_path(processed)
    with st.popover("Assistant", help="Ask about offers, prefs, board, area opinions"):
        conversation_id = _ensure_active_conversation(db)
        convs = list_conversations(db)
        labels = {c.id: f"{c.title} · {c.updated_at[:16]}" for c in convs}
        options = list(labels.keys())
        if options:
            pick = st.selectbox(
                "Conversation",
                options=options,
                format_func=lambda cid: labels.get(cid, cid),
                index=options.index(conversation_id)
                if conversation_id in options
                else 0,
                key="assistant_popover_conv_pick",
            )
            if pick != conversation_id:
                st.session_state[ACTIVE_CONV_KEY] = pick
                st.rerun()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("New chat", key="assistant_popover_new"):
                conv = create_conversation(db, title="New chat")
                st.session_state[ACTIVE_CONV_KEY] = conv.id
                st.rerun()
        with c2:
            if st.button("Delete", key="assistant_popover_del"):
                delete_conversation(db, conversation_id)
                st.session_state.pop(ACTIVE_CONV_KEY, None)
                st.rerun()
        _render_transcript(db, conversation_id)
        prompt = st.chat_input(
            "Ask about offers, prefs, areas…",
            key="assistant_popover_input",
        )
        if prompt:
            _handle_prompt(
                prompt=prompt,
                db=db,
                conversation_id=conversation_id,
                scored=scored,
                processed=processed,
                prefs=prefs,
                settings=settings,
            )
            st.rerun()


def render_assistant_tab(
    *,
    scored: pd.DataFrame,
    processed: Path,
    prefs: UserPreferences,
    settings: AppSettings,
) -> None:
    """Full Chat tab with conversation list and transcript."""
    if not settings.chat_enabled:
        st.info("Assistant disabled in Admin → AI settings.")
        return
    db = chat_db_path(processed)
    conversation_id = _ensure_active_conversation(db)
    left, right = st.columns([1, 2.4])
    with left:
        st.caption("Conversations")
        if st.button("New chat", key="assistant_tab_new", width="stretch"):
            conv = create_conversation(db, title="New chat")
            st.session_state[ACTIVE_CONV_KEY] = conv.id
            st.rerun()
        for conv in list_conversations(db):
            selected = conv.id == conversation_id
            label = conv.title or "Chat"
            if st.button(
                f"{'● ' if selected else ''}{label}",
                key=f"assistant_tab_pick_{conv.id}",
                width="stretch",
                type="primary" if selected else "secondary",
            ):
                st.session_state[ACTIVE_CONV_KEY] = conv.id
                st.rerun()
        if st.button("Delete current", key="assistant_tab_del"):
            delete_conversation(db, conversation_id)
            st.session_state.pop(ACTIVE_CONV_KEY, None)
            st.rerun()
    with right:
        st.caption(
            "LMS tools: search / recommend / prefs / stars / reviews / "
            "web / open offer. Conversations persist under processed data."
        )
        _render_transcript(db, conversation_id)
        prompt = st.chat_input(
            "Message the Assistant…",
            key="assistant_tab_input",
        )
        if prompt:
            _handle_prompt(
                prompt=prompt,
                db=db,
                conversation_id=conversation_id,
                scored=scored,
                processed=processed,
                prefs=prefs,
                settings=settings,
            )
            st.rerun()
