"""DuckDB persistence for Explore Assistant conversations."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatConversation:
    """One persisted assistant thread."""

    id: str
    """Stable conversation id."""
    title: str
    """Short display title."""
    created_at: str
    """ISO-8601 UTC created timestamp."""
    updated_at: str
    """ISO-8601 UTC last-message timestamp."""
    lms_history_json: str = ""
    """Serialized LM Studio chat history for continue."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One user/assistant (or system) message in a conversation."""

    conversation_id: str
    """Parent conversation id."""
    seq: int
    """1-based order within the conversation."""
    role: str
    """``user`` / ``assistant`` / ``system``."""
    content: str
    """Plain text body for UI."""
    created_at: str
    """ISO-8601 UTC timestamp."""
    meta_json: str = ""
    """Optional JSON blob (tool traces, open_offer ids, …)."""


def chat_db_path(processed: Path) -> Path:
    """DuckDB path for assistant conversations."""
    return processed / "chat_conversations.duckdb"


def clear_all_conversations(path: Path) -> None:
    """Delete the chat DuckDB file."""
    if path.exists():
        path.unlink()
        logger.info("Cleared chat conversations DB at %s", path)


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            lms_history_json VARCHAR NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            conversation_id VARCHAR NOT NULL,
            seq INTEGER NOT NULL,
            role VARCHAR NOT NULL,
            content VARCHAR NOT NULL,
            created_at VARCHAR NOT NULL,
            meta_json VARCHAR NOT NULL DEFAULT '',
            PRIMARY KEY (conversation_id, seq)
        )
        """
    )
    return conn


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def create_conversation(path: Path, title: str = "New chat") -> ChatConversation:
    """Insert an empty conversation and return it."""
    conv_id = str(uuid.uuid4())
    now = _now()
    clean_title = (title or "").strip() or "New chat"
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO conversations
                (id, title, created_at, updated_at, lms_history_json)
            VALUES (?, ?, ?, ?, '')
            """,
            [conv_id, clean_title, now, now],
        )
    finally:
        conn.close()
    return ChatConversation(
        id=conv_id,
        title=clean_title,
        created_at=now,
        updated_at=now,
        lms_history_json="",
    )


def list_conversations(path: Path) -> list[ChatConversation]:
    """Return conversations newest-first."""
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        frame = conn.execute(
            """
            SELECT id, title, created_at, updated_at, lms_history_json
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).df()
    finally:
        conn.close()
    if frame.empty:
        return []
    return [
        ChatConversation(
            id=str(row["id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            lms_history_json=str(row.get("lms_history_json") or ""),
        )
        for _, row in frame.iterrows()
    ]


def load_messages(path: Path, conversation_id: str) -> list[ChatMessage]:
    """Load messages for ``conversation_id`` in seq order."""
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        frame = conn.execute(
            """
            SELECT conversation_id, seq, role, content, created_at, meta_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq ASC
            """,
            [conversation_id],
        ).df()
    finally:
        conn.close()
    if frame.empty:
        return []
    return [
        ChatMessage(
            conversation_id=str(row["conversation_id"]),
            seq=int(row["seq"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
            meta_json=str(row.get("meta_json") or ""),
        )
        for _, row in frame.iterrows()
    ]


def append_message(
    path: Path,
    conversation_id: str,
    *,
    role: str,
    content: str,
    meta: dict[str, Any] | None = None,
) -> ChatMessage:
    """Append a message and bump conversation ``updated_at``."""
    now = _now()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = ?",
            [conversation_id],
        ).fetchone()
        seq = int(row[0] if row else 0) + 1
        conn.execute(
            """
            INSERT INTO messages
                (conversation_id, seq, role, content, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [conversation_id, seq, str(role), str(content), now, meta_json],
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            [now, conversation_id],
        )
    finally:
        conn.close()
    return ChatMessage(
        conversation_id=conversation_id,
        seq=seq,
        role=str(role),
        content=str(content),
        created_at=now,
        meta_json=meta_json,
    )


def save_lms_history(path: Path, conversation_id: str, history: dict[str, Any]) -> None:
    """Persist full LM Studio history JSON on the conversation row."""
    blob = json.dumps(history, ensure_ascii=False, default=str)
    now = _now()
    conn = _connect(path)
    try:
        conn.execute(
            """
            UPDATE conversations
            SET lms_history_json = ?, updated_at = ?
            WHERE id = ?
            """,
            [blob, now, conversation_id],
        )
    finally:
        conn.close()


def get_lms_history(path: Path, conversation_id: str) -> dict[str, Any] | None:
    """Load LM Studio history dict if present."""
    for conv in list_conversations(path):
        if conv.id == conversation_id:
            raw = (conv.lms_history_json or "").strip()
            if not raw:
                return None
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def update_conversation_title(path: Path, conversation_id: str, title: str) -> None:
    """Rename a conversation."""
    clean = (title or "").strip() or "Chat"
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            [clean[:80], _now(), conversation_id],
        )
    finally:
        conn.close()


def delete_conversation(path: Path, conversation_id: str) -> None:
    """Delete one conversation and its messages."""
    conn = _connect(path)
    try:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?", [conversation_id]
        )
        conn.execute("DELETE FROM conversations WHERE id = ?", [conversation_id])
    finally:
        conn.close()
