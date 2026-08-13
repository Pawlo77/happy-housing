"""Tests for DuckDB chat conversation persistence."""

from pathlib import Path

from happy_housing.app.chat_store import (
    append_message,
    clear_all_conversations,
    create_conversation,
    delete_conversation,
    list_conversations,
    load_messages,
    save_lms_history,
    update_conversation_title,
)


def test_chat_conversation_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "chat.duckdb"
    conv = create_conversation(db, title="First chat")
    assert conv.id
    assert conv.title == "First chat"
    append_message(db, conv.id, role="user", content="Hello")
    append_message(db, conv.id, role="assistant", content="Hi there")
    msgs = load_messages(db, conv.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "Hello"
    assert msgs[1].content == "Hi there"
    listed = list_conversations(db)
    assert len(listed) == 1
    assert listed[0].id == conv.id


def test_chat_continue_preserves_order(tmp_path: Path) -> None:
    db = tmp_path / "chat.duckdb"
    conv = create_conversation(db)
    for i in range(5):
        append_message(db, conv.id, role="user", content=f"u{i}")
        append_message(db, conv.id, role="assistant", content=f"a{i}")
    msgs = load_messages(db, conv.id)
    assert [m.content for m in msgs] == [
        "u0",
        "a0",
        "u1",
        "a1",
        "u2",
        "a2",
        "u3",
        "a3",
        "u4",
        "a4",
    ]


def test_chat_delete_and_clear(tmp_path: Path) -> None:
    db = tmp_path / "chat.duckdb"
    a = create_conversation(db, title="A")
    b = create_conversation(db, title="B")
    append_message(db, a.id, role="user", content="x")
    delete_conversation(db, a.id)
    ids = {c.id for c in list_conversations(db)}
    assert a.id not in ids
    assert b.id in ids
    clear_all_conversations(db)
    assert list_conversations(db) == []


def test_chat_lms_history_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "chat.duckdb"
    conv = create_conversation(db)
    history = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
    }
    save_lms_history(db, conv.id, history)
    loaded = list_conversations(db)[0]
    assert loaded.lms_history_json
    import json

    assert json.loads(loaded.lms_history_json)["messages"][1]["role"] == "user"


def test_update_conversation_title(tmp_path: Path) -> None:
    db = tmp_path / "chat.duckdb"
    conv = create_conversation(db, title="tmp")
    update_conversation_title(db, conv.id, "Renamed")
    assert list_conversations(db)[0].title == "Renamed"
