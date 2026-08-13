"""Tests for LM Studio model key resolution."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from happy_housing.app.lmstudio_models import (
    LmStudioModelNotFoundError,
    resolve_lmstudio_model_key,
)


def test_resolve_exact_key() -> None:
    assert (
        resolve_lmstudio_model_key(
            "gemma-3-4b-it",
            available=("qwen2-500m-instruct", "gemma-3-4b-it"),
        )
        == "gemma-3-4b-it"
    )


def test_resolve_case_insensitive() -> None:
    assert (
        resolve_lmstudio_model_key(
            "Gemma-3-4B-IT",
            available=("gemma-3-4b-it",),
        )
        == "gemma-3-4b-it"
    )


def test_resolve_alias_google_gemma() -> None:
    assert (
        resolve_lmstudio_model_key(
            "gemma-3-4b-it",
            available=("google/gemma-3-4b", "qwen2-500m-instruct"),
        )
        == "google/gemma-3-4b"
    )


def test_resolve_suffix_match() -> None:
    assert (
        resolve_lmstudio_model_key(
            "gemma-3-4b",
            available=("google/gemma-3-4b",),
        )
        == "google/gemma-3-4b"
    )


def test_resolve_missing_lists_available() -> None:
    with pytest.raises(LmStudioModelNotFoundError) as excinfo:
        resolve_lmstudio_model_key(
            "missing-model",
            available=("qwen2-500m-instruct",),
        )
    assert "missing-model" in str(excinfo.value)
    assert "qwen2-500m-instruct" in str(excinfo.value)


def test_get_lmstudio_llm_retries_after_path_not_found(monkeypatch: Any) -> None:
    from happy_housing.app import lmstudio_models as mod

    calls: list[str] = []
    llm = MagicMock(name="llm")

    def fake_llm(key: str) -> MagicMock:
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError(
                'Model not found: gemma-3-4b-it\ncode": "generic.pathNotFound"'
            )
        return llm

    monkeypatch.setattr(mod.lms, "llm", fake_llm)
    monkeypatch.setattr(mod, "list_lmstudio_llm_keys", lambda: ["gemma-3-4b-it"])
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    got = mod.get_lmstudio_llm("gemma-3-4b-it")
    assert got is llm
    assert calls == ["gemma-3-4b-it", "gemma-3-4b-it"]
