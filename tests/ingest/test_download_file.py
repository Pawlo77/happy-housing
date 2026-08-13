"""Tests for archive download reuse and URL-change refresh."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from happy_housing.ingest import download_file


def test_download_file_reuses_when_url_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "archive.zip"
    dest.write_bytes(b"old-bytes")
    (tmp_path / "archive.zip.url").write_text(
        "https://example.com/v1.zip\n", encoding="utf-8"
    )
    calls: list[str] = []

    def boom(*_args: Any, **_kwargs: Any) -> MagicMock:
        calls.append("hit")
        raise AssertionError("should not re-download")

    monkeypatch.setattr("happy_housing.ingest.resilient_requests_get", boom)
    out = download_file("https://example.com/v1.zip", dest)
    assert out == dest
    assert calls == []
    assert dest.read_bytes() == b"old-bytes"


def test_download_file_redownloads_when_url_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "archive.zip"
    dest.write_bytes(b"old-bytes")
    (tmp_path / "archive.zip.url").write_text(
        "https://example.com/v1.zip\n", encoding="utf-8"
    )

    class _FakeResponse:
        def iter_content(self, chunk_size: int = 1024) -> list[bytes]:
            _ = chunk_size
            return [b"new-bytes"]

        def close(self) -> None:
            return None

    def fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr("happy_housing.ingest.resilient_requests_get", fake_get)
    out = download_file("https://example.com/v2.zip", dest)
    assert out == dest
    assert dest.read_bytes() == b"new-bytes"
    assert (tmp_path / "archive.zip.url").read_text(encoding="utf-8").strip() == (
        "https://example.com/v2.zip"
    )
