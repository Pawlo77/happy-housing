"""Tests for pipeline step fingerprints and skip decisions."""

from pathlib import Path

from happy_housing.pipeline_cache import (
    StepManifest,
    file_token,
    fingerprint,
    outputs_fresh_vs_inputs,
    payload_token,
)


def test_fingerprint_stable_and_order_sensitive() -> None:
    assert fingerprint("a", "b") == fingerprint("a", "b")
    assert fingerprint("a", "b") != fingerprint("b", "a")


def test_file_token_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"one")
    first = file_token(path)
    path.write_bytes(b"two-bytes")
    second = file_token(path)
    assert first != second
    assert "missing" in file_token(tmp_path / "nope")


def test_outputs_fresh_vs_inputs(tmp_path: Path) -> None:
    src = tmp_path / "src.zip"
    out = tmp_path / "out.parquet"
    src.write_bytes(b"zip")
    assert not outputs_fresh_vs_inputs([out], [src])
    out.write_bytes(b"parquet")
    assert outputs_fresh_vs_inputs([out], [src])
    src.write_bytes(b"zip-changed")
    assert not outputs_fresh_vs_inputs([out], [src])


def test_step_manifest_skip_and_mark(tmp_path: Path) -> None:
    manifest = StepManifest.load(tmp_path / "pipeline_steps.json")
    out = tmp_path / "scored.parquet"
    out.write_text("x", encoding="utf-8")
    fp = fingerprint("inputs-v1")
    assert not manifest.should_skip("train_score", fp, [out])
    manifest.mark("train_score", fp)
    manifest.save()
    reloaded = StepManifest.load(tmp_path / "pipeline_steps.json")
    assert reloaded.should_skip("train_score", fp, [out])
    assert not reloaded.should_skip("train_score", fingerprint("other"), [out])


def test_step_manifest_mark_and_save_persists_immediately(tmp_path: Path) -> None:
    path = tmp_path / "pipeline_steps.json"
    manifest = StepManifest.load(path)
    out = tmp_path / "scored.parquet"
    out.write_text("x", encoding="utf-8")
    fp = fingerprint("inputs-v1")
    manifest.mark_and_save("train_score", fp)
    reloaded = StepManifest.load(path)
    assert reloaded.should_skip("train_score", fp, [out])


def test_payload_token_changes() -> None:
    assert payload_token({"a": 1}) != payload_token({"a": 2})
