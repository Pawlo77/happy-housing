"""Tests for pipeline tqdm helper."""

from happy_housing.pipeline_progress import tqdm_disabled


def test_tqdm_disabled_under_pytest(monkeypatch) -> None:
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_pipeline_progress.py::test")
    assert tqdm_disabled() is True


def test_tqdm_disabled_by_env(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TQDM_DISABLE", "1")
    assert tqdm_disabled() is True
