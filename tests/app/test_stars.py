"""Tests for persistent starred-offer store."""

from __future__ import annotations

from pathlib import Path

from happy_housing.app.stars import (
    is_starred,
    load_stars,
    star_marker,
    stars_path,
    toggle_star,
)


def test_stars_path(tmp_path: Path) -> None:
    assert stars_path(tmp_path).name == "starred_offers.json"


def test_load_stars_missing_is_empty(tmp_path: Path) -> None:
    assert load_stars(stars_path(tmp_path)) == frozenset()


def test_toggle_star_persists(tmp_path: Path) -> None:
    path = stars_path(tmp_path)
    assert toggle_star(path, "OTO-1") is True
    assert load_stars(path) == frozenset({"OTO-1"})
    assert is_starred(load_stars(path), "OTO-1")
    assert toggle_star(path, "OTO-1") is False
    assert load_stars(path) == frozenset()


def test_star_marker() -> None:
    assert star_marker("a", frozenset({"a"})) == "★"
    assert star_marker("b", frozenset({"a"})) == "☆"
