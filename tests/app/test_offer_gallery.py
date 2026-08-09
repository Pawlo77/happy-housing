"""Tests for offer photo carousel index helpers."""

from __future__ import annotations

from happy_housing.app.offer_gallery import (
    clamp_photo_index,
    photo_index_key,
    wrap_photo_index,
)


def test_wrap_photo_index_wraps_around() -> None:
    assert wrap_photo_index(0, 5, -1) == 4
    assert wrap_photo_index(4, 5, 1) == 0
    assert wrap_photo_index(2, 5, 1) == 3


def test_wrap_photo_index_empty() -> None:
    assert wrap_photo_index(0, 0, 1) == 0


def test_clamp_photo_index() -> None:
    assert clamp_photo_index(-1, 3) == 0
    assert clamp_photo_index(99, 3) == 2
    assert clamp_photo_index(1, 3) == 1
    assert clamp_photo_index(0, 0) == 0


def test_photo_index_key() -> None:
    assert photo_index_key("OTO-1") == "offer_photo_idx_OTO-1"
