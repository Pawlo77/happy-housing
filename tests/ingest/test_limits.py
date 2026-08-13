"""Tests for unlimited (None-cap) Otodom / RCN ingest helpers."""

from happy_housing.ingest.limits import reached_cap, resolve_ingest_cap


def test_resolve_ingest_cap_null_means_unlimited() -> None:
    """YAML null / missing / non-positive → unlimited (None)."""
    assert resolve_ingest_cap(None) is None
    assert resolve_ingest_cap(0) is None
    assert resolve_ingest_cap(-1) is None
    assert resolve_ingest_cap(50) == 50


def test_reached_cap_respects_unlimited() -> None:
    """Unlimited caps never trip; finite caps stop at the bound."""
    assert reached_cap(10_000, None) is False
    assert reached_cap(5, 5) is True
    assert reached_cap(4, 5) is False
