"""Tests for listed_at parsing."""

from happy_housing.ingest.listed_at import coalesce_listed_at, parse_listed_at


def test_parse_iso_and_z() -> None:
    assert parse_listed_at("2024-06-01T12:00:00Z").startswith("2024-06-01T12:00:00")
    assert parse_listed_at("2024-06-01T12:00:00+02:00") is not None


def test_parse_epoch_ms() -> None:
    # 2024-01-01 UTC roughly
    out = parse_listed_at(1_704_067_200_000)
    assert out is not None
    assert out.startswith("2024-01-01")


def test_coalesce_listed_at() -> None:
    assert coalesce_listed_at(None, "bad", "2023-05-01").startswith("2023-05-01")
    assert "2024-01-01" in coalesce_listed_at(None, None)
