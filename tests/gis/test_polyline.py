"""Tests for Google-encoded polyline decode used by OTP legGeometry."""

from happy_housing.gis.polyline import decode_google_polyline


def test_decode_google_polyline_known_sample() -> None:
    # Encoded form of (38.5, -120.2), (40.7, -120.95), (43.252, -126.453).
    points = decode_google_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    assert len(points) == 3
    assert abs(points[0][1] - 38.5) < 1e-4  # lat
    assert abs(points[0][0] - (-120.2)) < 1e-4  # lon
    assert abs(points[1][1] - 40.7) < 1e-4
    assert abs(points[2][0] - (-126.453)) < 1e-3


def test_decode_google_polyline_empty() -> None:
    assert decode_google_polyline("") == []
    assert decode_google_polyline(None) == []  # type: ignore[arg-type]
