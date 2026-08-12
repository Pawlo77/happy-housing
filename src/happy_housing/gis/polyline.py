"""Google-encoded polyline decode (OTP ``legGeometry.points``)."""

from __future__ import annotations


def decode_google_polyline(encoded: str | None) -> list[list[float]]:
    """Decode a Google polyline into ``[[lon, lat], …]`` for pydeck PathLayer.

    Args:
        encoded: Polyline string from OTP ``legGeometry.points``.

    Returns:
        Lon/lat pairs (empty when ``encoded`` is blank).
    """
    if not encoded:
        return []
    coords: list[list[float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)
    while index < length:
        result = 0
        shift = 0
        while True:
            if index >= length:
                return coords
            value = ord(encoded[index]) - 63
            index += 1
            result |= (value & 0x1F) << shift
            shift += 5
            if value < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += delta_lat

        result = 0
        shift = 0
        while True:
            if index >= length:
                return coords
            value = ord(encoded[index]) - 63
            index += 1
            result |= (value & 0x1F) << shift
            shift += 5
            if value < 0x20:
                break
        delta_lon = ~(result >> 1) if result & 1 else (result >> 1)
        lon += delta_lon
        coords.append([lon / 1e5, lat / 1e5])
    return coords
