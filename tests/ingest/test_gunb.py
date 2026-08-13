"""Tests for GUNB point GeoDataFrame helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import requests

from happy_housing.ingest.gunb import (
    _GUNB_WFS_LAYERS_ALL,
    _GUNB_WFS_LAYERS_RECENT,
    _centroid_lon_lat,
    _fetch_wfs_city_layer,
    _frame_from_wfs_geojson,
    _frame_from_wfs_gml,
    _keep_warsaw_metro_row,
    gunb_points_gdf,
    resolve_gunb_wfs_layers,
)


def test_gunb_points_gdf_from_lon_lat() -> None:
    """Lon/lat columns become a non-empty GeoDataFrame."""
    frame = pd.DataFrame({
        "lon": [21.0, 21.1],
        "lat": [52.2, 52.3],
        "adres": ["a", "b"],
    })
    gdf = gunb_points_gdf(frame)
    assert len(gdf) == 2
    assert gdf.crs.to_string() == "EPSG:4326"


def test_gunb_points_gdf_empty_without_coords() -> None:
    """Missing coordinate columns yield an empty GeoDataFrame."""
    frame = pd.DataFrame({"adres": ["x"]})
    gdf = gunb_points_gdf(frame)
    assert gdf.empty


def test_frame_from_wfs_geojson() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [21.0, 52.2]},
                "properties": {"adres": "x", "numer_decyzji": "1"},
            }
        ],
    }
    frame = _frame_from_wfs_geojson(payload)
    assert len(frame) == 1
    assert float(frame.iloc[0]["lon"]) == 21.0


def test_centroid_lon_lat_from_gml_pos_list() -> None:
    """GUGiK GML uses lat/lon pairs; centroid returns lon/lat."""
    lon, lat = _centroid_lon_lat("52.2 21.0 52.3 21.0 52.3 21.1 52.2 21.1")
    assert abs(lon - 21.05) < 1e-6
    assert abs(lat - 52.25) < 1e-6


def test_frame_from_wfs_gml_extracts_centroid() -> None:
    pos = "52.2 21.0 52.3 21.0 52.3 21.1 52.2 21.1"
    xml = f"""<?xml version='1.0' encoding="UTF-8" ?>
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"
      xmlns:gml="http://www.opengis.net/gml/3.2"
      xmlns:ms="http://mapserver.gis.umn.edu/mapserver"
      numberReturned="1">
      <wfs:member>
        <ms:pozwolenia_2025>
          <ms:msGeometry>
            <gml:Polygon srsName="urn:ogc:def:crs:EPSG::4326">
              <gml:exterior>
                <gml:LinearRing>
                  <gml:posList srsDimension="2">{pos}</gml:posList>
                </gml:LinearRing>
              </gml:exterior>
            </gml:Polygon>
          </ms:msGeometry>
          <ms:miasto>Warszawa</ms:miasto>
          <ms:ulica>Krucza</ms:ulica>
          <ms:nr_domu>38</ms:nr_domu>
          <ms:numer_ewidencyjny_system>ABC-1</ms:numer_ewidencyjny_system>
        </ms:pozwolenia_2025>
      </wfs:member>
    </wfs:FeatureCollection>
    """
    frame = _frame_from_wfs_gml(xml)
    assert len(frame) == 1
    assert frame.iloc[0]["miasto"] == "Warszawa"
    assert "Krucza" in str(frame.iloc[0]["adres"])
    assert abs(float(frame.iloc[0]["lon"]) - 21.05) < 1e-6
    assert frame.iloc[0]["numer_ewidencyjny_system"] == "ABC-1"


def test_fetch_wfs_city_layer_retries_on_timeout(monkeypatch: Any) -> None:
    """Geoportal read timeouts retry before failing the layer."""
    pos = "52.2 21.0 52.3 21.0 52.3 21.1 52.2 21.1"
    xml = f"""<?xml version='1.0' encoding="UTF-8" ?>
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"
      xmlns:gml="http://www.opengis.net/gml/3.2"
      xmlns:ms="http://mapserver.gis.umn.edu/mapserver"
      numberReturned="1">
      <wfs:member>
        <ms:pozwolenia_2026>
          <ms:msGeometry>
            <gml:Polygon srsName="urn:ogc:def:crs:EPSG::4326">
              <gml:exterior>
                <gml:LinearRing>
                  <gml:posList srsDimension="2">{pos}</gml:posList>
                </gml:LinearRing>
              </gml:exterior>
            </gml:Polygon>
          </ms:msGeometry>
          <ms:miasto>Warszawa</ms:miasto>
          <ms:ulica>Marszalkowska</ms:ulica>
          <ms:nr_domu>1</ms:nr_domu>
          <ms:numer_ewidencyjny_system>RETRY-1</ms:numer_ewidencyjny_system>
        </ms:pozwolenia_2026>
      </wfs:member>
    </wfs:FeatureCollection>
    """
    calls = {"n": 0}
    ok = MagicMock()
    ok.text = xml
    ok.status_code = 200
    ok.raise_for_status = MagicMock()

    def fake_session_get(self: object, *_a: object, **_k: object) -> MagicMock:
        _ = self
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return ok

    monkeypatch.setattr(requests.Session, "get", fake_session_get)
    monkeypatch.setattr(
        "happy_housing.ingest.http_client.time.sleep",
        lambda _s: None,
    )

    frame = _fetch_wfs_city_layer("ms:pozwolenia_2026", page_size=500)
    assert calls["n"] == 2
    assert len(frame) == 1
    assert frame.iloc[0]["numer_decyzji"] == "RETRY-1"


def test_keep_warsaw_metro_row_by_city_terc_or_bbox() -> None:
    assert _keep_warsaw_metro_row({"miasto": "Warszawa", "lon": 19.0, "lat": 50.0})
    assert _keep_warsaw_metro_row({
        "miasto": "Kraków",
        "terc": "1465011",
        "lon": 19.0,
        "lat": 50.0,
    })
    assert _keep_warsaw_metro_row({"miasto": "Piaseczno", "lon": 21.02, "lat": 52.07})
    assert not _keep_warsaw_metro_row({"miasto": "Kraków", "lon": 19.9, "lat": 50.1})


def test_fetch_wfs_filters_client_side_and_past_old_cap(monkeypatch: Any) -> None:
    """GUGiK ignores CQL — keep Warsaw-metro rows and scan past the old 200k stop."""

    def _member(city: str, lon: float, lat: float, decision: str) -> str:
        # GML posList is lat lon pairs.
        pos = (
            f"{lat} {lon} {lat + 0.001} {lon} "
            f"{lat + 0.001} {lon + 0.001} {lat} {lon + 0.001}"
        )
        return f"""
      <wfs:member>
        <ms:pozwolenia_2025>
          <ms:msGeometry>
            <gml:Polygon srsName="urn:ogc:def:crs:EPSG::4326">
              <gml:exterior>
                <gml:LinearRing>
                  <gml:posList srsDimension="2">{pos}</gml:posList>
                </gml:LinearRing>
              </gml:exterior>
            </gml:Polygon>
          </ms:msGeometry>
          <ms:miasto>{city}</ms:miasto>
          <ms:numer_ewidencyjny_system>{decision}</ms:numer_ewidencyjny_system>
        </ms:pozwolenia_2025>
      </wfs:member>"""

    page_size = 2
    # Three full pages then a short page — startIndex goes 0,2,4,6 (> old mental model).
    pages = [
        _member("Kraków", 19.9, 50.1, "K1") + _member("Warszawa", 21.0, 52.2, "W1"),
        _member("Gdańsk", 18.6, 54.3, "G1") + _member("Piaseczno", 21.02, 52.07, "P1"),
        _member("Warszawa", 21.1, 52.25, "W2") + _member("Łódź", 19.4, 51.7, "L1"),
        _member("Warszawa", 21.05, 52.23, "W3"),
    ]
    calls: list[int] = []

    def fake_session_get(self: object, _url: object, **kwargs: object) -> MagicMock:
        _ = self
        params = kwargs.get("params")
        assert isinstance(params, dict)
        start = int(params["startIndex"])  # type: ignore[index]
        calls.append(start)
        idx = start // page_size
        body = pages[idx] if idx < len(pages) else ""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = f"""<?xml version='1.0' encoding="UTF-8" ?>
    <wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"
      xmlns:gml="http://www.opengis.net/gml/3.2"
      xmlns:ms="http://mapserver.gis.umn.edu/mapserver"
      numberMatched="7" numberReturned="{page_size if idx < 3 else 1}">
      {body}
    </wfs:FeatureCollection>
    """
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(requests.Session, "get", fake_session_get)
    monkeypatch.setattr(
        "happy_housing.ingest.http_client.time.sleep",
        lambda _s: None,
    )

    frame = _fetch_wfs_city_layer("ms:pozwolenia_2025", page_size=page_size)
    assert calls == [0, 2, 4, 6]
    assert sorted(frame["numer_decyzji"].astype(str)) == ["P1", "W1", "W2", "W3"]


def test_resolve_gunb_wfs_layers_recent_by_default() -> None:
    recent = resolve_gunb_wfs_layers(full_history=False)
    assert recent == _GUNB_WFS_LAYERS_RECENT
    assert "ms:pozwolenia_pozostale" not in recent
    assert len(recent) == 3


def test_resolve_gunb_wfs_layers_full_includes_pozostale() -> None:
    full = resolve_gunb_wfs_layers(full_history=True)
    assert full == _GUNB_WFS_LAYERS_ALL
    assert full[-1] == "ms:pozwolenia_pozostale"
