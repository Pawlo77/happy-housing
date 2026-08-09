"""Tests for GUNB point GeoDataFrame helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import requests

from happy_housing.ingest.gunb import (
    _centroid_lon_lat,
    _fetch_wfs_city_layer,
    _frame_from_wfs_geojson,
    _frame_from_wfs_gml,
    gunb_points_gdf,
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
    ok.raise_for_status = MagicMock()

    def fake_get(*_a: object, **_k: object) -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return ok

    monkeypatch.setattr("happy_housing.ingest.gunb.requests.get", fake_get)
    monkeypatch.setattr("happy_housing.ingest.gunb.time.sleep", lambda _s: None)

    frame = _fetch_wfs_city_layer("ms:pozwolenia_2026", page_size=500)
    assert calls["n"] == 2
    assert len(frame) == 1
    assert frame.iloc[0]["numer_decyzji"] == "RETRY-1"
