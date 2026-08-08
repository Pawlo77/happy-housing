"""Tests for RCN GML streaming parser with synthetic PL-1992 coords."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from happy_housing.ingest.rcn import extract_rcn_transactions, parse_rcn_gml_stream


def _synthetic_gml() -> str:
    """Minimal GML member with PL-1992 point near central Warsaw + cena/pow."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml"
  xmlns:bt="http://example.pl/rcn">
  <gml:featureMember>
    <bt:Transakcja>
      <bt:posList>637800.50 486500.25</bt:posList>
      <bt:cena>850000</bt:cena>
      <bt:powierzchnia>48.5</bt:powierzchnia>
      <bt:dataTransakcji>2024-06-15</bt:dataTransakcji>
      <bt:rodzaj>lokal mieszkalny</bt:rodzaj>
    </bt:Transakcja>
  </gml:featureMember>
  <gml:featureMember>
    <bt:Transakcja>
      <bt:posList>637900.10 486600.40</bt:posList>
      <bt:cena>1200000</bt:cena>
      <bt:powierzchnia>72</bt:powierzchnia>
      <bt:rodzaj>lokal mieszkalny</bt:rodzaj>
    </bt:Transakcja>
  </gml:featureMember>
</gml:FeatureCollection>
"""


def _linked_gml() -> str:
    """Minimal linked transakcja / nieruchomosc / dzialka graph."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2"
  xmlns:rcn="urn:gugik:specyfikacje:gmlas:rejestrcennieruchomosci:1.0"
  xmlns:xlink="http://www.w3.org/1999/xlink">
  <gml:featureMember>
    <rcn:RCN_Transakcja gml:id="TX1" xmlns:gml="http://www.opengis.net/gml/3.2">
      <rcn:cenaTransakcjiBrutto>900000.00</rcn:cenaTransakcjiBrutto>
      <rcn:nieruchomosc xlink:href="PROP1"/>
    </rcn:RCN_Transakcja>
  </gml:featureMember>
  <gml:featureMember>
    <rcn:RCN_Nieruchomosc gml:id="PROP1" xmlns:gml="http://www.opengis.net/gml/3.2">
      <rcn:rodzajNieruchomosci>4</rcn:rodzajNieruchomosci>
      <rcn:polePowierzchniNieruchomosciGruntowej uom="ha"
        >55.00</rcn:polePowierzchniNieruchomosciGruntowej>
      <rcn:cenaNieruchomosciBrutto>900000.00</rcn:cenaNieruchomosciBrutto>
      <rcn:dzialka xlink:href="PARCEL1"/>
    </rcn:RCN_Nieruchomosc>
  </gml:featureMember>
  <gml:featureMember>
    <rcn:RCN_Dzialka gml:id="PARCEL1" xmlns:gml="http://www.opengis.net/gml/3.2">
      <rcn:polePowierzchniEwidencyjnej uom="m2">55.00</rcn:polePowierzchniEwidencyjnej>
      <rcn:dodatkoweInformacje>DANE Z MIGRACJI
[Dzielnica: Wola; Obręb: 1; Działka nr: 1]</rcn:dodatkoweInformacje>
    </rcn:RCN_Dzialka>
  </gml:featureMember>
</gml:FeatureCollection>
"""


def test_parse_rcn_gml_stream_projects_and_filters() -> None:
    """Parser projects EPSG:2180 → WGS84 and keeps priced Warsaw rows."""
    handle = io.BytesIO(_synthetic_gml().encode("utf-8"))
    frame = parse_rcn_gml_stream(handle, max_rows=10)
    assert len(frame) == 2
    assert frame["price_pln"].tolist() == [850000.0, 1200000.0]
    assert frame["area_m2"].tolist() == [48.5, 72.0]
    assert frame["lon"].between(20.5, 21.5).all()
    assert frame["lat"].between(51.9, 52.5).all()


def test_parse_linked_rcn_uses_district_centroid() -> None:
    """Linked dump without geometry falls back to district centroid."""
    handle = io.BytesIO(_linked_gml().encode("utf-8"))
    frame = parse_rcn_gml_stream(handle, max_rows=10)
    assert len(frame) == 1
    assert float(frame.iloc[0]["price_pln"]) == 900000.0
    assert frame.iloc[0]["district"] == "Wola"
    assert 20.5 < float(frame.iloc[0]["lon"]) < 21.5


def test_extract_rcn_transactions_from_zip(tmp_path: Path) -> None:
    """Zip member extraction writes parquet with price_per_m2."""
    zip_path = tmp_path / "rcn.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample.gml", _synthetic_gml())
    out = tmp_path / "rcn.parquet"
    frame = extract_rcn_transactions(zip_path, out, max_rows=10)
    assert out.exists()
    assert len(frame) == 2
    assert "price_per_m2" in frame.columns
    assert float(frame.iloc[0]["price_per_m2"]) == pytest.approx(850000 / 48.5)
