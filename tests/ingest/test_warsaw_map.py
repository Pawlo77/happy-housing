"""Tests for UM Warszawa WFS parks / schools ingest."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from happy_housing.ingest.warsaw_map import (
    WARSAW_WFS_URL,
    discover_education_type_names,
    fetch_warsaw_parks,
    fetch_warsaw_schools,
    fetch_warsaw_wfs_layer,
    normalize_warsaw_crs,
)

_SAMPLE_GML = b"""<?xml version="1.0" encoding="UTF-8"?>
<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs"
 xmlns:gml="http://www.opengis.net/gml"
 xmlns:ns92528565="http://fdo.osgeo.org/schemas/feature/ns92528565">
  <gml:featureMember>
    <ns92528565:PLACE_SKWERY>
      <ns92528565:ID>1</ns92528565:ID>
      <ns92528565:NAZWA_PODST>Park Testowy</ns92528565:NAZWA_PODST>
      <ns92528565:DZIELNICE>Wola</ns92528565:DZIELNICE>
      <ns92528565:GEOMETRY>
        <gml:Polygon srsName="EPSG:2178">
          <gml:outerBoundaryIs>
            <gml:LinearRing>
              <gml:coordinates>
                7500000,5788000 7500100,5788000
                7500100,5788100 7500000,5788100 7500000,5788000
              </gml:coordinates>
            </gml:LinearRing>
          </gml:outerBoundaryIs>
        </gml:Polygon>
      </ns92528565:GEOMETRY>
    </ns92528565:PLACE_SKWERY>
  </gml:featureMember>
</wfs:FeatureCollection>
"""


def test_normalize_warsaw_crs_sets_2178_for_projected_coords() -> None:
    """Projected PLACE_SKWERY coordinates without CRS → EPSG:2178."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(geometry=[Point(7500000, 5788000)])
    out = normalize_warsaw_crs(gdf)
    assert out.crs is not None
    assert "2178" in str(out.crs)


def test_fetch_warsaw_parks_parses_gml_and_caches(tmp_path: Path) -> None:
    """WFS GML parks land in EPSG:4326 GeoJSON cache."""
    mock_resp = MagicMock()
    mock_resp.content = _SAMPLE_GML
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    with patch(
        "happy_housing.ingest.warsaw_map.resilient_requests_get",
        return_value=mock_resp,
    ) as get:
        parks = fetch_warsaw_parks(tmp_path, force_refresh=True)
    assert get.called
    assert WARSAW_WFS_URL in str(get.call_args)
    assert len(parks) == 1
    assert parks.crs is not None
    assert "4326" in str(parks.crs)
    assert (tmp_path / "warsaw_parks.geojson").exists()
    assert "name" in parks.columns
    assert str(parks["name"].iloc[0]) == "Park Testowy"


def test_fetch_warsaw_parks_uses_cache(tmp_path: Path) -> None:
    """Existing cache skips the network."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    cached = gpd.GeoDataFrame(
        {"name": ["Cached Park"]},
        geometry=[
            Polygon([
                (20.98, 52.23),
                (20.99, 52.23),
                (20.99, 52.24),
                (20.98, 52.24),
                (20.98, 52.23),
            ])
        ],
        crs="EPSG:4326",
    )
    cached.to_file(tmp_path / "warsaw_parks.geojson", driver="GeoJSON")
    with patch("happy_housing.ingest.warsaw_map.resilient_requests_get") as get:
        out = fetch_warsaw_parks(tmp_path)
    get.assert_not_called()
    assert len(out) == 1


def test_fetch_warsaw_wfs_layer_caches_under_custom_name(tmp_path: Path) -> None:
    """Generic WFS fetch writes the requested cache filename."""
    mock_resp = MagicMock()
    mock_resp.content = _SAMPLE_GML
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    with patch(
        "happy_housing.ingest.warsaw_map.resilient_requests_get",
        return_value=mock_resp,
    ):
        frame = fetch_warsaw_wfs_layer(
            tmp_path,
            type_name="ns92528565:PLACE_SKWERY",
            cache_name="custom_layer.geojson",
            force_refresh=True,
        )
    assert len(frame) == 1
    assert (tmp_path / "custom_layer.geojson").exists()


def test_fetch_warsaw_schools_uses_fixture_when_wfs_fails(tmp_path: Path) -> None:
    """Schools fall back to fixture GeoJSON and cache warsaw_schools.geojson."""
    fixture = Path("data/fixtures/schools.geojson")
    with patch(
        "happy_housing.ingest.warsaw_map.resilient_requests_get",
        side_effect=RuntimeError("offline"),
    ):
        schools = fetch_warsaw_schools(
            tmp_path,
            fixture_path=fixture,
            force_refresh=True,
        )
    assert len(schools) >= 2
    assert schools.crs is not None
    assert "4326" in str(schools.crs)
    assert (tmp_path / "warsaw_schools.geojson").exists()
    assert "name" in schools.columns


def test_fetch_warsaw_schools_tries_candidate_type_names(tmp_path: Path) -> None:
    """First failing typeName is skipped; successful candidate is cached."""
    ok = MagicMock()
    ok.content = _SAMPLE_GML
    ok.status_code = 200
    ok.raise_for_status = MagicMock()
    with patch(
        "happy_housing.ingest.warsaw_map.resilient_requests_get",
        side_effect=[RuntimeError("404"), ok],
    ) as get:
        schools = fetch_warsaw_schools(
            tmp_path,
            type_names=(
                "ns92528565:SZKOLY",
                "ns92528565:OBWODY_SZKOL_PODSTAWOWYCH",
            ),
            force_refresh=True,
        )
    assert get.call_count == 2
    assert len(schools) == 1
    assert (tmp_path / "warsaw_schools.geojson").exists()


def test_discover_education_skips_catchments(monkeypatch: pytest.MonkeyPatch) -> None:
    """OBWODY catchment typeNames must not be treated as school POIs."""
    caps = MagicMock()
    caps.text = """
    <wfs:Name>ns92528565:OBWODY_SZKOL_PODSTAWOWYCH</wfs:Name>
    <wfs:Name>ns92528565:SZKOLY</wfs:Name>
    """
    monkeypatch.setattr(
        "happy_housing.ingest.warsaw_map.resilient_requests_get",
        lambda *a, **k: caps,
    )
    names = discover_education_type_names()
    assert "ns92528565:SZKOLY" in names
    assert all("OBWOD" not in n.upper() for n in names)
