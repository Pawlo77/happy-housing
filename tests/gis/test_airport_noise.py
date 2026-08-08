"""Aircraft Lden/Lnight levels from END contour bands."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from happy_housing.gis import (
    attach_aircraft_noise_levels,
    enrich_listings,
    listings_to_gdf,
    load_noise_contours,
)


def _band(lon0: float, lat0: float, lon1: float, lat1: float, db: int) -> dict:
    return {
        "lden_db": db,
        "geometry": Polygon([
            (lon0, lat0),
            (lon1, lat0),
            (lon1, lat1),
            (lon0, lat1),
            (lon0, lat0),
        ]),
    }


def test_attach_aircraft_noise_takes_max_containing_band() -> None:
    """Nested END bands → listing gets highest Lden band that contains it."""
    contours = gpd.GeoDataFrame(
        [
            _band(20.9, 52.1, 21.1, 52.3, 55),
            _band(20.95, 52.15, 21.05, 52.25, 65),
            _band(20.98, 52.18, 21.02, 52.22, 70),
        ],
        crs="EPSG:4326",
    )
    listings = listings_to_gdf(
        pd.DataFrame({
            "listing_id": ["loud", "mid", "quiet"],
            "lon": [21.00, 21.00, 21.20],
            "lat": [52.20, 52.16, 52.20],
        })
    )
    out = attach_aircraft_noise_levels(
        listings,
        lden=contours,
        lnight=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    )
    by_id = out.set_index("listing_id")
    assert float(by_id.loc["loud", "aircraft_lden_db"]) == 70.0
    assert float(by_id.loc["mid", "aircraft_lden_db"]) == 65.0
    assert float(by_id.loc["quiet", "aircraft_lden_db"]) == 0.0
    assert int(by_id.loc["loud", "in_oou"]) == 1
    assert int(by_id.loc["quiet", "in_oou"]) == 0


def test_load_noise_contours_reads_lden_property(tmp_path: Path) -> None:
    """Fixture GeoJSON with lden_db loads as contour frame."""
    path = tmp_path / "lden.geojson"
    gpd.GeoDataFrame(
        [_band(20.9, 52.1, 21.0, 52.2, 60)],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")
    loaded = load_noise_contours(path, value_col="lden_db")
    assert list(loaded.columns) == ["lden_db", "geometry"]
    assert int(loaded.iloc[0]["lden_db"]) == 60


def test_enrich_listings_uses_aircraft_noise_contours(tmp_path: Path) -> None:
    """enrich_listings attaches aircraft_lden_db from contour paths."""
    lden_path = tmp_path / "lden.geojson"
    lnight_path = tmp_path / "lnight.geojson"
    gpd.GeoDataFrame(
        [_band(20.95, 52.15, 21.05, 52.25, 60)],
        crs="EPSG:4326",
    ).to_file(lden_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        [
            {
                "lnight_db": 50,
                "geometry": Polygon([
                    (20.95, 52.15),
                    (21.05, 52.15),
                    (21.05, 52.25),
                    (20.95, 52.25),
                    (20.95, 52.15),
                ]),
            }
        ],
        crs="EPSG:4326",
    ).to_file(lnight_path, driver="GeoJSON")
    listings = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lon": [21.0, 21.2],
        "lat": [52.2, 52.2],
        "price_pln": [500_000, 600_000],
        "area_m2": [50.0, 60.0],
    })
    amenities = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    enriched = enrich_listings(
        listings,
        projected_crs="EPSG:2180",
        amenities=amenities,
        gios_stations=pd.DataFrame(columns=["lon", "lat"]),
        gtfs_stops=pd.DataFrame(columns=["stop_lon", "stop_lat"]),
        commute_lon=21.0,
        commute_lat=52.23,
        aircraft_lden_path=lden_path,
        aircraft_lnight_path=lnight_path,
    )
    assert float(enriched.loc[0, "aircraft_lden_db"]) == 60.0
    assert float(enriched.loc[0, "aircraft_lnight_db"]) == 50.0
    assert float(enriched.loc[1, "aircraft_lden_db"]) == 0.0
    assert int(enriched.loc[0, "in_oou"]) == 1
    assert int(enriched.loc[1, "in_oou"]) == 0
