"""Run metadata for the Home dashboard (refresh time, metrics, sources)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from happy_housing.app.io_cache import read_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataSource:
    """Attributed open-data / listing source shown on Home."""

    name: str
    """Short display name."""
    url: str
    """Canonical https homepage or docs URL."""
    role: str
    """What this source contributes to the model or UI."""
    key: str
    """Stable inventory key (``otodom``, ``gtfs``, …)."""
    unit: str
    """Count noun for inventory (``listings``, ``stops``, …)."""


@dataclass(slots=True)
class RunMetadata:
    """Snapshot of the last pipeline/training run shown on Home."""

    refreshed_at: str
    """ISO-8601 UTC timestamp of the completed run."""
    duration_s: float
    """Wall-clock seconds for the pipeline run."""
    n_offers: int
    """Rows in the scored universe."""
    n_by_source: dict[str, int] = field(default_factory=dict)
    """Offer counts keyed by source label."""
    model_metrics: dict[str, float] = field(default_factory=dict)
    """Hold-out metrics from LightGBM training."""
    model_path: str = ""
    """Relative or absolute path to the saved booster."""
    feature_columns: list[str] = field(default_factory=list)
    """Feature names used by the model."""
    sources: list[str] = field(default_factory=list)
    """Human-readable data source attributions."""
    methodology: str = ""
    """Short modeling methodology blurb."""


def write_run_metadata(path: Path, meta: RunMetadata) -> Path:
    """Persist run metadata as JSON next to scored parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote run metadata to %s", path)
    return path


def load_run_metadata(path: Path) -> RunMetadata | None:
    """Load run metadata JSON, or ``None`` if missing/invalid."""
    if not path.exists():
        return None
    try:
        payload_obj = read_json(str(path), path.stat().st_mtime)
        if not isinstance(payload_obj, dict):
            return None
        payload: dict[str, Any] = payload_obj
        return RunMetadata(
            refreshed_at=str(payload.get("refreshed_at", "")),
            duration_s=float(payload.get("duration_s", 0.0)),
            n_offers=int(payload.get("n_offers", 0)),
            n_by_source={
                str(k): int(v)
                for k, v in dict(payload.get("n_by_source") or {}).items()
            },
            model_metrics={
                str(k): float(v)
                for k, v in dict(payload.get("model_metrics") or {}).items()
            },
            model_path=str(payload.get("model_path", "")),
            feature_columns=[
                str(c) for c in list(payload.get("feature_columns") or [])
            ],
            sources=[str(s) for s in list(payload.get("sources") or [])],
            methodology=str(payload.get("methodology", "")),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load run metadata %s: %s", path, exc)
        return None


DEFAULT_SOURCE_CATALOG: tuple[DataSource, ...] = (
    DataSource(
        name="Otodom (public listing pages)",
        url="https://www.otodom.pl/",
        role="Asking prices, photos, and listing URLs for Warsaw flats.",
        key="otodom",
        unit="listings",
    ),
    DataSource(
        name="UM Warszawa RCN (transaction GML)",
        url="https://mapa.um.warszawa.pl/",
        role="Notarial transaction prices for fair-value context.",
        key="rcn",
        unit="transactions",
    ),
    DataSource(
        name="OpenStreetMap / OSMnx amenities",
        url="https://www.openstreetmap.org/",
        role="Hospital, fire, police, shops, and other amenity distances.",
        key="osm",
        unit="amenities",
    ),
    DataSource(
        name="UM Warszawa PLACE_SKWERY (testmapa WFS)",
        url="https://testmapa.um.warszawa.pl/",
        role="Official parks and squares — nearest-park distance (dist_park_m).",
        key="warsaw_parks",
        unit="parks",
    ),
    DataSource(
        name="ZTM Warsaw GTFS via mkuran.pl",
        url="https://mkuran.pl/gtfs/",
        role="Transit stop proximity from the public Warsaw GTFS feed.",
        key="gtfs",
        unit="stops",
    ),
    DataSource(
        name="GIOŚ air-quality stations (API v1)",
        url="https://powietrze.gios.gov.pl/",
        role=(
            "GIOŚ station measurements → continuous IDW AQI/PM/NO₂ field "
            "sampled onto listings (plus station distance); Explore BitmapLayer "
            "overlay mirrors strategic noise."
        ),
        key="gios",
        unit="stations",
    ),
    DataSource(
        name="GUNB building permits (CSV when available)",
        url="https://wyszukiwarka.gunb.gov.pl/",
        role="Nearby construction / permit activity proxy.",
        key="gunb",
        unit="permits",
    ),
    DataSource(
        name="GUS US Warszawa — Panorama dzielnic (crime)",
        url="https://warszawa.stat.gov.pl/",
        role=(
            "Police-ascertained crimes and detectability by Warsaw district "
            "(Panorama dzielnic Excel annex); rates per 1,000 residents."
        ),
        key="crime",
        unit="districts",
    ),
    DataSource(
        name="Chopin aircraft Lden / Lnight (EEA END)",
        url=(
            "https://noise.discomap.eea.europa.eu/arcgis/rest/services/"
            "Noise/2012_NOISE_END_LAEA_update2015_Contours/MapServer"
        ),
        role=(
            "Strategic aircraft noise contour bands (Lden/Lnight dB) for "
            "Poland major airports — END 2012/update2015."
        ),
        key="oou",
        unit="zones",
    ),
    DataSource(
        name="OSM traffic sample (roads)",
        url="https://www.openstreetmap.org/",
        role="Diurnal street-activity field for Explore Map / Spatial traffic.",
        key="traffic",
        unit="segments",
    ),
    DataSource(
        name="OSM bicycle infrastructure",
        url="https://www.openstreetmap.org/",
        role="Bike-path coverage features and Explore Map bike layer.",
        key="bike_paths",
        unit="ways",
    ),
)
"""Canonical data sources with URLs for the Home page."""

DEFAULT_SOURCES: list[str] = [src.name for src in DEFAULT_SOURCE_CATALOG]
"""Human-readable data source names stored in run metadata."""

DEFAULT_METHODOLOGY: str = (
    "LightGBM regresses asking/transaction zł/m² from property traits "
    "(area, rooms, year) and livability distances (hospital, fire, police, "
    "transit, air station, GUNB permits, commute to the Admin commute target) plus "
    "an aircraft Lden/Lnight feature from EEA END contours, nearest-station "
    "GIOŚ air quality (AQI + PM10/PM2.5/NO2, not distance alone), and "
    "district police crime rates (per 1,000 residents + detectability). SHAP "
    "TreeExplainer "
    "explains each offer. "
    "Commute minutes use local OSRM when available, else a 30 km/h distance proxy."
)
"""Short modeling methodology blurb (prefer ``format_methodology_note``)."""

METHODOLOGY_STEPS: tuple[str, ...] = (
    "Ingest Otodom asks and UM Warszawa RCN transactions into one universe.",
    "Enrich each pin with aircraft Lden/Lnight bands and nearest amenity / "
    "transit / air / GUNB distances.",
    "Attach commute minutes via local OSRM, or a 30 km/h distance proxy.",
    "Train LightGBM on hold-out zł/m²; log MAE/R² and the booster to MLflow.",
    "Score every offer; Explore explains residuals with SHAP TreeExplainer.",
)
"""Step-wise methodology bullets for the Home Model tab."""
