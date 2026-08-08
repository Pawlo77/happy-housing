"""Run metadata for the Home dashboard (refresh time, metrics, sources)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
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
    ),
    DataSource(
        name="UM Warszawa RCN (transaction GML)",
        url="https://mapa.um.warszawa.pl/",
        role="Notarial transaction prices for fair-value context.",
    ),
    DataSource(
        name="OpenStreetMap / OSMnx amenities",
        url="https://www.openstreetmap.org/",
        role="Hospital, fire, and police amenity distances.",
    ),
    DataSource(
        name="ZTM Warsaw GTFS via mkuran.pl",
        url="https://mkuran.pl/gtfs/",
        role="Transit stop proximity from the public Warsaw GTFS feed.",
    ),
    DataSource(
        name="GIOŚ air-quality stations (API v1)",
        url="https://powietrze.gios.gov.pl/",
        role="Nearest air-quality station distance.",
    ),
    DataSource(
        name="GUNB building permits (CSV when available)",
        url="https://wyszukiwarka.gunb.gov.pl/",
        role="Nearby construction / permit activity proxy.",
    ),
    DataSource(
        name="Chopin OOU restricted-use zone",
        url="https://www.lotnisko-chopina.pl/",
        role="Aircraft restricted-use zone flag (noise/livability proxy).",
    ),
)
"""Canonical data sources with URLs for the Home page."""

DEFAULT_SOURCES: list[str] = [src.name for src in DEFAULT_SOURCE_CATALOG]
"""Human-readable data source names stored in run metadata."""

DEFAULT_METHODOLOGY: str = (
    "LightGBM regresses asking/transaction zł/m² from property traits "
    "(area, rooms, year) and livability distances (hospital, fire, police, "
    "transit, air station, GUNB permits, commute to Rondo Daszyńskiego) plus "
    "an OOU aircraft-zone flag. SHAP TreeExplainer explains each offer. "
    "Commute minutes use local OSRM when available, else a 30 km/h distance proxy."
)
"""Short modeling methodology blurb."""

METHODOLOGY_STEPS: tuple[str, ...] = (
    "Ingest Otodom asks and UM Warszawa RCN transactions into one universe.",
    "Enrich each pin with OOU membership and nearest amenity / transit / "
    "air / GUNB distances.",
    "Attach commute minutes via local OSRM, or a 30 km/h distance proxy.",
    "Train LightGBM on hold-out zł/m²; log MAE/R² and the booster to MLflow.",
    "Score every offer; Explore explains residuals with SHAP TreeExplainer.",
)
"""Step-wise methodology bullets for the Home Model tab."""
