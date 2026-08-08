"""Run metadata for the Home dashboard (refresh time, metrics, sources)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


DEFAULT_SOURCES: list[str] = [
    "Otodom (public listing pages)",
    "UM Warszawa RCN (transaction GML)",
    "OpenStreetMap / OSMnx amenities",
    "ZTM Warsaw GTFS via mkuran.pl",
    "GIOŚ air-quality stations (API v1)",
    "GUNB building permits (CSV when available)",
    "Chopin OOU restricted-use zone (bootstrap/open polygon)",
]
"""Human-readable data source attributions."""

DEFAULT_METHODOLOGY: str = (
    "LightGBM regresses asking/transaction zł/m² from property traits "
    "(area, rooms, year) and livability distances (hospital, fire, police, "
    "transit, air station, GUNB permits, commute to Rondo Daszyńskiego) plus "
    "an OOU aircraft-zone flag. SHAP TreeExplainer explains each offer. "
    "Commute minutes use local OSRM when available, else a 30 km/h distance proxy."
)
"""Short modeling methodology blurb."""
