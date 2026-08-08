"""Typed experiment configuration loaded from YAML."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Self, cast

import yaml


class ConfigValidationError(ValueError):
    """Raised when a configuration value is invalid."""


def _require(condition: bool, message: str) -> None:
    """Raise a validation error when a condition is false."""
    if not condition:
        raise ConfigValidationError(message)


def _require_str(name: str, value: str) -> None:
    """Validate that a value is a non-empty string."""
    _require(isinstance(value, str), f"{name} must be a string.")
    _require(bool(value.strip()), f"{name} must not be empty.")


def _serialize(value: Any) -> Any:
    """Convert nested dataclasses and tuples into JSON-friendly values."""
    if is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _extract_mapping(data: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    """Validate and return a mapping for deserialization."""
    _require(isinstance(data, Mapping), f"{name} must be a mapping.")
    return data


@dataclass(frozen=True, slots=True)
class MLflowTrackingConfig:
    """Local MLflow tracking configuration."""

    enabled: bool = True
    """Enable MLflow tracking."""
    tracking_uri: str = "sqlite:///mlruns.db"
    """MLflow tracking URI (SQLite backend for SQL-based trace metrics support)."""
    experiment_name: str = "happy-housing"
    """MLflow experiment name."""
    run_name: str | None = None
    """Optional MLflow run name."""
    log_params: bool = True
    """Log hyperparameters to MLflow."""
    log_metrics: bool = True
    """Log metrics to MLflow."""
    log_artifacts: bool = True
    """Log artifacts to MLflow."""
    retain_local_checkpoints: bool = False
    """Keep local checkpoint files after they are uploaded to MLflow artifacts."""

    def __post_init__(self) -> None:
        if self.enabled:
            _require_str("tracking_uri", self.tracking_uri)
        _require_str("experiment_name", self.experiment_name)
        if self.run_name is not None:
            _require_str("run_name", self.run_name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the MLflow config."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an MLflow tracking config from a mapping."""
        mapping = _extract_mapping(data, name="MLflowTrackingConfig")
        if mapping.get("tracking_uri") == "mlruns":
            mapping = {**mapping, "tracking_uri": "sqlite:///mlruns.db"}
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class DataPathsConfig:
    """Local data directory layout for raw, processed, and fixture files."""

    root: str = "data"
    """Project-relative data root."""
    fixtures: str = "data/fixtures"
    """Committed small samples for offline tests."""
    raw: str = "data/raw"
    """Downloaded source dumps (gitignored)."""
    processed: str = "data/processed"
    """Enriched parquet / model outputs (gitignored)."""
    cache: str = "data/cache"
    """OSMnx and HTTP caches (gitignored)."""

    def root_path(self) -> Path:
        """Return the data root path."""
        return Path(self.root)

    def fixtures_path(self) -> Path:
        """Return the fixtures directory path."""
        return Path(self.fixtures)

    def raw_path(self) -> Path:
        """Return the raw downloads directory path."""
        return Path(self.raw)

    def processed_path(self) -> Path:
        """Return the processed outputs directory path."""
        return Path(self.processed)

    def cache_path(self) -> Path:
        """Return the cache directory path."""
        return Path(self.cache)

    def __post_init__(self) -> None:
        for name in ("root", "fixtures", "raw", "processed", "cache"):
            _require_str(name, str(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the data paths."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build data paths from a mapping."""
        mapping = _extract_mapping(data, name="DataPathsConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class WarsawBBoxConfig:
    """Geographic bounding box for Warsaw queries (WGS84)."""

    west: float = 20.85
    """Minimum longitude."""
    south: float = 52.10
    """Minimum latitude."""
    east: float = 21.28
    """Maximum longitude."""
    north: float = 52.37
    """Maximum latitude."""
    crs: str = "EPSG:4326"
    """Coordinate reference system."""
    projected_crs: str = "EPSG:2180"
    """Projected CRS for meter distances (PL-1992)."""

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)``."""
        return (self.west, self.south, self.east, self.north)

    def __post_init__(self) -> None:
        _require(self.west < self.east, "bbox west must be < east.")
        _require(self.south < self.north, "bbox south must be < north.")
        _require_str("crs", self.crs)
        _require_str("projected_crs", self.projected_crs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the bbox."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a Warsaw bbox from a mapping."""
        mapping = _extract_mapping(data, name="WarsawBBoxConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """LightGBM training hyperparameters for price-per-m² regression."""

    target: str = "price_per_m2"
    """Regression target column."""
    test_size: float = 0.2
    """Hold-out fraction for evaluation."""
    random_state: int = 42
    """RNG seed for splits."""
    n_estimators: int = 200
    """Number of boosting rounds."""
    learning_rate: float = 0.05
    """Boosting learning rate."""
    num_leaves: int = 31
    """LightGBM max leaves per tree."""

    def __post_init__(self) -> None:
        _require_str("target", self.target)
        _require(0.0 < self.test_size < 1.0, "test_size must be in (0, 1).")
        _require(self.n_estimators > 0, "n_estimators must be positive.")
        _require(self.learning_rate > 0.0, "learning_rate must be positive.")
        _require(self.num_leaves > 1, "num_leaves must be > 1.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the model config."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a model config from a mapping."""
        mapping = _extract_mapping(data, name="ModelConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class IngestUrlsConfig:
    """Remote free-data endpoints used by download helpers."""

    gtfs_zip: str = "https://mkuran.pl/gtfs/warsaw.zip"
    """Warsaw GTFS static feed (mkuran / Transitland)."""
    gios_stations: str = "https://api.gios.gov.pl/pjp-api/v1/rest/station/findAll"
    """GIOŚ air-quality stations list (v1)."""
    rcn_zip: str = "https://mapa.um.warszawa.pl/rcn_gml/rcn20260622.zip"
    """UM Warszawa RCN GML bulk zip."""
    gunb_zip: str = (
        "https://wyszukiwarka.gunb.gov.pl/pliki_pobranie/wynik_mazowieckie.zip"
    )
    """Official GUNB Mazowieckie bulk zip (attributes; coords via GUGiK WFS)."""
    crime_panorama_7z: str = (
        "https://warszawa.stat.gov.pl/download/gfx/warszawa/pl/"
        "defaultaktualnosci/760/5/26/1/"
        "panorama_dzielnic_warszawy_2024_aneks_excel.7z"
    )
    """US Warszawa Panorama dzielnic Excel annex (.7z) with police crime tables."""

    def __post_init__(self) -> None:
        for name in (
            "gtfs_zip",
            "gios_stations",
            "rcn_zip",
            "gunb_zip",
            "crime_panorama_7z",
        ):
            _require_str(name, str(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of ingest URLs."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build ingest URLs from a mapping."""
        mapping = _extract_mapping(data, name="IngestUrlsConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class IngestLimitsConfig:
    """Download caps; ``None`` / null means take everything the source returns."""

    otodom_max_pages: int | None = None
    """Otodom search pages; ``None`` walks every reported page."""
    otodom_max_listings: int | None = None
    """Geocoded Otodom listings; ``None`` keeps every successful detail."""
    otodom_incremental: bool = True
    """Merge into existing Otodom parquet; fetch only new/updated cards."""
    rcn_max_rows: int | None = None
    """RCN joined Warsaw rows; ``None`` parses the full GML."""

    def __post_init__(self) -> None:
        for name in ("otodom_max_pages", "otodom_max_listings", "rcn_max_rows"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                object.__setattr__(self, name, None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of ingest limits."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build ingest limits from a mapping (YAML null → unlimited)."""
        mapping = _extract_mapping(data, name="IngestLimitsConfig")

        def _opt_int(key: str) -> int | None:
            if key not in mapping or mapping[key] is None:
                return None
            value = int(mapping[key])
            return None if value <= 0 else value

        incremental = mapping.get("otodom_incremental", True)
        return cls(
            otodom_max_pages=_opt_int("otodom_max_pages"),
            otodom_max_listings=_opt_int("otodom_max_listings"),
            otodom_incremental=bool(incremental),
            rcn_max_rows=_opt_int("rcn_max_rows"),
        )


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Stdlib logging defaults for pipeline / app entrypoints."""

    level: str = "INFO"
    """Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    def __post_init__(self) -> None:
        _require_str("level", self.level)
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        _require(
            self.level.strip().upper() in allowed,
            f"level must be one of {sorted(allowed)}.",
        )
        object.__setattr__(self, "level", self.level.strip().upper())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of logging config."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build logging config from a mapping."""
        mapping = _extract_mapping(data, name="LoggingConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Local routing service endpoints (OSRM car, OTP transit)."""

    osrm_url: str = "http://127.0.0.1:5000"
    """OSRM table API root for car commute minutes."""
    otp_url: str = "http://127.0.0.1:8080"
    """OpenTripPlanner 2 HTTP root for transit+walk plans."""
    use_otp_transit: bool = False
    """When true, pipeline fills ``transit_commute_min`` via OTP (slow if large)."""

    def __post_init__(self) -> None:
        _require_str("osrm_url", self.osrm_url)
        _require_str("otp_url", self.otp_url)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of routing endpoints."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build routing config from a mapping."""
        mapping = _extract_mapping(data, name="RoutingConfig")
        return cls(
            osrm_url=str(mapping.get("osrm_url", "http://127.0.0.1:5000")),
            otp_url=str(mapping.get("otp_url", "http://127.0.0.1:8080")),
            use_otp_transit=bool(mapping.get("use_otp_transit", False)),
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "happy-housing"
    """Human-readable experiment name."""
    mlflow: MLflowTrackingConfig = field(default_factory=MLflowTrackingConfig)
    """MLflow tracking settings for this experiment."""
    data: DataPathsConfig = field(default_factory=DataPathsConfig)
    """Local data path layout."""
    bbox: WarsawBBoxConfig = field(default_factory=WarsawBBoxConfig)
    """Warsaw query bounding box."""
    model: ModelConfig = field(default_factory=ModelConfig)
    """Price model hyperparameters."""
    ingest_urls: IngestUrlsConfig = field(default_factory=IngestUrlsConfig)
    """Remote download endpoints."""
    ingest: IngestLimitsConfig = field(default_factory=IngestLimitsConfig)
    """Ingest download caps (default: unlimited)."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Package logging level for entrypoints."""
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    """Local OSRM / OTP endpoints."""
    commute_target_lon: float = 20.9830
    """Rondo Daszyńskiego longitude (commute destination)."""
    commute_target_lat: float = 52.2300
    """Rondo Daszyńskiego latitude (commute destination)."""

    def __post_init__(self) -> None:
        _require_str("name", self.name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the experiment config."""
        return cast(dict[str, Any], _serialize(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an experiment config from a mapping."""
        mapping = _extract_mapping(data, name="ExperimentConfig")
        mlflow_config = (
            MLflowTrackingConfig.from_dict(mapping["mlflow"])
            if mapping.get("mlflow")
            else MLflowTrackingConfig()
        )
        data_config = (
            DataPathsConfig.from_dict(mapping["data"])
            if mapping.get("data")
            else DataPathsConfig()
        )
        bbox_config = (
            WarsawBBoxConfig.from_dict(mapping["bbox"])
            if mapping.get("bbox")
            else WarsawBBoxConfig()
        )
        model_config = (
            ModelConfig.from_dict(mapping["model"])
            if mapping.get("model")
            else ModelConfig()
        )
        urls_config = (
            IngestUrlsConfig.from_dict(mapping["ingest_urls"])
            if mapping.get("ingest_urls")
            else IngestUrlsConfig()
        )
        ingest_config = (
            IngestLimitsConfig.from_dict(mapping["ingest"])
            if mapping.get("ingest")
            else IngestLimitsConfig()
        )
        logging_config = (
            LoggingConfig.from_dict(mapping["logging"])
            if mapping.get("logging")
            else LoggingConfig()
        )
        routing_config = (
            RoutingConfig.from_dict(mapping["routing"])
            if mapping.get("routing")
            else RoutingConfig()
        )
        name = str(mapping.get("name", "happy-housing"))
        return cls(
            name=name,
            mlflow=mlflow_config,
            data=data_config,
            bbox=bbox_config,
            model=model_config,
            ingest_urls=urls_config,
            ingest=ingest_config,
            logging=logging_config,
            routing=routing_config,
            commute_target_lon=float(mapping.get("commute_target_lon", 20.9830)),
            commute_target_lat=float(mapping.get("commute_target_lat", 52.2300)),
        )


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to a YAML file with ``experiment`` and ``mlflow`` sections.

    Returns:
        Parsed experiment configuration.

    Raises:
        ConfigValidationError: If the file structure or values are invalid.
        FileNotFoundError: If ``path`` does not exist.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), "Config root must be a mapping.")

    experiment_section = raw.get("experiment", {})
    _require(
        isinstance(experiment_section, Mapping),
        "experiment section must be a mapping.",
    )
    merged: dict[str, Any] = dict(experiment_section)
    for key in (
        "mlflow",
        "data",
        "bbox",
        "model",
        "ingest_urls",
        "ingest",
        "logging",
        "routing",
    ):
        if key in raw:
            merged[key] = raw[key]
    for key in ("commute_target_lon", "commute_target_lat"):
        if key in raw:
            merged[key] = raw[key]
    return ExperimentConfig.from_dict(merged)
