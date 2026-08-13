"""GIOŚ API v1 air-quality index and pollutant readings for stations."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from happy_housing.ingest.http_client import resilient_get
from happy_housing.ingest.proxy_pool import bind_http_proxy

logger = logging.getLogger(__name__)

GIOS_AQ_INDEX_URL = (
    "https://api.gios.gov.pl/pjp-api/v1/rest/aqindex/getIndex/{station_id}"
)
"""GIOŚ Polish AQI endpoint template."""

GIOS_SENSORS_URL = (
    "https://api.gios.gov.pl/pjp-api/v1/rest/station/sensors/{station_id}"
)
"""GIOŚ sensors-at-station endpoint template."""

GIOS_DATA_URL = "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/{sensor_id}"
"""GIOŚ sensor time-series endpoint template."""

AIR_QUALITY_COLUMNS: tuple[str, ...] = (
    "aq_index",
    "aq_pm10_index",
    "aq_pm25_index",
    "aq_no2_index",
    "pm10_ug_m3",
    "pm25_ug_m3",
    "no2_ug_m3",
)
"""Station-level AQ columns merged onto the GIOŚ stations table."""

_POLLUTANT_CODES: tuple[str, ...] = ("PM10", "PM2.5", "NO2")
"""Sensor codes fetched for concentration features."""

_CODE_TO_UG_COL: dict[str, str] = {
    "PM10": "pm10_ug_m3",
    "PM2.5": "pm25_ug_m3",
    "NO2": "no2_ug_m3",
}
"""Map GIOŚ sensor code → station column."""


def _as_aq_number(value: Any) -> float:
    """Coerce index values; ``-1`` / missing → NaN."""
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if number < 0:
        return float("nan")
    return number


def parse_gios_aq_index(payload: dict[str, Any]) -> dict[str, float]:
    """Parse ``aqindex/getIndex`` JSON-LD into numeric index fields."""
    block = payload.get("AqIndex")
    if not isinstance(block, dict):
        block = payload
    return {
        "aq_index": _as_aq_number(
            block.get("Wartość indeksu", block.get("stIndexLevel"))
        ),
        "aq_pm10_index": _as_aq_number(block.get("Wartość indeksu dla wskaźnika PM10")),
        "aq_pm25_index": _as_aq_number(
            block.get("Wartość indeksu dla wskaźnika PM2.5")
        ),
        "aq_no2_index": _as_aq_number(block.get("Wartość indeksu dla wskaźnika NO2")),
    }


def latest_sensor_value(rows: list[dict[str, Any]] | list[Any]) -> float | None:
    """Return first non-null ``Wartość`` in GIOŚ chronological sensor rows."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("Wartość", row.get("value"))
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def load_gios_air_quality_fixture(path: Path) -> pd.DataFrame:
    """Load offline station AQ rows keyed by ``station_id``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("GIOŚ air-quality fixture must be a JSON array.")
    frame = pd.DataFrame(payload)
    if frame.empty or "station_id" not in frame.columns:
        raise ValueError("GIOŚ air-quality fixture needs station_id rows.")
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce")
    frame = frame.dropna(subset=["station_id"])
    frame["station_id"] = frame["station_id"].astype(int)
    for col in AIR_QUALITY_COLUMNS:
        if col not in frame.columns:
            frame[col] = float("nan")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[["station_id", *AIR_QUALITY_COLUMNS]].drop_duplicates(
        subset=["station_id"], keep="last"
    )


def _fetch_station_air_quality(
    station_id: int,
    *,
    aq_index_url: str = GIOS_AQ_INDEX_URL,
    sensors_url: str = GIOS_SENSORS_URL,
    data_url: str = GIOS_DATA_URL,
    timeout_s: float = 30.0,
) -> dict[str, float]:
    """HTTP fetch of AQI + latest PM10/PM2.5/NO2 for one station."""
    out: dict[str, float] = {col: float("nan") for col in AIR_QUALITY_COLUMNS}
    session = requests.Session()
    bind_http_proxy(session)
    idx_resp = resilient_get(
        session,
        aq_index_url.format(station_id=station_id),
        timeout=timeout_s,
        retries=4,
        label=f"GIOŚ AQ index {station_id}",
    )
    if idx_resp is None:
        raise requests.RequestException(f"GIOŚ AQ index missing for {station_id}")
    out.update(parse_gios_aq_index(idx_resp.json()))

    sens_resp = resilient_get(
        session,
        sensors_url.format(station_id=station_id),
        timeout=timeout_s,
        retries=4,
        label=f"GIOŚ sensors {station_id}",
    )
    if sens_resp is None:
        raise requests.RequestException(f"GIOŚ sensors missing for {station_id}")
    sens_payload = sens_resp.json()
    sensors = (
        sens_payload.get("Lista stanowisk pomiarowych dla podanej stacji")
        or sens_payload.get("listaStanowiskPomiarowych")
        or []
    )
    seen_codes: set[str] = set()
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        code = str(sensor.get("Wskaźnik - kod") or sensor.get("paramCode") or "")
        if code not in _CODE_TO_UG_COL or code in seen_codes:
            continue
        sensor_id = sensor.get("Identyfikator stanowiska", sensor.get("id"))
        if sensor_id is None:
            continue
        data_resp = resilient_get(
            session,
            data_url.format(sensor_id=int(sensor_id)),
            timeout=timeout_s,
            retries=4,
            label=f"GIOŚ data {sensor_id}",
        )
        if data_resp is None:
            continue
        data_payload = data_resp.json()
        rows = (
            data_payload.get("Lista danych pomiarowych")
            or data_payload.get("values")
            or []
        )
        value = latest_sensor_value(rows if isinstance(rows, list) else [])
        if value is not None:
            out[_CODE_TO_UG_COL[code]] = float(value)
            seen_codes.add(code)
        if seen_codes.issuperset(_POLLUTANT_CODES):
            break
    return out


def _empty_aq_frame(station_ids: list[int]) -> pd.DataFrame:
    """NaN AQ columns for the given station ids."""
    frame = pd.DataFrame({"station_id": station_ids})
    for col in AIR_QUALITY_COLUMNS:
        frame[col] = float("nan")
    return frame


def enrich_stations_with_air_quality(
    stations: pd.DataFrame,
    *,
    fixture_path: Path | None = None,
    max_workers: int = 8,
    city_filter: str | None = "Warszawa",
) -> pd.DataFrame:
    """Attach AQI + pollutant µg/m³ columns to GIOŚ stations.

    Fetches live v1 endpoints for stations matching ``city_filter`` (None = all).
    On total HTTP failure, merges ``fixture_path`` when provided.
    """
    if stations.empty or "station_id" not in stations.columns:
        out = stations.copy()
        for col in AIR_QUALITY_COLUMNS:
            out[col] = float("nan")
        return out

    out = stations.copy()
    out["station_id"] = pd.to_numeric(out["station_id"], errors="coerce")
    fetch_mask = out["station_id"].notna()
    if city_filter and "city" in out.columns:
        fetch_mask &= (
            out["city"].astype(str).str.contains(city_filter, case=False, na=False)
        )
    fetch_ids = out.loc[fetch_mask, "station_id"].astype(int).drop_duplicates().tolist()
    aq_rows: list[dict[str, Any]] = []
    http_ok = False
    if fetch_ids:
        try:
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
                futures = {
                    pool.submit(_fetch_station_air_quality, sid): sid
                    for sid in fetch_ids
                }
                for fut in as_completed(futures):
                    sid = futures[fut]
                    try:
                        values = fut.result()
                        http_ok = True
                    except (
                        OSError,
                        requests.RequestException,
                        ValueError,
                        TypeError,
                    ) as exc:
                        logger.debug("GIOŚ AQ fetch failed for %s: %s", sid, exc)
                        values = {col: float("nan") for col in AIR_QUALITY_COLUMNS}
                    aq_rows.append({"station_id": sid, **values})
        except (OSError, requests.RequestException, ValueError, TypeError) as exc:
            logger.warning("GIOŚ AQ enrich aborted: %s", exc)
            aq_rows = []
            http_ok = False

    if aq_rows:
        aq = pd.DataFrame(aq_rows)
    elif fixture_path is not None and fixture_path.exists():
        logger.warning("GIOŚ AQ HTTP unavailable; using fixture %s", fixture_path)
        aq = load_gios_air_quality_fixture(fixture_path)
    else:
        aq = _empty_aq_frame(fetch_ids)

    if not http_ok and fixture_path is not None and fixture_path.exists():
        fixture = load_gios_air_quality_fixture(fixture_path)
        if aq.empty:
            aq = fixture
        else:
            aq = (
                aq
                .set_index("station_id")
                .combine_first(fixture.set_index("station_id"))
                .reset_index()
            )

    for col in AIR_QUALITY_COLUMNS:
        if col in out.columns:
            out = out.drop(columns=[col])
    if aq.empty:
        for col in AIR_QUALITY_COLUMNS:
            out[col] = float("nan")
        return out
    return out.merge(aq, on="station_id", how="left")
