"""Data ingest helpers for listings, GTFS, GIOŚ, and RCN."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET  # nosec B405 — local RCN GML only

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LISTING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "lat",
    "lon",
    "price_pln",
    "area_m2",
    "rooms",
    "year_built",
    "district",
    "listed_at",
)
"""Required columns in listing fixture / ingest tables."""


def load_listings_json(path: Path | str) -> pd.DataFrame:
    """Load listings from a JSON array file and validate required columns.

    Args:
        path: Path to a JSON file containing a list of listing objects.

    Returns:
        Normalized listings frame with ``price_per_m2``.

    Raises:
        ValueError: If required columns are missing or rows are empty.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Listings JSON must be a non-empty array.")
    frame = pd.DataFrame(payload)
    missing = [col for col in LISTING_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Listings missing columns: {missing}")
    frame = frame.copy()
    frame["price_per_m2"] = frame["price_pln"] / frame["area_m2"]
    return frame


def download_file(url: str, destination: Path, timeout_s: float = 120.0) -> Path:
    """Download a remote file to ``destination`` if missing.

    Args:
        url: HTTP(S) URL to fetch.
        destination: Local output path.
        timeout_s: Request timeout in seconds.

    Returns:
        Path to the downloaded file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Using cached download %s", destination)
        return destination
    logger.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def download_gtfs_stops(
    url: str,
    raw_dir: Path,
    output_parquet: Path,
) -> pd.DataFrame:
    """Download Warsaw GTFS and export ``stops.txt`` to parquet.

    Args:
        url: GTFS zip URL.
        raw_dir: Directory for the zip archive.
        output_parquet: Destination parquet path for stops.

    Returns:
        Stops dataframe.
    """
    zip_path = download_file(url, raw_dir / "gtfs_warsaw.zip", timeout_s=600.0)
    return export_gtfs_stops_from_zip(zip_path, output_parquet)


def export_gtfs_stops_from_zip(gtfs_zip: Path, output_parquet: Path) -> pd.DataFrame:
    """Export ``stops.txt`` from a local GTFS zip and build hourly counts."""
    with zipfile.ZipFile(gtfs_zip) as archive:
        stop_name = next(
            name for name in archive.namelist() if Path(name).name == "stops.txt"
        )
        with archive.open(stop_name) as handle:
            stops = pd.read_csv(handle)
    required = {"stop_id", "stop_lat", "stop_lon"}
    if not required.issubset(stops.columns):
        missing = required - set(stops.columns)
        raise ValueError(f"GTFS stops missing columns: {missing}")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    stops.to_parquet(output_parquet, index=False)
    logger.info("Wrote %s GTFS stops to %s", len(stops), output_parquet)
    hourly_path = output_parquet.with_name("gtfs_stop_hourly.parquet")
    build_gtfs_stop_hourly(gtfs_zip, hourly_path)
    return stops


def build_gtfs_stop_hourly(gtfs_zip: Path, output_parquet: Path) -> pd.DataFrame:
    """Aggregate GTFS ``stop_times`` departures by stop and hour-of-day.

    Args:
        gtfs_zip: Path to the Warsaw GTFS archive.
        output_parquet: Destination parquet with stop_id, hour, n_departures.

    Returns:
        Hourly departure counts per stop.
    """
    with zipfile.ZipFile(gtfs_zip) as archive:
        stop_times_name = next(
            name for name in archive.namelist() if Path(name).name == "stop_times.txt"
        )
        with archive.open(stop_times_name) as handle:
            times = pd.read_csv(handle, usecols=["stop_id", "departure_time"])
    hours = (
        times["departure_time"]
        .astype(str)
        .str.split(":", n=1)
        .str[0]
        .astype(int)
        .mod(24)
    )
    hourly = (
        times
        .assign(hour=hours)
        .groupby(["stop_id", "hour"], sort=False)
        .size()
        .reset_index(name="n_departures")
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_parquet, index=False)
    logger.info("Wrote %s GTFS stop-hour rows to %s", len(hourly), output_parquet)
    return hourly


def _parse_gios_station_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map one GIOŚ station object to a flat record (all useful API fields)."""
    station_id = row.get("Identyfikator stacji", row.get("id"))
    lat = row.get("WGS84 φ N", row.get("gegrLat", row.get("lat")))
    lon = row.get("WGS84 λ E", row.get("gegrLon", row.get("lon")))
    name = row.get("Nazwa stacji", row.get("stationName", row.get("name")))
    city_obj = row.get("Miejscowość") or row.get("city")
    if isinstance(city_obj, dict):
        city = str(city_obj.get("Nazwa miasta") or city_obj.get("name") or "")
    elif city_obj:
        city = str(city_obj)
    else:
        city = str(row.get("Nazwa miasta") or "")
    if station_id is None or lat is None or lon is None:
        return None
    return {
        "station_id": int(station_id),
        "station_code": str(row.get("Kod stacji") or row.get("stationCode") or ""),
        "lat": float(lat),
        "lon": float(lon),
        "name": str(name or station_id),
        "city": city,
        "street": str(row.get("Ulica") or row.get("addressStreet") or ""),
        "gmina": str(row.get("Gmina") or ""),
        "powiat": str(row.get("Powiat") or ""),
        "voivodeship": str(row.get("Województwo") or row.get("province") or ""),
        "city_id": row.get("Identyfikator miasta", row.get("cityId")),
    }


def fetch_gios_stations(
    url: str,
    output_parquet: Path,
    *,
    fixture_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch all GIOŚ v1 stations (paginated) into parquet.

    Args:
        url: GIOŚ stations endpoint.
        output_parquet: Destination parquet path.
        fixture_path: Optional offline JSON list used when the HTTP fetch fails.

    Returns:
        Stations dataframe with ``station_id``, ``lat``, ``lon``, ``name``.
    """
    try:
        frame = _fetch_gios_stations_http(url)
    except (OSError, requests.RequestException, ValueError, TypeError) as exc:
        if fixture_path is None or not fixture_path.exists():
            raise
        logger.warning("GIOŚ fetch failed (%s); using fixture %s", exc, fixture_path)
        frame = load_gios_stations_json(fixture_path)
    if frame.empty:
        raise ValueError("GIOŚ station response produced zero rows.")
    frame = frame.drop_duplicates(subset=["station_id"], keep="last")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)
    logger.info("Wrote %s GIOŚ stations to %s", len(frame), output_parquet)
    return frame


def load_gios_stations_json(path: Path) -> pd.DataFrame:
    """Load GIOŚ stations from a fixture JSON array of API-shaped rows."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("GIOŚ fixture must be a JSON array.")
    records: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        parsed = _parse_gios_station_row(row)
        if parsed is not None:
            records.append(parsed)
    return pd.DataFrame(records)


def _fetch_gios_stations_http(url: str) -> pd.DataFrame:
    """Paginated HTTP fetch of GIOŚ stations."""
    records: list[dict[str, Any]] = []
    page = 0
    page_size = 500
    while True:
        response = requests.get(
            url,
            params={"size": page_size, "page": page},
            timeout=60.0,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        rows = (
            payload.get("Lista stacji pomiarowych") or payload.get("stationList") or []
        )
        if not rows:
            break
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed = _parse_gios_station_row(row)
            if parsed is not None:
                records.append(parsed)
        total_pages = payload.get("totalPages") or payload.get("total_pages")
        if total_pages is not None and page + 1 >= int(total_pages):
            break
        if len(rows) < page_size:
            break
        page += 1
    return pd.DataFrame(records)


def sample_rcn_from_zip(
    url: str,
    raw_dir: Path,
    output_parquet: Path,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """Download RCN zip and sample WGS84-like points from GML/CSV members.

    Args:
        url: RCN zip URL.
        raw_dir: Directory for the zip archive.
        output_parquet: Destination parquet path.
        max_rows: Maximum sampled feature rows.

    Returns:
        Sampled transactions frame (best-effort column mapping).
    """
    zip_path = download_file(url, raw_dir / "rcn_warsaw.zip", timeout_s=1800.0)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        csv_members = [m for m in members if m.lower().endswith(".csv")]
        gml_members = [m for m in members if m.lower().endswith(".gml")]
        if csv_members:
            with archive.open(csv_members[0]) as handle:
                frame = pd.read_csv(handle, nrows=max_rows)
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output_parquet, index=False)
            return frame
        if not gml_members:
            raise ValueError("RCN zip contains neither CSV nor GML members.")
        with archive.open(gml_members[0]) as handle:
            # Trusted UM Warszawa RCN dump (local zip), not untrusted network XML.
            context = ET.iterparse(handle, events=("end",))  # nosec B314
            for _event, elem in context:
                tag = elem.tag.split("}")[-1].lower()
                if tag not in {
                    "featuremember",
                    "rcn",
                    "transakcja",
                    "nieruchomosc",
                    "member",
                }:
                    continue
                text_blob = " ".join(elem.itertext())
                coords = re.findall(r"(\d+\.\d+)\s+(\d+\.\d+)", text_blob)
                prices = re.findall(
                    r"(?:cena|price)[^\d]{0,20}(\d+[.,]?\d*)",
                    text_blob,
                    flags=re.IGNORECASE,
                )
                areas = re.findall(
                    r"(?:pow|area|powierzchnia)[^\d]{0,20}(\d+[.,]?\d*)",
                    text_blob,
                    flags=re.IGNORECASE,
                )
                if coords:
                    lon = float(coords[0][0])
                    lat = float(coords[0][1])
                    # Skip projected coordinates (PL-1992 values are >> 100).
                    if lon > 1000 or lat > 1000:
                        elem.clear()
                        continue
                    price = float(prices[0].replace(",", ".")) if prices else None
                    area = float(areas[0].replace(",", ".")) if areas else None
                    records.append({
                        "lon": lon,
                        "lat": lat,
                        "price_pln": price,
                        "area_m2": area,
                    })
                elem.clear()
                if len(records) >= max_rows:
                    break
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["lon", "lat", "price_pln", "area_m2"])
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)
    logger.info("Wrote %s RCN sample rows to %s", len(frame), output_parquet)
    return frame
