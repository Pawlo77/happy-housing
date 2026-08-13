"""GUS Panorama dzielnic Warszawy — police crime rates by district."""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from happy_housing.ingest import download_file

logger = logging.getLogger(__name__)

CRIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "crime_total",
    "crime_criminal",
    "population",
    "crime_rate_per_1000",
    "crime_criminal_rate_per_1000",
    "crime_detectability_pct",
)
"""Columns attached to listings by :func:`attach_crime_features`."""

_CITY_LABEL_RE = re.compile(
    r"^(m\.?\s*st\.?\s*warszawa|warsaw)\b",
    re.IGNORECASE,
)
"""City-total rows to skip when reading district tables."""

_DEFAULT_PANORAMA_URL = (
    "https://warszawa.stat.gov.pl/download/gfx/warszawa/pl/"
    "defaultaktualnosci/760/5/26/1/panorama_dzielnic_warszawy_2024_aneks_excel.7z"
)
"""US Warszawa Panorama dzielnic 2024 Excel annex (.7z)."""


def normalize_crime_district(value: object) -> str:
    """Casefold district label for joins (keeps Polish diacritics)."""
    text = str(value or "").strip()
    # Panorama footnotes sometimes append " a" / " 1" to city totals.
    text = re.sub(r"\s+[a-z0-9]$", "", text, flags=re.IGNORECASE).strip()
    return text.casefold()


def _cell_str(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


_EMPTY_NUMERIC_MARKERS: frozenset[str] = frozenset({
    "",
    "-",
    ".",
    "\u2013",  # en dash used as missing marker in GUS tables
})
"""Tokens treated as missing numeric cells in Panorama Excel tables."""


def _cell_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if number != number else number
    text = str(value).strip().replace(",", ".")
    if text in _EMPTY_NUMERIC_MARKERS:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if number != number else number


def _is_district_row_label(label: str) -> bool:
    """True when ``label`` looks like a Warsaw dzielnica name."""
    if not label or "\n" in label:
        return False
    folded = label.casefold()
    if _CITY_LABEL_RE.match(folded):
        return False
    # GUS sheet titles use the Polish abbreviation for "table" (T a b l.).
    if folded.startswith("tab") and "." in folded[:6]:
        return False
    if folded.startswith(("wyszczególnienie", "specification")):
        return False
    if "źródło" in folded or folded.startswith("source"):
        return False
    if folded.startswith(("a ", "1 ", "2 ")):
        return False
    # Require at least one letter (skip pure punctuation).
    return any(ch.isalpha() for ch in label)


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")


def _parse_crime_counts(safe_xlsx: Path) -> dict[str, dict[str, float | str]]:
    """Parse Panorama sheet 16 — ascertained crimes by district."""
    frame = _read_sheet(safe_xlsx, "16")
    out: dict[str, dict[str, float | str]] = {}
    for row in frame.itertuples(index=False):
        label = _cell_str(row[0])
        if not _is_district_row_label(label):
            continue
        total = _cell_float(row[1] if len(row) > 1 else None)
        criminal = _cell_float(row[2] if len(row) > 2 else None)
        if total is None:
            continue
        key = normalize_crime_district(label)
        out[key] = {
            "district": label.strip(),
            "crime_total": total,
            "crime_criminal": float(criminal) if criminal is not None else float("nan"),
        }
    return out


def _parse_detectability(safe_xlsx: Path) -> dict[str, float]:
    """Parse Panorama sheet 17 — overall detectability % by district."""
    frame = _read_sheet(safe_xlsx, "17")
    out: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        label = _cell_str(row[0])
        if not _is_district_row_label(label):
            continue
        pct = _cell_float(row[1] if len(row) > 1 else None)
        if pct is None:
            continue
        out[normalize_crime_district(label)] = pct
    return out


def _parse_population(pop_xlsx: Path) -> dict[str, tuple[str, float]]:
    """Parse Panorama sheet 23 — year-end population by district."""
    frame = _read_sheet(pop_xlsx, "23")
    out: dict[str, tuple[str, float]] = {}
    for row in frame.itertuples(index=False):
        label = _cell_str(row[0])
        if not _is_district_row_label(label):
            continue
        # Prefer 31.12 (col index 2); fall back to 30.06 (col 1).
        pop_end = _cell_float(row[2] if len(row) > 2 else None)
        pop_mid = _cell_float(row[1] if len(row) > 1 else None)
        population = pop_end if pop_end is not None else pop_mid
        if population is None or population <= 0:
            continue
        key = normalize_crime_district(label)
        out[key] = (label.strip(), float(population))
    return out


def build_crime_by_district(
    rows: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Normalize raw district crime rows and compute per-1000 rates."""
    frame = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "district",
                "district_key",
                "year",
                *CRIME_FEATURE_COLUMNS,
            ]
        )
    out = frame.copy()
    out["district"] = out["district"].map(lambda v: str(v).strip())
    out["district_key"] = out["district"].map(normalize_crime_district)
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out["crime_total"] = pd.to_numeric(out["crime_total"], errors="coerce")
    out["crime_criminal"] = pd.to_numeric(out["crime_criminal"], errors="coerce")
    if "crime_detectability_pct" in out.columns:
        out["crime_detectability_pct"] = pd.to_numeric(
            out["crime_detectability_pct"], errors="coerce"
        )
    else:
        out["crime_detectability_pct"] = float("nan")
    pop = out["population"]
    out["crime_rate_per_1000"] = out["crime_total"] / pop * 1000.0
    out["crime_criminal_rate_per_1000"] = out["crime_criminal"] / pop * 1000.0
    if "year" not in out.columns:
        out["year"] = pd.NA
    cols = [
        "district",
        "district_key",
        "year",
        *CRIME_FEATURE_COLUMNS,
    ]
    return out.loc[:, cols].reset_index(drop=True)


def parse_panorama_crime_tables(
    safety_xlsx: Path,
    population_xlsx: Path,
    *,
    year: int = 2024,
) -> pd.DataFrame:
    """Parse Panorama Bezpieczeństwo + Ludność workbooks into district rates."""
    counts = _parse_crime_counts(safety_xlsx)
    detect = _parse_detectability(safety_xlsx)
    population = _parse_population(population_xlsx)
    rows: list[dict[str, Any]] = []
    for key, count_row in counts.items():
        pop_row = population.get(key)
        if pop_row is None:
            logger.warning("Crime district %s missing population; skip", key)
            continue
        display, pop = pop_row
        rows.append({
            "district": str(count_row.get("district") or display),
            "year": year,
            "population": pop,
            "crime_total": float(count_row["crime_total"]),
            "crime_criminal": float(count_row["crime_criminal"]),
            "crime_detectability_pct": detect.get(key, float("nan")),
        })
    return build_crime_by_district(rows)


def _find_workbook(extracted: Path, stem_prefix: str) -> Path:
    """Locate a Panorama workbook by filename prefix (handles NFC/NFD names)."""
    matches = [
        path
        for path in extracted.rglob("*.xlsx")
        if path.name.casefold().startswith(stem_prefix.casefold())
    ]
    if not matches:
        raise FileNotFoundError(
            f"No workbook starting with {stem_prefix!r} under {extracted}"
        )
    return sorted(matches)[0]


def extract_panorama_crime_archive(archive_7z: Path, extract_dir: Path) -> pd.DataFrame:
    """Extract a Panorama .7z and parse crime + population tables."""
    import py7zr

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_7z, mode="r") as archive:
        archive.extractall(path=extract_dir)
    safety = _find_workbook(extract_dir, "04_")
    population = _find_workbook(extract_dir, "05_")
    year_match = re.search(r"(20\d{2})", archive_7z.name)
    year = int(year_match.group(1)) if year_match else 2024
    return parse_panorama_crime_tables(safety, population, year=year)


def load_crime_by_district(
    output_parquet: Path,
    fixture_path: Path | None = None,
) -> pd.DataFrame:
    """Load crime parquet, or materialize from a JSON fixture when missing."""
    if output_parquet.exists() and output_parquet.stat().st_size > 0:
        return build_crime_by_district(pd.read_parquet(output_parquet))
    if fixture_path is None or not fixture_path.exists():
        empty = build_crime_by_district([])
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(output_parquet, index=False)
        return empty
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Crime fixture must be a JSON array.")
    frame = build_crime_by_district(payload)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)
    logger.info("Wrote crime fixture → %s (%s districts)", output_parquet, len(frame))
    return frame


def download_panorama_crime(
    url: str,
    raw_dir: Path,
    output_parquet: Path,
    *,
    fixture_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download Panorama .7z, parse district crime rates, write parquet.

    On network/parse failure, falls back to ``fixture_path`` when provided.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "panorama_dzielnic_excel.7z"
    extract_dir = raw_dir / "panorama_dzielnic_excel"
    try:
        if force and archive.exists():
            archive.unlink()
        download_file(
            url or _DEFAULT_PANORAMA_URL,
            archive,
            timeout_s=300.0,
            force=force,
        )
        frame = extract_panorama_crime_archive(archive, extract_dir)
        if frame.empty:
            raise ValueError("Parsed crime table is empty.")
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_parquet, index=False)
        logger.info(
            "Wrote crime by district → %s (%s districts)",
            output_parquet,
            len(frame),
        )
        return frame
    except Exception as exc:
        logger.warning("Panorama crime download/parse failed: %s", exc)
        return load_crime_by_district(output_parquet, fixture_path=fixture_path)


def attach_crime_features(
    listings: pd.DataFrame,
    crime_by_district: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join district crime rates onto listings via casefolded district."""
    out = listings.copy()
    for col in CRIME_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    if out.empty or "district" not in out.columns or crime_by_district.empty:
        return out
    crime = build_crime_by_district(crime_by_district)
    if crime.empty:
        return out
    right = crime.set_index("district_key", drop=False)
    keys = out["district"].map(normalize_crime_district)
    for col in CRIME_FEATURE_COLUMNS:
        out[col] = keys.map(right[col])
    return out
