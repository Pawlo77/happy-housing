"""Compact LM Studio extraction of listing attrs from short description text."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmstudio as lms
import pandas as pd

from happy_housing.app.llm_metrics import PURPOSE_ATTR_EXTRACT, record_timed_llm_call
from happy_housing.app.lmstudio_models import get_lmstudio_llm

logger = logging.getLogger(__name__)

DEFAULT_EXTRACT_LM_MODEL: str = "gemma-3-4b-it"
"""Default lightweight LM Studio model for attribute extraction."""

EXTRACT_FIELDS: tuple[str, ...] = (
    "object_type",
    "floor_no",
    "is_attic",
    "has_garden",
    "has_garage",
    "has_outdoor_parking",
    "has_basement",
    "has_balcony",
    "ownership",
    "needs_renovation",
    "year_built",
    "year_renovated",
    "plumbing_replaced",
    "wiring_replaced",
)
"""CSV field order returned by the extractor (N = unknown)."""

_FLAG_FIELDS: frozenset[str] = frozenset({
    "is_attic",
    "has_garden",
    "has_garage",
    "has_outdoor_parking",
    "has_basement",
    "has_balcony",
    "needs_renovation",
    "plumbing_replaced",
    "wiring_replaced",
})

_OWNERSHIP_MAP: dict[str, str] = {
    "full": "full",
    "full_ownership": "full",
    "kw": "full",
    "usufruct": "usufruct",
    "wieczyste": "usufruct",
    "limited": "other",
    "limited_ownership": "other",
    "share": "other",
    "other": "other",
}

_DESC_ONLY_FIELDS: tuple[str, ...] = (
    "plumbing_replaced",
    "wiring_replaced",
    "year_renovated",
)
"""Fields that usually exist only after LM extract."""

_STRUCTURE_GAP_FLAGS: tuple[str, ...] = (
    "has_basement",
    "has_outdoor_parking",
    "is_attic",
)
"""Amenity flags that trigger LM only when missing (not when 0)."""


def _is_missing(value: object) -> bool:
    """True for None / NA / NaN / blank strings."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip().lower() in {
        "",
        "nan",
        "none",
        "null",
    }


def _as_int(value: object) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _as_ownership(value: object) -> str | None:
    if _is_missing(value):
        return None
    return _OWNERSHIP_MAP.get(str(value).strip().lower())


@dataclass(slots=True)
class ListingAttrExtract:
    """Sparse extracted listing attributes (None = unknown / N)."""

    object_type: str | None = None
    """``flat`` or ``house``."""
    floor_no: int | None = None
    """Floor number when known."""
    is_attic: int | None = None
    has_garden: int | None = None
    has_garage: int | None = None
    has_outdoor_parking: int | None = None
    has_basement: int | None = None
    has_balcony: int | None = None
    ownership: str | None = None
    """``full``, ``usufruct``, or ``other``."""
    needs_renovation: int | None = None
    year_built: int | None = None
    year_renovated: int | None = None
    plumbing_replaced: int | None = None
    """1 if plumbing renovated in the unit."""
    wiring_replaced: int | None = None
    """1 if electrical wiring renovated in the unit."""


def extract_system_prompt() -> str:
    """Ultra-short system prompt: one CSV line, N if unknown."""
    order = ",".join(EXTRACT_FIELDS)
    return (
        "Extract Warsaw housing listing facts. Reply ONE CSV line only, no words.\n"
        f"Order: {order}\n"
        "Rules: unknown→N; flags 0/1; years YYYY; "
        "object_type flat|house; ownership full|usufruct|other."
    )


def build_extract_user_prompt(description: str, *, max_chars: int = 900) -> str:
    """Truncate description to keep LM Studio prompts fast."""
    text = re.sub(r"\s+", " ", str(description or "")).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _parse_cell(raw: str, field: str) -> Any:
    token = raw.strip()
    if not token or token.upper() == "N":
        return None
    if field == "object_type":
        low = token.lower()
        if low in {"flat", "mieszkanie", "apartment"}:
            return "flat"
        if low in {"house", "dom", "home"}:
            return "house"
        return None
    if field == "ownership":
        return _OWNERSHIP_MAP.get(token.lower())
    if field in _FLAG_FIELDS:
        if token in {"1", "true", "yes", "y", "tak"}:
            return 1
        if token in {"0", "false", "no", "n", "nie"}:
            return 0
        return None
    try:
        number = int(float(token))
    except (TypeError, ValueError):
        return None
    if field in {"year_built", "year_renovated"} and not (1800 <= number <= 2100):
        return None
    return number


def parse_extract_line(text: str) -> ListingAttrExtract | None:
    """Parse one CSV extract line into ``ListingAttrExtract``."""
    blob = str(text or "").strip()
    if not blob:
        return None
    # Take first non-empty line; strip markdown fences if any.
    for line in blob.splitlines():
        line = line.strip().strip("`")
        if line and "," in line:
            blob = line
            break
    parts = [p.strip() for p in blob.split(",")]
    if len(parts) < len(EXTRACT_FIELDS):
        parts.extend(["N"] * (len(EXTRACT_FIELDS) - len(parts)))
    if len(parts) > len(EXTRACT_FIELDS):
        parts = parts[: len(EXTRACT_FIELDS)]
    values: dict[str, Any] = {}
    for field, cell in zip(EXTRACT_FIELDS, parts, strict=True):
        values[field] = _parse_cell(cell, field)
    # Reject all-unknown rows
    if all(v is None for v in values.values()):
        return None
    return ListingAttrExtract(**values)


def merge_structured_and_extract(
    structured: dict[str, Any],
    extract: ListingAttrExtract | None,
) -> dict[str, Any]:
    """Prefer Otodom structured fields; fill gaps from LM extract.

    Never clears an already-known structured / previously extracted value when
    ``extract`` is None or a field is ``N``.
    """
    estate = str(structured.get("estate") or "").upper()
    building_type = str(structured.get("building_type") or "").lower()
    object_type: str | None
    if estate in {"FLAT", "HOUSE"}:
        object_type = "flat" if estate == "FLAT" else "house"
    elif "house" in building_type or "dom" in building_type:
        object_type = "house"
    elif building_type and building_type not in {"nan", "none"}:
        object_type = "flat"
    else:
        existing_ot = structured.get("object_type")
        if not _is_missing(existing_ot):
            object_type = str(existing_ot).lower()
        else:
            object_type = extract.object_type if extract else None

    ownership = _as_ownership(structured.get("ownership"))
    if ownership is None:
        ownership = _as_ownership(structured.get("building_ownership"))
    if ownership is None and extract is not None:
        ownership = extract.ownership

    status = str(structured.get("construction_status") or "").lower()
    needs = _as_int(structured.get("needs_renovation"))
    if needs is None:
        if "renovation" in status or status == "to_renovation":
            needs = 1
        elif status in {"ready_to_use", "to_completion"}:
            needs = 0
        elif extract is not None:
            needs = extract.needs_renovation

    def _flag(col: str, extract_attr: str) -> int | None:
        current = _as_int(structured.get(col))
        if current is not None:
            return current
        if extract is None:
            return None
        value = getattr(extract, extract_attr)
        return int(value) if value is not None else None

    year_built_i = _as_int(structured.get("year_built"))
    if year_built_i is None and extract is not None:
        year_built_i = extract.year_built

    floor_i = _as_int(structured.get("floor_no"))
    if floor_i is None and extract is not None:
        floor_i = extract.floor_no

    year_renovated = _as_int(structured.get("year_renovated"))
    if year_renovated is None and extract is not None:
        year_renovated = extract.year_renovated

    plumbing = _as_int(structured.get("plumbing_replaced"))
    if plumbing is None and extract is not None:
        plumbing = extract.plumbing_replaced

    wiring = _as_int(structured.get("wiring_replaced"))
    if wiring is None and extract is not None:
        wiring = extract.wiring_replaced

    return {
        "object_type": object_type,
        "floor_no": floor_i,
        "is_attic": _flag("is_attic", "is_attic"),
        "has_garden": _flag("has_garden", "has_garden"),
        "has_garage": _flag("has_garage", "has_garage"),
        "has_outdoor_parking": _flag("has_outdoor_parking", "has_outdoor_parking"),
        "has_basement": _flag("has_basement", "has_basement"),
        "has_balcony": _flag("has_balcony", "has_balcony"),
        "ownership": ownership,
        "needs_renovation": needs,
        "year_built": year_built_i,
        "year_renovated": year_renovated,
        "plumbing_replaced": plumbing,
        "wiring_replaced": wiring,
    }


def row_needs_attr_llm(
    row: dict[str, Any] | pd.Series,
    columns: set[str],
    *,
    desc_fingerprint: str | None = None,
) -> bool:
    """Whether a row still has gaps that description extract can fill.

    Once ``attr_extract_fp`` matches the current description fingerprint, skip
    further LM calls even if plumbing/wiring stayed ``N`` (unknown).
    """
    desc = str(row.get("description") or row.get("short_description") or "").strip()
    if not desc:
        return False
    if desc_fingerprint and "attr_extract_fp" in columns:
        prior = row.get("attr_extract_fp")
        if not _is_missing(prior) and str(prior) == desc_fingerprint:
            return False

    ownership = _as_ownership(row.get("ownership"))
    if ownership is None:
        ownership = _as_ownership(row.get("building_ownership"))
    if ownership is None:
        return True
    for col in _STRUCTURE_GAP_FLAGS:
        if col not in columns or _is_missing(row.get(col)):
            return True
    if "object_type" not in columns or _is_missing(row.get("object_type")):
        estate = str(row.get("estate") or "").upper()
        if estate not in {"FLAT", "HOUSE"}:
            return True
    for col in _DESC_ONLY_FIELDS:
        if col not in columns or _is_missing(row.get(col)):
            return True
    return False


def description_fingerprint(text: str) -> str:
    """Stable hash for extract cache keys."""
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def extract_attrs_from_description(
    description: str,
    *,
    model: str = DEFAULT_EXTRACT_LM_MODEL,
    max_chars: int = 900,
    max_tokens: int = 48,
    metrics_path: Path | None = None,
) -> ListingAttrExtract | None:
    """Call LM Studio with a short prompt; parse CSV line or return None."""
    user = build_extract_user_prompt(description, max_chars=max_chars)
    if not user:
        return None
    system = extract_system_prompt()

    def _run() -> str:
        chat = lms.Chat(system)
        chat.add_user_message(user)
        llm = get_lmstudio_llm(model)
        result = llm.respond(
            chat,
            config={"temperature": 0.0, "maxTokens": int(max_tokens)},
        )
        return str(getattr(result, "content", result) or "")

    try:
        if metrics_path is not None:
            raw = record_timed_llm_call(
                metrics_path,
                purpose=PURPOSE_ATTR_EXTRACT,
                model=model,
                input_chars=len(system) + len(user),
                runner=_run,
            )
        else:
            raw = _run()
    except Exception as exc:
        logger.warning("Attr extract LM call failed: %s", exc)
        return None
    return parse_extract_line(str(raw))


def enrich_frame_listing_attrs(
    frame: pd.DataFrame,
    *,
    model: str = DEFAULT_EXTRACT_LM_MODEL,
    max_chars: int = 900,
    max_tokens: int = 48,
    use_llm: bool = True,
    metrics_path: Path | None = None,
    cache: dict[str, ListingAttrExtract | None] | None = None,
    max_llm_calls: int | None = None,
) -> pd.DataFrame:
    """Attach merged listing attrs; optional LLM only when gaps remain.

    Structured Otodom columns always win. LLM fills plumbing/wiring/year_renovated
    and missing flags from a truncated description.
    """
    out = frame.copy()
    cache = cache if cache is not None else {}
    llm_calls = 0
    columns = set(out.columns)
    if "attr_extract_fp" not in out.columns:
        out["attr_extract_fp"] = pd.NA
        columns.add("attr_extract_fp")
    rows: list[dict[str, Any]] = []
    fps: list[object] = []
    for _, row in out.iterrows():
        structured = row.to_dict()
        extract: ListingAttrExtract | None = None
        desc = str(
            structured.get("description") or structured.get("short_description") or ""
        ).strip()
        fp = description_fingerprint(desc[:max_chars]) if desc else ""
        needs_llm = bool(
            use_llm
            and desc
            and row_needs_attr_llm(structured, columns, desc_fingerprint=fp)
        )
        used_llm = False
        if needs_llm and (max_llm_calls is None or llm_calls < max_llm_calls):
            if fp in cache:
                extract = cache[fp]
            else:
                extract = extract_attrs_from_description(
                    desc,
                    model=model,
                    max_chars=max_chars,
                    max_tokens=max_tokens,
                    metrics_path=metrics_path,
                )
                cache[fp] = extract
                llm_calls += 1
            used_llm = True
        rows.append(merge_structured_and_extract(structured, extract))
        if used_llm:
            fps.append(fp)
        else:
            prior = structured.get("attr_extract_fp")
            fps.append(prior if not _is_missing(prior) else pd.NA)
    attrs = pd.DataFrame(rows, index=out.index)
    for col in attrs.columns:
        out[col] = attrs[col]
    out["attr_extract_fp"] = fps
    logger.info(
        "Listing attrs enriched · llm_calls=%s · rows=%s",
        llm_calls,
        len(out),
    )
    return out
