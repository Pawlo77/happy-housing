"""NBP FX rates: convert non-PLN listing prices to PLN."""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from happy_housing.ingest.http_client import resilient_get
from happy_housing.ingest.proxy_pool import bind_http_proxy

logger = logging.getLogger(__name__)

NBP_RATE_URL: str = "https://api.nbp.pl/api/exchangerates/rates/a/{code}/?format=json"
"""NBP table A mid-rate endpoint template (``{code}`` = ISO currency)."""

DEFAULT_FX_CACHE_NAME: str = "fx_nbp_cache.json"
"""Default FX cache filename under processed data."""


def fx_cache_path(processed: Path) -> Path:
    """Return the on-disk FX cache path under ``processed``."""
    return processed / DEFAULT_FX_CACHE_NAME


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


def load_fx_cache(path: Path) -> dict[str, Any]:
    """Load FX cache JSON; missing/invalid → empty dict."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("FX cache unreadable %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def save_fx_cache(path: Path, payload: dict[str, Any]) -> None:
    """Write FX cache JSON atomically-ish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clear_fx_cache(path: Path) -> None:
    """Delete the FX cache file if present."""
    if path.exists():
        path.unlink()
        logger.info("Cleared FX cache %s", path)


def fetch_nbp_mid_rate(
    currency: str,
    *,
    session: requests.Session | None = None,
    timeout_s: float = 15.0,
) -> tuple[float, str]:
    """Fetch NBP table A mid rate for ``currency`` → PLN.

    Returns:
        ``(mid_rate, effective_date)`` where rate multiplies foreign amount to PLN.

    Raises:
        ValueError: when the API response is missing a usable mid rate.
        requests.RequestException: on network/HTTP failure.
    """
    code = str(currency or "").strip().upper()
    if not code or code == "PLN":
        return 1.0, _utc_today()
    url = NBP_RATE_URL.format(code=code.lower())
    http = session or requests.Session()
    bind_http_proxy(http)
    response = resilient_get(
        http,
        url,
        timeout=timeout_s,
        retries=4,
        label=f"NBP FX {code}",
        headers={"Accept": "application/json"},
    )
    if response is None:
        raise requests.RequestException(f"NBP FX missing for {code}")
    payload = response.json()
    rates = payload.get("rates") if isinstance(payload, dict) else None
    if not isinstance(rates, list) or not rates:
        msg = f"NBP response missing rates for {code}"
        raise ValueError(msg)
    row = rates[0]
    mid = row.get("mid")
    effective = str(row.get("effectiveDate") or _utc_today())
    try:
        rate = float(mid)
    except (TypeError, ValueError) as exc:
        msg = f"NBP mid rate invalid for {code}: {mid!r}"
        raise ValueError(msg) from exc
    if rate <= 0:
        msg = f"NBP mid rate non-positive for {code}: {rate}"
        raise ValueError(msg)
    return rate, effective


def get_pln_rate(
    currency: str,
    cache_path: Path,
    *,
    session: requests.Session | None = None,
    force_refresh: bool = False,
) -> tuple[float, str]:
    """Return PLN mid rate for ``currency``, using a daily disk cache.

    Returns:
        ``(rate, effective_date)``. PLN always returns ``(1.0, today)``.
    """
    code = str(currency or "PLN").strip().upper() or "PLN"
    if code == "PLN":
        return 1.0, _utc_today()
    cache = load_fx_cache(cache_path)
    today = _utc_today()
    entry = cache.get(code)
    if (
        not force_refresh
        and isinstance(entry, dict)
        and entry.get("fetched_on") == today
        and entry.get("rate") is not None
    ):
        try:
            return float(entry["rate"]), str(entry.get("effective_date") or today)
        except (TypeError, ValueError):
            pass
    rate, effective = fetch_nbp_mid_rate(code, session=session)
    cache[code] = {
        "rate": rate,
        "effective_date": effective,
        "fetched_on": today,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    save_fx_cache(cache_path, cache)
    logger.info("NBP %s→PLN mid=%s (effective %s)", code, rate, effective)
    return rate, effective


def convert_amount_to_pln(
    amount: float,
    currency: str,
    cache_path: Path,
    *,
    session: requests.Session | None = None,
) -> tuple[float, float, str]:
    """Convert ``amount`` in ``currency`` to PLN.

    Returns:
        ``(pln_amount, rate_used, effective_date)``.
    """
    rate, effective = get_pln_rate(currency, cache_path, session=session)
    return float(amount) * rate, rate, effective


def currency_from_characteristics(characteristics: object) -> str | None:
    """Return ISO currency for the ``price`` characteristic when present."""
    if not isinstance(characteristics, list):
        return None
    for row in characteristics:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "").strip().lower()
        if key != "price":
            continue
        currency = row.get("currency")
        if isinstance(currency, str) and currency.strip():
            return currency.strip().upper()
    return None


def price_value_from_characteristics(characteristics: object) -> float | None:
    """Parse numeric ``price`` characteristic value when present."""
    if not isinstance(characteristics, list):
        return None
    for row in characteristics:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "").strip().lower()
        if key != "price":
            continue
        raw = row.get("value")
        try:
            return float(str(raw).replace(" ", "").replace(",", "."))
        except (TypeError, ValueError):
            return None
    return None


def normalize_price_fields(
    *,
    amount: float | None,
    currency: str | None,
    area_m2: float | None,
    price_per_m2: float | None,
    cache_path: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Build PLN-normalized price columns from a source amount + currency.

    When ``currency`` is non-PLN, recomputes ``price_per_m2`` from PLN / area
    (Otodom sometimes mixes EUR totals with PLN ppm).
    """
    code = (currency or "PLN").strip().upper() or "PLN"
    if amount is None:
        return {
            "price_original": None,
            "price_currency": code,
            "price_pln": None,
            "price_per_m2": price_per_m2,
            "fx_rate_to_pln": 1.0 if code == "PLN" else None,
            "fx_effective_date": None,
        }
    original = float(amount)
    if code == "PLN":
        pln = original
        rate = 1.0
        effective: str | None = None
        ppm = price_per_m2
        if ppm is None and area_m2 and area_m2 > 0:
            ppm = pln / float(area_m2)
    else:
        pln, rate, effective = convert_amount_to_pln(
            original, code, cache_path, session=session
        )
        ppm = None
        if area_m2 and area_m2 > 0:
            ppm = pln / float(area_m2)
    return {
        "price_original": original,
        "price_currency": code,
        "price_pln": pln,
        "price_per_m2": ppm,
        "fx_rate_to_pln": rate,
        "fx_effective_date": effective,
    }


def apply_fx_to_listings_frame(
    frame: pd.DataFrame,
    cache_path: Path,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Ensure ``price_pln`` is PLN; fill currency from characteristics if needed.

    Rows already PLN (or missing currency, treated as PLN) pass through.
    Non-PLN rows convert via NBP. Safe to call repeatedly.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    if "price_currency" not in out.columns:
        out["price_currency"] = None
    if "price_original" not in out.columns:
        out["price_original"] = None
    if "fx_rate_to_pln" not in out.columns:
        out["fx_rate_to_pln"] = None
    if "fx_effective_date" not in out.columns:
        out["fx_effective_date"] = None

    for idx, row in out.iterrows():
        currency = row.get("price_currency")
        if not isinstance(currency, str) or not currency.strip():
            chars_raw = row.get("characteristics_json")
            chars: object = None
            if isinstance(chars_raw, str) and chars_raw.strip():
                try:
                    chars = json.loads(chars_raw)
                except json.JSONDecodeError:
                    chars = None
            currency = currency_from_characteristics(chars) or "PLN"
            char_price = price_value_from_characteristics(chars)
            amount = (
                float(char_price)
                if char_price is not None
                else (
                    float(row["price_original"])
                    if pd.notna(row.get("price_original"))
                    else (
                        float(row["price_pln"])
                        if pd.notna(row.get("price_pln"))
                        else None
                    )
                )
            )
        else:
            currency = currency.strip().upper()
            if pd.notna(row.get("price_original")):
                amount = float(row["price_original"])
            elif pd.notna(row.get("price_pln")):
                # Legacy: EUR amount wrongly stored as price_pln.
                amount = float(row["price_pln"])
            else:
                amount = None

        area = float(row["area_m2"]) if pd.notna(row.get("area_m2")) else None
        ppm = float(row["price_per_m2"]) if pd.notna(row.get("price_per_m2")) else None
        normalized = normalize_price_fields(
            amount=amount,
            currency=currency,
            area_m2=area,
            price_per_m2=ppm,
            cache_path=cache_path,
            session=session,
        )
        for key, value in normalized.items():
            out.at[idx, key] = value
    return out


def format_original_price(amount: object, currency: object) -> str | None:
    """Human label for source currency amount (e.g. ``1 000 000 €``)."""
    if amount is None or (isinstance(amount, float) and amount != amount):
        return None
    try:
        number = float(amount)  # type: ignore[arg-type]  # pandas/object scalar
    except (TypeError, ValueError):
        return None
    code = str(currency or "PLN").strip().upper() or "PLN"
    if code == "PLN":
        return None
    symbols = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF"}
    symbol = symbols.get(code, code)
    formatted = f"{number:,.0f}".replace(",", " ")
    if symbol in {"€", "$", "£"}:
        return f"{formatted} {symbol}"
    return f"{formatted} {symbol}"


def fx_cache_summary(cache_path: Path) -> str:
    """Short Admin status line for the FX cache."""
    cache = load_fx_cache(cache_path)
    if not cache:
        return "No FX rates cached yet."
    parts: list[str] = []
    for code, entry in sorted(cache.items()):
        if not isinstance(entry, dict):
            continue
        rate = entry.get("rate")
        effective = entry.get("effective_date") or entry.get("fetched_on")
        parts.append(f"{code}={rate} (as of {effective})")
    return "; ".join(parts) if parts else "No FX rates cached yet."


def cache_is_fresh_for(
    cache_path: Path, currency: str, on_day: date | None = None
) -> bool:
    """True when ``currency`` was fetched on ``on_day`` (UTC today by default)."""
    code = str(currency or "").strip().upper()
    if not code or code == "PLN":
        return True
    day = (on_day or datetime.now(UTC).date()).isoformat()
    entry = load_fx_cache(cache_path).get(code)
    return isinstance(entry, dict) and entry.get("fetched_on") == day
