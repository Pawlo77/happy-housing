"""LM Studio natural-language → structured Explore preferences."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import lmstudio as lms

from happy_housing.app.app_settings import DEFAULT_LM_MODEL
from happy_housing.app.llm_metrics import PURPOSE_PREF_PARSE, record_timed_llm_call
from happy_housing.app.preferences import UserPreferences

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
"""Fenced JSON object matcher."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
"""First brace-to-brace JSON object (greedy within one blob)."""


def preferences_system_prompt() -> str:
    """System prompt asking for UserPreferences JSON only."""
    return (
        "You translate Warsaw flat-buyer preference notes into a JSON object.\n"
        "Return ONLY valid JSON (no markdown commentary outside a json fence).\n"
        "Allowed keys (omit unset fields):\n"
        "enabled (bool), districts (string[]), sources (string[]),\n"
        "oou one of: (any), prefer_outside, outside_only, inside_ok,\n"
        "min_price_pln, max_price_pln, min_price_per_m2, max_price_per_m2,\n"
        "min_area_m2, max_area_m2, min_rooms, max_rooms, min_year_built,\n"
        "max_commute_min, max_dist_transit_m, max_dist_hospital_m,\n"
        "max_dist_fire_m, max_dist_police_m, max_dist_air_m, max_dist_gunb_m,\n"
        "max_delta_price_per_m2, require_underpriced (bool), nl_text (string).\n"
        "Unknown keys are ignored. Prefer Polish district names when given."
    )


def extract_preferences_json(text: str) -> UserPreferences:
    """Parse model text into ``UserPreferences``; reject unknown-only / empty JSON.

    Raises:
        ValueError: When no JSON object can be parsed.
    """
    blob = text.strip()
    if not blob:
        raise ValueError("Empty model response; expected preferences JSON")
    match = _JSON_FENCE.search(blob)
    candidate = match.group(1) if match else None
    if candidate is None:
        obj_match = _JSON_OBJECT.search(blob)
        candidate = obj_match.group(0) if obj_match else None
    if not candidate:
        raise ValueError("No JSON object found in model response")
    try:
        payload: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid preferences JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Preferences JSON must be an object")
    return UserPreferences.from_dict(payload)


def merge_preferences(
    base: UserPreferences,
    overlay: UserPreferences,
    *,
    nl_text: str | None = None,
) -> UserPreferences:
    """Overlay non-default overlay fields onto ``base``; set ``nl_text`` if given."""
    merged = UserPreferences.from_dict(base.to_dict())
    overlay_dict = overlay.to_dict()
    blank = UserPreferences().to_dict()
    for key, value in overlay_dict.items():
        if key == "nl_text":
            continue
        if value == blank.get(key):
            continue
        setattr(merged, key, value)
    if nl_text is not None:
        merged.nl_text = nl_text
    return merged


def translate_preferences_nl(
    text: str,
    *,
    model: str = DEFAULT_LM_MODEL,
    metrics_path: Path | None = None,
) -> UserPreferences:
    """Ask LM Studio to turn free text into structured preferences.

    Raises:
        ValueError: When the model returns unparsable JSON.
        Exception: Propagated from the LM Studio client on hard failures.
    """
    note = text.strip()
    if not note:
        raise ValueError("Preference note is empty")
    system = preferences_system_prompt()
    user = f"Preference note:\n{note}"

    def _run() -> str:
        chat = lms.Chat(system)
        chat.add_user_message(user)
        llm = lms.llm(model)
        result = llm.respond(
            chat,
            config={"temperature": 0.2, "maxTokens": 800},
        )
        return str(getattr(result, "content", result) or "")

    if metrics_path is not None:
        raw = record_timed_llm_call(
            metrics_path,
            purpose=PURPOSE_PREF_PARSE,
            model=model,
            input_chars=len(system) + len(user),
            runner=_run,
        )
    else:
        raw = _run()
    prefs = extract_preferences_json(raw)
    prefs.nl_text = note
    if not prefs.enabled:
        prefs.enabled = True
    logger.info("Translated NL preferences (districts=%s)", prefs.districts)
    return prefs
