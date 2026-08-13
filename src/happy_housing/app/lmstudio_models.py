"""LM Studio model discovery, key resolution, and get-or-load helpers."""

import logging
import time
from typing import Any

import lmstudio as lms

logger = logging.getLogger(__name__)

_PATH_NOT_FOUND_MARKERS: tuple[str, ...] = (
    "pathNotFound",
    "Model not found",
    "noModelMatchingQuery",
)
"""Substrings that indicate a transient or wrong model-key failure."""

_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "gemma-3-4b-it": ("google/gemma-3-4b", "gemma-3-4b"),
    "google/gemma-3-4b": ("gemma-3-4b-it", "gemma-3-4b"),
    "gemma-3-4b": ("gemma-3-4b-it", "google/gemma-3-4b"),
}
"""Known alternate LMS keys for the same download family."""


class LmStudioModelNotFoundError(LookupError):
    """Requested model key is not among downloadable LMS LLM paths."""


def list_lmstudio_llm_keys() -> list[str]:
    """Return downloaded LLM model keys known to the LM Studio server."""
    keys: list[str] = []
    seen: set[str] = set()
    try:
        models = lms.list_downloaded_models()
    except Exception as exc:
        logger.warning("LM Studio list_downloaded_models failed: %s", exc)
        return []
    for model in models:
        key = str(getattr(model, "model_key", "") or "").strip()
        kind = str(getattr(model, "type", "") or "").lower()
        # Prefer LLM entries; skip pure embedding downloads.
        if key and key not in seen and "embedding" not in kind:
            # DownloadedEmbeddingModel.type is typically "embedding".
            type_name = type(model).__name__.lower()
            if "embedding" in type_name:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def resolve_lmstudio_model_key(
    requested: str,
    available: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Map a settings model id to a downloadable LMS key.

    Args:
        requested: Model id from app settings or caller.
        available: Optional explicit catalog (tests); else query LMS.

    Returns:
        Exact catalog key to pass to ``lms.llm``.

    Raises:
        LmStudioModelNotFoundError: When no exact, alias, or suffix match exists.
    """
    want = str(requested or "").strip()
    if not want:
        raise LmStudioModelNotFoundError("Empty LM Studio model id")
    catalog = list(available) if available is not None else list_lmstudio_llm_keys()
    if not catalog:
        # Server may be mid-index; keep requested and let get_or_load retry.
        return want

    by_lower = {k.lower(): k for k in catalog}
    if want in catalog:
        return want
    if want.lower() in by_lower:
        return by_lower[want.lower()]

    for alias in _MODEL_ALIASES.get(want.lower(), ()):
        if alias in catalog:
            return alias
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]

    # Suffix / path-tail match: "gemma-3-4b" → "google/gemma-3-4b"
    want_l = want.lower()
    suffix_hits = [
        key
        for key in catalog
        if key.lower() == want_l
        or key.lower().endswith("/" + want_l)
        or key.lower().endswith("-" + want_l)
        or want_l.endswith(key.lower().rsplit("/", 1)[-1])
    ]
    # Prefer shorter exact-ish keys; stable order.
    suffix_hits.sort(key=lambda k: (len(k), k))
    if len(suffix_hits) == 1:
        return suffix_hits[0]
    if len(suffix_hits) > 1:
        # Prefer alias family over unrelated suffix collisions.
        for alias in _MODEL_ALIASES.get(want_l, ()):
            for hit in suffix_hits:
                if hit.lower() == alias.lower() or hit.lower().endswith(
                    "/" + alias.lower()
                ):
                    return hit
        return suffix_hits[0]

    sample = ", ".join(catalog[:12])
    more = "" if len(catalog) <= 12 else f" (+{len(catalog) - 12} more)"
    raise LmStudioModelNotFoundError(
        f"LM Studio model not found: {want!r}. Available: {sample}{more}"
    )


def _is_path_not_found(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in _PATH_NOT_FOUND_MARKERS)


def get_lmstudio_llm(
    model: str, *, retries: int = 1, retry_delay_s: float = 0.75
) -> Any:
    """Resolve ``model`` against the LMS catalog and get-or-load an LLM handle.

    Retries once on ``pathNotFound`` — LMS sometimes returns a truncated catalog
    (e.g. ``totalModels: 1``) while the UI already shows the download.
    """
    last_exc: BaseException | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            key = resolve_lmstudio_model_key(model)
        except LmStudioModelNotFoundError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(retry_delay_s)
            continue
        try:
            return lms.llm(key)
        except Exception as exc:
            last_exc = exc
            if not _is_path_not_found(exc) or attempt + 1 >= attempts:
                raise
            logger.warning(
                "LM Studio pathNotFound for %r (attempt %s/%s); retrying",
                key,
                attempt + 1,
                attempts,
            )
            time.sleep(retry_delay_s)
    if last_exc is None:
        raise LmStudioModelNotFoundError(f"LM Studio model not found: {model!r}")
    raise last_exc
