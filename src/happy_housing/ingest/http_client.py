"""Shared resilient HTTP GET with optional free-proxy rotation on blocks."""

from __future__ import annotations

import logging
import time
from typing import Final

import requests

from happy_housing.ingest.proxy_pool import (
    active_proxy_pool,
    bind_http_proxy,
    block_statuses,
    http_proxies_enabled,
    rotate_http_proxy,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({
    408,
    425,
    429,
    500,
    502,
    503,
    504,
})
"""Transient HTTP statuses always worth retrying."""


def is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for timeouts, connection errors, and retryable HTTP codes."""
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = int(exc.response.status_code)
        return code in _RETRYABLE_STATUS or code in block_statuses()
    return isinstance(exc, requests.RequestException)


def retry_sleep_s(attempt: int, *, status_code: int | None = None) -> float:
    """Backoff delay; block statuses get a longer cool-down."""
    base = min(2.0 * attempt, 20.0)
    if status_code in block_statuses():
        return min(base * 2.5, 45.0)
    return base


def resilient_get(
    session: requests.Session,
    url: str,
    *,
    timeout: float | tuple[float, float] = 30.0,
    retries: int = 5,
    label: str,
    params: dict[str, object] | None = None,
    stream: bool = False,
    headers: dict[str, str] | None = None,
) -> requests.Response | None:
    """GET with retries; rotate free proxies on 403/407/429 when enabled.

    First attempt keeps the session's current proxies (often direct). A block
    status or proxy transport failure rotates to a new free proxy and retries.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        status_code: int | None = None
        try:
            get_kwargs: dict[str, object] = {"timeout": timeout}
            if params is not None:
                get_kwargs["params"] = params
            if stream:
                get_kwargs["stream"] = True
            if headers is not None:
                get_kwargs["headers"] = headers
            response = session.get(url, **get_kwargs)
            status_code = int(response.status_code)
            response.raise_for_status()
            return response
        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            last_exc = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status_code = int(exc.response.status_code)
            blocked = status_code in block_statuses()
            proxy_transport = isinstance(
                exc,
                (
                    requests.exceptions.ProxyError,
                    requests.ConnectionError,
                    requests.Timeout,
                ),
            ) and bool(getattr(session, "proxies", None))
            if http_proxies_enabled() and (blocked or proxy_transport):
                reason = (
                    f"HTTP {status_code}"
                    if blocked and status_code is not None
                    else type(exc).__name__
                )
                rotated = rotate_http_proxy(reason=f"{label}: {reason}")
                if rotated is not None or active_proxy_pool() is not None:
                    bind_http_proxy(session)
            retryable = (
                is_retryable_http_error(exc)
                or blocked
                or status_code in _RETRYABLE_STATUS
            )
            if not retryable or attempt >= retries:
                logger.warning(
                    "%s failed permanently (attempt %s/%s): %s",
                    label,
                    attempt,
                    retries,
                    exc,
                )
                break
            sleep_for = retry_sleep_s(attempt, status_code=status_code)
            logger.warning(
                "%s failed (attempt %s/%s): %s; retry in %.1fs",
                label,
                attempt,
                retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    if isinstance(last_exc, requests.HTTPError):
        raise last_exc
    if last_exc is not None:
        logger.debug("Last error for %s: %s", label, last_exc)
        raise last_exc
    return None


def resilient_requests_get(
    url: str,
    *,
    timeout: float | tuple[float, float] = 30.0,
    retries: int = 5,
    label: str,
    params: dict[str, object] | None = None,
    stream: bool = False,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """One-shot GET on a fresh session with free-proxy rotation on blocks."""
    session = requests.Session()
    if headers:
        session.headers.update(headers)
    bind_http_proxy(session)
    response = resilient_get(
        session,
        url,
        timeout=timeout,
        retries=retries,
        label=label,
        params=params,
        stream=stream,
    )
    if response is None:
        raise requests.RequestException(f"{label} failed with no response")
    return response
