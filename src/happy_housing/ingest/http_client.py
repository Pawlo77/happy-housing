"""Shared resilient HTTP GET with optional free-proxy rotation on blocks."""

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

_PROXY_ROTATION_SLEEP_S: Final[float] = 0.25
"""Short pause between dead-proxy rotations (avoid hammering)."""


def is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for timeouts, connection errors, and retryable HTTP codes."""
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = int(exc.response.status_code)
        return code in _RETRYABLE_STATUS or code in block_statuses()
    return isinstance(exc, requests.RequestException)


def retry_sleep_s(attempt: int, status_code: int | None = None) -> float:
    """Backoff delay; block statuses get a longer cool-down."""
    base = min(2.0 * attempt, 20.0)
    if status_code in block_statuses():
        return min(base * 2.5, 45.0)
    return base


def _is_proxy_transport_error(
    exc: BaseException,
    session: requests.Session,
) -> bool:
    """True when the failure is the proxy hop itself (not origin content)."""
    if not getattr(session, "proxies", None):
        return False
    if isinstance(exc, requests.exceptions.ProxyError):
        return True
    # Connection/timeout while a proxy is bound usually means a dead proxy.
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _default_max_proxy_rotations(retries: int) -> int:
    """Allow many dead free proxies before giving up on a URL."""
    return max(40, int(retries) * 12)


def resilient_get(
    session: requests.Session,
    url: str,
    *,
    timeout: float | tuple[float, float] = 30.0,
    retries: int = 5,
    max_proxy_rotations: int | None = None,
    label: str,
    params: dict[str, object] | None = None,
    stream: bool = False,
    headers: dict[str, str] | None = None,
) -> requests.Response | None:
    """GET with retries; rotate free proxies on 403/407/429 when enabled.

    First attempt keeps the session's current proxies (often direct). A block
    status burns a content retry (with backoff). A **proxy transport** failure
    rotates to the next free proxy without burning the content budget, so a
    run can walk many dead free proxies until one works.
    """
    last_exc: Exception | None = None
    content_attempt = 0
    proxy_rotations = 0
    proxy_budget = (
        max_proxy_rotations
        if max_proxy_rotations is not None
        else _default_max_proxy_rotations(retries)
    )

    while content_attempt < retries:
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
            proxy_transport = _is_proxy_transport_error(exc, session)

            if http_proxies_enabled() and proxy_transport:
                proxy_rotations += 1
                if proxy_rotations <= proxy_budget:
                    reason = type(exc).__name__
                    rotated = rotate_http_proxy(reason=f"{label}: {reason}")
                    if rotated is not None or active_proxy_pool() is not None:
                        bind_http_proxy(session)
                    if rotated is not None:
                        logger.info(
                            "%s dead proxy → rotate (%s/%s): %s",
                            label,
                            proxy_rotations,
                            proxy_budget,
                            exc,
                        )
                        time.sleep(_PROXY_ROTATION_SLEEP_S)
                        continue
                    # Pool empty: fall through and burn a content attempt.
                    logger.warning(
                        "%s proxy pool empty after %s rotations: %s",
                        label,
                        proxy_rotations,
                        exc,
                    )
                else:
                    logger.warning(
                        "%s exhausted %s proxy rotations: %s",
                        label,
                        proxy_budget,
                        exc,
                    )

            if http_proxies_enabled() and blocked:
                reason = (
                    f"HTTP {status_code}"
                    if status_code is not None
                    else type(exc).__name__
                )
                rotated = rotate_http_proxy(reason=f"{label}: {reason}")
                if rotated is not None or active_proxy_pool() is not None:
                    bind_http_proxy(session)

            retryable = (
                is_retryable_http_error(exc)
                or blocked
                or status_code in _RETRYABLE_STATUS
                or proxy_transport
            )
            content_attempt += 1
            if not retryable or content_attempt >= retries:
                logger.warning(
                    "%s failed permanently (attempt %s/%s, proxy_rotations=%s): %s",
                    label,
                    content_attempt,
                    retries,
                    proxy_rotations,
                    exc,
                )
                break
            sleep_for = retry_sleep_s(content_attempt, status_code=status_code)
            logger.warning(
                "%s failed (attempt %s/%s, proxy_rotations=%s): %s; retry in %.1fs",
                label,
                content_attempt,
                retries,
                proxy_rotations,
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
    max_proxy_rotations: int | None = None,
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
        max_proxy_rotations=max_proxy_rotations,
        label=label,
        params=params,
        stream=stream,
    )
    if response is None:
        raise requests.RequestException(f"{label} failed with no response")
    return response
