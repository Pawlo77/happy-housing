"""Process-wide free HTTP proxy pool for ingest downloads.

Uses the ``free-proxy`` library plus a ProxyScrape free list fallback.
Proxies are only engaged after a blocking response (403/407/429) or when
the active proxy itself fails; first attempt stays direct.
"""

import logging
import threading
from collections import deque
from collections.abc import Callable, Sequence
from typing import Final

import requests

logger = logging.getLogger(__name__)

ProxyFetcher = Callable[[], Sequence[str]]
"""Callable that returns candidate proxy URLs (``http://host:port``)."""

_BLOCK_STATUSES: Final[frozenset[int]] = frozenset({403, 407, 429})
"""Statuses that trigger free-proxy rotation."""

_PROXYSCRAPE_URL: Final[str] = (
    "https://api.proxyscrape.com/v2/"
    "?request=displayproxies&protocol=http&timeout=5000"
    "&country=all&ssl=all&anonymity=elite"
)
"""Public free HTTP proxy list (text, one host:port per line)."""

_lock = threading.RLock()
_enabled: bool = True
_pool: "FreeProxyPool | None" = None


def block_statuses() -> frozenset[int]:
    """HTTP statuses that should rotate the free-proxy pool."""
    return _BLOCK_STATUSES


def _normalize_proxy(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if "://" not in text:
        text = f"http://{text}"
    if not text.startswith(("http://", "https://", "socks4://", "socks5://")):
        return None
    return text


def fetch_proxyscrape_proxies(*, limit: int = 40) -> list[str]:
    """Pull a fresh free HTTP proxy list from ProxyScrape (direct, no proxy)."""
    try:
        response = requests.get(_PROXYSCRAPE_URL, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("ProxyScrape free list failed: %s", exc)
        return []
    out: list[str] = []
    for line in response.text.splitlines():
        normalized = _normalize_proxy(line)
        if normalized is None:
            continue
        out.append(normalized)
        if len(out) >= limit:
            break
    logger.info("ProxyScrape returned %s free proxies", len(out))
    return out


def fetch_free_proxy_lib_proxies(*, limit: int = 8) -> list[str]:
    """Pull working proxies via the ``free-proxy`` (``fp``) library."""
    try:
        from fp.fp import FreeProxy  # type: ignore[import-not-found]  # no stubs
    except ImportError:
        logger.debug("free-proxy package not installed; skip fp fetcher")
        return []
    out: list[str] = []
    for _ in range(max(1, limit)):
        try:
            raw = FreeProxy(timeout=1.5, rand=True).get()
        except Exception as exc:
            logger.debug("free-proxy fetch failed: %s", exc)
            break
        normalized = _normalize_proxy(str(raw) if raw is not None else "")
        if normalized is None:
            continue
        if normalized not in out:
            out.append(normalized)
    if out:
        logger.info("free-proxy library returned %s proxies", len(out))
    return out


def _default_fetchers() -> list[ProxyFetcher]:
    """Built-in free-proxy sources (library first, list API second)."""
    return [
        lambda: fetch_free_proxy_lib_proxies(limit=8),
        lambda: fetch_proxyscrape_proxies(limit=40),
    ]


class FreeProxyPool:
    """Thread-safe deque of free proxies with bad-proxy exclusion."""

    def __init__(
        self,
        *,
        fetchers: Sequence[ProxyFetcher] | None = None,
    ) -> None:
        self._fetchers = list(fetchers) if fetchers is not None else _default_fetchers()
        self._lock = threading.RLock()
        self._queue: deque[str] = deque()
        self._current: str | None = None
        self._bad: set[str] = set()

    @property
    def current(self) -> str | None:
        """Active proxy URL, if any."""
        with self._lock:
            return self._current

    @property
    def bad_proxies(self) -> frozenset[str]:
        """Proxies dropped after blocks / transport failures."""
        with self._lock:
            return frozenset(self._bad)

    def proxies_mapping(self) -> dict[str, str]:
        """``requests`` proxies dict for the current endpoint (maybe empty)."""
        with self._lock:
            if not self._current:
                return {}
            return {"http": self._current, "https": self._current}

    def bind(self, session: requests.Session) -> None:
        """Apply current proxy to ``session`` (clears proxies when none)."""
        mapping = self.proxies_mapping()
        session.proxies.clear()
        if mapping:
            session.proxies.update(mapping)

    def ensure(self) -> str | None:
        """Return current proxy, acquiring one when the pool has none."""
        with self._lock:
            if self._current is not None:
                return self._current
            return self._acquire_unlocked()

    def rotate(self, *, reason: str = "") -> str | None:
        """Drop the current proxy and pick the next free one."""
        with self._lock:
            if self._current is not None:
                self._bad.add(self._current)
                logger.info(
                    "Rotating free proxy away from %s (%s)",
                    self._current,
                    reason or "blocked",
                )
                self._current = None
            return self._acquire_unlocked()

    def _acquire_unlocked(self) -> str | None:
        while True:
            while self._queue:
                candidate = self._queue.popleft()
                if candidate in self._bad:
                    continue
                self._current = candidate
                logger.info("Using free proxy %s", candidate)
                return candidate
            if not self._refill_unlocked():
                logger.warning("Free proxy pool empty — continuing without proxy")
                self._current = None
                return None

    def _refill_unlocked(self) -> bool:
        added = 0
        for fetcher in self._fetchers:
            try:
                batch = list(fetcher())
            except Exception as exc:
                logger.warning("Proxy fetcher failed: %s", exc)
                continue
            for raw in batch:
                normalized = _normalize_proxy(str(raw))
                if normalized is None or normalized in self._bad:
                    continue
                if normalized == self._current or normalized in self._queue:
                    continue
                self._queue.append(normalized)
                added += 1
        return added > 0


def configure_http_proxies(*, enabled: bool) -> None:
    """Enable or disable process-wide free-proxy rotation."""
    global _enabled, _pool
    with _lock:
        _enabled = bool(enabled)
        if not _enabled:
            _pool = None
            logger.info("Free HTTP proxies disabled")
            return
        if _pool is None:
            _pool = FreeProxyPool()
        logger.info("Free HTTP proxies enabled")


def http_proxies_enabled() -> bool:
    """Whether free-proxy rotation is currently enabled."""
    with _lock:
        return _enabled


def active_proxy_pool() -> FreeProxyPool | None:
    """Return the live pool when free proxies are enabled."""
    with _lock:
        if not _enabled:
            return None
        global _pool
        if _pool is None:
            _pool = FreeProxyPool()
        return _pool


def rotate_http_proxy(*, reason: str = "") -> str | None:
    """Rotate the process pool when enabled; otherwise no-op."""
    pool = active_proxy_pool()
    if pool is None:
        return None
    return pool.rotate(reason=reason)


def bind_http_proxy(session: requests.Session) -> None:
    """Bind the current free proxy (if any) onto ``session``."""
    pool = active_proxy_pool()
    if pool is None:
        session.proxies.clear()
        return
    pool.bind(session)


def reset_http_proxies_for_tests() -> None:
    """Drop process proxy state (tests only)."""
    global _enabled, _pool
    with _lock:
        _enabled = True
        _pool = None
