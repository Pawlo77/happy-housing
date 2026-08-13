"""Process-wide free HTTP proxy pool for ingest downloads.

Sources: ``free-proxy`` library, ProxyScrape, Proxifly, and HProxy public
lists. Proxies engage after a blocking response (403/407/429) or when the
active proxy fails; first attempt stays direct. Candidates are CONNECT-probed
before use so obviously dead tunnels are skipped.
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

ProxyProbeFn = Callable[[str], bool]
"""Return True when ``proxy_url`` can tunnel HTTPS."""

_BLOCK_STATUSES: Final[frozenset[int]] = frozenset({403, 407, 429})
"""Statuses that trigger free-proxy rotation."""

_PROXYSCRAPE_URL: Final[str] = (
    "https://api.proxyscrape.com/v2/"
    "?request=displayproxies&protocol=http&timeout=5000"
    "&country=all&ssl=all&anonymity=elite"
)
"""Public free HTTP proxy list (text, one host:port per line)."""

_PROXIFLY_HTTPS_URL: Final[str] = (
    "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/"
    "proxies/protocols/https/data.txt"
)
"""Proxifly HTTPS-capable proxies (refreshed ~every 5 minutes)."""

_PROXIFLY_HTTP_URL: Final[str] = (
    "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/"
    "proxies/protocols/http/data.txt"
)
"""Proxifly HTTP proxies (CONNECT fallback)."""

_HPROXY_HTTPS_URL: Final[str] = (
    "https://cdn.jsdelivr.net/gh/hproxy-com/free-proxy-list@main/https.txt"
)
"""HProxy continuously verified HTTPS proxy list."""

_PROBE_URL: Final[str] = "https://example.com/"
"""Cheap HTTPS target for CONNECT pre-checks."""

_PROBE_TIMEOUT_S: Final[float] = 3.0
"""CONNECT probe timeout (seconds)."""

_lock = threading.RLock()
_enabled: bool = True
_pool: "FreeProxyPool | None" = None


def block_statuses() -> frozenset[int]:
    """HTTP statuses that should rotate the free-proxy pool."""
    return _BLOCK_STATUSES


def _normalize_proxy(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text or text.startswith("#") or text.startswith("{"):
        return None
    if "://" in text:
        # Already a URL; keep only supported schemes.
        if not text.startswith(("http://", "https://", "socks4://", "socks5://")):
            return None
        return text
    # host:port
    if ":" not in text or " " in text:
        return None
    return f"http://{text}"


def _parse_proxy_list_text(text: str, *, limit: int) -> list[str]:
    """Parse ``host:port`` / URL lines into normalized proxy URLs."""
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        normalized = _normalize_proxy(line)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _fetch_text_proxy_list(url: str, *, label: str, limit: int) -> list[str]:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("%s free list failed: %s", label, exc)
        return []
    out = _parse_proxy_list_text(response.text, limit=limit)
    if out:
        logger.info("%s returned %s free proxies", label, len(out))
    return out


def fetch_proxyscrape_proxies(*, limit: int = 80) -> list[str]:
    """Pull a fresh free HTTP proxy list from ProxyScrape (direct, no proxy)."""
    return _fetch_text_proxy_list(_PROXYSCRAPE_URL, label="ProxyScrape", limit=limit)


def fetch_proxifly_proxies(*, limit: int = 80) -> list[str]:
    """Pull Proxifly HTTPS list (then HTTP) via jsDelivr CDN."""
    https = _fetch_text_proxy_list(
        _PROXIFLY_HTTPS_URL, label="Proxifly HTTPS", limit=limit
    )
    if len(https) >= limit:
        return https
    http = _fetch_text_proxy_list(
        _PROXIFLY_HTTP_URL,
        label="Proxifly HTTP",
        limit=max(1, limit - len(https)),
    )
    merged: list[str] = []
    seen: set[str] = set()
    for item in https + http:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def fetch_hproxy_proxies(*, limit: int = 80) -> list[str]:
    """Pull HProxy continuously verified HTTPS proxy list."""
    return _fetch_text_proxy_list(_HPROXY_HTTPS_URL, label="HProxy HTTPS", limit=limit)


def fetch_free_proxy_lib_proxies(*, limit: int = 12) -> list[str]:
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


def probe_https_proxy(
    proxy_url: str,
    *,
    timeout: float = _PROBE_TIMEOUT_S,
    probe_url: str = _PROBE_URL,
) -> bool:
    """Return True when ``proxy_url`` can tunnel a simple HTTPS GET."""
    mapping = {"http": proxy_url, "https": proxy_url}
    try:
        response = requests.get(probe_url, proxies=mapping, timeout=timeout)
    except requests.RequestException:
        return False
    return int(response.status_code) < 500


def _default_fetchers() -> list[ProxyFetcher]:
    """Built-in free-proxy sources (library + multiple public lists)."""
    return [
        lambda: fetch_free_proxy_lib_proxies(limit=12),
        lambda: fetch_proxyscrape_proxies(limit=80),
        lambda: fetch_proxifly_proxies(limit=80),
        lambda: fetch_hproxy_proxies(limit=80),
    ]


class FreeProxyPool:
    """Thread-safe deque of free proxies with bad-proxy exclusion."""

    def __init__(
        self,
        *,
        fetchers: Sequence[ProxyFetcher] | None = None,
        probe: bool = True,
        probe_fn: ProxyProbeFn | None = None,
    ) -> None:
        self._fetchers = list(fetchers) if fetchers is not None else _default_fetchers()
        self._probe = bool(probe)
        self._probe_fn = probe_fn or probe_https_proxy
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
        refill_attempts = 0
        while True:
            while self._queue:
                candidate = self._queue.popleft()
                if candidate in self._bad:
                    continue
                if self._probe and not self._probe_candidate(candidate):
                    self._bad.add(candidate)
                    continue
                self._current = candidate
                logger.info("Using free proxy %s", candidate)
                return candidate
            refill_attempts += 1
            if not self._refill_unlocked():
                if refill_attempts >= 2:
                    logger.warning(
                        "Free proxy pool empty (%s bad) — continuing without proxy",
                        len(self._bad),
                    )
                    self._current = None
                    return None
                continue

    def _probe_candidate(self, candidate: str) -> bool:
        """Run CONNECT probe outside the lock? Keep simple: call under lock.

        Probe is short (3s); holding the pool lock during probe serializes
        acquire, which is fine for ingest throughput.
        """
        try:
            ok = bool(self._probe_fn(candidate))
        except Exception as exc:
            logger.debug("Proxy probe error for %s: %s", candidate, exc)
            return False
        if not ok:
            logger.info("CONNECT probe failed for %s — skipping", candidate)
        return ok

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
