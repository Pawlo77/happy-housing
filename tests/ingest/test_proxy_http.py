"""Free-proxy rotation on HTTP 403 for shared ingest client."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from happy_housing.ingest import http_client, proxy_pool
from happy_housing.ingest.http_client import resilient_get
from happy_housing.ingest.proxy_pool import (
    FreeProxyPool,
    configure_http_proxies,
    reset_http_proxies_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_proxy_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid real CONNECT probes against fake 10.x hosts in unit tests.
    monkeypatch.setattr(proxy_pool, "probe_https_proxy", lambda *_a, **_k: True)
    reset_http_proxies_for_tests()
    yield
    reset_http_proxies_for_tests()


def test_pool_rotate_marks_bad_and_binds_next() -> None:
    fetched = ["http://1.1.1.1:8080", "http://2.2.2.2:8080"]

    def _fetcher() -> list[str]:
        return list(fetched)

    pool = FreeProxyPool(fetchers=[_fetcher], probe=False)
    first = pool.ensure()
    assert first == "http://1.1.1.1:8080"
    session = requests.Session()
    pool.bind(session)
    assert session.proxies["http"] == first

    second = pool.rotate(reason="403")
    assert second == "http://2.2.2.2:8080"
    pool.bind(session)
    assert session.proxies["https"] == second
    assert first in pool.bad_proxies


def test_resilient_get_rotates_proxy_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_http_proxies(enabled=True)
    proxies = ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
    monkeypatch.setattr(
        proxy_pool,
        "_default_fetchers",
        lambda: [lambda: list(proxies)],
    )
    reset_http_proxies_for_tests()
    configure_http_proxies(enabled=True)

    sleeps: list[float] = []
    monkeypatch.setattr(http_client.time, "sleep", lambda s: sleeps.append(float(s)))

    calls: list[dict[str, Any]] = []

    class SeqSession:
        def __init__(self) -> None:
            self.proxies: dict[str, str] = {}

        def get(self, url: str, timeout: object = None, **kwargs: object) -> MagicMock:
            _ = timeout, kwargs
            calls.append({"url": url, "proxies": dict(self.proxies)})
            response = MagicMock()
            if len(calls) == 1:
                response.status_code = 403
                response.raise_for_status.side_effect = requests.HTTPError(
                    "403", response=response
                )
                return response
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.text = "ok"
            return response

    session = SeqSession()
    out = resilient_get(
        session,  # type: ignore[arg-type]
        "https://example.test/blocked",
        retries=4,
        label="probe",
    )
    assert out is not None
    assert out.status_code == 200
    assert len(calls) >= 2
    assert calls[0]["proxies"] == {}
    assert calls[1]["proxies"].get("http") in proxies
    assert sleeps  # backoff still applied


def test_resilient_get_skips_rotation_when_proxies_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_http_proxies(enabled=False)
    rotates = 0

    def _boom_rotate(**_kwargs: object) -> str | None:
        nonlocal rotates
        rotates += 1
        return None

    monkeypatch.setattr(http_client, "rotate_http_proxy", _boom_rotate)
    monkeypatch.setattr(http_client.time, "sleep", lambda _s: None)

    class Always403:
        def __init__(self) -> None:
            self.proxies: dict[str, str] = {}

        def get(self, url: str, timeout: object = None, **kwargs: object) -> MagicMock:
            _ = url, timeout, kwargs
            response = MagicMock()
            response.status_code = 403
            response.raise_for_status.side_effect = requests.HTTPError(
                "403", response=response
            )
            return response

    with pytest.raises(requests.HTTPError):
        resilient_get(Always403(), "https://example.test/x", retries=2, label="x")  # type: ignore[arg-type]
    assert rotates == 0


def test_resilient_get_proxy_errors_do_not_burn_content_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead free proxies rotate until one works; content retries stay intact."""
    configure_http_proxies(enabled=True)
    proxies = [f"http://10.0.0.{i}:8000" for i in range(1, 8)]
    monkeypatch.setattr(
        proxy_pool,
        "_default_fetchers",
        lambda: [lambda: list(proxies)],
    )
    reset_http_proxies_for_tests()
    configure_http_proxies(enabled=True)
    monkeypatch.setattr(http_client.time, "sleep", lambda _s: None)

    calls: list[dict[str, Any]] = []

    class SeqSession:
        def __init__(self) -> None:
            self.proxies: dict[str, str] = {}

        def get(self, url: str, timeout: object = None, **kwargs: object) -> MagicMock:
            _ = url, timeout, kwargs
            calls.append({"proxies": dict(self.proxies)})
            # Direct → 403, then 5 dead proxies, then success.
            n = len(calls)
            response = MagicMock()
            if n == 1:
                response.status_code = 403
                response.raise_for_status.side_effect = requests.HTTPError(
                    "403", response=response
                )
                return response
            if n < 7:
                raise requests.exceptions.ProxyError(f"dead-{n}")
            response.status_code = 200
            response.raise_for_status.return_value = None
            return response

    out = resilient_get(
        SeqSession(),  # type: ignore[arg-type]
        "https://example.test/offer",
        retries=3,
        label="detail",
    )
    assert out is not None
    assert out.status_code == 200
    assert len(calls) == 7
    assert calls[0]["proxies"] == {}
    assert calls[-1]["proxies"].get("http") in proxies


def test_pool_refills_when_queue_exhausted() -> None:
    batches = [
        ["http://1.1.1.1:1", "http://1.1.1.2:1"],
        ["http://2.2.2.1:1"],
    ]
    state = {"i": 0}

    def _fetcher() -> list[str]:
        i = state["i"]
        state["i"] = min(i + 1, len(batches) - 1)
        return list(batches[i])

    pool = FreeProxyPool(fetchers=[_fetcher], probe=False)
    assert pool.ensure() == "http://1.1.1.1:1"
    assert pool.rotate(reason="bad") == "http://1.1.1.2:1"
    # Queue empty → refill pulls next batch.
    assert pool.rotate(reason="bad") == "http://2.2.2.1:1"


def test_pool_skips_proxies_that_fail_connect_probe() -> None:
    fetched = ["http://dead.proxy:1", "http://live.proxy:2"]

    def _probe(url: str) -> bool:
        return url.endswith(":2")

    pool = FreeProxyPool(
        fetchers=[lambda: list(fetched)],
        probe=True,
        probe_fn=_probe,
    )
    assert pool.ensure() == "http://live.proxy:2"
    assert "http://dead.proxy:1" in pool.bad_proxies


def test_fetch_proxifly_and_hproxy_parse_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies = {
        "proxifly": "1.2.3.4:8080\n5.6.7.8:3128\n",
        "hproxy": "9.9.9.9:80\n8.8.8.8:8080\n",
    }

    def _fake_get(url: str, timeout: object = None, **_kwargs: object) -> MagicMock:
        _ = timeout
        response = MagicMock()
        response.raise_for_status.return_value = None
        if "proxifly" in url:
            response.text = bodies["proxifly"]
        else:
            response.text = bodies["hproxy"]
        return response

    monkeypatch.setattr(proxy_pool.requests, "get", _fake_get)
    proxifly = proxy_pool.fetch_proxifly_proxies(limit=10)
    hproxy = proxy_pool.fetch_hproxy_proxies(limit=10)
    assert proxifly == ["http://1.2.3.4:8080", "http://5.6.7.8:3128"]
    assert hproxy == ["http://9.9.9.9:80", "http://8.8.8.8:8080"]


def test_otodom_http_get_rotates_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    from happy_housing.ingest import otodom

    configure_http_proxies(enabled=True)
    monkeypatch.setattr(
        proxy_pool,
        "_default_fetchers",
        lambda: [lambda: ["http://9.9.9.9:9999"]],
    )
    reset_http_proxies_for_tests()
    configure_http_proxies(enabled=True)
    monkeypatch.setattr(otodom.time, "sleep", lambda _s: None)

    calls: list[dict[str, str]] = []

    class SeqSession:
        def __init__(self) -> None:
            self.proxies: dict[str, str] = {}

        def get(self, url: str, timeout: object = None, **kwargs: object) -> MagicMock:
            _ = url, timeout, kwargs
            calls.append(dict(self.proxies))
            response = MagicMock()
            if len(calls) == 1:
                response.status_code = 403
                response.raise_for_status.side_effect = requests.HTTPError(
                    "403", response=response
                )
                return response
            response.status_code = 200
            response.raise_for_status.return_value = None
            return response

    response = otodom._http_get(
        SeqSession(),  # type: ignore[arg-type]
        "https://www.otodom.pl/pl/oferta/x",
        retries=3,
        label="detail",
    )
    assert response is not None
    assert calls[0] == {}
    assert calls[1].get("http") == "http://9.9.9.9:9999"
