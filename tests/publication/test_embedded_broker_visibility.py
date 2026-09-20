from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


PAYLOAD = {
    "schema": "combined-gemma.broker-metric.v1",
    "queue": {"depth": 0, "fresh": True},
    "members": [],
}


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "candidate_gpu_manager_embedded_visibility", CONTROLLER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_visibility_uses_in_process_proxy_without_http_session(monkeypatch):
    module = _load_controller()
    module._services_config = {
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    module.session = None
    proxy = SimpleNamespace(broker_metrics=Mock(return_value=PAYLOAD))
    module._combined_gemma_api_proxy = proxy

    payload, error = asyncio.run(module._fetch_combined_gemma_visibility())

    assert payload is PAYLOAD
    assert error is None
    proxy.broker_metrics.assert_called_once_with()


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self, **_kwargs):
        return PAYLOAD


class _Session:
    closed = False

    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return _Response()


def test_external_visibility_retains_loopback_http_path(monkeypatch):
    module = _load_controller()
    module._services_config = {
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {
            "mode": "external",
            "endpoint": "http://127.0.0.1:18095",
        },
    }
    session = _Session()
    module.session = session
    monkeypatch.setenv(
        "COMBINED_GEMMA_BROKER_METRICS_URL",
        "http://127.0.0.1:18095/v1/metrics/broker?durable=0",
    )

    payload, error = asyncio.run(module._fetch_combined_gemma_visibility())

    assert payload is PAYLOAD
    assert error is None
    assert session.urls == [
        "http://127.0.0.1:18095/v1/metrics/broker?durable=0"
    ]


def test_disabled_broker_short_circuits_without_proxy_or_http(monkeypatch):
    """Broker disabled: must return (None, 'broker_disabled') before any IO."""
    module = _load_controller()
    # No combined_gemma_broker block at all (or explicitly disabled).
    module._services_config = {
        "combined_gemma_broker": {"enabled": False},
    }
    proxy = SimpleNamespace(broker_metrics=Mock(side_effect=AssertionError(
        "disabled broker must not invoke the embedded proxy"
    )))
    module._combined_gemma_api_proxy = proxy

    session = _Session()

    class _GuardSession(_Session):
        def get(self, url, **_kwargs):
            session.urls.append(url)
            raise AssertionError(
                "disabled broker must not perform HTTP requests"
            )

    module.session = _GuardSession()
    # Make sure the metrics env override would route somewhere different if
    # the implementation wrongly fell through to HTTP.
    monkeypatch.setenv(
        "COMBINED_GEMMA_BROKER_METRICS_URL",
        "http://127.0.0.1:65535/v1/metrics/broker?durable=0",
    )

    payload, error = asyncio.run(module._fetch_combined_gemma_visibility())

    assert payload is None
    assert error == "broker_disabled"
    proxy.broker_metrics.assert_not_called()
    assert module.session.urls == []


def test_external_visibility_uses_broker_metrics_url_never_alt_constant(monkeypatch):
    """External mode must request COMBINED_GEMMA_BROKER_METRICS_URL, not _COMBINED_GEMMA_URL."""
    module = _load_controller()
    module._services_config = {
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {
            "mode": "external",
            "endpoint": "http://127.0.0.1:18095",
        },
    }
    session = _Session()
    module.session = session
    metrics_url = "http://127.0.0.1:18095/v1/metrics/broker?durable=0"
    # Inject a sentinel _COMBINED_GEMMA_URL to prove the implementation never
    # uses it as the metrics endpoint. The implementation must read from
    # COMBINED_GEMMA_BROKER_METRICS_URL regardless of this sentinel.
    module._COMBINED_GEMMA_URL = "http://127.0.0.1:1/alt/should/not/be/used"
    monkeypatch.setenv("COMBINED_GEMMA_BROKER_METRICS_URL", metrics_url)

    payload, error = asyncio.run(module._fetch_combined_gemma_visibility())

    assert payload is PAYLOAD
    assert error is None
    # The requested URL is exactly the env-controlled metrics URL.
    assert session.urls == [metrics_url]
    # The sentinel alt constant must never have been requested.
    assert module._COMBINED_GEMMA_URL not in session.urls
    # The session must not have observed any URL other than the metrics URL.
    assert all(url == metrics_url for url in session.urls)


def test_member_refresh_uses_configured_ordered_members_only():
    module = _load_controller()
    configured = ("member-primary", "member-secondary", "member-fallback")
    calls: list[str] = []

    class _Cache:
        _configs = {
            name: {"enabled": True, "cpu_only": True}
            for name in configured
        }
        _snapshots = {}

        def update(self, name, _probes, *, now):
            calls.append(name)
            raise RuntimeError("bounded test stop after member selection")

    class _Repository:
        @staticmethod
        def lease_count(_name):
            return 0

    module._services_config = {
        "services": {name: {"enabled": True, "cpu_only": True} for name in configured},
        "combined_gemma_broker": {
            "enabled": True,
            "ordered_members": list(configured),
        },
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    module._combined_gemma_member_cache = _Cache()
    module._combined_gemma_broker = SimpleNamespace(
        dispatch_loop=SimpleNamespace(is_running=True),
        _repository=_Repository(),
    )
    module.session = SimpleNamespace(closed=False)

    asyncio.run(module._refresh_combined_gemma_member_snapshots())

    assert calls == list(configured)
    assert not {"LLM-Primary", "LLM-Secondary", "LLM-CPU"}.intersection(calls)


@pytest.mark.parametrize(
    "ordered_members",
    [None, [], "member-primary", ["member-primary", "member-primary"], [""], ["member-primary", 7]],
)
def test_member_refresh_refuses_invalid_ordered_members(ordered_members, caplog):
    module = _load_controller()
    calls: list[str] = []

    class _Cache:
        def update(self, name, _probes, *, now):
            calls.append(name)

    class _Repository:
        @staticmethod
        def lease_count(_name):
            return 0

    broker_config = {"enabled": True}
    if ordered_members is not None:
        broker_config["ordered_members"] = ordered_members
    module._services_config = {
        "services": {},
        "combined_gemma_broker": broker_config,
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    module._combined_gemma_member_cache = _Cache()
    module._combined_gemma_broker = SimpleNamespace(
        dispatch_loop=SimpleNamespace(is_running=True),
        _repository=_Repository(),
    )
    module.session = SimpleNamespace(closed=False)

    with caplog.at_level("ERROR", logger="gpu-manager"):
        asyncio.run(module._refresh_combined_gemma_member_snapshots())

    assert calls == []
    assert any("refused invalid ordered_members" in record.message for record in caplog.records)


def test_summarize_requires_non_stale_accepting_member_with_capacity():
    """available=true only when schema valid AND ≥1 non-stale member accepting work with positive capacity."""
    module = _load_controller()

    # Case A: schema valid but all members are non-accepting (draining) → not available.
    draining_payload = {
        "schema": "combined-gemma.broker-metric.v1",
        "queue": {"depth": 0, "fresh": True},
        "dispatch_loop": {"fencing_token": 7, "fail_counters": {"reservation_stale": 0}},
        "members": [
            {
                "name": "alpha",
                "fresh": True,
                "stale": False,
                "unknown": False,
                "accepting": False,
                "backend_slots_total": 8,
                "backend_slots_busy": 3,
            },
        ],
    }
    result = module._summarize_combined_router_readiness(draining_payload, None)
    assert result["schema_valid"] is True
    assert result["available"] is False
    assert result["accepting_member_count"] == 0
    assert result["reason"] == "no_live_constituents"

    # Case B: schema valid, member live+accepting with positive capacity → available.
    accepting_payload = {
        "schema": "combined-gemma.broker-metric.v1",
        "queue": {"depth": 0, "fresh": True},
        "dispatch_loop": {"fencing_token": 7, "fail_counters": {"reservation_stale": 0}},
        "members": [
            {
                "name": "alpha",
                "fresh": True,
                "stale": False,
                "unknown": False,
                "accepting": True,
                "backend_slots_total": 8,
                "backend_slots_busy": 3,
            },
        ],
    }
    result = module._summarize_combined_router_readiness(accepting_payload, None)
    assert result["schema_valid"] is True
    assert result["available"] is True
    assert result["accepting_member_count"] == 1
    assert result["reason"] == "ready"

    # Case C: schema valid, member live but stale-marked → not available.
    stale_payload = {
        "schema": "combined-gemma.broker-metric.v1",
        "queue": {"depth": 0, "fresh": True},
        "dispatch_loop": {"fencing_token": 7, "fail_counters": {"reservation_stale": 0}},
        "members": [
            {
                "name": "alpha",
                "fresh": False,
                "stale": True,
                "unknown": False,
                "accepting": True,
                "backend_slots_total": 8,
            },
        ],
    }
    result = module._summarize_combined_router_readiness(stale_payload, None)
    assert result["schema_valid"] is True
    assert result["available"] is False
    assert result["reason"] == "no_live_constituents"

    # Case D: schema valid, member live+accepting but zero total capacity → not available.
    zero_capacity_payload = {
        "schema": "combined-gemma.broker-metric.v1",
        "queue": {"depth": 0, "fresh": True},
        "dispatch_loop": {"fencing_token": 7, "fail_counters": {"reservation_stale": 0}},
        "members": [
            {
                "name": "alpha",
                "fresh": True,
                "stale": False,
                "unknown": False,
                "accepting": True,
                "backend_slots_total": 0,
            },
        ],
    }
    result = module._summarize_combined_router_readiness(zero_capacity_payload, None)
    assert result["schema_valid"] is True
    assert result["available"] is False
    assert result["reason"] == "no_live_constituents"

    # Case E: wrong schema → not available.
    wrong_schema_payload = {
        "schema": "something.else.v1",
        "queue": {"depth": 0, "fresh": True},
        "members": [],
    }
    result = module._summarize_combined_router_readiness(wrong_schema_payload, None)
    assert result["schema_valid"] is False
    assert result["available"] is False
    assert result["reason"] == "invalid_broker_schema"
