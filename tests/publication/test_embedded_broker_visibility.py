from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobRecord, MemberState
from gemma_broker.member_cache import StrictMemberSnapshotCache
from gemma_broker.redis_store import InMemoryJobRepository


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


def test_embedded_broker_uses_process_independent_epoch_leadership_clock():
    source = CONTROLLER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    initializer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_initialize_combined_gemma_broker"
    )
    repository_calls = [
        node
        for node in ast.walk(initializer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RedisJobRepository"
    ]

    assert len(repository_calls) == 1
    leadership_clock = next(
        (
            keyword.value
            for keyword in repository_calls[0].keywords
            if keyword.arg == "leadership_clock"
        ),
        None,
    )
    assert isinstance(leadership_clock, ast.Attribute)
    assert isinstance(leadership_clock.value, ast.Name)
    assert leadership_clock.value.id == "time"
    assert leadership_clock.attr == "time"


def test_embedded_api_proxy_exposes_live_repository_for_attempt_status():
    module = _load_controller()
    repository = object()
    setattr(
        module,
        "_combined_gemma_broker",
        SimpleNamespace(api_service=SimpleNamespace(repository=repository)),
    )

    proxy = module._CombinedGemmaAPIProxy()

    assert proxy.repository is repository


def test_embedded_start_rejoins_member_with_generation_fenced_cas():
    module = _load_controller()
    service_name = "Gemma-MI50"
    calls: list[tuple] = []
    member = SimpleNamespace(
        name=service_name,
        state=SimpleNamespace(value="offline"),
        state_version=17,
        generation_fence=1_000,
        accepting=False,
    )

    class _Repository:
        @staticmethod
        def get_member(name):
            assert name == service_name
            return member

        @staticmethod
        def begin_member_rejoin(
            name, *, expected_state_version, generation_fence
        ):
            assert name == service_name
            assert expected_state_version == 17
            assert generation_fence > 1_000
            calls.append((name, expected_state_version, generation_fence))
            member.state = SimpleNamespace(value="joining")
            member.state_version = 18
            member.generation_fence = generation_fence
            return member

    async def _refresh():
        assert member.state.value == "joining"
        member.state = SimpleNamespace(value="ready_accepting")
        member.state_version = 19
        member.accepting = True

    module._services_config = {
        "services": {
            service_name: {
                "enabled": True,
                "type": "llm_backend",
                "routing_group": "Gemma",
            }
        },
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    module._combined_gemma_broker = SimpleNamespace(
        _repository=_Repository()
    )
    module._refresh_combined_gemma_member_snapshots = _refresh

    rejoined = asyncio.run(
        module._rejoin_combined_gemma_member_after_start(
            service_name, timeout=0.1
        )
    )

    assert rejoined is True
    assert len(calls) == 1
    assert member.state.value == "ready_accepting"
    assert member.accepting is True


def test_embedded_start_completes_real_repository_rejoin_and_reserves(
    monkeypatch,
):
    module = _load_controller()
    service_name = "member-secondary"
    service_config = {
        "enabled": True,
        "type": "llm_backend",
        "routing_group": "Gemma",
        "port": 18100,
        "gpu_id": "synthetic-secondary",
        "parallel": 1,
        "context_per_slot": 4096,
        "model_path": "/models/synthetic-secondary.gguf",
        "capabilities": ["chat"],
    }
    cache_config = {
        **service_config,
        "idle_service_configured": service_name,
    }

    class _Repository(InMemoryJobRepository):
        def __init__(self):
            super().__init__()
            self.complete_rejoin_calls = 0

        def complete_member_rejoin(self, *args, **kwargs):
            self.complete_rejoin_calls += 1
            return super().complete_member_rejoin(*args, **kwargs)

    class _Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self, **_kwargs):
            return self.payload

    class _ProbeSession:
        closed = False

        def get(self, url, **_kwargs):
            if url.endswith("/health"):
                return _Response({"status": "ok"})
            if url.endswith("/slots"):
                return _Response(
                    [{"id": 0, "n_ctx": 4096, "is_processing": False}]
                )
            if url.endswith("/v1/models"):
                return _Response(
                    {
                        "data": [
                            {
                                "id": "synthetic-secondary.gguf",
                                "meta": {"n_ctx": 4096},
                            }
                        ]
                    }
                )
            raise AssertionError(url)

    async def _systemd_facts(_config):
        return {"systemd_active": True, "main_pid": 12345}

    repository = _Repository()
    cache = StrictMemberSnapshotCache({service_name: cache_config})
    offline = replace(
        cache.snapshots()[0],
        state=MemberState.OFFLINE,
        accepting=False,
        state_version=17,
        generation_fence=1_000,
        compatible_free_slots=0,
        blockers=("member rejoin required",),
    )
    repository.put_member(offline)
    module._services_config = {
        "services": {service_name: service_config},
        "combined_gemma_broker": {
            "enabled": True,
            "ordered_members": [service_name],
        },
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    module._combined_gemma_member_cache = cache
    module._combined_gemma_broker = SimpleNamespace(
        _repository=repository,
        dispatch_loop=SimpleNamespace(is_running=True),
    )
    module.session = _ProbeSession()
    monkeypatch.setattr(
        module, "_combined_gemma_systemd_facts", _systemd_facts
    )

    rejoined = asyncio.run(
        module._rejoin_combined_gemma_member_after_start(
            service_name, timeout=1.0
        )
    )

    ready = repository.get_member(service_name)
    assert rejoined is True
    assert repository.complete_rejoin_calls == 1
    assert ready is not None
    assert ready.state is MemberState.READY_ACCEPTING
    assert ready.accepting is True
    assert ready.compatible_free_slots == 1

    job = JobRecord.new(
        job_id="job-after-rejoin",
        idempotency_key="idem-after-rejoin",
        request_sha256="0" * 64,
        request_body={"messages": [{"role": "user", "content": "synthetic"}]},
        submitted_at=10.0,
        enqueue_sequence=1,
        required_capabilities=("chat",),
    )
    repository.submit(job, caller_scope="test")
    reservation = repository.reserve_next(
        now=11.0,
        ordered_members=[ready],
        requirements=RequestRequirements(
            input_tokens_estimate=8,
            max_output_tokens=16,
            required_capabilities=("chat",),
            estimate_source="synthetic",
        ),
        member_lookup=repository.get_member,
    )
    assert reservation is not None
    assert reservation.member.name == service_name


def test_bundle_start_rejoins_only_broker_owned_members(monkeypatch):
    module = _load_controller()
    calls: list[tuple[str, float]] = []
    services = {
        "member-secondary": {
            "enabled": True,
            "type": "llm_backend",
            "routing_group": "Gemma",
        },
        "unrelated-service": {
            "enabled": True,
            "type": "audio",
        },
    }
    module._services_config = {
        "services": services,
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {"mode": "embedded"},
    }

    async def _rejoin(name, *, timeout):
        calls.append((name, timeout))
        return True

    monkeypatch.setattr(
        module, "_rejoin_combined_gemma_member_after_start", _rejoin
    )

    result = asyncio.run(
        module._rejoin_combined_gemma_bundle_members_after_start(
            ["member-secondary", "unrelated-service"],
            services,
            timeout=12.0,
        )
    )

    assert result is True
    assert calls == [("member-secondary", 12.0)]


def test_load_bundle_invokes_broker_member_rejoin_gate():
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    load_bundle = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_load_bundle"
    )
    called_names = {
        call.func.id
        for call in ast.walk(load_bundle)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
    }

    assert "_rejoin_combined_gemma_bundle_members_after_start" in called_names


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
