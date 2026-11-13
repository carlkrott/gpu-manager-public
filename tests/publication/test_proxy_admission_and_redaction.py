"""Regression tests for catch-all proxy admission and Redis header persistence."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
CONTROLLER_PATH = SCRIPTS / "gpu-manager.py"


def _load_controller():
    scripts_path = str(SCRIPTS)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location("gpu_manager_proxy_admission_test", CONTROLLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _request(*, service=None, path="/v1/chat/completions", headers=None):
    hdrs = dict(headers or {})
    query = {"service": service} if service is not None else {}
    return SimpleNamespace(
        path=path,
        method="POST",
        headers=hdrs,
        query=query,
        can_read_body=False,
        read=lambda: asyncio.sleep(0, result=b""),
        app={},
    )


def _configure(monkeypatch, services=None):
    from aiohttp import web

    module = _load_controller()
    module._services_config = {"services": services or {}}
    module._providers_config = {"providers": {}}
    module._model_validator = None
    module.routing_engine = None
    module._scoped_gemma_request_is_authorized = lambda _request: False
    module._combined_gemma_ingress_service_name = lambda name: name
    module._maybe_handle_combined_gemma_ingress = _none
    module.llm.total_requests = 0
    module.llm._llm_healthy = True
    module._load_services_config = lambda: {"services": services or {}}
    module._forward_llm = _forwarded
    module.LLM_BACKEND_URL = "http://backend"
    module.session = SimpleNamespace(request=lambda **_kwargs: None)
    module.web = web
    return module


async def _none(**_kwargs):
    return None


async def _forwarded(*_args, **_kwargs):
    from aiohttp import web
    return web.Response(status=200, text="forwarded")


@pytest.mark.parametrize("service,path", [("not-a-service", "/v1/chat/completions"), ("known", "/v1/not-a-route")])
def test_unknown_service_or_path_returns_404_without_enqueue(monkeypatch, service, path):
    module = _configure(monkeypatch, {"known": {"enabled": True, "type": "llm_backend", "port": 8123}})
    writes = []
    module.queue_engine = SimpleNamespace(enqueue=lambda *args, **kwargs: writes.append((args, kwargs)))
    response = asyncio.run(module.handle_llm_proxy(_request(service=service, path=path)))
    assert response.status == 404
    assert writes == []


def test_missing_explicit_service_binding_is_rejected_without_enqueue(monkeypatch):
    module = _configure(monkeypatch, {"known": {"enabled": True, "type": "llm_backend", "port": 8123}})
    writes = []
    module.queue_engine = SimpleNamespace(enqueue=lambda *args, **kwargs: writes.append((args, kwargs)))
    response = asyncio.run(module.handle_llm_proxy(_request()))
    assert response.status in (400, 404, 503)
    assert writes == []


@pytest.mark.parametrize("binding", ["header", "query"])
@pytest.mark.parametrize("resolution", [None, "missing-service"])
def test_unresolved_known_routing_group_is_rejected_before_enqueue(monkeypatch, binding, resolution):
    module = _configure(
        monkeypatch,
        {"known": {"enabled": True, "type": "llm_backend", "port": 8123}},
    )
    queue = _AdmissionQueue(_healthy_lanes(), enabled=True)
    module.queue_engine = queue
    writes = queue.writes
    module._queue_depth_for_group = lambda _group: 0
    module.routing_engine = SimpleNamespace(
        get_group_members=lambda group: ["known"] if group == "known-group" else [],
        find_service_for_group=lambda *_args, **_kwargs: resolution,
    )
    request = (
        _request(headers={"X-Backend": "known-group"})
        if binding == "header"
        else _request(service="known-group")
    )

    response = asyncio.run(module.handle_llm_proxy(request))

    assert response.status == 404
    assert json.loads(response.text)["code"] == "unknown_service"
    assert writes == []


def test_serialized_redis_headers_drop_credentials_and_preserve_forwarding_headers(monkeypatch):
    module = _configure(
        monkeypatch,
        {"known": {"enabled": True, "type": "llm_backend", "port": 8123, "gpu_id": "gpu0", "routing_group": "known-group"}},
    )
    payloads = []
    module._admission_gate = lambda *_args: (None, {})

    class Redis:
        def ping(self):
            return True

        def xinfo_groups(self, _stream):
            return [{"name": "proxy-workers"}]

        def xinfo_consumers(self, _stream, _group):
            return [{"name": "worker-1", "idle": 10}]

    class Queue:
        r = Redis()
        GROUP = "proxy-workers"

        def _stream(self, service, routing_group=None):
            return f"queue:{routing_group or service}"

        def _stream_specs(self, service, routing_group=None):
            return [(self._stream(service, routing_group), None)]

        def enqueue(self, service, payload, routing_group=None):
            payloads.append(payload)
            return "1-0"

    class Tracker:
        def next_job_id(self):
            return "job-test"

        def create_job(self, **_kwargs):
            pass

        def update_status(self, *_args, **_kwargs):
            pass

        def get_job(self, _job_id):
            return {"status": "completed", "result": "{}"}

    module.queue_engine = Queue()
    module.job_tracker = Tracker()
    module.slot_manager = object()
    module._COMBINED_GEMMA_ROUTING_GROUP = "known-group"
    module.routing_engine = SimpleNamespace(
        get_group_members=lambda _name: ["known"],
        find_service_for_group=lambda *_args, **_kwargs: "known",
    )
    module._queue_depth_for_group = lambda _name: 0
    class Healthy:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    module.session = SimpleNamespace(get=lambda *_args, **_kwargs: Healthy())
    request = _request(
        headers={
            "X-Backend": "known-group",
            "aUtHoRiZaTiOn": "Bearer secret-token",
            "Cookie": "session=secret-cookie",
            "pRoXy-AuThOrIzAtIoN": "Basic secret-proxy-token",
            "X-Api-Key": "secret-api-key",
            "X-Forwarded-For": "192.0.2.7",
            "X-Request-Id": "req-123",
        }
    )
    response = asyncio.run(module.handle_llm_proxy(request))
    assert response.status == 200, response.text
    assert len(payloads) == 1
    persisted = json.loads(payloads[0]["headers"])
    persisted_lower = {name.lower(): value for name, value in persisted.items()}
    assert not ({"authorization", "cookie", "proxy-authorization", "x-api-key"} & persisted_lower.keys())
    assert persisted_lower["x-forwarded-for"] == "192.0.2.7"
    assert persisted_lower["x-request-id"] == "req-123"


class _AdmissionRedis:
    def __init__(self, metadata):
        self.metadata = metadata

    def ping(self):
        return True

    def xinfo_groups(self, stream):
        return [{"name": "workers"}] if stream in self.metadata else []

    def xinfo_consumers(self, stream, _group):
        return self.metadata.get(stream, [])


class _AdmissionQueue:
    GROUP = "workers"

    def __init__(self, metadata, enabled=True):
        from queue_engine import RedisQueue

        self.priority_lanes_enabled = enabled
        self.r = _AdmissionRedis(metadata)
        self._real_queue = RedisQueue(r=self.r, priority_lanes_enabled=enabled)
        self.writes = []

    def _stream(self, service, routing_group=None):
        return self._real_queue._stream(service, routing_group=routing_group)

    def _stream_specs(self, service, routing_group=None):
        return self._real_queue._stream_specs(service, routing_group=routing_group)

    def enqueue(self, *args, **kwargs):
        self.writes.append((args, kwargs))
        return "1-0"


def _healthy_lanes():
    return {f"queue:group:lane:{lane}": [{"name": "worker", "idle": 100}]
            for lane in ("interactive", "normal", "background")}


def test_priority_admission_accepts_recent_consumers_on_writable_lanes():
    module = _load_controller()
    queue = _AdmissionQueue(_healthy_lanes(), enabled=True)
    module.queue_engine = queue
    assert module._proxy_queue_admission_error("service", "group") is None


def test_priority_admission_rejects_legacy_only_consumer_without_enqueue():
    module = _load_controller()
    queue = _AdmissionQueue({"queue:group": [{"name": "legacy", "idle": 20}]}, enabled=True)
    module.queue_engine = queue
    response = module._proxy_queue_admission_error("service", "group")
    assert response.status == 503
    assert queue.writes == []


@pytest.mark.parametrize("lane_state", ["stale", "missing"])
def test_priority_admission_rejects_stale_or_missing_lane_consumer(lane_state):
    module = _load_controller()
    metadata = _healthy_lanes()
    if lane_state == "stale":
        metadata["queue:group:lane:background"] = [{"name": "worker", "idle": 120_000}]
    else:
        del metadata["queue:group:lane:background"]
    queue = _AdmissionQueue(metadata, enabled=True)
    module.queue_engine = queue
    response = module._proxy_queue_admission_error("service", "group")
    assert response.status == 503
    assert queue.writes == []


def test_legacy_admission_uses_legacy_stream_when_priority_lanes_disabled():
    module = _load_controller()
    queue = _AdmissionQueue({"queue:group": [{"name": "legacy", "idle": 10}]}, enabled=False)
    module.queue_engine = queue
    assert module._proxy_queue_admission_error("service", "group") is None


@pytest.mark.parametrize("ping_ok,consumers", [(False, [{"idle": 10}]), (True, []), (True, [{"idle": 120_000}])])
def test_queue_admission_fails_closed_when_redis_or_consumer_is_unhealthy(monkeypatch, ping_ok, consumers):
    module = _configure(monkeypatch)

    class Redis:
        def ping(self):
            return ping_ok

        def xinfo_groups(self, _stream):
            return [{"name": "workers"}]

        def xinfo_consumers(self, _stream, _group):
            return consumers

    class Queue:
        r = Redis()
        GROUP = "workers"

        def _stream(self, service, routing_group=None):
            return f"queue:{routing_group or service}"

        def _stream_specs(self, service, routing_group=None):
            return [(self._stream(service, routing_group), None)]

    module.queue_engine = Queue()
    response = module._proxy_queue_admission_error("explicit-service", "group")
    assert response.status == 503
    assert json.loads(response.text)["code"] == "queue_not_admitting"
