"""Scoped KDA Gemma token: member pins and broker-job routing."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from api_auth import (
    HONCHO_GEMMA_TOKEN_FILE_ENV,
    KDA_GEMMA_DEFAULT_PIN_APP_KEY,
    KDA_GEMMA_MEMBER_PINS_APP_KEY,
    KDA_GEMMA_SCOPED_GATE_APP_KEY,
    api_auth_middleware,
    scoped_gemma_request_is_authorized,
)
from gemma_broker.api import APIError, DurableBrokerService
from gemma_broker.compat import CompatibilityWrapper
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import (
    JobRecord,
    JobState,
    MemberSnapshot,
    MemberState,
    Priority,
    ReasonCode,
)
from gemma_broker.dispatcher import Dispatcher
from gemma_broker.runtime import PollingTerminalWait, _CandidateCompatAdapter
from gemma_broker.redis_store import (
    InMemoryJobRepository,
    LeadershipLost,
    RepositoryUnavailable,
    ReservationStale,
)
from test_broker_pin_503_regressions import _idle_dispatch_loop

# Synthetic values only; these are never real credentials.
ADMIN = "synthetic-admin-token"
SCOPED = "synthetic-scoped-token"
HONCHO = "synthetic-honcho-token"
MEMBERS = ("Gemma-CPU", "Gemma-MI50")
CHAT = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def tokens(tmp_path, monkeypatch):
    for name, value, env in (
        ("admin", ADMIN, "GPU_MANAGER_API_TOKEN_FILE"),
        ("scoped", SCOPED, "GPU_MANAGER_KDA_GEMMA_TOKEN_FILE"),
        ("honcho", HONCHO, HONCHO_GEMMA_TOKEN_FILE_ENV),
    ):
        path = tmp_path / name
        path.write_text(value)
        path.chmod(0o600)
        monkeypatch.setenv(env, str(path))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)


def _gate_app(members=MEMBERS) -> tuple[web.Application, list]:
    seen: list = []

    async def ok(request):
        seen.append((request.method, request.path, request.headers.get("X-Backend")))
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[api_auth_middleware])
    app[KDA_GEMMA_SCOPED_GATE_APP_KEY] = lambda: True
    app[KDA_GEMMA_MEMBER_PINS_APP_KEY] = lambda: frozenset(members)
    for method, path in (
        ("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"),
        ("GET", "/v1/other"), ("POST", "/v1/admin"),
    ):
        app.router.add_route(method, path, ok)
    return app, seen


def _call(method, path, *, token=SCOPED, backend=None, extra=None):
    async def go():
        app, seen = _gate_app()
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {token}"}
            if backend:
                headers["X-Backend"] = backend
            headers.update(extra or {})
            response = await client.request(method, path, headers=headers)
            return response.status, seen
    return asyncio.run(go())


def test_scoped_token_accepts_group_and_member_pins(tokens):
    for backend in ("Gemma", "Gemma-CPU", "Gemma-MI50"):
        for method, path in (("GET", "/health"), ("GET", "/v1/models"),
                             ("POST", "/v1/chat/completions")):
            status, seen = _call(method, path, backend=backend)
            assert status == 200 and seen[-1][2] == backend


@pytest.mark.parametrize("backend", ["Gemma-Unknown", "gemma-cpu", "", None, "Luna-6-Low"])
def test_scoped_token_rejects_unknown_member_pins(tokens, backend):
    status, seen = _call("POST", "/v1/chat/completions", backend=backend)
    assert status == 401 and not seen


def test_scoped_token_rejects_other_paths_and_query_string(tokens):
    assert _call("GET", "/v1/other", backend="Gemma-CPU")[0] == 401
    assert _call("POST", "/v1/admin", backend="Gemma-CPU")[0] == 401
    assert _call("POST", "/v1/chat/completions?service=x", backend="Gemma-CPU")[0] == 401
    assert _call("GET", "/health?x=1", backend="Gemma-CPU")[0] == 401


def test_scoped_token_rejects_non_loopback_peer(tokens):
    app, _ = _gate_app()
    request = SimpleNamespace(
        app=app, remote="10.1.2.3", path="/v1/chat/completions", raw_path="/v1/chat/completions",
        query_string="", method="POST",
        headers={"X-Backend": "Gemma-CPU", "Authorization": f"Bearer {SCOPED}"},
    )
    assert scoped_gemma_request_is_authorized(request) is False
    request.remote = "127.0.0.1"
    assert scoped_gemma_request_is_authorized(request) is True


def test_member_set_comes_from_controller_not_client(tokens):
    # A client-supplied X-Client/X-Members style header cannot widen the set.
    status, _ = _call("POST", "/v1/chat/completions", backend="Evil",
                      extra={"X-Client": "Evil", "X-Members": "Evil"})
    assert status == 401


def test_admin_token_behaviour_unchanged(tokens):
    for backend in (None, "Gemma-CPU", "anything"):
        assert _call("GET", "/v1/other", token=ADMIN, backend=backend)[0] == 200
    assert _call("GET", "/v1/other", token="wrong")[0] == 401


# ── broker: member-pinned jobs share the queue, priority and lease accounting ──

def _member(name, order, slots=1):
    return MemberSnapshot(
        name=name, gpu_id=None, state=MemberState.READY_ACCEPTING, accepting=True,
        configured_slots=slots, context_per_slot=4096, state_version=1,
        observed_at=1.0, compatible_free_slots=slots, capabilities=("chat",),
    )


def _service():
    repo = InMemoryJobRepository()
    members = [repo.put_member(_member(n, i)) for i, n in enumerate(MEMBERS)]
    clock = lambda: 2.0
    return repo, members, DurableBrokerService(repo, members=lambda: members, clock=clock)


def test_pinned_job_is_queued_and_only_reserves_its_member():
    repo, members, service = _service()
    job = service.submit({"idempotency_key": "p1", "request": CHAT, "pinned_member": "Gemma-CPU"})
    assert job.pinned_member == "Gemma-CPU" and job.state is JobState.QUEUED
    # Offered the GPU-first order, the pin still lands on Gemma-CPU.
    reservation = repo.reserve_next(
        now=3.0, ordered_members=list(reversed(members)),
        requirements=RequestRequirements(8, 16, ("chat",), "t", pinned_member="Gemma-CPU"),
    )
    assert reservation.member.name == "Gemma-CPU"
    assert repo.lease_count("Gemma-CPU") == 1 and repo.lease_count("Gemma-MI50") == 0


def test_unknown_member_pin_rejected_at_submit():
    _, _, service = _service()
    with pytest.raises(APIError) as caught:
        service.submit({"idempotency_key": "p2", "request": CHAT, "pinned_member": "Nope"})
    assert caught.value.status == 404 and caught.value.code == "unknown_member"


def test_pinned_member_busy_blocks_only_by_slot_accounting():
    repo, members, service = _service()
    for key in ("a", "b"):
        service.submit({"idempotency_key": key, "request": CHAT, "pinned_member": "Gemma-CPU"})
    req = lambda: RequestRequirements(8, 16, ("chat",), "t", pinned_member="Gemma-CPU")
    first = repo.reserve_next(now=3.0, ordered_members=members, requirements=req())
    assert first is not None
    # The single CPU slot is leased; the second pinned job must wait, not fall through.
    # The stale snapshot still shows a free slot; the lease count refuses it.
    with pytest.raises(ReservationStale):
        repo.reserve_next(now=3.0, ordered_members=members, requirements=req())
    assert repo.lease_count("Gemma-MI50") == 0  # never falls through to another member


def test_priority_ordering_across_pinned_and_unpinned_jobs():
    repo, members, service = _service()
    low = service.submit({"idempotency_key": "low", "request": CHAT, "priority": int(Priority.BACKGROUND)})
    pinned = service.submit({"idempotency_key": "hi", "request": CHAT,
                             "priority": int(Priority.INTERACTIVE), "pinned_member": "Gemma-CPU"})
    normal = service.submit({"idempotency_key": "nm", "request": CHAT, "priority": int(Priority.NORMAL)})
    order = []
    for _ in range(3):
        head = repo.peek_next(now=3.0)
        order.append(head.job_id)
        res = repo.reserve_next(
            now=3.0, ordered_members=members,
            requirements=RequestRequirements(8, 16, ("chat",), "t", pinned_member=head.pinned_member),
        )
        assert res is not None
        repo.replace_job(replace(res.job, state=JobState.COMPLETED))
        repo.release_member_lease(res.member.name, res.job.current_attempt_id,
                                  claimed_member_state_version=res.job.member_state_version)
    assert order == [pinned.job_id, normal.job_id, low.job_id]


class _Transport:
    def __init__(self, exc=None):
        self.exc = exc

    async def send(self, member, body, on_accepted):
        if self.exc:
            raise self.exc
        on_accepted()
        return {"status": 200, "body": {}, "content_type": "application/json"}


def _pinned_reservation(repo, members, service):
    service.submit({"idempotency_key": "t1", "request": CHAT, "pinned_member": "Gemma-CPU"})
    return repo.reserve_next(
        now=3.0, ordered_members=members,
        requirements=RequestRequirements(8, 16, ("chat",), "t", pinned_member="Gemma-CPU"),
    )


def _dispatch(repo, transport):
    return Dispatcher(repo=repo, transport=transport, request_timeout=5.0,
                      clock=lambda: 4.0, terminal_write_delay=0.0)


@pytest.mark.parametrize("exc,final", [
    (None, JobState.COMPLETED),
    (RuntimeError("backend died after accept"), JobState.QUEUED),  # pre-acceptance requeue
])
def test_lease_released_on_terminal_paths(exc, final):
    repo, members, service = _service()
    res = _pinned_reservation(repo, members, service)
    out = asyncio.run(_dispatch(repo, _Transport(exc)).dispatch(res))
    assert out.state is final
    assert repo.lease_count("Gemma-CPU") == 0


class _AcceptThenFail(_Transport):
    async def send(self, member, body, on_accepted):
        on_accepted()
        raise RuntimeError("backend died after accept")


class _AcceptThenCancel(_Transport):
    def __init__(self, repo, job_id):
        super().__init__()
        self.repo, self.job_id = repo, job_id

    async def send(self, member, body, on_accepted):
        on_accepted()
        job = self.repo.get(self.job_id)
        self.repo.replace_job(replace(job, cancel_requested=True))
        return {"status": 200, "body": {}, "content_type": "application/json"}


def test_lease_released_on_outcome_unknown_after_acceptance():
    repo, members, service = _service()
    res = _pinned_reservation(repo, members, service)
    out = asyncio.run(_dispatch(repo, _AcceptThenFail()).dispatch(res))
    assert out.state is JobState.OUTCOME_UNKNOWN
    assert out.pinned_member == "Gemma-CPU" and repo.lease_count("Gemma-CPU") == 0


def test_lease_released_on_cancel_after_acceptance():
    repo, members, service = _service()
    res = _pinned_reservation(repo, members, service)
    out = asyncio.run(_dispatch(repo, _AcceptThenCancel(repo, res.job.job_id)).dispatch(res))
    assert out.state is JobState.CANCELLED and repo.lease_count("Gemma-CPU") == 0


def test_lease_released_on_cancel_before_acceptance():
    repo, members, service = _service()
    res = _pinned_reservation(repo, members, service)
    repo.replace_job(replace(repo.get(res.job.job_id), cancel_requested=True))
    out = asyncio.run(_dispatch(repo, _Transport()).dispatch(res))
    assert out.state is JobState.CANCELLED and repo.lease_count("Gemma-CPU") == 0


def test_f3_leadership_loss_in_dispatch_leaves_lease_for_successor_recovery():
    """Dispatcher re-raises LeadershipLost without releasing (a successor owns it).

    The lease is then freed by the successor's stale-claim recovery
    (``requeue_claim``), never leaked and never double-released.
    """
    repo, members, service = _service()
    res = _pinned_reservation(repo, members, service)
    dispatcher = _dispatch(repo, _Transport(LeadershipLost("fenced")))
    with pytest.raises(LeadershipLost):
        asyncio.run(dispatcher.dispatch(res))
    assert repo.lease_count("Gemma-CPU") == 1
    from gemma_broker.contracts import ReasonCode
    requeued = repo.requeue_claim(
        repo.get(res.job.job_id), reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE, terminal_at=99.0
    )
    assert requeued.state is JobState.QUEUED and repo.lease_count("Gemma-CPU") == 0
    assert requeued.pinned_member == "Gemma-CPU"  # the pin survives a requeue


def test_f3_repository_unavailable_in_dispatch_is_reconciled():
    repo, members, service = _service()
    res = _pinned_reservation(repo, members, service)
    dispatcher = _dispatch(repo, _Transport(RepositoryUnavailable("redis_state_unknown")))
    with pytest.raises(RepositoryUnavailable):
        asyncio.run(dispatcher.dispatch(res))
    assert repo.lease_count("Gemma-CPU") == 1 and dispatcher.unreleased
    asyncio.run(dispatcher.reconcile_unreleased())
    assert repo.lease_count("Gemma-CPU") == 0 and not dispatcher.unreleased


# ── compat handler: X-Backend pin + X-Client reach the broker, key separates pins ──

class _CompatService:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def submit_compat(self, body, *, idempotency_key, pinned_member=None):
        self.calls.append((idempotency_key, pinned_member))
        payload = {"idempotency_key": idempotency_key, "request": body}
        if pinned_member:
            payload["pinned_member"] = pinned_member
        return self.inner.submit(payload)


def _compat_post(headers):
    async def go():
        repo, members, service = _service()
        compat_service = _CompatService(service)
        wrapper = CompatibilityWrapper(compat_service, timeout=5.0)

        async def handler(request):
            return await wrapper.handle_proxy(
                method="POST", path="/v1/chat/completions", body=await request.read(),
                headers=request.headers, request=request,
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions", json=CHAT,
                headers={**headers, "X-Async": "1", "Prefer": "respond-async"},
            )
            return response.status, await response.json(), compat_service.calls, repo
    return asyncio.run(go())


def test_compat_handler_queues_member_pinned_job_with_client_attribution(caplog):
    caplog.set_level("INFO", logger="gemma_broker.compat")
    status, body, calls, repo = _compat_post({"X-Backend": "Gemma-CPU", "X-Client": "kda"})
    assert status in (200, 202), body
    assert calls[0][1] == "Gemma-CPU"
    assert repo.get(body["job_id"]).pinned_member == "Gemma-CPU"
    assert "client='kda'" in caplog.text


def test_compat_handler_logical_gemma_is_unpinned_and_unknown_member_rejected():
    status, body, calls, repo = _compat_post({"X-Backend": "Gemma"})
    assert calls[0][1] is None and repo.get(body["job_id"]).pinned_member is None
    status, body, calls, _ = _compat_post({"X-Backend": "Gemma-Unknown"})
    assert status == 404 and body["code"] == "unknown_member"


def test_default_idempotency_key_separates_pinned_from_unpinned():
    _, _, pinned, _ = _compat_post({"X-Backend": "Gemma-CPU"})
    _, _, plain, _ = _compat_post({"X-Backend": "Gemma"})
    assert pinned[0][0] != plain[0][0]


# --- handle_llm_proxy: member pins reach the broker, never _forward_llm --------

import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aiohttp.test_utils import make_mocked_request

from combined_gemma_broker_service import build_member_configs
from gemma_broker.config import BrokerConfig

_ROOT = Path(__file__).parents[2]
_CONTROLLER = _ROOT / "scripts" / "gpu-manager.py"


@pytest.fixture(scope="module")
def controller():
    spec = importlib.util.spec_from_file_location("candidate_gpm_member_pins", _CONTROLLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proxy_world(controller, monkeypatch, *, scoped: bool):
    """Controller with an enabled embedded broker and a CPU-only Gemma-CPU service."""
    monkeypatch.setattr(controller, "_services_config", {
        "services": {
            "Gemma-CPU": {
                "enabled": True, "type": "llm_backend", "port": 18099,
                "routing_group": "Gemma",
            },
        },
        "combined_gemma_broker": {"enabled": True, "ordered_members": list(MEMBERS)},
    })
    broker = MagicMock()
    broker.handle_proxy = AsyncMock(return_value=web.json_response({"job_id": "j1"}, status=202))
    forward = AsyncMock(return_value=web.json_response({"direct": True}))
    monkeypatch.setattr(controller, "_combined_gemma_broker", broker)
    monkeypatch.setattr(controller, "_combined_gemma_deployment_config", lambda: {"mode": "embedded"})
    monkeypatch.setattr(controller, "_forward_llm", forward)
    monkeypatch.setattr(controller, "_model_validator", None)
    monkeypatch.setattr(controller, "_scoped_gemma_request_is_authorized", lambda request: scoped)
    return broker, forward


class _Payload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _run_proxy(controller, request):
    return asyncio.run(controller.handle_llm_proxy(request))


def _patched_request(backend):
    request = make_mocked_request(
        "POST", "/v1/chat/completions",
        headers={"X-Backend": backend, "X-Client": "kda"},
    )
    body = json.dumps({"model": "gemma", "messages": []}).encode()
    return SimpleNamespace(
        path=request.path, method=request.method, headers=request.headers,
        query=request.query, can_read_body=True, app=request.app,
        read=_Payload(body).read,
    )


@pytest.mark.parametrize("backend", ["Gemma-CPU", "Gemma-MI50"])
def test_scoped_member_pin_becomes_broker_job_not_forward_llm(controller, monkeypatch, backend):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _patched_request(backend))
    assert response.status == 202
    forward.assert_not_called()
    kwargs = broker.handle_proxy.await_args.kwargs
    assert kwargs["headers"]["X-Backend"] == backend
    assert kwargs["headers"]["X-Client"] == "kda"


def test_admin_member_pin_keeps_direct_forward(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    response = _run_proxy(controller, _patched_request("Gemma-CPU"))
    assert response.status == 200
    broker.handle_proxy.assert_not_called()
    forward.assert_awaited_once()


def test_x_client_is_attribution_only_never_auth(tokens):
    # Claiming a privileged client neither admits a bad pin nor a wrong token.
    status, _ = _call("POST", "/v1/chat/completions", backend="Gemma-Unknown",
                      extra={"X-Client": "admin"})
    assert status == 401 or status == 403
    status, _ = _call("POST", "/v1/chat/completions", token="wrong", backend="Gemma-CPU",
                      extra={"X-Client": "kda"})
    assert status in (401, 403)


# --- F4: cloud members call the router's own routes, never its gemma route -----

def _router_member(model: str) -> dict:
    return {
        "enabled": True, "member_type": "openai_compatible",
        "endpoint": "http://127.0.0.1:18646/v1/chat/completions", "model": model,
        "forward_timeout": 900, "systemd_unit": "fleet-llm-router.service",
    }


def test_cloud_members_target_router_routes_that_are_not_gemma():
    block = json.loads((_ROOT / "examples/combined-gemma/services.json").read_text())
    config = BrokerConfig.from_dict(block["combined_gemma_broker"], candidate=True)
    services = dict(block["services"])
    cloud = {"Luna-6-Low": "gpt-6-luna", "Gemini-3.8-Flash-Low": "gemini-3.8-flash"}
    for name, model in zip(config.ordered_members[:2], cloud.values()):
        services[name] = _router_member(model)
    members, _ = build_member_configs(services, config)
    router_models = [
        members[name]["model"] for name in config.ordered_members[:2]
    ]
    assert router_models == list(cloud.values())
    # The router's "gemma" route points back at gpu-manager :8091; a member on it
    # would loop broker -> router -> broker.
    assert "gemma" not in router_models


# --- default pin: header-less scoped client (Hermes aux) becomes an unpinned Gemma-group job ---

DEFAULT_PIN = ("gemma-4-e4b", "Gemma")  # production since r188 (CPU member retired)
MEMBER_PIN = ("gemma-4-e4b", "Gemma-CPU")  # a member target stays supported


def _default_pin_call(method, path, *, body=None, pin=DEFAULT_PIN, backend=None, token=SCOPED,
                      query=""):
    async def go():
        app, seen = _gate_app()
        if pin is not None:
            app[KDA_GEMMA_DEFAULT_PIN_APP_KEY] = lambda: pin
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {token}"}
            if backend:
                headers["X-Backend"] = backend
            response = await client.request(method, path + query, headers=headers, json=body)
            return response.status, seen
    return asyncio.run(go())


def test_default_pin_admits_headerless_chat_only(tokens):
    status, seen = _default_pin_call("POST", "/v1/chat/completions", body=CHAT)
    assert status == 200 and seen[-1][2] is None
    # Still loopback-only / three routes / no query string / right token.
    assert _default_pin_call("GET", "/v1/models")[0] == 401
    assert _default_pin_call("GET", "/health")[0] == 401
    assert _default_pin_call("POST", "/v1/admin", body=CHAT)[0] == 401
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, query="?service=x")[0] == 401
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, token="wrong")[0] == 401
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, token=ADMIN)[0] == 200


def test_default_pin_needs_controller_config_and_real_member(tokens):
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, pin=None)[0] == 401
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT,
                             pin=("gemma-4-e4b", "Not-A-Member"))[0] == 401
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, pin=("", "Gemma-CPU"))[0] == 401


def test_routing_group_default_pin_needs_members_and_any_casing(tokens):
    for pin in (("gemma-4-e4b", "Gemma"), ("gemma-4-e4b", "gemma"), ("gemma-4-e4b", "GEMMA")):
        assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, pin=pin)[0] == 200
    # A member target still works, as before r188.
    assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, pin=MEMBER_PIN)[0] == 200

    async def no_members():
        app, _ = _gate_app(members=())
        app[KDA_GEMMA_DEFAULT_PIN_APP_KEY] = lambda: DEFAULT_PIN
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", json=CHAT,
                                         headers={"Authorization": f"Bearer {SCOPED}"})
            return response.status
    # With no broker members configured there is nothing to route to: fail closed.
    assert asyncio.run(no_members()) == 401


def test_controller_default_pin_targets_the_routing_group():
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[2] / "scripts/gpu-manager.py").read_text())
    values = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    pin = values["_SCOPED_DEFAULT_PIN"]
    assert isinstance(pin, ast.Tuple) and ast.literal_eval(pin.elts[0]) == "gemma-4-e4b"
    assert isinstance(pin.elts[1], ast.Name) and pin.elts[1].id == "_COMBINED_GEMMA_ROUTING_GROUP"
    assert ast.literal_eval(values["_COMBINED_GEMMA_ROUTING_GROUP"]) == "Gemma"


@pytest.mark.parametrize("model", ["gemma", "gemma-4-e4b"])
def test_headerless_scoped_member_default_pin_still_pins_that_member(controller, monkeypatch, model):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request(model, pin=MEMBER_PIN))
    assert response.status == 202
    assert broker.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == "Gemma-CPU"


def test_default_pin_does_not_widen_explicit_backend_rules(tokens):
    for backend in ("Gemma-Unknown", "Luna-6-Low", "gemma-cpu"):
        assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, backend=backend)[0] == 401
    for backend in ("Gemma", "Gemma-CPU", "Gemma-MI50"):
        assert _default_pin_call("POST", "/v1/chat/completions", body=CHAT, backend=backend)[0] == 200


def test_default_pin_non_loopback_peer_still_rejected(tokens):
    app, _ = _gate_app()
    app[KDA_GEMMA_DEFAULT_PIN_APP_KEY] = lambda: DEFAULT_PIN
    request = SimpleNamespace(
        app=app, remote="10.1.2.3", path="/v1/chat/completions", raw_path="/v1/chat/completions",
        query_string="", method="POST", headers={"Authorization": f"Bearer {SCOPED}"},
    )
    assert scoped_gemma_request_is_authorized(request) is False
    request.remote = "127.0.0.1"
    assert scoped_gemma_request_is_authorized(request) is True


def _headerless_request(model, *, pin=DEFAULT_PIN, raw=None, extra_headers=None):
    request = make_mocked_request(
        "POST", "/v1/chat/completions",
        headers={"X-Client": "hermes", **(extra_headers or {})},
    )
    app = {KDA_GEMMA_MEMBER_PINS_APP_KEY: lambda: frozenset(MEMBERS)}
    if pin is not None:
        app[KDA_GEMMA_DEFAULT_PIN_APP_KEY] = lambda: pin
    body = raw if raw is not None else json.dumps({"model": model, "messages": []}).encode()
    return SimpleNamespace(
        path=request.path, method=request.method, headers=request.headers,
        query=request.query, can_read_body=True, app=app, read=_Payload(body).read,
    )


@pytest.mark.parametrize("model", ["gemma", "gemma-4-E4B", "gemma-4-e4b"])
def test_headerless_scoped_default_model_becomes_gemma_group_broker_job(controller, monkeypatch, model):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request(model))
    assert response.status == 202
    forward.assert_not_called()
    kwargs = broker.handle_proxy.await_args.kwargs
    assert kwargs["headers"]["X-Backend"] == "Gemma"
    assert kwargs["headers"]["X-Client"] == "hermes"


@pytest.mark.parametrize("model", ["gemma", "gemma-4-e4b"])
def test_headerless_kda_gemma_reaches_real_broker_unpinned(controller, monkeypatch, model):
    """End to end: header-less KDA call -> controller default pin -> real compat -> real broker.

    Nothing between the controller and the broker is mocked: the group default pin
    must become an UNPINNED job, never a Gemma-CPU pin, and must not depend on which
    member is free.
    """
    _proxy_world(controller, monkeypatch, scoped=True)
    repo, _members, service = _service()
    compat_service = _CompatService(service)
    monkeypatch.setattr(controller, "_combined_gemma_broker",
                        CompatibilityWrapper(compat_service, timeout=5.0))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()
    request = _headerless_request(
        model, pin=controller._SCOPED_DEFAULT_PIN, raw=body,
        extra_headers={"X-Async": "1", "Prefer": "respond-async"},
    )
    response = _run_proxy(controller, request)
    assert response.status == 202, response.status
    assert compat_service.calls, "no submission reached the broker"
    assert all(pin is None for _key, pin in compat_service.calls), compat_service.calls
    job = repo.get(json.loads(response.body)["job_id"])
    assert job.pinned_member is None


@pytest.mark.parametrize("model", ["Gemma-CPU", "gpt-6-luna", "gemma-4-e4b ", "", None, 42, {}, []])
def test_headerless_scoped_other_models_are_rejected_not_routed(controller, monkeypatch, model):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request(model))
    assert response.status == 403
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"gemma-4-e4b"'])
def test_headerless_scoped_bad_body_is_rejected(controller, monkeypatch, raw):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request(None, raw=raw))
    assert response.status == 403
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


def test_headerless_scoped_without_default_pin_config_is_rejected(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request("gemma-4-e4b", pin=None))
    assert response.status == 403
    broker.handle_proxy.assert_not_called()


def test_headerless_scoped_never_falls_through_to_direct_when_broker_declines(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    monkeypatch.setattr(controller, "_combined_gemma_broker_enabled", lambda: False)
    response = _run_proxy(controller, _headerless_request("gemma-4-e4b"))
    assert response.status == 503
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


def test_explicit_x_backend_scoped_request_is_unchanged_by_default_pin(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    request = _headerless_request("gemma-4-e4b", extra_headers={"X-Backend": "Gemma"})
    response = _run_proxy(controller, request)
    assert response.status == 202
    assert broker.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == "Gemma"


# --- X-Backend: Gemma is an UNPINNED submission, so it must not skip the gate ---
#
# api_auth exempts the logical routing-group name from the scoped member check,
# and gemma_broker/compat.py maps any backend whose name casefolds to "gemma"
# back to pinned_member=None. So a scoped KDA credential sending
# "X-Backend: Gemma" with ANY body model was admitted, skipped the default-pin
# gate (which was guarded by "X-Backend not in request.headers") and reached
# handle_proxy as an unpinned, model-unchecked job. Such a request is gated
# exactly like a header-less one. Casing must not decide it: compat.py compares
# casefold(), so the controller must recognise the same group names.


@pytest.mark.parametrize("backend", ["Gemma", "gemma", "GEMMA"])
def test_scoped_logical_gemma_pin_cannot_bypass_model_scope(controller, monkeypatch, backend):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request("gpt-6-luna", extra_headers={"X-Backend": backend}))
    assert response.status == 403, (backend, response.status)
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


@pytest.mark.parametrize("backend", ["Gemma", "gemma", "GEMMA"])
@pytest.mark.parametrize("model", ["gemma", "gemma-4-E4B", "gemma-4-e4b"])
def test_scoped_logical_gemma_pin_is_pinned_for_the_in_scope_model(controller, monkeypatch, backend, model):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    response = _run_proxy(controller, _headerless_request(model, extra_headers={"X-Backend": backend}))
    assert response.status == 202, (backend, model, response.status)
    forward.assert_not_called()
    # In scope means the server-side default target (the Gemma routing group since
    # r188): unpinned, but only after the model scope check above passed.
    kwargs = broker.handle_proxy.await_args.kwargs
    assert kwargs["headers"]["X-Backend"] == "Gemma"
    assert kwargs["headers"]["X-Client"] == "hermes"


def test_scoped_real_member_and_unscoped_credential_are_not_narrowed(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=True)
    # A concrete member is an explicit, checked choice: any model still reaches it.
    member = _run_proxy(controller, _headerless_request("gpt-6-luna", extra_headers={"X-Backend": "Gemma-CPU"}))
    assert member.status == 202
    assert broker.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == "Gemma-CPU"
    broker.handle_proxy.reset_mock()
    # An unknown member is still 404, and lower-casing a real member name does not
    # turn it into the routing group.
    for backend in ("not-a-member", "gemma-cpu"):
        unknown = _run_proxy(controller, _headerless_request("gpt-6-luna", extra_headers={"X-Backend": backend}))
        assert unknown.status == 404, (backend, unknown.status)
    broker.handle_proxy.assert_not_called()

    # An unscoped/admin credential keeps the logical group and any model, whatever
    # the casing -- the routing-group name is recognised the same way for everyone.
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    for backend in ("Gemma", "gemma", "GEMMA"):
        broker.handle_proxy.reset_mock()
        response = _run_proxy(controller, _headerless_request("gpt-6-luna", extra_headers={"X-Backend": backend}))
        assert response.status == 202, (backend, response.status)
        assert broker.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == backend


def test_unscoped_headerless_request_never_gets_default_pin(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    _run_proxy(controller, _headerless_request("gemma-4-e4b"))
    for call in broker.handle_proxy.await_args_list:
        assert call.kwargs["headers"].get("X-Backend") not in ("Gemma-CPU", "Gemma")


# --- Honcho: logical Gemma ingress remains unpinned and fails closed ---

@pytest.mark.parametrize("backend", [None, "Gemma", "gemma", "GEMMA"])
def test_honcho_gemma_reaches_logical_broker_without_member_pin(controller, monkeypatch, backend):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    monkeypatch.setattr(controller, "_honcho_gemma_request_is_authorized", lambda request, body=None: body is not None)
    request = _headerless_request("gemma", extra_headers={"X-Backend": backend} if backend else None)
    response = _run_proxy(controller, request)
    assert response.status == 202
    forward.assert_not_called()
    kwargs = broker.handle_proxy.await_args.kwargs
    assert kwargs["headers"].get("X-Backend") is None
    assert json.loads(kwargs["body"])["model"] == "gemma"


@pytest.mark.parametrize("model", ["gemma-4-e4b", "gpt-6-luna", ""])
def test_honcho_out_of_scope_never_submits_or_routes(controller, monkeypatch, model):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    monkeypatch.setattr(controller, "_honcho_gemma_request_is_authorized", lambda request, body=None: False)
    monkeypatch.setattr(controller, "_scoped_gemma_request_is_authorized", lambda request: False)
    response = _run_proxy(controller, _headerless_request(model))
    assert response.status != 202
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


def test_honcho_broker_decline_returns_503_without_direct_fallback(controller, monkeypatch):
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    monkeypatch.setattr(controller, "_honcho_gemma_request_is_authorized", lambda request, body=None: True)
    monkeypatch.setattr(controller, "_combined_gemma_broker_enabled", lambda: False)
    response = _run_proxy(controller, _headerless_request("gemma"))
    assert response.status == 503
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


def _honcho_controller_app(controller, monkeypatch):
    """Run controller routes behind the real API-auth middleware and fake broker."""
    broker, forward = _proxy_world(controller, monkeypatch, scoped=False)
    ingress_calls = []
    original_ingress = controller._maybe_handle_combined_gemma_ingress

    async def capture_ingress(**kwargs):
        ingress_calls.append(kwargs)
        return await original_ingress(**kwargs)

    monkeypatch.setattr(controller, "_maybe_handle_combined_gemma_ingress", capture_ingress)
    app = web.Application(middlewares=[api_auth_middleware])
    app[KDA_GEMMA_SCOPED_GATE_APP_KEY] = lambda: True
    app[KDA_GEMMA_MEMBER_PINS_APP_KEY] = lambda: frozenset(MEMBERS)
    app.router.add_post("/v1/chat/completions", controller.handle_llm_proxy)
    app.router.add_get("/health", controller.handle_health)
    return app, broker, forward, ingress_calls


def _honcho_controller_request(controller, monkeypatch, *, path="/v1/chat/completions",
                               model="gemma", backend=None, raw_body=None):
    async def go():
        app, broker, forward, ingress_calls = _honcho_controller_app(controller, monkeypatch)
        headers = {"Authorization": f"Bearer {HONCHO}"}
        if backend is not None:
            headers["X-Backend"] = backend
        body = raw_body if raw_body is not None else json.dumps(
            {"model": model, "messages": []}
        ).encode()
        async with TestClient(TestServer(app)) as client:
            response = await client.request(
                "GET" if path == "/health" else "POST", path,
                data=None if path == "/health" else body,
                headers=headers,
            )
            return response.status, await response.json(), broker, forward, ingress_calls
    return asyncio.run(go())


@pytest.mark.parametrize("model,backend", [
    ("gemma", None),
    ("Gemma", "Gemma"),
    ("GeMmA", "GEMMA"),
])
def test_honcho_real_middleware_routes_logical_gemma_unpinned(controller, monkeypatch, tokens,
                                                              model, backend):
    status, body, broker, forward, ingress_calls = _honcho_controller_request(
        controller, monkeypatch, model=model, backend=backend
    )
    assert status == 202, body
    forward.assert_not_called()
    assert len(ingress_calls) == 1
    assert ingress_calls[0]["service_name"] == "Gemma"
    assert ingress_calls[0]["allow_member_pin"] is False
    kwargs = broker.handle_proxy.await_args.kwargs
    assert kwargs["headers"].get("X-Backend") is None
    assert json.loads(kwargs["body"])["model"] == model


@pytest.mark.parametrize("raw_body,backend", [
    (b"{", None),
    (b'{"model":"gemma-4-e4b","messages":[]}', None),
    (b'{"model":"gpt-6-luna","messages":[]}', None),
    (b'{"model":"gemma","messages":[]}', "Gemma-CPU"),
    (b'{"model":"gemma","messages":[]}', "Gemma-MI50"),
])
def test_honcho_real_middleware_rejects_bad_body_and_member_pin_before_submission(
    controller, monkeypatch, tokens, raw_body, backend
):
    status, _, broker, forward, ingress_calls = _honcho_controller_request(
        controller, monkeypatch, raw_body=raw_body, backend=backend
    )
    assert status == 401
    assert ingress_calls == []
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


def test_honcho_real_middleware_health_is_sanitized(controller, monkeypatch, tokens):
    monkeypatch.setattr(
        controller, "_CombinedGemmaAPIProxy",
        lambda: SimpleNamespace(health_ready=lambda: ({"private": "omitted"}, 200)),
    )
    status, body, _, _, ingress_calls = _honcho_controller_request(
        controller, monkeypatch, path="/health"
    )
    assert status == 200 and body == {"status": "ok"}
    assert ingress_calls == []


def test_honcho_real_middleware_broker_unavailable_returns_503_without_fallback(
    controller, monkeypatch, tokens
):
    monkeypatch.setattr(controller, "_combined_gemma_broker_enabled", lambda: False)
    status, body, broker, forward, ingress_calls = _honcho_controller_request(
        controller, monkeypatch
    )
    assert status == 503 and body["code"] == "combined_gemma_broker_unavailable"
    assert len(ingress_calls) == 1 and ingress_calls[0]["allow_member_pin"] is False
    broker.handle_proxy.assert_not_called()
    forward.assert_not_called()


# --- controller broker metrics route: real create_app router + auth middleware ---

BROKER_METRICS = {
    "members": [{
        "name": "Gemma-CPU", "state": "ready_accepting", "configured_slots": 2,
    }],
    "source": "synthetic-test",
}


def _real_controller_app(controller, monkeypatch, tmp_path):
    """Build the actual controller routes with isolated config and fake Redis."""
    services_path = tmp_path / "services.json"
    services_path.write_text(json.dumps(controller.empty_services_config()), encoding="utf-8")
    providers_path = tmp_path / "providers.json"
    providers_path.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    monkeypatch.setattr(controller, "SERVICES_CONFIG_PATH", str(services_path))
    monkeypatch.setattr(controller, "PROVIDERS_CONFIG_PATH", str(providers_path))
    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    fake_redis.connection_pool.connection_kwargs = {"decode_responses": True}
    monkeypatch.setattr(controller._redis_mod, "Redis", MagicMock(return_value=fake_redis))
    app = controller.create_app()
    # Exercise the real auth middleware with the synthetic gate enabled; the
    # empty services fixture otherwise correctly disables Honcho admission.
    app[controller._KDA_GEMMA_SCOPED_GATE_APP_KEY] = lambda: True
    app[controller._KDA_GEMMA_MEMBER_PINS_APP_KEY] = lambda: frozenset(MEMBERS)
    app.on_startup.clear()
    app.on_cleanup.clear()
    return app


async def _real_route_request(app, path, token=HONCHO):
    async with TestClient(TestServer(app)) as client:
        response = await client.get(path, headers={"Authorization": f"Bearer {token}"})
        return response.status, await response.json()


@pytest.mark.parametrize("query", ["?durable=0", ""])
def test_honcho_controller_broker_metrics_route_feeds_real_compute_capacity(
    controller, monkeypatch, tokens, tmp_path, query
):
    metrics_path = Path("/home/user/docker-services/honcho/src/src/deriver/adaptive_concurrency.py")
    if not metrics_path.is_file():
        pytest.skip(f"Honcho adaptive_concurrency.py is absent: {metrics_path}")
    app = _real_controller_app(controller, monkeypatch, tmp_path)
    proxy = MagicMock()
    proxy.broker_metrics.return_value = BROKER_METRICS
    monkeypatch.setattr(controller, "_combined_gemma_api_proxy", proxy)
    status, body = asyncio.run(_real_route_request(app, "/v1/metrics/broker" + query))
    assert status == 200
    assert body["members"] == BROKER_METRICS["members"]
    assert "durable_counts" not in body and "durable" not in body

    spec = importlib.util.spec_from_file_location("honcho_adaptive_concurrency_readonly", metrics_path)
    assert spec and spec.loader
    honcho_metrics = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = honcho_metrics
    try:
        spec.loader.exec_module(honcho_metrics)
    finally:
        sys.modules.pop(spec.name, None)
    capacity, breakdown = honcho_metrics.compute_capacity(body, minimum=1, maximum=8)
    assert capacity == 2
    assert breakdown == {"Gemma-CPU": 2}
    proxy.broker_metrics.assert_called_once_with()


def test_honcho_broker_metrics_rejections_happen_before_proxy(controller, monkeypatch, tokens, tmp_path):
    app = _real_controller_app(controller, monkeypatch, tmp_path)
    proxy = MagicMock()
    proxy.broker_metrics.return_value = BROKER_METRICS
    monkeypatch.setattr(controller, "_combined_gemma_api_proxy", proxy)

    async def exercise():
        async with TestClient(TestServer(app)) as client:
            def headers(token):
                return {"Authorization": f"Bearer {token}"}
            for path in (
                "/v1/metrics/broker?foo=1",
                "/v1/metrics/broker?durable=0&durable=0",
                "/v1/metrics/broker?durable=1",
            ):
                response = await client.get(path, headers=headers(HONCHO))
                assert response.status == 401
            response = await client.get("/v1/metrics/broker", headers=headers(SCOPED))
            assert response.status == 403
        proxy.broker_metrics.assert_not_called()

    asyncio.run(exercise())


def test_honcho_broker_metrics_proxy_runs_off_event_loop(controller, monkeypatch, tokens, tmp_path):
    app = _real_controller_app(controller, monkeypatch, tmp_path)
    loop_thread = __import__("threading").get_ident()
    observed = []

    class Proxy:
        def broker_metrics(self):
            observed.append(__import__("threading").get_ident())
            return BROKER_METRICS

    monkeypatch.setattr(controller, "_combined_gemma_api_proxy", Proxy())
    status, body = asyncio.run(_real_route_request(app, "/v1/metrics/broker?durable=0"))
    assert status == 200 and body["members"] == BROKER_METRICS["members"]
    assert observed and observed[0] != loop_thread


def test_honcho_broker_metrics_repository_unavailable_is_503_json(controller, monkeypatch, tokens, tmp_path):
    app = _real_controller_app(controller, monkeypatch, tmp_path)
    proxy = MagicMock()
    # Raise the message the REAL repository raises. The wire code is a stable
    # contract string, never the internal ReasonCode carried by the exception.
    proxy.broker_metrics.side_effect = RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
    monkeypatch.setattr(controller, "_combined_gemma_api_proxy", proxy)
    status, body = asyncio.run(_real_route_request(app, "/v1/metrics/broker?durable=0"))
    assert status == 503
    assert body == {"error": "repository unavailable", "code": "redis_state_unknown"}
    assert "traceback" not in json.dumps(body).lower()
    proxy.broker_metrics.assert_called_once_with()


def test_honcho_broker_metrics_absent_embedded_broker_is_typed_503(controller, monkeypatch, tokens, tmp_path):
    """No embedded runtime is a typed 503, not a 500 with a stack trace."""
    app = _real_controller_app(controller, monkeypatch, tmp_path)
    # The real proxy against no initialised runtime raises APIError(503).
    monkeypatch.setattr(controller, "_combined_gemma_broker", None)
    status, body = asyncio.run(_real_route_request(app, "/v1/metrics/broker?durable=0"))
    assert status == 503
    assert body["error"] == "repository unavailable"
    assert "traceback" not in json.dumps(body).lower()


def test_controller_metrics_sibling_routes_keep_their_handlers(controller, monkeypatch, tokens, tmp_path):
    async def metrics(_request):
        return web.json_response({"kind": "base"})

    async def summary(_request):
        return web.json_response({"kind": "summary"})

    async def service(request):
        return web.json_response({"kind": "service", "service": request.match_info["service"]})

    monkeypatch.setattr(controller, "handle_get_metrics", metrics)
    monkeypatch.setattr(controller, "handle_get_metrics_summary", summary)
    monkeypatch.setattr(controller, "handle_get_metrics_service", service)
    app = _real_controller_app(controller, monkeypatch, tmp_path)

    async def exercise():
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {ADMIN}"}
            expected = {
                "/v1/metrics": {"kind": "base"},
                "/v1/metrics/summary": {"kind": "summary"},
                "/v1/metrics/example": {"kind": "service", "service": "example"},
            }
            for path, body in expected.items():
                response = await client.get(path, headers=headers)
                assert response.status == 200 and await response.json() == body

    asyncio.run(exercise())


# ── broker 2.6: head-of-line dispatch window + compat orphan cancel ──────────

def _full(member):
    """Snapshot of a member whose only slot is leased."""
    return replace(member, compatible_free_slots=0, dispatcher_leases=1)


def _req_for(job):
    return RequestRequirements(8, 16, ("chat",), "t", pinned_member=job.pinned_member)


def _cpu_full_setup():
    """Gemma-CPU has its single slot leased; Gemma-MI50 is free."""
    repo, members, service = _service()
    service.submit({"idempotency_key": "filler", "request": CHAT, "pinned_member": "Gemma-CPU"})
    filler = repo.reserve_next(
        now=2.0, ordered_members=members,
        requirements=RequestRequirements(8, 16, ("chat",), "t", pinned_member="Gemma-CPU"),
    )
    assert filler is not None and repo.lease_count("Gemma-CPU") == 1
    return repo, members, service


def _submit_pinned(service, key, member):
    return service.submit({"idempotency_key": key, "request": CHAT, "pinned_member": member})


def _submit_free(service, key):
    return service.submit({"idempotency_key": key, "request": CHAT})


def _dispatch_loop(repo, members):
    loop = _idle_dispatch_loop(repo, lambda: members, lambda: 2.0)
    loop._fencing_token = repo.acquire_leadership(
        owner="test-dispatcher", now=2.0, ttl_seconds=30.0
    )
    return loop


def test_full_pinned_head_does_not_block_dispatch_for_free_member():
    repo, members, service = _cpu_full_setup()
    head = _submit_pinned(service, "hol-head", "Gemma-CPU")
    _submit_free(service, "hol-behind")
    assert repo.peek_next(now=2.0).job_id == head.job_id
    assert _dispatch_loop(repo, members)._has_dispatchable_head() is True


def test_reserve_pass_reserves_unpinned_job_behind_full_pinned_head():
    repo, members, service = _cpu_full_setup()
    head = _submit_pinned(service, "hol-head", "Gemma-CPU")
    behind = _submit_free(service, "hol-behind")
    cpu, mi50 = members
    reservation = repo.reserve_next(
        now=2.0,
        ordered_members=[_full(cpu), mi50],
        requirements=RequestRequirements(8, 16, ("chat",), "t"),
        requirements_for=_req_for,
    )
    assert reservation is not None
    assert reservation.job.job_id == behind.job_id
    assert reservation.member.name == "Gemma-MI50"
    assert repo.lease_count("Gemma-MI50") == 1
    assert repo.get(head.job_id).state is JobState.QUEUED


def test_pinned_head_for_full_member_is_not_reserved_and_keeps_its_place():
    repo, members, service = _cpu_full_setup()
    head = _submit_pinned(service, "hol-head", "Gemma-CPU")
    cpu, mi50 = members
    reservation = repo.reserve_next(
        now=2.0,
        ordered_members=[_full(cpu), mi50],
        requirements=_req_for(head),
        requirements_for=_req_for,
    )
    assert reservation is None
    assert repo.queue_total == 1
    assert repo.peek_next(now=2.0).job_id == head.job_id


def test_dispatch_window_stops_at_64_queued_jobs():
    repo, members, service = _cpu_full_setup()
    for index in range(64):
        _submit_pinned(service, f"blocked-{index}", "Gemma-CPU")
    beyond = _submit_free(service, "beyond-window")  # queue position 65
    assert _dispatch_loop(repo, members)._has_dispatchable_head() is False
    cpu, mi50 = members
    assert repo.reserve_next(
        now=2.0,
        ordered_members=[_full(cpu), mi50],
        requirements=RequestRequirements(8, 16, ("chat",), "t"),
        requirements_for=_req_for,
    ) is None
    assert repo.get(beyond.job_id).state is JobState.QUEUED


def test_dispatch_window_includes_job_at_position_64():
    repo, members, service = _cpu_full_setup()
    for index in range(63):
        _submit_pinned(service, f"blocked-{index}", "Gemma-CPU")
    _submit_free(service, "last-in-window")  # queue position 64
    assert _dispatch_loop(repo, members)._has_dispatchable_head() is True


def _compat(repo, service, *, timeout, adapter_cls=_CandidateCompatAdapter):
    from gemma_broker.compat import CompatibilityWrapper

    polling = PollingTerminalWait(repository=repo, clock=time.monotonic)
    return CompatibilityWrapper(adapter_cls(service, polling_wait=polling), timeout=timeout)


def _compat_call(wrapper):
    return wrapper.handle_proxy(
        method="POST", path="/v1/chat/completions",
        body=json.dumps(CHAT).encode(), headers={},
    )


def test_compat_timeout_cancels_its_queued_job():
    repo, members, service = _service()
    response = asyncio.run(_compat_call(_compat(repo, service, timeout=0.05)))
    assert response.status == 504
    assert repo.queue_total == 0
    assert repo.get(response.headers["X-Job-ID"]).state is JobState.CANCELLED


def test_compat_client_disconnect_cancels_its_queued_job():
    repo, members, service = _service()
    wrapper = _compat(repo, service, timeout=30.0)

    async def disconnect_mid_wait():
        task = asyncio.ensure_future(_compat_call(wrapper))
        for _ in range(2000):
            if repo.queue_total == 1:
                break
            await asyncio.sleep(0.001)
        assert repo.queue_total == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return repo.queue_total

    assert asyncio.run(disconnect_mid_wait()) == 0


def test_compat_broker_503_cancels_its_queued_job():
    from gemma_broker.compat import CompatibilityWrapper

    class _UnavailableWait(_CandidateCompatAdapter):
        async def wait_for_terminal(self, job_id, *, timeout):
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)

    repo, members, service = _service()
    polling = PollingTerminalWait(repository=repo, clock=time.monotonic)
    wrapper = CompatibilityWrapper(_UnavailableWait(service, polling_wait=polling), timeout=5.0)
    response = asyncio.run(_compat_call(wrapper))
    assert response.status == 503
    assert repo.queue_total == 0
