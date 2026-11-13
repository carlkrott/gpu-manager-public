"""Strict-red regressions for broker 503 suppression and pinned admission."""
from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from gemma_broker.api import APIError, DurableBrokerService
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobRecord, JobState, MemberSnapshot, MemberState, ReasonCode
from gemma_broker.dispatcher import Dispatcher
from gemma_broker.member_cache import StrictMemberSnapshotCache
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import BrokerRuntime, DispatchLoop
from gemma_broker.member_observer import MemberObserver
from gemma_broker.readiness import reduce_member_readiness

CHAT = {"messages": [{"role": "user", "content": "ping"}]}


def _member(name: str, *, state=MemberState.READY_ACCEPTING, capabilities=("chat",)):
    return MemberSnapshot(
        name=name,
        gpu_id=None,
        state=state,
        accepting=state is MemberState.READY_ACCEPTING,
        configured_slots=1,
        context_per_slot=4096,
        state_version=1,
        observed_at=1.0,
        compatible_free_slots=1,
        capabilities=capabilities,
    )


def _idle_dispatch_loop(repository, members_provider, clock):
    class _UnusedTransport:
        async def send(self, *_args):
            raise AssertionError("no dispatch should reach transport")

    transport = _UnusedTransport()
    dispatcher = Dispatcher(
        repo=repository,
        transport=transport,
        request_timeout=1.0,
        clock=clock,
    )
    return DispatchLoop(
        repository=repository,
        transport=transport,
        members_provider=members_provider,
        dispatcher=dispatcher,
        owner="test-dispatcher",
        leader_ttl_seconds=30.0,
        heartbeat_seconds=1.0,
        clock=clock,
        requirements_factory=lambda _body: RequestRequirements(8, 16, ("chat",), "test"),
    )


@pytest.mark.parametrize(
    "response_body",
    [
        "service unavailable",
        '{"error":{"code":"RESOURCE_EXHAUSTED","message":"quota exceeded for this model"}}',
    ],
    ids=["generic-503", "quota-bearing-503"],
)
def test_dispatch_503_suppression_survives_embedded_successful_readiness_refresh(response_body):
    """Run a real 503 dispatch, then the embedded readiness writer, under a controlled clock."""
    now = [100.0]
    member = _member("Gemini-3.8-Flash-Low")
    repository = InMemoryJobRepository()
    repository.put_member(member)
    repository.submit(
        JobRecord(
            job_id="dispatch-503",
            idempotency_key="dispatch-503-idempotency",
            request_sha256="dispatch-503-sha",
            request_body={**CHAT, "model": "gemini", "max_tokens": 16},
            submitted_at=now[0],
            enqueue_sequence=1,
        ),
        caller_scope="test",
    )

    class _Http503Transport:
        async def send(self, _member, _body, _on_accepted):
            raise RuntimeError(f"TRANSPORT_HTTP_503:{response_body}")

    dispatcher = Dispatcher(
        repo=repository,
        transport=_Http503Transport(),
        request_timeout=1.0,
        clock=lambda: now[0],
    )
    loop = DispatchLoop(
        repository=repository,
        transport=_Http503Transport(),
        members_provider=lambda: [member],
        dispatcher=dispatcher,
        owner="test-dispatcher",
        leader_ttl_seconds=30.0,
        heartbeat_seconds=1.0,
        clock=lambda: now[0],
        requirements_factory=lambda _body: RequestRequirements(8, 16, ("chat",), "test"),
    )
    loop._fencing_token = repository.acquire_leadership(
        owner="test-dispatcher", now=now[0], ttl_seconds=30.0
    )

    # Exercise the actual GPU Manager embedded-refresh caller and durable member write.
    controller_path = Path(__file__).resolve().parents[2] / "scripts" / "gpu-manager.py"
    spec = importlib.util.spec_from_file_location("test_503_embedded_refresh", controller_path)
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    member_config = {
        "enabled": True,
        "member_type": "openai_compatible",
        "endpoint": "http://provider.test/v1/chat/completions",
        "health_url": "http://provider.test/health/cloud",
        "models_url": "http://provider.test/v1/models",
        "model": "synthetic-gemini-model",
        "parallel": 1,
        "context_per_slot": 4096,
        "cpu_only": True,
        "capabilities": ("chat", "completion"),
    }
    cache = StrictMemberSnapshotCache({member.name: member_config}, freshness_ms=5_000)

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

    class _HealthySession:
        closed = False

        def get(self, url, **_kwargs):
            if url.endswith("/health/cloud"):
                return _Response({"status": "ok"})
            if url.endswith("/v1/models"):
                return _Response({"data": [{"id": member_config["model"]}]})
            raise AssertionError(url)

    async def _no_local_systemd(_config):
        return {"systemd_active": False, "main_pid": None}

    controller._combined_gemma_broker_embedded = lambda: True
    controller._combined_gemma_member_cache = cache
    controller._combined_gemma_broker = SimpleNamespace(
        dispatch_loop=loop,
        _repository=repository,
    )
    controller._services_config = {
        "combined_gemma_broker": {"enabled": True, "ordered_members": [member.name]},
        "services": {member.name: member_config},
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    controller._combined_gemma_refresh_logged_state = {}
    controller._combined_gemma_systemd_facts = _no_local_systemd
    controller.session = _HealthySession()
    controller.time = SimpleNamespace(time=lambda: now[0])
    loop._task = SimpleNamespace(done=lambda: False)

    async def scenario():
        assert await loop._dispatch_one()
        failed = repository.get("dispatch-503")
        assert failed is not None and failed.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE
        suppression = loop.active_dispatch_suppression(member.name)
        assert suppression is not None
        expected_kind = "quota_wall" if response_body.startswith("{") else "preaccept_cooldown"
        assert suppression["kind"] == expected_kind
        cooldown_until = suppression["deadline_monotonic"]
        await controller._refresh_combined_gemma_member_snapshots()
        refreshed_during_cooldown = repository.get_member(member.name)
        now[0] = cooldown_until
        await controller._refresh_combined_gemma_member_snapshots()
        refreshed_after_expiry = repository.get_member(member.name)

        assert refreshed_after_expiry is not None
        assert refreshed_after_expiry.state is MemberState.READY_ACCEPTING
        assert refreshed_after_expiry.accepting
        assert refreshed_during_cooldown is not None
        assert (
            refreshed_during_cooldown.state is not MemberState.READY_ACCEPTING
            and not refreshed_during_cooldown.accepting
        ), (
            "successful readiness refresh republished a 503-suppressed member as "
            f"{refreshed_during_cooldown.state.value}/accepting={refreshed_during_cooldown.accepting} "
            f"before cooldown expiry (response={response_body!r})"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "member",
    [
        _member("Gemma-CPU", state=MemberState.UNHEALTHY),
        _member("Gemma-CPU", state=MemberState.DRAINING),
    ],
    ids=["unhealthy", "draining"],
)
def test_pinned_undispatchable_member_is_rejected_before_enqueue_within_bound(member):
    """Pins that cannot dispatch now should not sit queued until a client timeout."""
    service = DurableBrokerService(
        InMemoryJobRepository(), members=lambda: [member], clock=lambda: 2.0
    )

    async def submit_and_require_rejection():
        with pytest.raises(APIError) as caught:
            service.submit({
                "idempotency_key": "pin-undispatchable",
                "request": CHAT,
                "pinned_member": "Gemma-CPU",
            })
        assert caught.value.status == 503
        assert caught.value.code == "pinned_member_unavailable"

    asyncio.run(asyncio.wait_for(submit_and_require_rejection(), timeout=1.0))


def test_pinned_missing_route_is_cleanly_rejected():
    service = DurableBrokerService(
        InMemoryJobRepository(),
        members=lambda: [_member("Gemma-CPU", capabilities=())],
        clock=lambda: 2.0,
    )
    with pytest.raises(APIError) as caught:
        service.submit({
            "idempotency_key": "pin-no-route",
            "request": CHAT,
            "pinned_member": "Gemma-CPU",
        })
    assert caught.value.status == 503 and caught.value.code == "pinned_member_unavailable"


def test_ready_busy_parallel_one_pin_remains_queued_normally():
    """Capacity saturation differs from an unavailable member: preserve ordinary queueing."""
    member = _member("Gemma-CPU")
    repository = InMemoryJobRepository()
    service = DurableBrokerService(repository, members=lambda: [member], clock=lambda: 2.0)
    job = service.submit({
        "idempotency_key": "pin-busy-control",
        "request": CHAT,
        "pinned_member": "Gemma-CPU",
    })
    waiting = service.submit({
        "idempotency_key": "pin-busy-control-2",
        "request": CHAT,
        "pinned_member": "Gemma-CPU",
    })
    assert job.pinned_member == waiting.pinned_member == "Gemma-CPU"
    assert job.state.value == waiting.state.value == "queued"
    requirements = RequestRequirements(8, 16, ("chat",), "test", pinned_member="Gemma-CPU")
    reservation = repository.reserve_next(now=3.0, ordered_members=[member], requirements=requirements)
    assert reservation is not None
    from gemma_broker.redis_store import ReservationStale

    with pytest.raises(ReservationStale):
        repository.reserve_next(now=3.0, ordered_members=[member], requirements=requirements)


@pytest.mark.parametrize("wall_kind", ["preaccept-cooldown", "quota-wall"])
def test_pinned_wall_member_is_rejected_before_enqueue_within_bound(wall_kind):
    """A live dispatch cooldown or quota wall must be visible to pinned API admission."""
    now = [2.0]
    member = _member("Gemini-3.8-Flash-Low")
    repository = InMemoryJobRepository()
    repository.put_member(member)
    loop = _idle_dispatch_loop(repository, lambda: [member], lambda: now[0])
    if wall_kind == "preaccept-cooldown":
        loop._record_preaccept_failure(member.name)
    else:
        loop._record_quota_wall(member.name, "gemini_resource_exhausted", 900)
    assert loop._members_eligible([member]) == []

    service = DurableBrokerService(repository, members=lambda: [member], clock=lambda: now[0])
    service.member_admission = lambda name: any(
        candidate.name == name for candidate in loop._members_eligible([member])
    )

    async def submit_and_require_rejection():
        with pytest.raises(APIError) as caught:
            service.submit({
                "idempotency_key": f"pin-{wall_kind}",
                "request": CHAT,
                "pinned_member": member.name,
            })
        assert caught.value.status == 503
        assert caught.value.code == "pinned_member_unavailable"

    asyncio.run(asyncio.wait_for(submit_and_require_rejection(), timeout=1.0))


@pytest.mark.parametrize("member_remains", [True, False], ids=["unhealthy-member", "zero-members"])
def test_readiness_loss_after_enqueue_terminalizes_without_reservation(member_remains):
    """Once a queued pin loses readiness, return a terminal reason, not a silent timeout."""
    now = [2.0]
    member = _member("Gemma-CPU")
    members = [member]
    repository = InMemoryJobRepository()
    repository.put_member(member)
    service = DurableBrokerService(repository, members=lambda: members, clock=lambda: now[0])
    job = service.submit({
        "idempotency_key": "pin-loses-readiness",
        "request": CHAT,
        "pinned_member": member.name,
    })

    unhealthy = replace(member, state=MemberState.UNHEALTHY, accepting=False)
    repository.put_member(unhealthy)
    members[:] = [unhealthy] if member_remains else []
    loop = _idle_dispatch_loop(repository, lambda: members, lambda: now[0])
    loop._fencing_token = 1

    async def dispatch_and_require_terminal():
        assert not await loop._dispatch_one()
        current = repository.get(job.job_id)
        assert current is not None
        assert current.state is JobState.FAILED, (
            "readiness loss left pinned request nonterminal: "
            f"state={current.state.value}, reason={current.reason_code}"
        )
        assert current.reason_code is ReasonCode.NO_ELIGIBLE_MEMBER
        assert current.current_attempt_id is None and current.attempt_count == 0
        assert repository.lease_count(member.name) == 0

    asyncio.run(asyncio.wait_for(dispatch_and_require_terminal(), timeout=1.0))


@pytest.mark.parametrize("limit", [5], ids=["bounded"])
@pytest.mark.parametrize(
    "requeues_done",
    [1, 4, 5, 101],
    ids=["fresh", "near_limit", "at_limit", "over_limit"],
)
def test_pinned_pre_acceptance_requeue_bounded_by_gate_limit(limit, requeues_done):
    """Pinned job with N >= limit pre-acceptance requeues must terminalize, not loop forever.

    Regression for live job a85d1013 (101 attempts, pinned Gemini-3.8-Flash-Low, 502/503).

    Drives repository.requeue_claim() DIRECTLY with the job in CLAIMED state (matches the
    real dispatcher path that calls requeue_claim after a 502/503); bypasses
    _fail_unavailable_pinned_head which only fires when member goes UNHEALTHY at dispatch.

    The pre_acceptance_requeue_count field is seeded directly to simulate prior cycles;
    the test covers the requeue-exhaustion branch in requeue_claim regardless of how
    attempt_count is managed upstream.
    """
    from scripts.gemma_broker.gate_limits import BrokerGateLimits

    now = [2.0]
    member_name = "Gemini-3.8-Flash-Low"
    member = _member(member_name)
    members = [member]
    repository = InMemoryJobRepository(
        gate_limits=BrokerGateLimits(pre_acceptance_requeue_limit=limit)
    )
    repository.put_member(member)
    service = DurableBrokerService(
        repository, members=lambda: members, clock=lambda: now[0]
    )
    job = service.submit(
        {
            "idempotency_key": f"pin-requeue-limit-{requeues_done}",
            "request": CHAT,
            "pinned_member": member_name,
        }
    )

    # Transition QUEUED -> CLAIMED via reserve_next (pattern at line 270-271).
    # Without this, the state guard at requeue_claim raises
    # ContractError("REQUEUE_INVALID_STATE:queued") before the exhaustion
    # check runs (redis_store.py:785).
    requirements = RequestRequirements(
        8, 16, ("chat",), "test", pinned_member=member_name
    )
    reservation = repository.reserve_next(
        now=now[0], ordered_members=[member], requirements=requirements
    )
    assert reservation is not None, "reserve_next must succeed for a healthy pinned member"
    claimed = repository.get(job.job_id)
    assert claimed.state is JobState.CLAIMED, (
        f"expected CLAIMED, got {claimed.state.value}"
    )

    # Seed pre_acceptance_requeue_count to simulate prior attempts.
    repository.replace_job(
        replace(claimed, pre_acceptance_requeue_count=requeues_done)
    )

    # Drive requeue_claim once to exercise the exhaustion path.
    result = repository.requeue_claim(
        repository.get(job.job_id),
        reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
    )
    current = repository.get(job.job_id)
    assert current is not None

    if requeues_done >= limit:
        assert current.state is JobState.FAILED, (
            f"Expected terminal FAILED after {requeues_done} requeues (limit={limit}), "
            f"got {current.state.value} reason={current.reason_code}"
        )
        assert current.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE_EXHAUSTED, (
            f"expected PRE_ACCEPTANCE_REQUEUE_EXHAUSTED, got {current.reason_code}"
        )
        # Queue slot must be freed: job no longer in any priority queue.
        for prio_queue in repository._queues.values():
            assert current.job_id not in prio_queue, (
                "terminalized job must not occupy a queue slot"
            )
    else:
        assert current.state is JobState.QUEUED, (
            f"Expected QUEUED, got {current.state.value}"
        )
        assert current.pre_acceptance_requeue_count == requeues_done + 1


_AUDIO_SERVICES = (
    ("apollo-audio-restoration", "apollo-restoration-gen"),
    ("resemble-enhance", "resemble-enhance-gen"),
    ("audio-separator-rocm", "audio-separation-gen"),
)


@pytest.mark.parametrize("service_name,routing_group", _AUDIO_SERVICES, ids=lambda s: s)
def test_audio_routing_group_is_in_legacy_generation_whitelist(service_name, routing_group):
    controller_path = Path(__file__).resolve().parents[2] / "scripts" / "gpu-manager.py"
    spec = importlib.util.spec_from_file_location("test_audio_ingress_whitelist", controller_path)
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    assert routing_group in controller._GENERATION_ROUTING_GROUPS, (
        f"{service_name} routing group {routing_group!r} missing from legacy whitelist"
    )


@pytest.mark.parametrize("service_name,routing_group", _AUDIO_SERVICES, ids=lambda s: s)
def test_audio_routing_group_is_admitted_when_registry_has_no_declared_groups(
    service_name, routing_group
):
    controller_path = Path(__file__).resolve().parents[2] / "scripts" / "gpu-manager.py"
    spec = importlib.util.spec_from_file_location("test_audio_declared_groups", controller_path)
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    admitted = controller._declared_generation_routing_groups({"routing_groups": {}})
    assert routing_group in admitted, (
        f"{service_name} routing group {routing_group!r} missing from resolved ingress groups"
    )


# CPU busy-timeout reducer and real fake-server observer regressions.
"""Behavioral coverage for CPU probe timeouts while broker leases are active."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gemma_broker.api import DurableBrokerService
from gemma_broker.contracts import JobState, MemberState, ReasonCode
from gemma_broker.member_observer import MemberObserver
from gemma_broker.member_cache import StrictMemberSnapshotCache
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.readiness import reduce_member_readiness

CPU = {
    "enabled": True, "cpu_only": True, "gpu_id": None, "parallel": 2,
    "context_per_slot": 4096, "capabilities": ("chat",),
}
GPU = {
    "enabled": True, "cpu_only": False, "gpu_id": "gpu0", "parallel": 2,
    "context_per_slot": 4096, "idle_service_configured": "Gemma-GPU",
    "capabilities": ("chat",),
}


def _probes(*, lease=0, error: str | None="probe_timeout", observed_at=10.0, semantic=False, **extra):
    return {
        "observed_at": observed_at, "systemd_active": True, "main_pid": 42,
        "dispatcher_registered": True, "dispatcher_leases": lease,
        "probe_error": error, "semantic_health": semantic,
        "backend_slots_total": None, "backend_slots_busy": None, **extra,
    }


def test_fresh_cpu_probe_timeout_with_lease_is_capacity_only_ready_busy():
    snapshot = reduce_member_readiness(
        "Gemma-CPU", CPU, _probes(lease=1), now=11.0, freshness_ms=3000
    )
    assert snapshot.state is MemberState.READY_BUSY
    assert not snapshot.accepting
    assert snapshot.compatible_free_slots == 0
    assert snapshot.dispatcher_leases == 1
    assert snapshot.backend_slots_total is None
    assert snapshot.backend_slots_busy is None
    assert snapshot.reason_codes == (ReasonCode.MEMBER_NO_CAPACITY,)


@pytest.mark.parametrize(
    "config, probes, expected, reason",
    [
        (CPU, _probes(lease=0), MemberState.UNHEALTHY, ReasonCode.MEMBER_PROBE_TIMEOUT),
        (CPU, _probes(lease=1, observed_at=1.0), MemberState.UNHEALTHY, ReasonCode.TELEMETRY_STALE),
        (GPU, _probes(lease=1), MemberState.UNHEALTHY, ReasonCode.MEMBER_PROBE_TIMEOUT),
        (CPU, _probes(lease=1, other_blocker=True), MemberState.JOINING, ReasonCode.MEMBER_DISPATCHER_UNREGISTERED),
    ],
    ids=["dead-no-lease", "stale-lease", "gpu-unchanged", "additional-blocker"],
)
def test_timeout_remains_fail_closed_outside_fresh_cpu_lease(config, probes, expected, reason):
    if probes.pop("other_blocker", False):
        probes["dispatcher_registered"] = False
    snapshot = reduce_member_readiness(
        "member", config, probes, now=11.0, freshness_ms=3000
    )
    assert snapshot.state is expected
    assert reason in snapshot.reason_codes


def test_healthy_cpu_free_slot_stays_ready_accepting():
    snapshot = reduce_member_readiness(
        "Gemma-CPU", CPU,
        _probes(lease=0, error=None, semantic=True, backend_slots_total=2, backend_slots_busy=0),
        now=11.0, freshness_ms=3000,
    )
    assert snapshot.state is MemberState.READY_ACCEPTING
    assert snapshot.accepting
    assert snapshot.compatible_free_slots == 2


def test_pinned_busy_member_admission_queues_without_claiming_capacity():
    busy = reduce_member_readiness(
        "Gemma-CPU", CPU, _probes(lease=1), now=11.0, freshness_ms=3000
    )
    repo = InMemoryJobRepository()
    repo.put_member(busy)
    service = DurableBrokerService(repo, members=lambda: [busy], clock=lambda: 11.0)
    loop = _idle_dispatch_loop(repo, lambda: [busy], lambda: 11.0)
    runtime = SimpleNamespace(
        _repository=repo,
        dispatch_loop=loop,
        _clock=lambda: 11.0,
        _wall_clock=lambda: 11.0,
        config=SimpleNamespace(freshness_seconds=3.0),
    )
    service.member_admission = lambda name: BrokerRuntime._pinned_member_admission(runtime, name)
    service.member_capacity = lambda name: BrokerRuntime._pinned_member_capacity(runtime, name)
    job = service.submit({
        "idempotency_key": "busy-cpu-timeout-test",
        "request": {"messages": [{"role": "user", "content": "synthetic"}]},
        "pinned_member": "Gemma-CPU",
    })
    assert job.state is JobState.QUEUED
    assert job.pinned_member == "Gemma-CPU"
    assert repo.get_member("Gemma-CPU").compatible_free_slots == 0


@pytest.mark.asyncio
async def test_busy_fake_server_endpoint_responses_and_observer_timeout_shape():
    blocked = [True]
    requested = []

    async def endpoint(request):
        requested.append(request.path)
        if blocked[0]:
            await asyncio.sleep(0.2)
        payloads = {
            "/health": {"status": "ok", "slots_idle": 2, "slots_processing": 0},
            "/slots": [
                {"id": 0, "n_ctx": 4096, "is_processing": False},
                {"id": 1, "n_ctx": 4096, "is_processing": False},
            ],
            "/v1/models": {"data": [{"id": "gemma-3-27b.gguf", "meta": {"n_ctx": 4096}}]},
        }
        return web.json_response(payloads[request.path])

    app = web.Application()
    app.router.add_get("/health", endpoint)
    app.router.add_get("/slots", endpoint)
    app.router.add_get("/v1/models", endpoint)
    server = TestServer(app)
    await server.start_server()
    try:
        config = {**CPU, "port": server.port, "probe_host": "127.0.0.1", "model_path": "/models/gemma-3-27b.gguf"}
        async with aiohttp.ClientSession() as session:
            observer = MemberObserver(
                {"Gemma-CPU": config}, session=session,
                systemd_probe=lambda _cfg: _facts(),
                dispatcher_state=lambda _name: (True, 2), timeout=0.03,
            )
            busy = await observer.observe("Gemma-CPU", now=100.0)
            assert set(requested) == {"/health", "/slots", "/v1/models"}
            assert busy["probe_error"] == "probe_timeout"
            assert busy["backend_slots_total"] is None
            assert busy["backend_slots_busy"] is None
            assert busy["semantic_health"] is False
            assert busy["dispatcher_leases"] == 2

            blocked[0] = False
            requested.clear()
            responsive = await MemberObserver(
                {"Gemma-CPU": config}, session=session,
                systemd_probe=lambda _cfg: _facts(),
                dispatcher_state=lambda _name: (True, 0), timeout=0.3,
            ).observe("Gemma-CPU", now=101.0)
            assert set(requested) == {"/health", "/slots", "/v1/models"}
            assert responsive["probe_error"] is None
            assert responsive["semantic_health"] is True
            assert responsive["backend_slots_total"] == 2
            assert responsive["backend_slots_busy"] == 0
    finally:
        await server.close()


async def _facts():
    return {"systemd_active": True, "main_pid": 42}


@pytest.mark.asyncio
async def test_embedded_sweep_stamps_observation_after_probe_completion(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    controller_path = root / "scripts" / "gpu-manager.py"
    spec = importlib.util.spec_from_file_location("cpu_busy_sweep_test", controller_path)
    assert spec and spec.loader
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    clock = [100.0]
    probes = _probes(lease=0, error=None, observed_at=1.0, semantic=True,
                     backend_slots_total=2, backend_slots_busy=0)

    class _Observer:
        def __init__(self, *_args, **_kwargs):
            pass

        async def observe(self, _name, *, now):
            assert now == 100.0
            clock[0] = 104.0
            return dict(probes)

    class _Loop:
        def active_dispatch_suppression(self, _name):
            return None

    repository = SimpleNamespace(get_member=lambda _name: None)
    config = {"Gemma-CPU": CPU}
    cache = StrictMemberSnapshotCache(config, freshness_ms=3000)
    controller.MemberObserver = _Observer
    controller.time = SimpleNamespace(time=lambda: clock[0])
    controller._combined_gemma_broker_embedded = lambda: True
    controller._combined_gemma_member_cache = cache
    controller._combined_gemma_broker = SimpleNamespace(
        dispatch_loop=_Loop(), _repository=repository
    )
    controller._services_config = {"combined_gemma_broker": {"ordered_members": ["Gemma-CPU"]}}
    controller._combined_gemma_refresh_logged_state = {}
    controller.session = SimpleNamespace(closed=False)
    controller._combined_gemma_systemd_facts = _facts

    await controller._refresh_combined_gemma_member_snapshots()
    snapshot = cache.snapshots()[0]
    assert snapshot.observed_at == 104.0
    assert snapshot.snapshot_age_ms == 0
    assert snapshot.state is MemberState.READY_ACCEPTING
