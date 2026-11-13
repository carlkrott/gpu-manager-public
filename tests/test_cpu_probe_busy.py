"""Real cache-path regressions for CPU probe-timeout busy readiness."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import pytest
from types import SimpleNamespace
import time

from gemma_broker.api import APIError, DurableBrokerService
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import MemberState, ReasonCode
from gemma_broker.dispatcher import Dispatcher
from gemma_broker.member_cache import StrictMemberSnapshotCache
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import BrokerRuntime, DispatchLoop


CPU = {
    "enabled": True,
    "cpu_only": True,
    "gpu_id": None,
    "parallel": 2,
    "context_per_slot": 4096,
    "capabilities": ("chat",),
}
GPU = {
    "enabled": True,
    "cpu_only": False,
    "gpu_id": "gpu0",
    "parallel": 2,
    "context_per_slot": 4096,
    "idle_service_configured": "Gemma-GPU",
    "capabilities": ("chat",),
}


def probes(**overrides):
    values = {
        "observed_at": 10.0,
        "systemd_active": True,
        "main_pid": 42,
        "dispatcher_registered": True,
        "dispatcher_leases": 1,
        "probe_error": "probe_timeout",
        "semantic_health": False,
        "backend_slots_total": None,
        "backend_slots_busy": None,
    }
    values.update(overrides)
    return values


def test_fresh_cpu_timeout_with_lease_is_busy_through_real_snapshot_cache():
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)

    snapshot = cache.update("Gemma-CPU", probes(), now=11.0)

    assert cache.snapshots(now=11.0) == [snapshot]
    assert snapshot.state is MemberState.READY_BUSY
    assert snapshot.semantic_health is False
    assert not snapshot.accepting
    assert snapshot.compatible_free_slots == 0
    assert snapshot.dispatcher_leases == 1
    assert snapshot.backend_slots_total is None
    assert snapshot.backend_slots_busy is None
    assert snapshot.reason_codes == (ReasonCode.MEMBER_NO_CAPACITY,)


@pytest.mark.parametrize(
    "config, probe_values, now, expected_reason",
    [
        (CPU, probes(dispatcher_leases=0), 11.0, ReasonCode.MEMBER_PROBE_TIMEOUT),
        (CPU, probes(observed_at=1.0), 11.0, ReasonCode.TELEMETRY_STALE),
        (GPU, probes(), 11.0, ReasonCode.MEMBER_PROBE_TIMEOUT),
        (CPU, probes(dispatcher_registered=False), 11.0, ReasonCode.MEMBER_DISPATCHER_UNREGISTERED),
        (CPU, probes(systemd_active=False), 11.0, ReasonCode.MEMBER_SYSTEMD_INACTIVE),
        ({**CPU, "enabled": False}, probes(), 11.0, ReasonCode.MEMBER_CONFIG_DISABLED),
    ],
    ids=["no-lease", "stale", "gpu-member", "extra-blocker", "inactive-process", "disabled"],
)
def test_timeout_outside_narrow_busy_exception_stays_fail_closed(
    config, probe_values, now, expected_reason
):
    cache = StrictMemberSnapshotCache({"member": config}, freshness_ms=3000)

    snapshot = cache.update("member", probe_values, now=now)

    assert snapshot.state is MemberState.UNHEALTHY
    assert not snapshot.accepting
    assert snapshot.compatible_free_slots == 0
    assert expected_reason in snapshot.reason_codes


def test_healthy_cpu_cache_update_remains_accepting():
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)

    snapshot = cache.update(
        "Gemma-CPU",
        probes(
            dispatcher_leases=0,
            probe_error=None,
            semantic_health=True,
            backend_slots_total=2,
            backend_slots_busy=0,
        ),
        now=11.0,
    )

    assert snapshot.state is MemberState.READY_ACCEPTING
    assert snapshot.accepting
    assert snapshot.compatible_free_slots == 2


def _real_hooks(repository, members, now, *, monotonic_now=None):
    class NoDispatchTransport:
        async def send(self, *_args):
            raise AssertionError("capacity-only snapshot must not dispatch")

    transport = NoDispatchTransport()
    clock = lambda: now[0] if monotonic_now is None else monotonic_now
    dispatcher = Dispatcher(repo=repository, transport=transport, request_timeout=1.0, clock=clock)
    loop = DispatchLoop(
        repository=repository,
        transport=transport,
        members_provider=lambda: members,
        dispatcher=dispatcher,
        owner="cpu-busy-test",
        leader_ttl_seconds=30.0,
        heartbeat_seconds=1.0,
        clock=clock,
    )
    loop._wall_clock = lambda: now[0]
    loop._readiness_freshness_seconds = 3.0
    runtime = SimpleNamespace(
        _repository=repository,
        dispatch_loop=loop,
        _clock=clock,
        _wall_clock=lambda: now[0],
        config=SimpleNamespace(freshness_seconds=3.0),
    )
    service = DurableBrokerService(repository, members=lambda: members, clock=lambda: now[0])
    service.member_admission = lambda name: BrokerRuntime._pinned_member_admission(runtime, name)
    service.member_capacity = lambda name: BrokerRuntime._pinned_member_capacity(runtime, name)
    return service, loop


def test_real_pinned_hooks_queue_cpu_timeout_without_reserving_until_free():
    now = [11.0]
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)
    accepting = cache.update(
        "Gemma-CPU",
        probes(dispatcher_leases=0, probe_error=None, semantic_health=True,
               backend_slots_total=2, backend_slots_busy=0),
        now=now[0],
    )
    repository = InMemoryJobRepository()
    repository.put_member(accepting)
    service, loop = _real_hooks(repository, [accepting], now)
    first = service.submit({
        "idempotency_key": "cpu-timeout-held-capacity-seed-1",
        "request": {"messages": [{"role": "user", "content": "synthetic"}]},
        "pinned_member": "Gemma-CPU",
    })
    second = service.submit({
        "idempotency_key": "cpu-timeout-held-capacity-seed-2",
        "request": {"messages": [{"role": "user", "content": "synthetic"}]},
        "pinned_member": "Gemma-CPU",
    })
    requirements = RequestRequirements(8, 16, ("chat",), "test", pinned_member="Gemma-CPU")
    held = [
        repository.reserve_next(
            now=now[0], ordered_members=[accepting], requirements=requirements,
            member_lookup=repository.get_member,
        )
        for _ in range(2)
    ]
    assert all(item is not None for item in held)
    assert {item.job.job_id for item in held if item is not None} == {first.job_id, second.job_id}
    assert repository.lease_count("Gemma-CPU") == 2

    busy = cache.update("Gemma-CPU", probes(dispatcher_leases=2), now=now[0])
    repository.put_member(busy)
    queued = service.submit({
        "idempotency_key": "cpu-timeout-pinned-real-hook",
        "request": {"messages": [{"role": "user", "content": "synthetic"}]},
        "pinned_member": "Gemma-CPU",
    })
    assert queued.state.value == "queued"
    assert repository.get_member("Gemma-CPU").compatible_free_slots == 0
    assert repository.lease_count("Gemma-CPU") == 2
    assert repository.reserve_next(
        now=now[0], ordered_members=[busy], requirements=requirements,
        member_lookup=repository.get_member,
    ) is None

    # The embedded dispatcher holds a positive leadership fence. A pinned
    # request against this exact capacity-only busy snapshot must remain queued
    # through its next dispatch tick, rather than being terminalized as absent.
    loop._fencing_token = repository.acquire_leadership(
        owner="cpu-busy-test", now=now[0], ttl_seconds=30.0
    )
    assert loop._fencing_token is not None and loop._fencing_token > 0
    assert asyncio.run(loop._dispatch_one()) is False
    still_queued = repository.get(queued.job_id)
    assert still_queued is not None
    assert still_queued.state.value == "queued"
    assert still_queued.reason_code is None
    assert repository.lease_count("Gemma-CPU") == 2

    for reservation in held:
        assert reservation is not None
        repository.release_member_lease(
            "Gemma-CPU",
            str(reservation.job.current_attempt_id),
            claimed_member_state_version=int(reservation.job.member_state_version),
        )
    assert repository.lease_count("Gemma-CPU") == 0
    free = cache.update(
        "Gemma-CPU",
        probes(dispatcher_leases=0, probe_error=None, semantic_health=True,
               backend_slots_total=2, backend_slots_busy=0),
        now=now[0],
    )
    repository.put_member(free)
    reservation = repository.reserve_next(
        now=now[0], ordered_members=[free], requirements=requirements,
        member_lookup=repository.get_member,
    )
    assert reservation is not None
    assert reservation.job.job_id == queued.job_id
    assert repository.lease_count("Gemma-CPU") == 1
    assert loop._members_eligible([free]) == [free]


@pytest.mark.parametrize(
    "config, probe_values, now",
    [
        (CPU, probes(dispatcher_leases=0), 11.0),
        (CPU, probes(observed_at=1.0), 11.0),
        (GPU, probes(), 11.0),
        (CPU, probes(dispatcher_registered=False), 11.0),
    ],
    ids=["no-lease", "stale", "gpu", "extra-blocker"],
)
def test_real_capacity_hook_rejects_nonexception_cache_snapshots(config, probe_values, now):
    cache = StrictMemberSnapshotCache({"member": config}, freshness_ms=3000)
    snapshot = cache.update("member", probe_values, now=now)
    repository = InMemoryJobRepository()
    repository.put_member(snapshot)
    service, _loop = _real_hooks(repository, [snapshot], [now])

    assert not service.member_capacity("member")


def test_real_capacity_hook_uses_wall_time_not_monotonic_for_observed_at():
    wall_now = time.time()
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)
    healthy = cache.update(
        "Gemma-CPU",
        probes(
            observed_at=wall_now,
            dispatcher_leases=0,
            probe_error=None,
            semantic_health=True,
            backend_slots_total=2,
            backend_slots_busy=0,
        ),
        now=wall_now,
    )
    snapshot = replace(
        healthy,
        state=MemberState.READY_BUSY,
        accepting=False,
        dispatcher_leases=1,
        backend_slots_total=None,
        backend_slots_busy=None,
        semantic_health=False,
        compatible_free_slots=0,
        reason_codes=(ReasonCode.MEMBER_NO_CAPACITY,),
    )
    repository = InMemoryJobRepository()
    repository.put_member(snapshot)
    service, _loop = _real_hooks(
        repository, [snapshot], [wall_now], monotonic_now=time.monotonic()
    )

    queued = service.submit({
        "idempotency_key": "cpu-timeout-monotonic-wall-clock-regression",
        "request": {"messages": [{"role": "user", "content": "synthetic"}]},
        "pinned_member": "Gemma-CPU",
    })

    assert queued.state.value == "queued"
    assert repository.lease_count("Gemma-CPU") == 0


def test_real_capacity_hook_rejects_expired_published_busy_snapshot():
    now = [11.0]
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)
    snapshot = cache.update("Gemma-CPU", probes(), now=now[0])
    repository = InMemoryJobRepository()
    repository.put_member(snapshot)
    service, _loop = _real_hooks(repository, [snapshot], now)

    now[0] = 15.0
    assert not service.member_capacity("Gemma-CPU")


def test_dispatch_fails_pinned_busy_snapshot_that_expires_after_enqueue():
    now = [11.0]
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)
    healthy = cache.update(
        "Gemma-CPU",
        probes(
            dispatcher_leases=0,
            probe_error=None,
            semantic_health=True,
            backend_slots_total=2,
            backend_slots_busy=0,
        ),
        now=now[0],
    )
    busy = replace(
        healthy,
        state=MemberState.READY_BUSY,
        accepting=False,
        dispatcher_leases=1,
        backend_slots_total=None,
        backend_slots_busy=None,
        semantic_health=False,
        compatible_free_slots=0,
        reason_codes=(ReasonCode.MEMBER_NO_CAPACITY,),
    )
    repository = InMemoryJobRepository()
    repository.put_member(healthy)
    members = [healthy]
    service, loop = _real_hooks(repository, members, now)
    queued = service.submit({
        "idempotency_key": "cpu-timeout-pin-expired-before-dispatch",
        "request": {"messages": [{"role": "user", "content": "synthetic"}]},
        "pinned_member": "Gemma-CPU",
    })
    assert queued.state.value == "queued"
    members[0] = busy
    repository.put_member(busy)
    loop._fencing_token = repository.acquire_leadership(
        owner="cpu-busy-test", now=now[0], ttl_seconds=30.0
    )

    now[0] = 15.0
    assert asyncio.run(loop._dispatch_one()) is False
    failed = repository.get(queued.job_id)
    assert failed is not None
    assert failed.state.value == "failed"
    assert failed.reason_code is ReasonCode.NO_ELIGIBLE_MEMBER


@pytest.mark.parametrize("wall_kind", ["cooldown", "quota"], ids=["cooldown", "quota-wall"])
def test_real_admission_hook_keeps_cooldown_and_quota_suppression(wall_kind):
    now = [11.0]
    cache = StrictMemberSnapshotCache({"Gemma-CPU": CPU}, freshness_ms=3000)
    snapshot = cache.update("Gemma-CPU", probes(), now=now[0])
    repository = InMemoryJobRepository()
    repository.put_member(snapshot)
    service, loop = _real_hooks(repository, [snapshot], now)
    if wall_kind == "cooldown":
        loop._record_preaccept_failure("Gemma-CPU")
    else:
        loop._record_quota_wall("Gemma-CPU", "quota", 900)

    with pytest.raises(APIError) as rejected:
        service.submit({
            "idempotency_key": f"cpu-timeout-suppressed-{wall_kind}",
            "request": {"messages": [{"role": "user", "content": "synthetic"}]},
            "pinned_member": "Gemma-CPU",
        })
    assert rejected.value.status == 503
