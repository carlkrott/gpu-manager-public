"""t_51d919c5: a new leader reconciles a dead predecessor's leases at once.

Real ``redis-server`` because acquire_leader.lua must report the superseded
owner; fakeredis cannot run ``eval``.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import replace

import pytest
import redis

from gemma_broker.api import DurableBrokerService
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobState, MemberSnapshot, MemberState
from gemma_broker.redis_store import RedisJobRepository
from gemma_broker.runtime import DispatchLoop

from test_broker_window_redis import pytestmark, redis_port  # noqa: F401

CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def _req_for(job):
    return RequestRequirements(8, 16, ("chat",), "t", pinned_member=job.pinned_member)


def _strand_jobs(client, old_owner):
    """Old leader claims 3 jobs on a 3-slot member, 2 cross acceptance, then dies."""
    now = [2.0]
    repo = RedisJobRepository(client, prefix="test:", leadership_clock=lambda: now[0])
    member = repo.put_member(MemberSnapshot(
        name="Qwen-Dual", gpu_id=None, state=MemberState.READY_ACCEPTING, accepting=True,
        configured_slots=3, context_per_slot=4096, state_version=1,
        observed_at=1.0, compatible_free_slots=3, capabilities=("chat",),
    ))
    service = DurableBrokerService(repo, members=lambda: [member], clock=lambda: 2.0)
    fence = repo.acquire_leadership(owner=old_owner, now=2.0, ttl_seconds=30.0)
    for key in ("a", "b", "c"):
        service.submit({"idempotency_key": key, "request": CHAT})
    reservations = []
    for _ in range(3):
        reservations.append(repo.reserve_next(
            now=2.0, ordered_members=[repo.get_member("Qwen-Dual")], leader_owner=old_owner,
            fencing_token=fence, requirements_for=_req_for,
        ))
    for r in reservations[:2]:
        repo.replace_job(replace(
            r.job, state=JobState.ACCEPTED, accepted_boundary_crossed=True,
            state_version=r.job.state_version + 1,
        ))
    assert repo.lease_count("Qwen-Dual") == 3
    assert repo.get_member("Qwen-Dual").compatible_free_slots == 0
    now[0] = 40.0  # old lease expired
    return repo, reservations


def _refresh(repo):
    return repo.put_member(replace(repo.get_member("Qwen-Dual"), backend_slots_busy=0))


def _new_loop(repo, owner):
    return DispatchLoop(
        repository=repo, transport=object(), members_provider=lambda: [],
        dispatcher=object(), owner=owner, leader_ttl_seconds=30.0,
        heartbeat_seconds=1.0, clock=lambda: 50.0,
        startup_recovery_grace_seconds=3600.0,
        requirements_factory=lambda _body: RequestRequirements(8, 16, ("chat",), "t"),
    )


@pytest.fixture
def client(redis_port):  # noqa: F811
    c = redis.Redis(host="127.0.0.1", port=redis_port)
    c.flushall()
    yield c
    c.close()


def test_dead_prior_leader_leases_reconciled_without_grace(client):
    dead = subprocess.Popen(["true"])
    dead.wait()
    repo, reservations = _strand_jobs(client, f"gpu-manager-embedded:{dead.pid}:1")
    loop = _new_loop(repo, f"gpu-manager-embedded:{os.getpid()}:2")
    assert asyncio.run(loop._acquire_leader_until_available(reason="initial"))
    assert loop._prior_leader_dead()
    loop._recover_dead_leader_now()
    assert loop._startup_recovery_complete
    states = [repo.get(r.job.job_id).state for r in reservations]
    assert states == [JobState.OUTCOME_UNKNOWN, JobState.OUTCOME_UNKNOWN, JobState.QUEUED]
    assert repo.lease_count("Qwen-Dual") == 0
    # The 2 s readiness refresh restores capacity once no lease is live.
    assert _refresh(repo).compatible_free_slots == 3


def test_live_prior_leader_keeps_grace(client):
    alive = subprocess.Popen(["sleep", "30"])
    try:
        repo, reservations = _strand_jobs(client, f"gpu-manager-embedded:{alive.pid}:1")
        loop = _new_loop(repo, f"gpu-manager-embedded:{os.getpid()}:2")
        assert asyncio.run(loop._acquire_leader_until_available(reason="initial"))
        assert not loop._prior_leader_dead()
        assert repo.lease_count("Qwen-Dual") == 3
        assert _refresh(repo).compatible_free_slots == 0  # held until the grace
    finally:
        alive.kill()
        alive.wait()


def test_same_owner_reacquire_is_not_a_dead_handover(client):
    repo, _ = _strand_jobs(client, "gpu-manager-embedded:999999:1")
    loop = _new_loop(repo, "gpu-manager-embedded:999999:1")
    repo.last_minted_leadership = None
    repo._leadership_clock = lambda: 3.0  # old lease still held by the same owner
    assert asyncio.run(loop._acquire_leader_until_available(reason="initial"))
    assert repo.last_minted_leadership is None
    assert not loop._prior_leader_dead()


def test_lease_held_first_then_mint_still_reconciles(client):
    """Production order: the dead owner's lease is still valid for ~12 s first."""
    dead = subprocess.Popen(["true"])
    dead.wait()
    repo, reservations = _strand_jobs(client, f"gpu-manager-embedded:{dead.pid}:1")
    loop = _new_loop(repo, f"gpu-manager-embedded:{os.getpid()}:2")
    repo._leadership_clock = lambda: 3.0  # old lease (expires 32.0) still held
    fence_before = client.get("test:dispatcher:fence-counter")
    assert asyncio.run(loop._try_acquire_leader()) is None
    assert client.get("test:dispatcher:fence-counter") == fence_before
    assert repo.lease_count("Qwen-Dual") == 3
    repo._leadership_clock = lambda: 40.0
    assert asyncio.run(loop._acquire_leader_until_available(reason="initial"))
    assert loop._prior_leader_dead()
    loop._recover_dead_leader_now()
    states = [repo.get(r.job.job_id).state for r in reservations]
    assert states == [JobState.OUTCOME_UNKNOWN, JobState.OUTCOME_UNKNOWN, JobState.QUEUED]
    assert repo.lease_count("Qwen-Dual") == 0


def test_slow_recovery_scan_keeps_the_fence(client):
    """Live 10-23: a 27 s inline scan outlived the TTL and the fence was re-minted."""
    import time

    repo = RedisJobRepository(client, prefix="test:", leadership_clock=time.time)
    loop = DispatchLoop(
        repository=repo, transport=object(), members_provider=lambda: [],
        dispatcher=object(), owner="gpu-manager-embedded:1:1", leader_ttl_seconds=0.5,
        heartbeat_seconds=0.1, clock=time.time,
        requirements_factory=lambda _body: RequestRequirements(8, 16, ("chat",), "t"),
    )
    assert asyncio.run(loop._acquire_leader_until_available(reason="initial"))
    token = loop._fencing_token
    asyncio.run(loop._run_renewing_leadership(lambda: time.sleep(1.5)))
    assert int(client.get("test:dispatcher:fence-counter")) == token
    repo.renew_leadership(
        owner="gpu-manager-embedded:1:1", fencing_token=token, now=time.time(), ttl_seconds=0.5,
    )
