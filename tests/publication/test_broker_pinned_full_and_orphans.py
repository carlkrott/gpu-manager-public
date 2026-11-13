"""Plan 2.6 fix loop 1: a full GPU pin stays queued; failed pins leave the queue.

Live 2026-10-11 04:3x (HOL drill, R2 cf2afe5): four pinned requests filled
Gemma-RX6950XT (4 slots, READY_BUSY). The fifth pin was failed with
``pinned_member_unavailable`` after ~6 s instead of waiting for a slot, and its
id stayed in the priority zset, so the broker queue-depth gauge never returned
to 0 (two such orphans were live: one from this drill, one older Gemma-CPU pin).
"""
from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time
from dataclasses import replace

import pytest

from gemma_broker.api import DurableBrokerService
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobState, MemberSnapshot, MemberState, ReasonCode
from gemma_broker.dispatcher import Dispatcher
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import DispatchLoop

CHAT = {"messages": [{"role": "user", "content": "ping"}]}
RX = "Gemma-RX6950XT"


def _gpu_member(name=RX, *, state=MemberState.READY_ACCEPTING, leases=0):
    full = leases >= 4
    return MemberSnapshot(
        name=name,
        gpu_id="rx6950xt",
        state=state,
        accepting=state is MemberState.READY_ACCEPTING,
        configured_slots=4,
        context_per_slot=4096,
        state_version=1,
        observed_at=1.0,
        compatible_free_slots=0 if full else 4 - leases,
        dispatcher_leases=leases,
        backend_slots_busy=leases,
        backend_slots_total=4,
        capabilities=("chat",),
    )


def _loop(repository, members_provider, clock):
    class _UnusedTransport:
        async def send(self, *_args):
            raise AssertionError("no dispatch should reach transport")

    transport = _UnusedTransport()
    dispatcher = Dispatcher(repo=repository, transport=transport, request_timeout=1.0, clock=clock)
    loop = DispatchLoop(
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
    loop._fencing_token = 1
    return loop


def _queued_pin(repository, members):
    service = DurableBrokerService(repository, members=lambda: members, clock=lambda: 2.0)
    return service.submit({"idempotency_key": "pin-5", "request": CHAT, "pinned_member": RX})


def test_pin_to_full_ready_busy_gpu_member_stays_queued():
    """Live shape: 4/4 slots leased, member READY_BUSY. Busy is not unavailable."""
    members = [_gpu_member(leases=0)]
    repository = InMemoryJobRepository()
    repository.put_member(members[0])
    job = _queued_pin(repository, members)
    members[:] = [_gpu_member(state=MemberState.READY_BUSY, leases=4)]
    repository.put_member(members[0])
    loop = _loop(repository, lambda: members, lambda: 2.0)

    asyncio.run(asyncio.wait_for(loop._dispatch_one(), timeout=1.0))

    current = repository.get(job.job_id)
    assert current.state is JobState.QUEUED, f"full busy pin terminalized: {current.reason_code}"
    assert repository.queue_total == 1


def test_pin_to_unhealthy_gpu_member_fails_and_leaves_the_queue():
    members = [_gpu_member(leases=0)]
    repository = InMemoryJobRepository()
    repository.put_member(members[0])
    job = _queued_pin(repository, members)
    members[:] = [replace(_gpu_member(leases=4), state=MemberState.UNHEALTHY, accepting=False)]
    repository.put_member(members[0])
    loop = _loop(repository, lambda: members, lambda: 2.0)

    asyncio.run(asyncio.wait_for(loop._dispatch_one(), timeout=1.0))

    current = repository.get(job.job_id)
    assert current.state is JobState.FAILED
    assert current.reason_code is ReasonCode.NO_ELIGIBLE_MEMBER
    assert repository.queue_total == 0, "failed pin left an orphan id in the priority queue"


def test_pin_to_full_busy_member_behind_a_quota_wall_still_fails():
    members = [_gpu_member(leases=0)]
    repository = InMemoryJobRepository()
    repository.put_member(members[0])
    job = _queued_pin(repository, members)
    members[:] = [_gpu_member(state=MemberState.READY_BUSY, leases=4)]
    repository.put_member(members[0])
    loop = _loop(repository, lambda: members, lambda: 2.0)
    loop._record_quota_wall(RX, "test_wall", 900)

    asyncio.run(asyncio.wait_for(loop._dispatch_one(), timeout=1.0))

    assert repository.get(job.job_id).state is JobState.FAILED
    assert repository.queue_total == 0


def test_peek_window_reaps_terminal_ids_and_keeps_queued_ones():
    members = [_gpu_member(leases=0)]
    repository = InMemoryJobRepository()
    repository.put_member(members[0])
    service = DurableBrokerService(repository, members=lambda: members, clock=lambda: 2.0)
    orphan = service.submit({"idempotency_key": "orphan", "request": CHAT})
    live = service.submit({"idempotency_key": "live", "request": {**CHAT, "max_tokens": 9}})
    current = repository.get(orphan.job_id)
    # Pre-fix failure path: terminal record written without leaving the zset.
    repository.replace_job(replace(current, state=JobState.FAILED, completed_at=2.0,
                                   state_version=current.state_version + 1))
    assert repository.queue_total == 2

    window = repository.peek_window(now=2.0)

    assert [job.job_id for job in window] == [live.job_id]
    assert repository.queue_total == 1
    assert repository.get(live.job_id).state is JobState.QUEUED


# ---- the same contracts against reserve/peek Lua on a throwaway redis-server ----

redis = pytest.importorskip("redis")
needs_redis = pytest.mark.skipif(shutil.which("redis-server") is None,
                                 reason="redis-server not installed")


@pytest.fixture(scope="module")
def redis_port(tmp_path_factory):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    workdir = tmp_path_factory.mktemp("redis-orphans")
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--bind", "127.0.0.1",
         "--save", "", "--appendonly", "no", "--dir", str(workdir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield port
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def redis_repo(redis_port):
    from gemma_broker.redis_store import RedisJobRepository

    client = redis.Redis(host="127.0.0.1", port=redis_port)
    client.flushall()
    yield RedisJobRepository(client, prefix="test:", leadership_clock=lambda: 2.0)
    client.close()


@needs_redis
def test_redis_failed_pin_leaves_the_queue(redis_repo):
    members = [redis_repo.put_member(_gpu_member(leases=0))]
    job = _queued_pin(redis_repo, members)
    members[:] = [redis_repo.put_member(
        replace(_gpu_member(leases=4), state=MemberState.UNHEALTHY, accepting=False))]
    loop = _loop(redis_repo, lambda: members, lambda: 2.0)
    loop._fencing_token = redis_repo.acquire_leadership(
        owner="test-dispatcher", now=2.0, ttl_seconds=30.0)

    asyncio.run(asyncio.wait_for(loop._dispatch_one(), timeout=2.0))

    assert redis_repo.get(job.job_id).state is JobState.FAILED
    assert redis_repo.queue_total == 0


@needs_redis
def test_redis_peek_window_reaps_terminal_ids_only(redis_repo):
    members = [redis_repo.put_member(_gpu_member(leases=0))]
    service = DurableBrokerService(redis_repo, members=lambda: members, clock=lambda: 2.0)
    orphan = service.submit({"idempotency_key": "orphan", "request": CHAT})
    live = service.submit({"idempotency_key": "live", "request": {**CHAT, "max_tokens": 9}})
    current = redis_repo.get(orphan.job_id)
    redis_repo.replace_job(replace(current, state=JobState.FAILED, completed_at=2.0,
                                   state_version=current.state_version + 1))
    assert redis_repo.queue_total == 2

    window = redis_repo.peek_window(now=2.0)

    assert [job.job_id for job in window] == [live.job_id]
    assert redis_repo.queue_total == 1
    assert redis_repo.get(live.job_id).state is JobState.QUEUED
    assert redis_repo.get(orphan.job_id).state is JobState.FAILED
