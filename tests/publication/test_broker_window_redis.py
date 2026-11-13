"""Broker head-of-line window (plan 2.6) against a real ``redis-server``.

reserve_job.lua checks queue membership, not head-ness. fakeredis cannot run
``eval``, so this executes the Lua on a throwaway server.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import redis

from gemma_broker.api import DurableBrokerService
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobState, MemberSnapshot, MemberState
from gemma_broker.redis_store import RedisJobRepository, RepositoryUnavailable
from gemma_broker.runtime import NO_RESERVATION_BACKOFF_SECONDS, DispatchLoop

pytestmark = pytest.mark.skipif(
    shutil.which("redis-server") is None, reason="redis-server not installed"
)

CHAT = {"messages": [{"role": "user", "content": "hi"}]}
OWNER = "test-dispatcher"


@pytest.fixture(scope="module")
def redis_port(tmp_path_factory):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    workdir = tmp_path_factory.mktemp("redis")
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


def _member(name):
    return MemberSnapshot(
        name=name, gpu_id=None, state=MemberState.READY_ACCEPTING, accepting=True,
        configured_slots=1, context_per_slot=4096, state_version=1,
        observed_at=1.0, compatible_free_slots=1, capabilities=("chat",),
    )


def _req_for(job):
    return RequestRequirements(8, 16, ("chat",), "t", pinned_member=job.pinned_member)


@pytest.fixture
def broker(redis_port):
    client = redis.Redis(host="127.0.0.1", port=redis_port)
    client.flushall()
    repo = RedisJobRepository(client, prefix="test:", leadership_clock=lambda: 2.0)
    members = [repo.put_member(_member(n)) for n in ("Member-A", "Member-B")]
    service = DurableBrokerService(repo, members=lambda: members, clock=lambda: 2.0)
    fence = repo.acquire_leadership(owner=OWNER, now=2.0, ttl_seconds=30.0)
    yield repo, members, service, fence
    client.close()


def _reserve(repo, members, fence, **kw):
    return repo.reserve_next(
        now=2.0, ordered_members=members, leader_owner=OWNER, fencing_token=fence,
        requirements_for=_req_for, **kw,
    )


def test_job_behind_full_pinned_head_is_reserved_by_lua(broker):
    repo, members, service, fence = broker
    filler = service.submit({"idempotency_key": "f", "request": CHAT, "pinned_member": "Member-A"})
    assert _reserve(repo, members, fence).job.job_id == filler.job_id
    head = service.submit({"idempotency_key": "h", "request": CHAT, "pinned_member": "Member-A"})
    behind = service.submit({"idempotency_key": "b", "request": CHAT})
    full_a = replace(members[0], compatible_free_slots=0, dispatcher_leases=1)
    reservation = _reserve(repo, [full_a, members[1]], fence)
    assert reservation is not None
    assert (reservation.job.job_id, reservation.member.name) == (behind.job_id, "Member-B")
    assert repo.get(head.job_id).state is JobState.QUEUED
    assert repo.peek_next(now=2.0).job_id == head.job_id
    assert repo.lease_count("Member-B") == 1


def test_lua_refuses_a_job_no_longer_in_its_queue(broker):
    repo, members, service, fence = broker
    job = service.submit({"idempotency_key": "j", "request": CHAT})
    first = _reserve(repo, members, fence)
    assert first.job.job_id == job.job_id
    # A second claim of the same (now claimed) job must not succeed.
    queued_view = replace(first.job, state=JobState.QUEUED, current_attempt_id=None,
                          state_version=first.job.state_version - 1)
    outcome = repo._claim_on_member(
        queued_view, members[1], now=2.0, fencing_token=fence,
        leader_owner=OWNER, backend_url=None,
    )
    assert outcome == "job_raced"
    assert repo.lease_count("Member-B") == 0


def test_lua_membership_guard_refuses_queued_job_missing_from_zset(broker):
    repo, members, service, fence = broker
    job = service.submit({"idempotency_key": "m", "request": CHAT})
    repo.client.zrem(repo._queue_key(job.base_priority), job.job_id)  # throwaway server only
    outcome = repo._claim_on_member(
        repo.get(job.job_id), members[0], now=2.0, fencing_token=fence,
        leader_owner=OWNER, backend_url=None,
    )
    assert outcome == "job_raced"
    assert repo.lease_count("Member-A") == 0
    assert repo.get(job.job_id).state is JobState.QUEUED


class _RecordingDispatcher:
    def __init__(self):
        self.reservations = []

    async def dispatch(self, reservation):
        self.reservations.append(reservation)
        return SimpleNamespace(state=JobState.COMPLETED, reason_code=None, error=None)


def _dispatch_loop(repo, members, fence, dispatcher):
    loop = DispatchLoop(
        repository=repo,
        transport=object(),
        members_provider=lambda: members,
        dispatcher=dispatcher,
        owner=OWNER,
        leader_ttl_seconds=30.0,
        heartbeat_seconds=1.0,
        clock=lambda: 2.0,
        requirements_factory=lambda _body: RequestRequirements(8, 16, ("chat",), "t"),
    )
    loop._fencing_token = fence
    return loop


def _poison(repo, job, state=JobState.FAILED):
    """Rewrite a job record as terminal but leave it in its queue zset (live 2026-10-11)."""
    key = repo._job_key(job.job_id)
    record = json.loads(repo.client.hget(key, "record"))
    record["state"] = state.value
    repo.client.hset(key, mapping={"record": json.dumps(record), "state": state.value})
    assert repo.client.zscore(repo._queue_key(job.base_priority), job.job_id) is not None


def test_dispatch_reserves_job_behind_terminal_zset_head(broker):
    repo, members, service, fence = broker
    stale = service.submit({"idempotency_key": "s", "request": CHAT})
    behind = service.submit({"idempotency_key": "b", "request": CHAT})
    _poison(repo, stale)
    with pytest.raises(RepositoryUnavailable):
        repo.peek_next(now=2.0)  # the head-only peek the loop used before the fix
    dispatcher = _RecordingDispatcher()
    loop = _dispatch_loop(repo, members, fence, dispatcher)
    assert asyncio.run(loop._dispatch_one()) is True
    assert [r.job.job_id for r in dispatcher.reservations] == [behind.job_id]
    assert loop._reservation_retry_not_before_monotonic == 0.0


def test_dispatch_pass_that_reserves_nothing_sets_retry_gate(broker):
    repo, members, service, fence = broker
    stale = service.submit({"idempotency_key": "s", "request": CHAT})
    _poison(repo, stale)
    dispatcher = _RecordingDispatcher()
    loop = _dispatch_loop(repo, members, fence, dispatcher)
    before = time.monotonic()
    assert asyncio.run(loop._dispatch_one()) is False
    assert dispatcher.reservations == []
    assert loop._reservation_retry_not_before_monotonic >= before + NO_RESERVATION_BACKOFF_SECONDS
