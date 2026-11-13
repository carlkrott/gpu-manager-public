"""Real-Redis proof of the stale-orphan fence and immutable submission identity.

The terminalize script is Lua, so it is executed by a real ``redis-server``
started on a private port in a temp dir (no live Redis is touched).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import time

import pytest
import redis.asyncio as aioredis

from queue_engine import AsyncRedisQueue, _dead_letter_stream

pytestmark = pytest.mark.skipif(
    shutil.which("redis-server") is None, reason="redis-server not installed"
)

SVC = "comfyui"
RG = "comfyui-gen"
TRACKER = "job:orphan-1"
SLOTS = "slots:machine:comfyui"


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


def _run(coro_fn, port):
    async def main():
        r = aioredis.Redis(host="127.0.0.1", port=port, decode_responses=True)
        await r.flushall()
        try:
            return await coro_fn(r, AsyncRedisQueue(r))
        finally:
            await r.aclose()

    return asyncio.run(main())


async def _claimed_entry(r, q, job_id="orphan-1", tracker_status="in_flight"):
    await q.enqueue(SVC, {"job_id": job_id, "payload": "audit-me"}, routing_group=RG)
    got = await q.dequeue(SVC, "worker-0", routing_group=RG)
    assert got, "entry must be claimed"
    await r.hset(f"job:{job_id}", mapping={"job_id": job_id, "status": tracker_status})
    pending = await q.get_pending(SVC, routing_group=RG)
    assert len(pending) == 1
    return pending[0]


async def _terminalize(q, entry, min_idle=0, tracker=TRACKER):
    return await q.nack(
        SVC, entry["entry_id"], routing_group=RG, max_retries=1,
        dead_letter_reason="stale_orphan_ttl", orphan_tracker_key=tracker,
        orphan_slot_key=SLOTS, minimum_idle_ms=min_idle,
    )


async def _state(r, q):
    pending = await q.get_pending(SVC, routing_group=RG)
    dlq = await r.xrange(_dead_letter_stream(SVC))
    return pending, dlq, await r.hgetall(TRACKER)


def test_aged_orphan_is_dead_lettered_with_audit_and_tracker_failed_without_requeue(redis_port):
    async def fn(r, q):
        entry = await _claimed_entry(r, q)
        assert await _terminalize(q, entry) is True
        pending, dlq, tracker = await _state(r, q)
        assert pending == []
        assert await r.xlen(f"queue:{RG}") == 0  # not replayed anywhere
        assert len(dlq) == 1
        fields = dlq[0][1]
        assert fields["job_id"] == "orphan-1" and fields["payload"] == "audit-me"
        assert fields["_dead_letter_reason"] == "stale_orphan_ttl"
        assert tracker["status"] == "failed"
        assert tracker["error_code"] == "stale_orphan_ttl"
        assert await r.ttl(TRACKER) > 0

    _run(fn, redis_port)


@pytest.mark.parametrize("scenario", [
    "terminal", "cancelled", "tracker_missing", "other_job", "active_slot",
    "slot_unreadable", "renewed_lease",
])
def test_fence_fails_closed_and_leaves_entry_pending_untouched(redis_port, scenario):
    async def fn(r, q):
        entry = await _claimed_entry(r, q)
        min_idle = 0
        if scenario == "terminal":
            await r.hset(TRACKER, "status", "completed")
        elif scenario == "cancelled":
            await r.hset(TRACKER, "status", "cancelled")
        elif scenario == "tracker_missing":
            await r.delete(TRACKER)
        elif scenario == "other_job":
            await r.hset(TRACKER, "job_id", "someone-else")
        elif scenario == "active_slot":
            await r.hset(SLOTS, "slot-0", json.dumps({"status": "busy", "job_id": "orphan-1"}))
        elif scenario == "slot_unreadable":
            await r.hset(SLOTS, "slot-0", "{not json")
        elif scenario == "renewed_lease":
            min_idle = 10 ** 9  # PEL idle is below the required floor
        before = await r.hgetall(TRACKER)
        assert await _terminalize(q, entry, min_idle=min_idle) is False
        pending, dlq, after = await _state(r, q)
        assert len(pending) == 1 and dlq == []
        assert await r.xlen(f"queue:{RG}") == 1
        assert after == before  # tracker not mutated by a refused recovery

    _run(fn, redis_port)


def test_recovery_flag_alone_cannot_bypass_missing_fence_inputs(redis_port):
    async def fn(r, q):
        entry = await _claimed_entry(r, q)
        refused = await q.nack(
            SVC, entry["entry_id"], routing_group=RG, max_retries=1,
            dead_letter_reason="stale_orphan_ttl",
        )
        assert refused is False
        pending, dlq, tracker = await _state(r, q)
        assert len(pending) == 1 and dlq == [] and tracker["status"] == "in_flight"

    _run(fn, redis_port)


def test_submission_identity_survives_requeues_and_fresh_stuck_retries_four_times(redis_port):
    async def fn(r, q):
        await q.enqueue(SVC, {"job_id": "orphan-1", "payload": "x"}, routing_group=RG)
        first = (await r.xrange(f"queue:{RG}"))[0][1]
        submitted = first["_original_submitted_at_ms"]
        assert submitted.isdigit()
        await asyncio.sleep(0.01)
        for attempt in range(1, 4):
            got = await q.dequeue(SVC, "worker-0", routing_group=RG)
            assert got
            pending = await q.get_pending(SVC, routing_group=RG)
            assert await q.nack(SVC, pending[0]["entry_id"], routing_group=RG, max_retries=4)
            entries = await r.xrange(f"queue:{RG}")
            assert len(entries) == 1
            assert entries[0][1]["_nack_count"] == str(attempt)
            assert entries[0][1]["_original_submitted_at_ms"] == submitted
        # Reaper-visible age is the submission age, not the new delivery id.
        got = await q.dequeue(SVC, "worker-0", routing_group=RG)
        pending = await q.get_pending(SVC, routing_group=RG)
        assert pending[0]["entry_created_ms"] == int(submitted)
        assert int(pending[0]["entry_id"].split("-")[0]) >= int(submitted)
        # The fourth stuck delivery exhausts the budget and is dead-lettered.
        assert await q.nack(SVC, pending[0]["entry_id"], routing_group=RG, max_retries=4)
        assert await r.xlen(f"queue:{RG}") == 0
        assert len(await r.xrange(_dead_letter_stream(SVC))) == 1

    _run(fn, redis_port)
