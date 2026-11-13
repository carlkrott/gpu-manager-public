"""A same-key retry must re-run a job cancelled before acceptance.

Live 2026-10-17 (pibook e2e #7): one Idempotency-Key produced job 95933b45,
which sat in the queue past the 300 s compatibility timeout. compat.py
cancelled the still-queued job and answered 504 compatibility_timeout with
retry_safe=true, but all 11 retries with the SAME key replayed the cancelled
job instantly, so the promise the 504 made could never be kept.

The accepted boundary is ``JobRecord.accepted_boundary_crossed`` (set by the
dispatcher's on_accepted path and by api.accept_for_test). A job that is
CANCELLED and never crossed that boundary is safe to re-run, so the same key
must create a NEW job; anything accepted, in flight or completed keeps
replaying, and a different body under the same key stays 409.

The reservation tests run the real Lua against a throwaway ``redis-server`` on
loopback (fakeredis has no EVAL).
"""
from __future__ import annotations

from dataclasses import replace
import shutil
import socket
import subprocess
import threading
import time

import pytest
import redis

from gemma_broker.api import APIError, DurableBrokerService
from gemma_broker.contracts import (
    JobState,
    MemberSnapshot,
    MemberState,
    ReasonCode,
)
from gemma_broker.redis_store import InMemoryJobRepository, RedisJobRepository

CHAT = {"messages": [{"role": "user", "content": "ping"}]}
OTHER_CHAT = {"messages": [{"role": "user", "content": "pong"}]}
SCOPE = "api"
NOW = 2.0


def _member():
    return MemberSnapshot(
        name="Gemma-CPU",
        gpu_id=None,
        state=MemberState.READY_ACCEPTING,
        accepting=True,
        configured_slots=1,
        context_per_slot=8192,
        state_version=1,
        observed_at=1.0,
        compatible_free_slots=1,
        capabilities=("chat",),
    )


def _service(repository):
    return DurableBrokerService(
        repository, members=lambda: [_member()], clock=lambda: NOW
    )


@pytest.fixture
def redis_port(tmp_path_factory):
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server not installed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
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


@pytest.fixture(params=["redis", "memory"])
def service(request, redis_port):
    """The repository under test, or an InMemory mirror for the same rule."""
    client = None
    if request.param == "redis":
        client = redis.Redis(host="127.0.0.1", port=redis_port)
        client.flushall()
        repository = RedisJobRepository(
            client, prefix="idem:", leadership_clock=lambda: NOW
        )
    else:
        repository = InMemoryJobRepository()
    yield _service(repository)
    if client is not None:
        client.close()


def _cancel_before_acceptance(service, job_id):
    return service.cancel(job_id)


def _cancel_after_acceptance(service, job_id):
    accepted = service.accept_for_test(job_id)
    return service.repository.replace_job(
        replace(
            accepted,
            state=JobState.CANCELLED,
            completed_at=NOW,
            reason_code=ReasonCode.CANCELLED_AFTER_ACCEPTANCE,
            state_version=accepted.state_version + 1,
        )
    )


def _resubmit(service, key="key-1", request=CHAT):
    return service.submit(
        {"idempotency_key": key, "request": request, "caller_scope": SCOPE}
    )


# ------------------------------------------------------- the failing case


def test_same_key_retry_after_cancel_before_acceptance_runs_a_new_job(service):
    """The 504 says retry_safe=true, so the retry must actually re-run."""
    first = _resubmit(service)
    assert first.state is JobState.QUEUED
    cancelled = _cancel_before_acceptance(service, first.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.accepted_boundary_crossed is False
    assert cancelled.reason_code is ReasonCode.CANCELLED_BEFORE_ACCEPTANCE

    second = _resubmit(service)

    assert second.job_id != cancelled.job_id, (
        "the resubmitted job_id differs from the cancelled job"
    )
    assert second.state is JobState.QUEUED

    completed = service.complete_for_test(second.job_id, {"text": "pong"})
    assert completed.state is JobState.COMPLETED
    assert completed.result == {"text": "pong"}
    # The cancelled job keeps its own terminal record; it is not resurrected.
    assert service.status(cancelled.job_id).state is JobState.CANCELLED


# --------------------------------------------------------- regressions


def test_accepted_job_replays_the_same_job_id(service):
    accepted = service.accept_for_test(_resubmit(service).job_id)
    replay = _resubmit(service)
    assert replay.job_id == accepted.job_id
    assert replay.state is JobState.ACCEPTED


def test_completed_job_replays_the_stored_result(service):
    completed = service.complete_for_test(
        _resubmit(service).job_id, {"text": "stored"}
    )
    replay = _resubmit(service)
    assert replay.job_id == completed.job_id
    assert replay.result == {"text": "stored"}


def test_same_key_with_a_different_body_is_still_refused(service):
    _resubmit(service)
    with pytest.raises(APIError) as caught:
        _resubmit(service, request=OTHER_CHAT)
    assert caught.value.status == 409
    assert caught.value.code == "idempotency_key_reused"


def test_cancelled_after_acceptance_still_replays(service):
    cancelled = _cancel_after_acceptance(service, _resubmit(service).job_id)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.accepted_boundary_crossed is True
    replay = _resubmit(service)
    assert replay.job_id == cancelled.job_id


def test_cancelled_before_acceptance_retry_then_runs_twice(service):
    """A second retry after the new job is cancelled starts over again."""
    first = _resubmit(service)
    _cancel_before_acceptance(service, first.job_id)
    second = _resubmit(service)
    _cancel_before_acceptance(service, second.job_id)
    third = _resubmit(service)
    assert len({first.job_id, second.job_id, third.job_id}) == 3


# ----------------------------------------------------------------- race


def test_concurrent_same_key_resubmits_after_cancel_create_one_job(service):
    first = _resubmit(service)
    _cancel_before_acceptance(service, first.job_id)

    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def retry() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(_resubmit(service).job_id)
        except BaseException as exc:  # noqa: BLE001 - reported by the assert
            errors.append(exc)

    threads = [threading.Thread(target=retry) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(results) == 2
    assert len(set(results)) == 1, f"both retries must share one job: {results}"
    assert set(results) != {first.job_id}, (
        "the winner must not be the cancelled job"
    )

def test_terminal_job_expires_and_same_key_runs_again(redis_port):
    """Terminal jobs get a 30 min TTL; once gone, the same key is a new job."""
    from gemma_broker.redis_store import TERMINAL_JOB_RETENTION_SECONDS

    client = redis.Redis(host="127.0.0.1", port=redis_port)
    client.flushall()
    service = _service(
        RedisJobRepository(client, prefix="idem:", leadership_clock=lambda: NOW)
    )
    done = service.complete_for_test(_resubmit(service).job_id, {"text": "x"})
    ttl = client.ttl(f"idem:job:{done.job_id}")
    assert 0 < ttl <= TERMINAL_JOB_RETENTION_SECONDS == 1800
    client.delete(f"idem:job:{done.job_id}")  # what the expiry does
    again = _resubmit(service)
    assert again.job_id != done.job_id
    assert again.state is JobState.QUEUED
    client.close()
