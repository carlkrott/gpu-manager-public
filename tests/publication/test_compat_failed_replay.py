"""A compat retry of a FAILED job with a derived key must re-run.

Live 2026-10-21 (card t_96e02041): broker job 88b47a0d failed at 300 s and the
client's retries at +5 s and +10 s, sent without an Idempotency-Key, got the
same failed job instantly. compat.py derives ``compat-<sha256(body)>`` for such
requests and the store replays any known key, so a FAILED job was sticky.

Only DERIVED keys re-run a FAILED job; a caller-supplied Idempotency-Key keeps
exactly-once replay, and COMPLETED jobs keep replaying under either key.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import threading

from gemma_broker.api import DurableBrokerService
from gemma_broker.compat import CompatibilityWrapper
from gemma_broker.contracts import JobState, MemberSnapshot, MemberState
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import _CandidateCompatAdapter

BODY = json.dumps({"messages": [{"role": "user", "content": "ping"}]}).encode()
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


def _setup():
    service = DurableBrokerService(
        InMemoryJobRepository(), members=lambda: [_member()], clock=lambda: NOW
    )
    wrapper = CompatibilityWrapper(
        _CandidateCompatAdapter(service, polling_wait=None), timeout=5
    )
    return service, wrapper


def _post(wrapper, headers=None):
    response = asyncio.run(
        wrapper.handle_proxy(
            method="POST",
            path="/v1/chat/completions",
            body=BODY,
            headers={"Prefer": "respond-async", **(headers or {})},
        )
    )
    assert response.status == 202, response.text
    return response.headers["X-Job-ID"]


def _fail(service, job_id):
    current = service.accept_for_test(job_id)
    return service.repository.replace_job(
        replace(
            current,
            state=JobState.FAILED,
            completed_at=NOW,
            state_version=current.state_version + 1,
        )
    )


def test_derived_key_retry_after_failure_runs_a_new_job():
    service, wrapper = _setup()
    first = _post(wrapper)
    assert _fail(service, first).state is JobState.FAILED

    retry = _post(wrapper)

    assert retry != first, "a derived-key retry must not replay the failed job"
    assert service.status(retry).state is JobState.QUEUED
    assert service.status(first).state is JobState.FAILED


def test_concurrent_derived_key_retries_after_failure_share_one_new_job():
    service, wrapper = _setup()
    first = _post(wrapper)
    _fail(service, first)

    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def retry() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(_post(wrapper))
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
    assert results[0] != first


def test_repeated_failures_retry_a_bounded_number_of_times():
    service, wrapper = _setup()
    seen = []
    job = _post(wrapper)
    for _ in range(10):
        seen.append(job)
        if service.status(job).state is JobState.FAILED:
            break
        _fail(service, job)
        job = _post(wrapper)
    # 1 original + bounded re-runs; then the last failure replays.
    assert 2 <= len(set(seen)) <= 4
    assert seen[-1] == seen[-2]


def test_caller_key_still_replays_the_failed_job():
    service, wrapper = _setup()
    headers = {"Idempotency-Key": "caller-key-1"}
    first = _post(wrapper, headers)
    _fail(service, first)

    assert _post(wrapper, headers) == first


def test_derived_key_still_replays_a_completed_job():
    service, wrapper = _setup()
    first = _post(wrapper)
    service.complete_for_test(first, {"text": "stored"})

    assert _post(wrapper) == first
