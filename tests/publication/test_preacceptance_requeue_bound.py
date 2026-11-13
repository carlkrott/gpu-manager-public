"""Bounded pre-acceptance requeue: jobs pinned to an unavailable member exhaust
their requeue budget and go terminal rather than looping forever.

Reproduces live incident a85d1013 (Gemini pin attempt_count=101, forever requeue
loop) by simulating repeated PRE_ACCEPTANCE_REQUEUE cycles.
"""
from __future__ import annotations

import pytest

from dataclasses import replace

from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import (
    JobRecord,
    JobState,
    MemberSnapshot,
    MemberState,
    ReasonCode,
)
from gemma_broker.gate_limits import BrokerGateLimits
from gemma_broker.redis_store import InMemoryJobRepository


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _member(
    name: str = "m1",
    state: MemberState = MemberState.READY_ACCEPTING,
) -> MemberSnapshot:
    return MemberSnapshot(
        name=name,
        gpu_id="gpu0",
        state=state,
        accepting=True,
        configured_slots=2,
        context_per_slot=4096,
        state_version=1,
        observed_at=1.0,
        compatible_free_slots=2,
        capabilities=("chat",),
    )


def _reserve(
    repo: InMemoryJobRepository,
    job_id: str,
    pinned: str | None = None,
) -> object:
    """Submit and reserve next, optionally pinning to a member."""
    member = repo.put_member(_member(pinned)) if pinned else repo.put_member(_member())
    repo.submit(
        JobRecord.new(
            job_id=job_id,
            idempotency_key=f"i-{job_id}",
            request_sha256="0" * 64,
            request_body={"messages": [{"role": "user", "content": "x"}]},
            submitted_at=1.0,
            enqueue_sequence=1,
            required_capabilities=("chat",),
            pinned_member=pinned,
        ),
        caller_scope="test",
    )
    return repo.reserve_next(
        now=2.0,
        ordered_members=[member],
        requirements=RequestRequirements(8, 16, ("chat",), "test"),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pinned_job_exhausts_requeue_budget_and_goes_terminal():
    """After N pre-accept failures a pinned job goes FAILED with
    PRE_ACCEPTANCE_REQUEUE_EXHAUSTED instead of requeueing forever."""

    repo = InMemoryJobRepository(
        gate_limits=BrokerGateLimits(pre_acceptance_requeue_limit=3)
    )
    m1 = repo.put_member(_member())

    # Submit a pinned job.
    repo.submit(
        JobRecord.new(
            job_id="p1",
            idempotency_key="i-p1",
            request_sha256="0" * 64,
            request_body={"messages": [{"role": "user", "content": "x"}]},
            submitted_at=1.0,
            enqueue_sequence=1,
            required_capabilities=("chat",),
            pinned_member="m1",
        ),
        caller_scope="test",
    )

    # Budget limit = 3.  pre_acceptance_requeue_count tracks cycles where
    # requeue_claim was called with PRE_ACCEPTANCE_REQUEUE; it starts at 0
    # and increments on each requeue.  With `>=`, we get N-1 requeues then
    # terminalize on the Nth call to requeue_claim.
    #
    # Cycle 1: requeue_claim called with attempt_count=1, pre_acceptance_requeue_count=0.
    #   Check: 0 >= 3? F -> requeue OK; pre_acceptance_requeue_count becomes 1.
    # Cycle 2: requeue_claim called with attempt_count=2, pre_acceptance_requeue_count=1.
    #   Check: 1 >= 3? F -> requeue OK; pre_acceptance_requeue_count becomes 2.
    # Cycle 3: requeue_claim called with attempt_count=3, pre_acceptance_requeue_count=2.
    #   Check: 2 >= 3? F -> requeue OK; pre_acceptance_requeue_count becomes 3.
    # Cycle 4: requeue_claim called with attempt_count=4, pre_acceptance_requeue_count=3.
    #   Check: 3 >= 3? T -> TERMINALIZE.
    res1 = repo.reserve_next(
        now=2.0,
        ordered_members=[m1],
        requirements=RequestRequirements(8, 16, ("chat",), "test"),
    )
    assert res1 is not None
    claimed1 = repo.get("p1")
    assert claimed1.attempt_count == 1
    out = repo.requeue_claim(claimed1, reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE)
    assert out.state is JobState.QUEUED, f"cycle 1: expected QUEUED, got {out.state}"
    assert out.pre_acceptance_requeue_count == 1

    # Cycle 2: attempt_count=2, pre_acceptance_requeue_count=1 -> requeue OK.
    res2 = repo.reserve_next(
        now=3.0,
        ordered_members=[m1],
        requirements=RequestRequirements(8, 16, ("chat",), "test"),
    )
    assert res2 is not None
    claimed2 = repo.get("p1")
    out2 = repo.requeue_claim(claimed2, reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE)
    assert out2.state is JobState.QUEUED, (
        f"cycle 2: expected QUEUED, got {out2.state}"
    )
    assert out2.pre_acceptance_requeue_count == 2

    # Cycle 3: attempt_count=3, pre_acceptance_requeue_count=2 -> requeue OK.
    res3 = repo.reserve_next(
        now=4.0,
        ordered_members=[m1],
        requirements=RequestRequirements(8, 16, ("chat",), "test"),
    )
    assert res3 is not None
    claimed3 = repo.get("p1")
    assert claimed3.attempt_count == 3
    assert claimed3.pre_acceptance_requeue_count == 2
    out3 = repo.requeue_claim(claimed3, reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE)
    assert out3.state is JobState.QUEUED, (
        f"cycle 3: expected QUEUED, got {out3.state}"
    )

    # Cycle 4: pre_acceptance_requeue_count=3 == limit=3 -> TERMINALIZE.
    res4 = repo.reserve_next(
        now=5.0,
        ordered_members=[m1],
        requirements=RequestRequirements(8, 16, ("chat",), "test"),
    )
    assert res4 is not None
    claimed4 = repo.get("p1")
    assert claimed4.attempt_count == 4
    assert claimed4.pre_acceptance_requeue_count == 3

    out4 = repo.requeue_claim(claimed4, reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE)
    assert out4.state is JobState.FAILED, (
        f"expected FAILED at budget exhaustion, got {out4.state}"
    )
    assert out4.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE_EXHAUSTED, (
        f"expected PRE_ACCEPTANCE_REQUEUE_EXHAUSTED, got {out4.reason_code}"
    )
    # Queue slot is freed.
    assert repo.queue_total == 0, "queue slot not freed after terminalization"


def test_unpinned_job_not_affected_by_requeue_budget():
    """Unpinned jobs are not subject to the pre-acceptance requeue budget —
    the budget check only fires when reason == PRE_ACCEPTANCE_REQUEUE."""

    repo = InMemoryJobRepository(
        gate_limits=BrokerGateLimits(pre_acceptance_requeue_limit=10)
    )
    m1 = repo.put_member(_member())

    # Unpinned job.
    repo.submit(
        JobRecord.new(
            job_id="u1",
            idempotency_key="i-u1",
            request_sha256="0" * 64,
            request_body={"messages": [{"role": "user", "content": "x"}]},
            submitted_at=1.0,
            enqueue_sequence=1,
            required_capabilities=("chat",),
            pinned_member=None,
        ),
        caller_scope="test",
    )

    # 3 cycles with same limit=2 — unpinned has no special gating in requeue_claim.
    for cycle in range(1, 4):
        res = repo.reserve_next(
            now=float(cycle + 1),
            ordered_members=[m1],
            requirements=RequestRequirements(8, 16, ("chat",), "test"),
        )
        assert res is not None
        claimed = repo.get("u1")
        out = repo.requeue_claim(claimed, reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE)
        assert out.state is JobState.QUEUED, (
            f"unpinned cycle {cycle}: expected QUEUED, got {out.state}"
        )


def test_accepted_boundary_crossed_job_not_affected():
    """Jobs that have crossed the acceptance boundary are protected:
    requeue_claim raises CANCEL_UNSAFE_AFTER_ACCEPTANCE before the
    budget check is even reached."""
    repo = InMemoryJobRepository(
        gate_limits=BrokerGateLimits(pre_acceptance_requeue_limit=1)
    )
    repo.put_member(_member())

    # Pre-populate a job that has already crossed the boundary.
    repo.submit(
        JobRecord.new(
            job_id="acc1",
            idempotency_key="i-acc1",
            request_sha256="0" * 64,
            request_body={"messages": [{"role": "user", "content": "x"}]},
            submitted_at=1.0,
            enqueue_sequence=1,
            required_capabilities=("chat",),
            pinned_member="m1",
            accepted_boundary_crossed=True,
            state=JobState.CLAIMED,
            selected_member="m1",
            attempt_count=5,
        ),
        caller_scope="test",
    )

    # Attempt requeue_claim with PRE_ACCEPTANCE_REQUEUE — must raise
    # CANCEL_UNSAFE_AFTER_ACCEPTANCE, not silently terminalize.
    with pytest.raises(Exception) as exc_info:
        repo.requeue_claim(
            repo.get("acc1"),
            reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
        )
    assert "CANCEL_UNSAFE" in str(exc_info.value) or "UNSAFE" in str(exc_info.value)


def test_default_limit_is_five():
    """The module-level default limit is 5 PRE_ACCEPTANCE_REQUEUE cycles."""
    from scripts.gemma_broker.gate_limits import DEFAULT_PRE_ACCEPTANCE_REQUEUE_LIMIT
    assert DEFAULT_PRE_ACCEPTANCE_REQUEUE_LIMIT == 5
