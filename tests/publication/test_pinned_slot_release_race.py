"""R1: a pinned parallel=1 member must admit the request sent just after the previous one completes.

Live shape (go2-record.jsonl, sha256 34f6b2989b1424744916846de0aacfadd9fab17803d8d5c91252d3b5ac0fb26c):
sends 1 and 2 answered 200, send 3 answered 503 650 ms after send 1 completed.

The durable member mirror knows the instant a lease is released, but the
embedded readiness cache only republishes on its own 2 s tick. A pinned submit
that lands inside that window used to read the stale, capacity-derived
``accepting=False`` snapshot and refuse a member whose slot was already free.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gemma_broker.api import APIError, DurableBrokerService
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobState, MemberState, ReasonCode
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import BrokerRuntime

# Reuse the fakes already established by the pinned-admission regressions
# rather than inventing a second repository double.
from test_broker_pin_503_regressions import CHAT, _idle_dispatch_loop, _member

MEMBER_NAME = "Gemini-3.8-Flash-Low"
# The embedded readiness loop republishes every 2 s; a submit inside one tick
# sees the snapshot the previous tick published, not the post-release state.
REFRESH_TICK_SECONDS = 2.0


def _busy_snapshot(member, *, leases: int):
    """The capacity-only snapshot a refresh tick publishes while a lease is held."""
    return replace(
        member,
        state=MemberState.READY_BUSY,
        accepting=False,
        dispatcher_leases=leases,
        compatible_free_slots=0,
        reason_codes=(ReasonCode.MEMBER_NO_CAPACITY,),
        blockers=("no proved immediate slot",),
    )


def _wire(service, loop, repository):
    """Attach both production runtime admission hooks to a bare API service.

    Calls the real ``BrokerRuntime`` methods unbound over a stub, so the test
    exercises the shipped wiring rather than a re-implementation of it.
    """
    stub = SimpleNamespace(dispatch_loop=loop, _repository=repository)
    service.member_admission = lambda name: BrokerRuntime._pinned_member_admission(
        stub, name
    )
    service.member_capacity = lambda name: BrokerRuntime._pinned_member_capacity(
        stub, name
    )
    return service


def _scenario(observer_state_after_lease: str):
    """Drive send 1 -> dispatch -> refresh tick -> completion -> send 3.

    ``observer_state_after_lease`` selects what the readiness cache still holds
    once the lease has been released: ``"stale"`` reproduces the live race (the
    republish tick has not run yet), ``"refreshed"`` is the already-recovered case.
    """
    now = [2.0]
    clock = lambda: now[0]  # noqa: E731 - matches the clock style in the sibling file
    member = _member(MEMBER_NAME)
    repository = InMemoryJobRepository()
    repository.put_member(member)

    observer = [member]
    loop = _idle_dispatch_loop(repository, lambda: observer, clock)
    service = DurableBrokerService(repository, members=lambda: observer, clock=clock)
    _wire(service, loop, repository)

    # send 1: a pinned request against an idle parallel=1 member is admitted.
    first = service.submit(
        {
            "idempotency_key": "pin-release-race-1",
            "request": CHAT,
            "pinned_member": MEMBER_NAME,
        }
    )
    assert first.state is JobState.QUEUED

    requirements = RequestRequirements(
        8, 16, ("chat",), "test", pinned_member=MEMBER_NAME
    )
    reservation = repository.reserve_next(
        now=now[0], ordered_members=observer, requirements=requirements,
        member_lookup=repository.get_member,
    )
    assert reservation is not None, "the pinned dispatch must be able to claim the slot"

    # A readiness tick lands while the lease is held: capacity-only, not lifecycle.
    busy = _busy_snapshot(member, leases=1)
    observer[:] = [busy]
    repository.put_member(busy)

    # Completion: the lease is released and the durable mirror recomputes free.
    repository.release_member_lease(
        MEMBER_NAME,
        str(reservation.job.current_attempt_id),
        claimed_member_state_version=int(reservation.job.member_state_version),
    )
    if observer_state_after_lease == "refreshed":
        recovered = replace(
            busy,
            state=MemberState.READY_ACCEPTING,
            accepting=True,
            dispatcher_leases=0,
            compatible_free_slots=1,
            reason_codes=(),
            blockers=(),
        )
        observer[:] = [recovered]
        repository.put_member(recovered)

    mirror = repository.get_member(MEMBER_NAME)
    assert mirror is not None
    # The durable mirror is the authority for capacity and is already correct.
    assert mirror.compatible_free_slots == 1, (
        "durable mirror must prove the slot free before the pinned submit is judged"
    )
    assert repository.lease_count(MEMBER_NAME) == 0
    return service


def test_pinned_request_sent_right_after_completion_is_admitted():
    """send 3 = 503 in go2-record.jsonl: the released slot must be re-admitted."""
    service = _scenario(observer_state_after_lease="stale")
    now = [2.0]

    third = service.submit(
        {
            "idempotency_key": "pin-release-race-3",
            "request": CHAT,
            "pinned_member": MEMBER_NAME,
        }
    )
    assert third.state is JobState.QUEUED, (
        "a pinned parallel=1 member must admit a request sent immediately after "
        f"the previous completion, got {third.state.value}"
    )
    assert third.pinned_member == MEMBER_NAME
    assert now[0] == 2.0


def test_pinned_submit_does_not_wait_on_the_readiness_refresh_tick():
    """Admission must read the released mirror, not sleep for the next 2 s publish."""
    service = _scenario(observer_state_after_lease="stale")

    import time as _time

    started = _time.monotonic()
    service.submit(
        {
            "idempotency_key": "pin-release-race-3-latency",
            "request": CHAT,
            "pinned_member": MEMBER_NAME,
        }
    )
    elapsed = _time.monotonic() - started
    assert elapsed < REFRESH_TICK_SECONDS, (
        f"pinned admission blocked on the {REFRESH_TICK_SECONDS}s readiness tick "
        f"({elapsed:.3f}s)"
    )


def test_pinned_submit_is_still_refused_fast_while_the_slot_is_genuinely_held():
    """A real saturation is not the race: no free slot must still answer 503."""
    now = [2.0]
    member = _member(MEMBER_NAME)
    repository = InMemoryJobRepository()
    repository.put_member(member)
    observer = [member]
    loop = _idle_dispatch_loop(repository, lambda: observer, lambda: now[0])
    service = DurableBrokerService(repository, members=lambda: observer, clock=lambda: now[0])
    _wire(service, loop, repository)

    service.submit(
        {
            "idempotency_key": "pin-still-busy-1",
            "request": CHAT,
            "pinned_member": MEMBER_NAME,
        }
    )
    requirements = RequestRequirements(8, 16, ("chat",), "test", pinned_member=MEMBER_NAME)
    reservation = repository.reserve_next(
        now=now[0], ordered_members=observer, requirements=requirements,
        member_lookup=repository.get_member,
    )
    assert reservation is not None
    busy = _busy_snapshot(member, leases=1)
    observer[:] = [busy]
    repository.put_member(busy)
    assert repository.get_member(MEMBER_NAME).compatible_free_slots == 0

    with pytest.raises(APIError) as caught:
        service.submit(
            {
                "idempotency_key": "pin-still-busy-3",
                "request": CHAT,
                "pinned_member": MEMBER_NAME,
            }
        )
    assert caught.value.status == 503
    assert caught.value.code == "pinned_member_unavailable"


@pytest.mark.parametrize(
    "lifecycle",
    [
        MemberState.UNHEALTHY,
        MemberState.DRAINING,
        MemberState.JOINING,
        MemberState.OFFLINE,
    ],
    ids=lambda state: state.value,
)
def test_pinned_request_still_refused_when_member_is_lifecycle_unavailable(lifecycle):
    """The fix must not reopen an unavailable member."""
    now = [2.0]
    member = replace(
        _member(MEMBER_NAME),
        state=lifecycle,
        accepting=False,
        compatible_free_slots=1,
    )
    repository = InMemoryJobRepository()
    repository.put_member(member)
    observer = [member]
    loop = _idle_dispatch_loop(repository, lambda: observer, lambda: now[0])
    service = DurableBrokerService(repository, members=lambda: observer, clock=lambda: now[0])
    _wire(service, loop, repository)

    with pytest.raises(APIError) as caught:
        service.submit(
            {
                "idempotency_key": f"pin-lifecycle-{lifecycle.value}",
                "request": CHAT,
                "pinned_member": MEMBER_NAME,
            }
        )
    assert caught.value.status == 503
    assert caught.value.code == "pinned_member_unavailable"


def test_unpinned_routing_is_unchanged_by_the_pinned_admission_path():
    """The fix is scoped to pinned admission: unpinned submits stay unconstrained."""
    service = _scenario(observer_state_after_lease="stale")
    job = service.submit(
        {"idempotency_key": "unpinned-after-release", "request": CHAT}
    )
    assert job.state is JobState.QUEUED
    assert job.pinned_member is None