"""Durable bounded event producer for the combined-Gemma broker.

This module implements the producer-side contract for the
``combined-gemma.lease-event.v1`` / ``combined-gemma.member-rank.v1`` /
``combined-gemma.lifecycle-event.v1`` schemas (the §6.2 schema table in
the reconciled plan). The producer is a bounded ring buffer that holds
AT MOST ``max_entries`` events (the canonical cap is 100); older events
are evicted FIFO when the cap is hit.

The producer is intentionally lock-free for the recording path (the
caller already serializes through the dispatch loop's coroutine); the
:func:`snapshot` method returns a list copy so the consumer can iterate
without contending with the producer.

The producer is also safe to use in hermetic tests: the test suite
constructs it directly with a 100-entry cap and asserts the eviction
behaviour.
"""
from __future__ import annotations

from collections import deque
from typing import Any


# Canonical schema literals — the producer MUST emit these so consumers
# can version-match without parsing the body.
LEASE_EVENT_SCHEMA = "combined-gemma.lease-event.v1"
MEMBER_RANK_SCHEMA = "combined-gemma.member-rank.v1"
LIFECYCLE_EVENT_SCHEMA = "combined-gemma.lifecycle-event.v1"

# Default cap from the reconciled plan §6.2 — the producer may never
# exceed 100 entries; this is the only contract value.
DEFAULT_MAX_ENTRIES = 100


class BoundedEventProducer:
    """Bounded ring buffer of broker events (max 100 entries).

    The producer is wired into the runtime's terminal branches and the
    dispatch loop's failure counters; it captures the producer-side
    wire shape independently of any Redis side-channel.
    """

    def __init__(self, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = int(max_entries)
        self._events: deque[dict[str, Any]] = deque(maxlen=self._max_entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def size(self) -> int:
        return len(self._events)

    def record_lease(
        self,
        *,
        kind: str,
        member: str,
        mode: str,
        attempt_id: str,
        fence: int,
        leases_after: int,
        free_after: int,
        ts: float,
    ) -> dict[str, Any]:
        """Append a lease-event (``combined-gemma.lease-event.v1``)."""
        event = {
            "schema": LEASE_EVENT_SCHEMA,
            "kind": kind,
            "ts": float(ts),
            "member": str(member),
            "mode": str(mode),
            "attempt_id": str(attempt_id),
            "fence": int(fence),
            "leases_after": int(leases_after),
            "free_after": int(free_after),
        }
        self._events.append(event)
        return event

    def record_member(
        self,
        *,
        kind: str,
        member: str,
        state: str,
        state_version: int,
        compatible_free_slots: int,
        dispatcher_leases: int,
        backend_slots_busy: int,
        configured_slots: int,
        reason_codes: tuple[str, ...] | list[str],
        ts: float,
    ) -> dict[str, Any]:
        """Append a member-rank event (``combined-gemma.member-rank.v1``)."""
        event = {
            "schema": MEMBER_RANK_SCHEMA,
            "kind": str(kind),
            "ts": float(ts),
            "member": str(member),
            "state": str(state),
            "state_version": int(state_version),
            "compatible_free_slots": int(compatible_free_slots),
            "dispatcher_leases": int(dispatcher_leases),
            "backend_slots_busy": int(backend_slots_busy),
            "configured_slots": int(configured_slots),
            "reason_codes": list(reason_codes),
        }
        self._events.append(event)
        return event

    def record_lifecycle(
        self,
        *,
        kind: str,
        from_owner: str,
        to_owner: str,
        from_token: int,
        to_token: int,
        fence: int,
        expires_at: float,
        ts: float,
    ) -> dict[str, Any]:
        """Append a lifecycle event (``combined-gemma.lifecycle-event.v1``)."""
        event = {
            "schema": LIFECYCLE_EVENT_SCHEMA,
            "kind": str(kind),
            "ts": float(ts),
            "from_owner": str(from_owner),
            "to_owner": str(to_owner),
            "from_token": int(from_token),
            "to_token": int(to_token),
            "fence": int(fence),
            "expires_at": float(expires_at),
        }
        self._events.append(event)
        return event

    def record_failure(
        self,
        *,
        kind: str,
        member: str | None,
        attempt_id: str | None,
        error: str | None,
        ts: float,
    ) -> dict[str, Any]:
        """Append a generic dispatch-one failure event.

        The dispatch loop's failure counter discipline is exposed through
        this shape so the producer side mirrors the runtime's
        ``_dispatch_one_fail_counters`` mapping.
        """
        event = {
            "schema": LIFECYCLE_EVENT_SCHEMA,
            "kind": str(kind),
            "ts": float(ts),
            "member": member if member is None else str(member),
            "attempt_id": attempt_id if attempt_id is None else str(attempt_id),
            "error": error if error is None else str(error),
        }
        self._events.append(event)
        return event

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a list copy of the buffered events.

        The copy is order-preserving so consumers can render the
        chronological ring buffer without contending with the
        producer.
        """
        return list(self._events)

    def clear(self) -> None:
        """Drop all buffered events.  Used by the runtime's tests."""
        self._events.clear()
