"""Secret-free broker metrics projections.

The snapshot layer is the consumer-side surface of the Track 3
observability contract. Every authoritative field is annotated with

  * ``live_lease_count`` (the SCARD authoritative value)
  * ``live_lease_count_source`` (a labelled path so the consumer can
    verify the source is the SCARD read, not a cached mirror)
  * ``fresh`` / ``stale`` / ``unknown`` — the three-way boolean
    semantics that the route must report verbatim
  * ``compatible_free_slots_derived`` / ``dispatcher_leases_derived`` —
    the cache-mirror fields are labelled derived so the consumer can
    reject them as ground truth.

The handler is intentionally schema-versioned: ``schema``,
``namespace``, and ``leader`` are pass-through kwargs so the route can
inject the runtime's namespace and the durable ``owner_token`` (never
``id(self)``).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .contracts import JobRecord, MemberSnapshot


SCHEMA_VERSION = "combined-gemma.broker-metric.v1"


def _lease_count_key(member_name: str) -> str:
    return f"SCARD leases:{member_name}"


class BrokerMetrics:
    def __init__(self, *, freshness_seconds: float = 5.0) -> None:
        if freshness_seconds <= 0:
            raise ValueError("freshness_seconds must be positive")
        self.freshness_seconds = freshness_seconds
        self._jobs_by_state: Counter[str] = Counter()
        self._jobs_observed = 0

    def observe_job(self, job: JobRecord) -> None:
        self._jobs_by_state[job.state.value] += 1
        self._jobs_observed += 1

    def snapshot(
        self,
        *,
        members: list[MemberSnapshot],
        queue_depth: int,
        now: float,
        lease_counts: dict[str, int] | None = None,
        schema: str = SCHEMA_VERSION,
        namespace: str | None = None,
        leader: dict[str, Any] | None = None,
        queue_depths_by_priority: dict[str, int] | None = None,
        attempts_total: int | None = None,
        attempts_by_state: dict[str, int] | None = None,
        jobs_total: int | None = None,
        jobs_by_state: dict[str, int] | None = None,
        dispatch_loop: dict[str, Any] | None = None,
    ) -> dict:
        """Project the metrics snapshot.

        The caller's ``lease_counts`` map is the SCARD projection for
        every member. The ``leader`` dict is the runtime's
        ``owner_token``-keyed identity (the durable leader record).
        Both are required for the route's payload — they are
        separately testable so the snapshot layer does not silently
        fall back to process-local identity.
        """
        lease_counts = lease_counts or {}
        member_metrics: dict[str, dict[str, Any]] = {}
        for member in members:
            obs_age = now - member.observed_at
            fresh = obs_age <= self.freshness_seconds
            # Lease count is authoritative when the route (or the
            # snapshot caller) supplied a SCARD projection key for the
            # member. Anything else is unknown — the SCARD read erred
            # or the key was missing.
            unknown = member.name not in lease_counts
            live_count = lease_counts.get(member.name)
            member_metrics[member.name] = {
                "name": member.name,
                "state": member.state.value,
                "accepting": member.accepting,
                "state_version": member.state_version,
                "generation_fence": member.generation_fence,
                "live_lease_count": int(live_count) if live_count is not None else None,
                "live_lease_count_source": _lease_count_key(member.name),
                "configured_slots": int(member.configured_slots),
                "backend_slots_total": (
                    int(member.backend_slots_total)
                    if member.backend_slots_total is not None
                    else None
                ),
                "backend_slots_busy": (
                    int(member.backend_slots_busy)
                    if member.backend_slots_busy is not None
                    else None
                ),
                "compatible_free_slots": int(member.compatible_free_slots or 0),
                "compatible_free_slots_derived": True,
                "dispatcher_leases": int(member.dispatcher_leases),
                "dispatcher_leases_derived": True,
                "freshness_window_s": self.freshness_seconds,
                "fresh": bool(fresh) and not unknown,
                "stale": (not fresh) or unknown,
                "unknown": unknown,
                "observed_at": float(member.observed_at),
                "reason_codes": [item.value for item in member.reason_codes],
                "blockers": list(member.blockers),
            }
        return {
            "schema": schema,
            "namespace": namespace,
            "leader": leader if leader is not None else {},
            "queue": {
                "depth": max(0, int(queue_depth)),
                "by_priority": dict(sorted((queue_depths_by_priority or {}).items())),
                "fresh": True,
            },
            "members": member_metrics,
            "dispatch_loop": dispatch_loop if dispatch_loop is not None else {},
            "attempts": {
                "total": int(attempts_total) if attempts_total is not None else 0,
                "by_state": dict(sorted((attempts_by_state or {}).items())),
                "fresh": True,
            },
            "jobs": {
                "depth": max(0, int(queue_depth)),
                "total": int(jobs_total) if jobs_total is not None else self._jobs_observed,
                "jobs_observed": self._jobs_observed,
                "by_state": dict(
                    sorted(
                        (jobs_by_state if jobs_by_state is not None else self._jobs_by_state).items()
                    )
                ),
                "fresh": jobs_by_state is not None,
            },
        }
