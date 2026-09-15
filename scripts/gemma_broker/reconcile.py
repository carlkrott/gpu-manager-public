"""Corroborated crash evidence and exactly-once stale-lease repair."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import RLock
from typing import Any, Mapping

from .contracts import JobRecord, JobState, ReasonCode


_LOG = logging.getLogger(__name__)


#: Default threshold (seconds) beyond which a non-active lease is
#: classified as ``orphan`` rather than ``stale_attempt``. The plan
#: pins this at 30 seconds so a transient transport blip is not treated
#: as a recoverable in-flight claim.
DEFAULT_STALE_LEASE_SECONDS: float = 30.0


#: Reason-codes whose lease is NEVER auto-released by the reconciler.
#: These are the active-claim states per the plan: the dispatcher is
#: responsible for terminal release, and the reconciler must not race.
ACTIVE_LEASE_STATES: frozenset[str] = frozenset(
    {"claimed", "accepted", "in_flight"}
)


class CrashVerdict(StrEnum):
    ALIVE = "alive"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"
    PROVEN_DEAD = "proven_dead"


@dataclass(frozen=True, slots=True)
class CrashEvidence:
    service_name: str
    failed_probe_rounds: int
    systemd_active: bool | None
    main_pid_alive: bool | None
    process_present: bool | None
    gpu_utilization_percent: float | None = None
    cpu_utilization_percent: float | None = None
    oom_exit: bool = False


@dataclass(frozen=True, slots=True)
class CrashAssessment:
    verdict: CrashVerdict
    evidence: tuple[str, ...]
    reason_codes: tuple[ReasonCode, ...] = ()


def reduce_crash_evidence(value: CrashEvidence) -> CrashAssessment:
    evidence: list[str] = []
    if value.oom_exit:
        evidence.append("oom_exit")
    evidence.append(f"failed_probe_rounds:{max(0, value.failed_probe_rounds)}")
    for name in ("systemd_active", "main_pid_alive", "process_present"):
        state = getattr(value, name)
        evidence.append(f"{name}:{'unknown' if state is None else str(state).lower()}")

    alive_signals = (
        value.systemd_active is True,
        value.main_pid_alive is True,
        value.process_present is True,
    )
    dead_signals = (
        value.systemd_active is False,
        value.main_pid_alive is False,
        value.process_present is False,
    )
    if any(alive_signals) and any(dead_signals):
        return CrashAssessment(
            CrashVerdict.CONFLICTING,
            tuple(evidence),
            (ReasonCode.CRASH_EVIDENCE_CONFLICT,),
        )
    if all(alive_signals):
        return CrashAssessment(CrashVerdict.ALIVE, tuple(evidence))
    if value.oom_exit and value.process_present is False and value.main_pid_alive is False:
        return CrashAssessment(CrashVerdict.PROVEN_DEAD, tuple(evidence))
    if value.failed_probe_rounds >= 3 and all(dead_signals):
        return CrashAssessment(CrashVerdict.PROVEN_DEAD, tuple(evidence))
    return CrashAssessment(CrashVerdict.UNKNOWN, tuple(evidence))


class InMemoryRepairFence:
    """Semantic reference for one terminal repair per job/attempt."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._claims: dict[str, tuple[int, str, bool]] = {}

    def acquire(self, job_id: str, *, owner: str, fence: int) -> bool:
        if not owner or fence <= 0:
            raise ValueError("repair owner and positive fence are required")
        with self._lock:
            prior = self._claims.get(job_id)
            if prior is not None:
                prior_fence, _prior_owner, completed = prior
                if completed or fence <= prior_fence:
                    return False
            self._claims[job_id] = (fence, owner, False)
            return True

    def complete(self, job_id: str, *, owner: str, fence: int) -> None:
        with self._lock:
            if self._claims.get(job_id) != (fence, owner, False):
                return
            self._claims[job_id] = (fence, owner, True)


class Reconciler:
    def __init__(self, repository, repair_fence: InMemoryRepairFence) -> None:
        self.repository = repository
        self.repair_fence = repair_fence

    def repair(
        self,
        job_id: str,
        evidence: CrashEvidence,
        *,
        repair_owner: str,
        repair_fence: int,
    ) -> JobRecord:
        current = self.repository.get(job_id)
        if current is None:
            raise ValueError(f"UNKNOWN_JOB_ID:{job_id}")
        assessment = reduce_crash_evidence(evidence)
        if assessment.verdict is not CrashVerdict.PROVEN_DEAD:
            return current
        if current.state in {
            JobState.QUEUED,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.OUTCOME_UNKNOWN,
        }:
            return current
        if not self.repair_fence.acquire(
            job_id, owner=repair_owner, fence=repair_fence
        ):
            return self.repository.get(job_id) or current

        if current.accepted_boundary_crossed or current.state in {
            JobState.ACCEPTED,
            JobState.IN_FLIGHT,
        }:
            repaired = replace(
                current,
                state=JobState.OUTCOME_UNKNOWN,
                reason_code=ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN,
                error="proven_dead_after_acceptance",
                state_version=current.state_version + 1,
            )
            repaired = self.repository.replace_job(repaired)
        else:
            repaired = self.repository.requeue_claim(
                current, reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE
            )
        self.repair_fence.complete(
            job_id, owner=repair_owner, fence=repair_fence
        )
        return repaired


class LeaseReconcilerUnavailable(RuntimeError):
    """Raised when the Redis-side reconcile tick cannot complete.

    The plan's fail-closed contract: any Redis error during reconciliation
    aborts the tick and surfaces to the runtime; no partial write is
    committed, no auto-retry is attempted. The next tick (driven by the
    runtime's readiness-refresh loop) retries with the same inputs.
    """


@dataclass(frozen=True, slots=True)
class LeaseClassification:
    """Output of :meth:`LeaseReconciler.classify_leases`.

    ``active`` are attempt_ids whose lease the reconciler must NEVER
    auto-release (the dispatcher owns terminal release). ``orphan`` are
    leases whose attempt is in a non-active state and whose
    last_heartbeat_at is past ``stale_threshold_seconds`` — surfaced
    for the operator-`GO`'d repair hook, never auto-deleted.
    ``stale_attempt`` is the in-flight-but-stale partition (state is
    still ``claimed`` / ``accepted`` but the heartbeat is past the
    threshold) — kept distinct so the runtime can log and continue
    without taking action.
    """

    active: tuple[str, ...] = ()
    orphan: tuple[str, ...] = ()
    stale_attempt: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LeaseReconciler:
    """Authoritative seam for member lease / free-slot reconciliation.

    Funnels fenced release, live SCARD, mirror update, active-claim
    preservation, orphan classification, drain/rejoin fence respect, and
    Redis-error fail-closed behaviour through a single Python method
    backed by atomic Redis primitives (Lua scripts wherever a
    read-then-write or multi-key mutation is required).

    The seam is intentionally minimal and stateless beyond the injected
    repository. It is **idempotent** by construction: a tick that finds
    no change produces no writes, and a tick that observes the same
    state twice produces the same writes.

    Behavioural guarantees (plan §4.2, restated):

      * At most one ``put_member`` per member per tick.
      * Never touches ``state``, ``accepting``, or ``generation_fence``
        while the member is ``DRAINING`` or ``JOINING``. The Lua's
        state-version CAS remains the sole authority on the drain /
        rejoin fence.
      * Orphan classification never auto-deletes leases. The
        operator-`GO`'d ``_repair_orphan_leases`` startup hook is the
        only consumer.
      * All Lua invocations are wrapped so a Redis error raises
        :class:`LeaseReconcilerUnavailable` and aborts the tick.
    """

    repository: Any
    stale_threshold_seconds: float = DEFAULT_STALE_LEASE_SECONDS

    # The repository protocol the runtime uses is duck-typed; tests
    # inject in-memory fakes, the candidate uses RedisJobRepository. We
    # only require the methods we actually call.
    _REQUIRED_METHODS: frozenset[str] = frozenset(
        {
            "get_member",
            "lease_count",
            "put_member",
            "add_member_lease",
            "_attempt_key",
        }
    )

    def _require_repository(self) -> None:
        missing = tuple(
            sorted(
                name
                for name in self._REQUIRED_METHODS
                if not callable(getattr(self.repository, name, None))
            )
        )
        if missing:
            raise TypeError(
                f"LeaseReconciler repository missing required methods: {missing}"
            )

    def _attempt_state(self, attempt_id: str) -> str | None:
        """Read the attempt:<id> HGET state field. Returns None when the
        attempt key is absent or the Redis read fails."""
        try:
            raw = self.repository.client.hget(
                self.repository._attempt_key(attempt_id), "state"
            )
        except Exception as exc:  # noqa: BLE001 — broad on purpose, see contract
            raise LeaseReconcilerUnavailable(str(exc)) from exc
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def _attempt_last_heartbeat(self, attempt_id: str) -> float | None:
        """Read the attempt:<id> HGET last_heartbeat_at field (Unix seconds).
        Returns None when the field is absent or unparseable."""
        try:
            raw = self.repository.client.hget(
                self.repository._attempt_key(attempt_id), "last_heartbeat_at"
            )
        except Exception as exc:  # noqa: BLE001
            raise LeaseReconcilerUnavailable(str(exc)) from exc
        if raw is None:
            return None
        try:
            return float(raw if not isinstance(raw, bytes) else raw.decode("utf-8"))
        except (TypeError, ValueError):
            return None

    def classify_leases(
        self,
        member_name: str,
        *,
        now: float | None = None,
    ) -> LeaseClassification:
        """Walk ``leases:<member>`` and classify each attempt_id.

        Active claims (state in ACTIVE_LEASE_STATES) go to ``active``.
        Non-active leases with a stale heartbeat go to ``orphan``;
        non-active leases without a heartbeat (or with a fresh one)
        are reported as ``stale_attempt`` to distinguish "never seen"
        from "definitely orphaned".
        """
        self._require_repository()
        if now is None:
            import time as _time  # local import keeps reconcile import surface lean

            now = _time.time()
        try:
            members_set = list(
                self.repository.client.smembers(
                    self.repository._lease_key(member_name)
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise LeaseReconcilerUnavailable(str(exc)) from exc

        active: list[str] = []
        orphan: list[str] = []
        stale: list[str] = []
        for raw in members_set:
            attempt_id = (
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            )
            state = self._attempt_state(attempt_id)
            if state in ACTIVE_LEASE_STATES:
                active.append(attempt_id)
                continue
            heartbeat = self._attempt_last_heartbeat(attempt_id)
            if heartbeat is not None and (now - heartbeat) > self.stale_threshold_seconds:
                orphan.append(attempt_id)
            else:
                stale.append(attempt_id)
        return LeaseClassification(
            active=tuple(active),
            orphan=tuple(orphan),
            stale_attempt=tuple(stale),
        )

    def reconcile_member(
        self,
        member_name: str,
        *,
        now: float | None = None,
    ) -> Any | None:
        """Reconcile a member's lease set and free-slot mirror.

        Behaviour:

          * Reads the authoritative lease count via
            :meth:`Repository.lease_count` (``SCARD leases:<member>``).
          * If the mirror says ``compatible_free_slots == 0`` but
            lease_count is 0 AND the backend reports no busy slots,
            repairs the mirror to ``max(0, configured - backend_busy)``.
          * Classifies every lease and emits WARN logs for orphans
            (no auto-delete — the operator-`GO`'d repair hook is the
            only consumer).
          * Returns the live MemberSnapshot (post any mirror write) or
            ``None`` if the member is unknown.

        Returns the live MemberSnapshot or None if the member is unknown.
        """
        self._require_repository()
        if now is None:
            import time as _time

            now = _time.time()
        # Classify first so we can surface orphan observations even when
        # the member hash is fresh and put_member is a no-op.
        classification = self.classify_leases(member_name, now=now)
        for attempt_id in classification.orphan:
            _LOG.warning(
                "lease_reconciler.orphan_surfaced member=%s attempt_id=%s",
                member_name,
                attempt_id,
            )
        for attempt_id in classification.stale_attempt:
            _LOG.info(
                "lease_reconciler.stale_attempt_surfaced member=%s attempt_id=%s",
                member_name,
                attempt_id,
            )

        # Read the live snapshot. If the member key is absent (clean
        # wipe, no put_member has written it yet), return None and do
        # not invent capacity.
        try:
            snapshot = self.repository.get_member(member_name)
        except Exception as exc:  # noqa: BLE001
            raise LeaseReconcilerUnavailable(str(exc)) from exc
        if snapshot is None:
            return None
        # Drain / rejoin fence: never touch state_version, state,
        # accepting, or generation_fence while the member is in a
        # fenced state. The Lua's state-version CAS is the sole authority.
        from .contracts import MemberState as _MemberState  # local to avoid cycle

        if snapshot.state in {_MemberState.DRAINING, _MemberState.JOINING}:
            return snapshot

        # Free-slot repair: if the mirror is latched to 0 but the
        # authoritative inputs (lease_count + backend_busy) say there
        # is capacity, the runtime's refresh has clobbered the mirror
        # with a stale view. Repair via the same put_member write path
        # the runtime uses, but only when lease_count == 0 AND
        # backend_busy == 0 (a stale latch only).
        try:
            lease_count = int(
                self.repository.lease_count(member_name)
            )
        except Exception as exc:  # noqa: BLE001
            raise LeaseReconcilerUnavailable(str(exc)) from exc

        mirror_free = int(getattr(snapshot, "compatible_free_slots", 0) or 0)
        backend_busy = getattr(snapshot, "backend_slots_busy", None)
        configured = int(getattr(snapshot, "configured_slots", 0) or 0)
        backend_total = getattr(snapshot, "backend_slots_total", None)
        if configured <= 0 and backend_total is not None:
            configured = int(backend_total)
        if (
            lease_count == 0
            and mirror_free <= 0
            and snapshot.accepting
            and configured > 0
            and backend_busy is not None
            and int(backend_busy) == 0
        ):
            # Authoritative capacity. LeaseReconciler enforces idempotence:
            # put_member is at most one per member per tick, and a second
            # tick with the same inputs produces the same write.
            repaired = replace(snapshot, compatible_free_slots=configured)
            try:
                self.repository.put_member(repaired)
                return self.repository.get_member(member_name)
            except Exception as exc:  # noqa: BLE001
                raise LeaseReconcilerUnavailable(str(exc)) from exc
        return snapshot
