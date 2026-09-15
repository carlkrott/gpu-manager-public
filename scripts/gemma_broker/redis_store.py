"""Repository contracts and deterministic in-memory reference implementation.

The in-memory implementation is the semantic oracle for unit tests.  Task 6 adds
an equivalent fenced Redis transaction without changing these public rules.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Callable
from uuid import uuid4

from redis.exceptions import RedisError, ResponseError

from .compatibility import RequestRequirements, evaluate
from .contracts import (
    AttemptRecord,
    AttemptState,
    ContractError,
    JobRecord,
    JobState,
    MemberSnapshot,
    MemberState,
    Priority,
    ReasonCode,
    validate_member_transition,
)


LUA_DIR = Path(__file__).resolve().with_name("lua")


def _load_lua(name: str) -> str:
    path = LUA_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"BROKER_LUA_RESOURCE_MISSING:{path}") from exc


RESERVE_JOB_LUA = _load_lua("reserve_job.lua")
SUBMIT_JOB_LUA = _load_lua("submit_job.lua")
REPLACE_JOB_LUA = _load_lua("replace_job.lua")
MEMBER_DRAIN_LUA = _load_lua("member_drain.lua")
REPAIR_DEAD_JOB_LUA = _load_lua("repair_dead_job.lua")
CANCEL_QUEUED_LUA = _load_lua("cancel_queued.lua")
REQUEUE_CLAIM_LUA = _load_lua("requeue_claim.lua")
REFRESH_MEMBER_MIRROR_LUA = """
-- combined-gemma.member-refresh.v1
-- KEYS: member lease set, member hash
-- ARGV: record_json, state, accepting, state_version, generation_fence,
--       requested_free_slots
local live_leases = redis.call('SCARD', KEYS[1])
local current_free = tonumber(redis.call('HGET', KEYS[2], 'compatible_free_slots') or '-1')
local free_slots = tonumber(ARGV[6]) or 0
-- A live lease owns the currently advertised reservation capacity. Preserve
-- that value during refresh; an empty lease set is the safe repair boundary
-- for stale positive capacity mirrors.
if live_leases > 0 and current_free >= 0 then
  free_slots = current_free
end
local record = cjson.decode(ARGV[1])
record.dispatcher_leases = live_leases
record.compatible_free_slots = free_slots
redis.call('HSET', KEYS[2],
  'record', cjson.encode(record),
  'state', ARGV[2],
  'accepting', ARGV[3],
  'state_version', ARGV[4],
  'generation_fence', ARGV[5],
  'compatible_free_slots', tostring(free_slots),
  'configured_slots', tostring(record.configured_slots or 0))
return {tostring(live_leases), tostring(free_slots)}
"""
MEMBER_RESERVATION_VIEW_LUA = """
-- combined-gemma.member-reservation-view.v1
-- KEYS: member hash, member lease set
local record = redis.call('HGET', KEYS[1], 'record')
if not record then
  return {}
end
local state = redis.call('HGET', KEYS[1], 'state')
local accepting = redis.call('HGET', KEYS[1], 'accepting')
local state_version = redis.call('HGET', KEYS[1], 'state_version')
local generation_fence = redis.call('HGET', KEYS[1], 'generation_fence')
local free = redis.call('HGET', KEYS[1], 'compatible_free_slots')
local configured = redis.call('HGET', KEYS[1], 'configured_slots')
if not state or not accepting or not state_version or not generation_fence
   or not free or not configured then
  return {err='MEMBER_RESERVATION_VIEW_INCOMPLETE'}
end
return {
  record, state, accepting, state_version, generation_fence,
  free, configured, tostring(redis.call('SCARD', KEYS[2]))
}
"""
REPAIR_MEMBER_FENCE_LUA = """
-- combined-gemma.member-fence-repair.v1
-- KEYS: member hash, member lease set
-- ARGV: expected_state_version, expected_record, repaired_record
local current_record = redis.call('HGET', KEYS[1], 'record')
if not current_record then
  return redis.error_reply('UNKNOWN_MEMBER')
end
if current_record ~= ARGV[2] then
  return redis.error_reply('MEMBER_RECORD_CHANGED')
end
if redis.call('HGET', KEYS[1], 'state_version') ~= ARGV[1] then
  return redis.error_reply('RESERVATION_STATE_STALE')
end
if redis.call('SCARD', KEYS[2]) ~= 0 then
  return redis.error_reply('MEMBER_LEASES_REMAIN')
end
local repaired = cjson.decode(ARGV[3])
if repaired.generation_fence ~= 0 then
  return redis.error_reply('REPAIR_FENCE_MUST_BE_ZERO')
end
redis.call('HSET', KEYS[1],
  'record', ARGV[3],
  'state', tostring(repaired.state),
  'accepting', repaired.accepting and '1' or '0',
  'state_version', tostring(repaired.state_version),
  'generation_fence', '0',
  'compatible_free_slots', tostring(repaired.compatible_free_slots or 0),
  'configured_slots', tostring(repaired.configured_slots or 0))
return ARGV[3]
"""
ACQUIRE_LEADER_LUA = _load_lua("acquire_leader.lua")
RENEW_LEADER_LUA = _load_lua("renew_leader.lua")


class ReservationStale(RuntimeError):
    """A lifecycle CAS conflicted with a newer durable member snapshot."""

    status = 409

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = {
            "RESERVATION_STATE_STALE": "reservation_state_stale",
            "LEASE_NOT_OWNED": "lease_not_owned",
            "MEMBER_LEASES_REMAIN": "member_leases_remain",
        }.get(message, "reservation_stale")


class LeadershipLost(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Reservation:
    job: JobRecord
    member: MemberSnapshot
    attempt: AttemptRecord | None = None


class RepositoryUnavailable(RuntimeError):
    pass


def _derive_compatible_free_slots(
    *,
    configured_slots: int,
    backend_slots_total: int | None = None,
    backend_slots_busy: int,
    live_leases: int,
) -> int:
    """Fail-closed capacity without counting the same work twice.

    Backend busy slots and dispatcher leases are independent observations of
    the same in-flight requests.  The safe overlap is therefore the minimum of
    their independently proved free capacities, not their sum.
    """
    configured = max(0, int(configured_slots))
    if configured <= 0:
        return 0
    observed_total = (
        configured
        if backend_slots_total is None
        else max(0, min(configured, int(backend_slots_total)))
    )
    backend_free = max(0, observed_total - max(0, int(backend_slots_busy)))
    dispatcher_free = max(0, configured - max(0, int(live_leases)))
    return min(configured, backend_free, dispatcher_free)


def effective_priority(
    base_priority: Priority | int,
    *,
    submitted_at: float,
    now: float,
    aging_seconds: float = 300.0,
) -> int:
    if aging_seconds <= 0:
        raise ValueError("aging_seconds must be positive")
    base = int(Priority(base_priority))
    wait = max(0.0, now - submitted_at)
    tiers = int(wait // aging_seconds)
    return max(int(Priority.INTERACTIVE), base - (10 * tiers))


class InMemoryJobRepository:
    """Thread-safe semantic reference for priority/FIFO and idempotency."""

    def __init__(
        self,
        *,
        available: bool = True,
        aging_seconds: float = 300.0,
        terminal_notifier=None,
    ):
        self.available = available
        self.aging_seconds = aging_seconds
        self._terminal_notifier = terminal_notifier
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._queues: dict[Priority, deque[str]] = {priority: deque() for priority in Priority}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._members: dict[str, MemberSnapshot] = {}
        self._leases: dict[str, set[str]] = {}
        # Single-process leadership state. The InMemory repo never
        # contends with another process so a constant fence token is
        # the minimally-valid semantics — the B2 brief pins this
        # contract. RedisJobRepository retains its Lua-fenced path.
        self._leader_owner: str | None = None
        self._leader_token: int = 0
        self._leader_expires_at: float = 0.0

    def _require_available(self) -> None:
        if not self.available:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)

    def set_terminal_notifier(self, notifier) -> None:
        self._terminal_notifier = notifier

    def _notify_terminal(self, job_id: str) -> None:
        notifier = self._terminal_notifier
        notify = getattr(notifier, "notify_terminal", None)
        if callable(notify):
            notify(job_id)

    @property
    def queue_total(self) -> int:
        with self._lock:
            return sum(len(queue) for queue in self._queues.values())

    def durable_counts(self) -> dict:
        """Return an authoritative process-local projection for test runtimes."""
        self._require_available()
        with self._lock:
            jobs_by_state = Counter(job.state.value for job in self._jobs.values())
            attempts_total = sum(int(job.attempt_count) for job in self._jobs.values())
            attempts_by_state: Counter[str] = Counter()
            for job in self._jobs.values():
                if not job.current_attempt_id:
                    continue
                attempt_state = {
                    JobState.CLAIMED: AttemptState.RESERVED.value,
                    JobState.ACCEPTED: AttemptState.ACCEPTED.value,
                    JobState.IN_FLIGHT: AttemptState.FORWARDING.value,
                    JobState.COMPLETED: AttemptState.COMPLETED.value,
                    JobState.FAILED: AttemptState.FAILED.value,
                    JobState.CANCELLED: AttemptState.FAILED.value,
                    JobState.OUTCOME_UNKNOWN: AttemptState.FAILED.value,
                }.get(job.state)
                if attempt_state is not None:
                    attempts_by_state[attempt_state] += 1
            return {
                "jobs_total": len(self._jobs),
                "jobs_by_state": dict(jobs_by_state),
                "attempts_total": attempts_total,
                "attempts_by_state": dict(attempts_by_state),
            }

    def submit(self, job: JobRecord, *, caller_scope: str) -> JobRecord:
        self._require_available()
        key = (caller_scope, job.idempotency_key)
        with self._lock:
            prior = self._idempotency.get(key)
            if prior is not None:
                prior_job_id, prior_hash = prior
                if prior_hash != job.request_sha256:
                    raise ContractError(ReasonCode.IDEMPOTENCY_KEY_REUSED.value)
                return self._jobs[prior_job_id]
            if job.job_id in self._jobs:
                raise ContractError(f"DUPLICATE_JOB_ID:{job.job_id}")
            stored = replace(job, effective_priority=int(job.base_priority))
            self._jobs[stored.job_id] = stored
            self._queues[stored.base_priority].append(stored.job_id)
            self._idempotency[key] = (stored.job_id, stored.request_sha256)
            return stored

    def get(self, job_id: str) -> JobRecord | None:
        self._require_available()
        with self._lock:
            return self._jobs.get(job_id)

    def peek_next(self, *, now: float) -> JobRecord | None:
        self._require_available()
        with self._lock:
            heads: list[JobRecord] = []
            for queue in self._queues.values():
                if queue:
                    heads.append(self._jobs[queue[0]])
            if not heads:
                return None
            selected = min(
                heads,
                key=lambda job: (
                    effective_priority(
                        job.base_priority,
                        submitted_at=job.submitted_at,
                        now=now,
                        aging_seconds=self.aging_seconds,
                    ),
                    job.enqueue_sequence,
                ),
            )
            return replace(
                selected,
                effective_priority=effective_priority(
                    selected.base_priority,
                    submitted_at=selected.submitted_at,
                    now=now,
                    aging_seconds=self.aging_seconds,
                ),
            )

    def reserve_next(
        self,
        *,
        now: float,
        ordered_members: list[MemberSnapshot],
        requirements: RequestRequirements,
        member_lookup: Callable[[str], MemberSnapshot] | None = None,
        leader_owner: str | None = None,
        fencing_token: int | None = None,
        backend_url: str | None = None,
    ) -> Reservation | None:
        """Select queue head and member, then commit both under one lock.

        The optional lookup models the Redis commit-time state-version re-read.
        Any mismatch leaves the job in its original queue.

        The ``leader_owner`` / ``fencing_token`` / ``backend_url`` kwargs
        are accepted for parity with ``RedisJobRepository.reserve_next``;
        the InMemory variant is single-process and does not enforce
        leadership at the repo boundary (it has no second contender),
        so it accepts and ignores them. Tests that pass these kwargs
        therefore work against both implementations.
        """
        self._require_available()
        with self._lock:
            selected = self.peek_next(now=now)
            if selected is None:
                return None
            member = next(
                (
                    candidate
                    for candidate in ordered_members
                    if candidate.state is MemberState.READY_ACCEPTING
                    and candidate.accepting
                    and (candidate.compatible_free_slots or 0) > 0
                    and evaluate(requirements, candidate).compatible
                ),
                None,
            )
            if member is None:
                return None
            current = member_lookup(member.name) if member_lookup else member
            live_leases = len(self._leases.get(current.name, set()))
            configured_slots = int(current.configured_slots or 0)
            if configured_slots <= 0 or live_leases >= configured_slots:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            if (
                current.state_version != member.state_version
                or current.state is not MemberState.READY_ACCEPTING
                or not current.accepting
                or (current.compatible_free_slots or 0) <= 0
            ):
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)

            queue = self._queues[selected.base_priority]
            if not queue or queue[0] != selected.job_id:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            queue.popleft()
            claimed = replace(
                self._jobs[selected.job_id],
                state=JobState.CLAIMED,
                selected_member=member.name,
                member_state_version=member.state_version,
                attempt_count=self._jobs[selected.job_id].attempt_count + 1,
                current_attempt_id=str(uuid4()),
                queue_claimed_at=now,
                effective_priority=selected.effective_priority,
                state_version=self._jobs[selected.job_id].state_version + 1,
            )
            self._jobs[claimed.job_id] = claimed
            # Mirror the Lua reserve_job.lua's SADD to the lease set so the
            # terminal-release path can find and remove the attempt.
            attempt_id: str = claimed.current_attempt_id or ""
            self._leases.setdefault(member.name, set()).add(attempt_id)
            reserved_member = replace(
                current,
                dispatcher_leases=live_leases + 1,
                compatible_free_slots=max(
                    0,
                    min(
                        configured_slots - live_leases - 1,
                        (current.compatible_free_slots or 0) - 1,
                    ),
                ),
            )
            self._members[member.name] = reserved_member
            return Reservation(job=claimed, member=reserved_member)

    def put_member(self, member: MemberSnapshot) -> MemberSnapshot:
        self._require_available()
        with self._lock:
            current = self._members.get(member.name)
            if (
                current is not None
                and (current.compatible_free_slots or 0) > 0
                and (member.compatible_free_slots or 0) <= 0
                and member.configured_slots > 0
            ):
                member = replace(
                    member,
                    compatible_free_slots=current.compatible_free_slots,
                )
            self._members[member.name] = member
            self._leases.setdefault(member.name, set())
            return member

    def get_member(self, member_name: str) -> MemberSnapshot | None:
        self._require_available()
        with self._lock:
            return self._members.get(member_name)

    def add_member_lease(self, member_name: str, attempt_id: str) -> None:
        self._require_available()
        with self._lock:
            if member_name not in self._members:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
            self._leases.setdefault(member_name, set()).add(attempt_id)

    def lease_count(self, member_name: str) -> int:
        self._require_available()
        with self._lock:
            return len(self._leases.get(member_name, set()))

    def acquire_leadership(
        self, *, owner: str, now: float, ttl_seconds: float
    ) -> int:
        """Single-process leadership: always grant a positive fence token.

        The InMemory variant never contends with another process — it is
        the in-process reference implementation used by tests and the
        dispatch loop fence. The token is a monotonic counter so a
        caller can observe renewal across ``renew_leadership`` calls,
        and is reused across repeated acquires by the same owner
        until the TTL elapses (matching the Lua-fenced Redis variant
        that returns the existing token while the lease is held).
        """
        if not owner or ttl_seconds <= 0:
            raise ValueError(
                "acquire_leadership: owner and positive ttl_seconds are required"
            )
        self._require_available()
        with self._lock:
            if (
                self._leader_owner == owner
                and self._leader_token > 0
                and now < self._leader_expires_at
            ):
                # Refresh the TTL and return the held token.
                self._leader_expires_at = now + float(ttl_seconds)
                return int(self._leader_token)
            # New owner (or expired): bump the token.
            self._leader_token += 1
            self._leader_owner = owner
            self._leader_expires_at = now + float(ttl_seconds)
            return int(self._leader_token)

    def renew_leadership(
        self,
        *,
        owner: str,
        fencing_token: int,
        now: float,
        ttl_seconds: float,
    ) -> int:
        """Refresh the held lease if the fence token still matches.

        Returns the (same) fencing token on success. Raises
        ``LeadershipLost`` if no lease is held by ``owner`` or the
        supplied token does not match — matching the
        ``RedisJobRepository`` semantics on which the dispatch loop
        depends for clean shutdown.
        """
        if (
            not owner
            or isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token <= 0
            or ttl_seconds <= 0
        ):
            raise ValueError(
                "renew_leadership: owner, positive fencing_token, and "
                "positive ttl_seconds are required"
            )
        self._require_available()
        with self._lock:
            if (
                self._leader_owner != owner
                or self._leader_token != int(fencing_token)
                or now >= self._leader_expires_at
            ):
                raise LeadershipLost("DISPATCHER_FENCE_STALE")
            self._leader_expires_at = now + float(ttl_seconds)
            return int(self._leader_token)

    def begin_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot:
        self._require_available()
        with self._lock:
            current = self._members.get(member_name)
            if current is None:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
            if current.state_version != expected_state_version:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            # GPUManager may repeat the same drain request while the broker
            # still owns the lifecycle transition. Treat DRAINING at the
            # expected version as an idempotent no-op; the coordinator will
            # continue waiting and complete the transition to OFFLINE.
            if current.state is MemberState.DRAINING:
                return current
            validate_member_transition(current.state, MemberState.DRAINING)
            draining = replace(
                current,
                state=MemberState.DRAINING,
                accepting=False,
                compatible_free_slots=0,
                state_version=current.state_version + 1,
                blockers=tuple(dict.fromkeys((*current.blockers, "member_draining"))),
                reason_codes=tuple(
                    dict.fromkeys((*current.reason_codes, ReasonCode.MEMBER_DRAINING))
                ),
            )
            self._members[member_name] = draining
            return draining

    def release_member_lease(
        self,
        member_name: str,
        attempt_id: str,
        *,
        claimed_member_state_version: int,
    ) -> MemberSnapshot:
        """Release a lease following a terminal outcome.

        Semantics:
        - Happy-path (terminal COMPLETED / CANCELLED_AFTER_ACCEPTANCE / OUTCOME_UNKNOWN):
          the dispatcher knows the attempt_id it owns. The state_version fence is
          intentionally SKIPPED — readiness-refresh may have advanced the member
          hash while the terminal release was in flight (the original 14:17
          incident class). The release is fail-closed on lease ownership only.
        - Drain-mode path (drain.py): the operator is preparing the member for
          offline. The fence is enforced — a stale claim must NOT silently
          release into a freshly-re-prepared member.
        - Rejoin-mode path: the member is transitioning back online. The fence
          is enforced — a stale claim from the prior generation must NOT leak.

        The fence check is automatically retained whenever the member is in
        DRAINING or JOINING state (which is what the drain/rejoin flow leaves
        behind).
        """
        self._require_available()
        with self._lock:
            current = self._members.get(member_name)
            if current is None:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
            leases = self._leases.setdefault(member_name, set())
            if attempt_id not in leases:
                raise ReservationStale("LEASE_NOT_OWNED")
            # Enforce the state_version fence only when the member is currently
            # in DRAINING or JOINING (drain / rejoin flows). Happy-path terminal
            # releases from the dispatcher do not enforce the fence; the original
            # readiness-refresh race must not pin stale leases.
            if current.state is MemberState.DRAINING or current.state is MemberState.JOINING:
                if claimed_member_state_version != current.state_version:
                    raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            elif claimed_member_state_version > current.state_version:
                # Backward-compat path: the old "claimed > current" heuristic,
                # which catches the most egregious case (claim from the future)
                # without blocking otherwise-correct happy-path releases.
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            leases.remove(attempt_id)
            # Re-derive ``dispatcher_leases`` from the LIVE lease set so
            # the cache cannot self-propagate a stale value. The previous
            # implementation used ``current.dispatcher_leases - 1`` which
            # relied on every prior release having already written back
            # into ``current``; production races (e.g. release writes that
            # were dropped, or releases via paths that bypass the cache)
            # left the cache holding a value that didn't match the live
            # lease set. This bug surfaced as CPU staying latched at
            # ``state=joining, accepting=0`` despite the live lease set
            # being empty.
            live_count = len(leases)
            if current.state is MemberState.DRAINING:
                free_slots = 0
            else:
                configured = int(current.configured_slots or 0)
                backend_busy = int(current.backend_slots_busy or 0)
                if configured > 0:
                    free_slots = _derive_compatible_free_slots(
                        configured_slots=configured,
                        backend_slots_total=current.backend_slots_total,
                        backend_slots_busy=backend_busy,
                        live_leases=live_count,
                    )
                else:
                    free_slots = min(
                        int(current.configured_slots or 0),
                        max(0, (current.compatible_free_slots or 0) + 1),
                    )
            if current.state is MemberState.DRAINING:
                updated = replace(
                    current,
                    dispatcher_leases=live_count,
                    drain_disabled_slots=current.drain_disabled_slots + 1,
                )
            else:
                updated = replace(
                    current,
                    dispatcher_leases=live_count,
                    compatible_free_slots=free_slots,
                )
            self._members[member_name] = updated
            return updated

    def complete_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot:
        self._require_available()
        with self._lock:
            current = self._members.get(member_name)
            if current is None:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
            if current.state_version != expected_state_version:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            if current.state is not MemberState.DRAINING:
                raise ContractError(f"DRAIN_INVALID_STATE:{current.state.value}")
            if self._leases.get(member_name):
                raise ReservationStale("MEMBER_LEASES_REMAIN")
            validate_member_transition(current.state, MemberState.OFFLINE)
            offline = replace(
                current,
                state=MemberState.OFFLINE,
                accepting=False,
                compatible_free_slots=0,
                state_version=current.state_version + 1,
            )
            self._members[member_name] = offline
            return offline

    def begin_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        generation_fence: int,
    ) -> MemberSnapshot:
        self._require_available()
        with self._lock:
            current = self._members.get(member_name)
            if current is None:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
            if current.state_version != expected_state_version:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            if generation_fence <= current.generation_fence:
                raise ContractError("GENERATION_FENCE_STALE")
            validate_member_transition(current.state, MemberState.JOINING)
            joining = replace(
                current,
                state=MemberState.JOINING,
                accepting=False,
                compatible_free_slots=0,
                generation_fence=generation_fence,
                state_version=current.state_version + 1,
            )
            self._members[member_name] = joining
            return joining

    def complete_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        readiness: MemberSnapshot,
    ) -> MemberSnapshot:
        self._require_available()
        with self._lock:
            current = self._members.get(member_name)
            if current is None:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
            if current.state_version != expected_state_version:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            validate_member_transition(current.state, MemberState.READY_ACCEPTING)
            ready = replace(
                readiness,
                state=MemberState.READY_ACCEPTING,
                accepting=True,
                generation_fence=current.generation_fence,
                state_version=current.state_version + 1,
                drain_disabled_slots=0,
                dispatcher_leases=0,
            )
            self._members[member_name] = ready
            return ready

    def cancel_queued(self, job_id: str, *, completed_at: float) -> JobRecord:
        self._require_available()
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise ContractError(f"UNKNOWN_JOB_ID:{job_id}")
            if current.state is not JobState.QUEUED:
                return current
            queue = self._queues[current.base_priority]
            try:
                queue.remove(job_id)
            except ValueError:
                raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
            cancelled = replace(
                current,
                state=JobState.CANCELLED,
                completed_at=completed_at,
                reason_code=ReasonCode.CANCELLED_BEFORE_ACCEPTANCE,
                state_version=current.state_version + 1,
            )
            self._jobs[job_id] = cancelled
            self._notify_terminal(cancelled.job_id)
            return cancelled

    def replace_job(self, job: JobRecord) -> JobRecord:
        self._require_available()
        with self._lock:
            if job.job_id not in self._jobs:
                raise ContractError(f"UNKNOWN_JOB_ID:{job.job_id}")
            self._jobs[job.job_id] = job
            if job.state in {
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.OUTCOME_UNKNOWN,
            }:
                self._notify_terminal(job.job_id)
            return job

    def requeue_claim(
        self,
        job: JobRecord,
        *,
        reason: ReasonCode,
        terminal_at: float | None = None,
    ) -> JobRecord:
        del terminal_at  # Redis persists attempt timing; this oracle tracks job semantics.
        self._require_available()
        with self._lock:
            current = self._jobs.get(job.job_id)
            if current is None:
                raise ContractError("UNKNOWN_JOB_ID")
            if current.state not in {JobState.CLAIMED, JobState.ACCEPTED, JobState.IN_FLIGHT}:
                raise ContractError(f"REQUEUE_INVALID_STATE:{current.state.value}")
            if current.accepted_boundary_crossed:
                raise ContractError(ReasonCode.CANCEL_UNSAFE_AFTER_ACCEPTANCE.value)
            queued = replace(
                current,
                state=JobState.QUEUED,
                selected_member=None,
                member_state_version=None,
                current_attempt_id=None,
                queue_claimed_at=None,
                forward_started_at=None,
                reason_code=reason,
                error=None,
                state_version=current.state_version + 1,
            )
            self._jobs[queued.job_id] = queued
            queue = self._queues[queued.base_priority]
            if queued.job_id not in queue:
                queue.appendleft(queued.job_id)
            # Mirror the Lua requeue_claim.lua's SREM: the pre-accept requeue
            # releases the lease so capacity is restored. The dispatcher's
            # post-accept terminal handler should NOT call release again
            # (it'd hit LEASE_NOT_OWNED). A double-release is a 5xx
            # signal that the bug is caught.
            if current.selected_member and current.current_attempt_id:
                leases = self._leases.setdefault(current.selected_member, set())
                leases.discard(current.current_attempt_id)
                member = self._members.get(current.selected_member)
                if member is not None:
                    self._members[current.selected_member] = replace(
                        member,
                        dispatcher_leases=max(0, member.dispatcher_leases - 1),
                        compatible_free_slots=min(
                            member.configured_slots,
                            (member.compatible_free_slots or 0) + 1,
                        ),
                        state_version=member.state_version + 1,
                    )
            return queued


class RedisJobRepository:
    """Durable Redis repository with Lua-fenced submission and claims."""

    def __init__(
        self,
        client,
        *,
        prefix: str,
        aging_seconds: float = 300.0,
        terminal_attempt_retention_seconds: int = 604800,
        terminal_notifier=None,
        leadership_clock: Callable[[], float] | None = None,
    ):
        if not prefix or not prefix.endswith(":"):
            raise ValueError("Redis repository prefix must be non-empty and end with ':'")
        if aging_seconds <= 0:
            raise ValueError("aging_seconds must be positive")
        if terminal_attempt_retention_seconds <= 0:
            raise ValueError("terminal attempt retention must be positive")
        self.client = client
        self.prefix = prefix
        self.aging_seconds = aging_seconds
        self.terminal_attempt_retention_seconds = int(
            terminal_attempt_retention_seconds
        )
        self._terminal_notifier = terminal_notifier
        self._leadership_clock = leadership_clock

    def set_terminal_notifier(self, notifier) -> None:
        self._terminal_notifier = notifier

    def _notify_terminal(self, job_id: str) -> None:
        notifier = self._terminal_notifier
        notify = getattr(notifier, "notify_terminal", None)
        if callable(notify):
            notify(job_id)

    @staticmethod
    def _text(value) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _json(value: dict) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def _job_key(self, job_id: str) -> str:
        return f"{self.prefix}job:{job_id}"

    def _queue_key(self, priority: Priority | int) -> str:
        return f"{self.prefix}queue:{int(Priority(priority))}"

    def _member_key(self, member_name: str) -> str:
        return f"{self.prefix}member:{member_name}"

    def _lease_key(self, member_name: str) -> str:
        return f"{self.prefix}leases:{member_name}"

    def _attempt_key(self, attempt_id: str) -> str:
        return f"{self.prefix}attempt:{attempt_id}"

    def _repair_key(self, job_id: str) -> str:
        return f"{self.prefix}repair:{job_id}"

    def _leader_key(self) -> str:
        return f"{self.prefix}dispatcher:leader"

    def _fence_counter_key(self) -> str:
        return f"{self.prefix}dispatcher:fence-counter"

    def _idempotency_key(self, caller_scope: str, key: str) -> str:
        digest = hashlib.sha256(f"{caller_scope}\0{key}".encode()).hexdigest()
        return f"{self.prefix}idempotency:{digest}"

    def _redis(self, operation):
        try:
            return operation()
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc

    def acquire_leadership(self, *, owner: str, now: float, ttl_seconds: float) -> int:
        if not owner or ttl_seconds <= 0:
            raise ValueError("leader owner and positive ttl_seconds are required")
        lease_now = float(self._leadership_clock()) if self._leadership_clock else now
        try:
            result = self.client.eval(
                ACQUIRE_LEADER_LUA,
                2,
                self._leader_key(),
                self._fence_counter_key(),
                owner,
                lease_now,
                ttl_seconds,
            )
        except ResponseError as exc:
            if "DISPATCHER_LEASE_HELD" in str(exc):
                raise LeadershipLost("DISPATCHER_LEASE_HELD") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        return int(self._text(result[0]))

    def renew_leadership(
        self,
        *,
        owner: str,
        fencing_token: int,
        now: float,
        ttl_seconds: float,
    ) -> int:
        if not owner or isinstance(fencing_token, bool) or fencing_token <= 0 or ttl_seconds <= 0:
            raise ValueError("valid leader owner, fencing token, and ttl are required")
        lease_now = float(self._leadership_clock()) if self._leadership_clock else now
        try:
            result = self.client.eval(
                RENEW_LEADER_LUA,
                1,
                self._leader_key(),
                owner,
                fencing_token,
                lease_now,
                ttl_seconds,
            )
        except ResponseError as exc:
            if "DISPATCHER_FENCE_STALE" in str(exc):
                raise LeadershipLost("DISPATCHER_FENCE_STALE") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        return int(self._text(result[0]))

    @property
    def queue_total(self) -> int:
        def read() -> int:
            pipe = self.client.pipeline(transaction=False)
            for priority in Priority:
                pipe.zcard(self._queue_key(priority))
            return sum(int(value) for value in pipe.execute())
        return self._redis(read)

    def durable_counts(self) -> dict:
        """Count durable job/attempt records from the active namespace."""

        def read() -> dict:
            jobs_by_state: Counter[str] = Counter()
            attempts_by_state: Counter[str] = Counter()

            def count_batch(keys: list[str | bytes], counter: Counter[str]) -> None:
                if not keys:
                    return
                pipe = self.client.pipeline(transaction=False)
                for key in keys:
                    pipe.hget(key, "record")
                records = pipe.execute()
                if len(records) != len(keys) or any(raw is None for raw in records):
                    raise RepositoryUnavailable(
                        ReasonCode.REDIS_STATE_UNKNOWN.value
                    )
                for raw in records:
                    try:
                        payload = json.loads(self._text(raw))
                        state = str(payload["state"])
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise RepositoryUnavailable(
                            ReasonCode.REDIS_STATE_UNKNOWN.value
                        ) from exc
                    counter[state] += 1

            for kind, counter in (
                ("job", jobs_by_state),
                ("attempt", attempts_by_state),
            ):
                batch: list[str | bytes] = []
                for key in self.client.scan_iter(
                    match=f"{self.prefix}{kind}:*",
                    count=1000,
                ):
                    batch.append(key)
                    if len(batch) >= 1000:
                        count_batch(batch, counter)
                        batch.clear()
                count_batch(batch, counter)
            return {
                "jobs_total": sum(jobs_by_state.values()),
                "jobs_by_state": dict(jobs_by_state),
                "attempts_total": sum(attempts_by_state.values()),
                "attempts_by_state": dict(attempts_by_state),
            }

        return self._redis(read)

    def recover_stale_pre_acceptance_claims(
        self,
        *,
        now: float,
        max_age_seconds: float = 120.0,
        limit: int = 256,
    ) -> tuple[JobRecord, ...]:
        """Requeue claims abandoned before the backend acceptance boundary.

        A broker restart can leave a job in ``claimed`` with its lease still
        held even though no dispatcher task remains to finish it.  Only the
        unambiguous pre-acceptance shape is eligible: no acceptance boundary,
        no forward start, a current attempt, and a claim older than the
        bounded recovery age.  ``requeue_claim`` performs the member/job CAS
        atomically, so a live dispatcher racing the recovery either wins or
        is rejected without releasing the wrong lease.
        """
        if max_age_seconds <= 0 or limit <= 0:
            raise ValueError("positive recovery age and limit are required")
        recovered: list[JobRecord] = []
        try:
            keys = self.client.scan_iter(
                match=f"{self.prefix}job:*",
                count=min(max(limit * 2, 32), 1000),
            )
            for key in keys:
                if len(recovered) >= limit:
                    break
                job_id = self._text(key).rsplit(":", 1)[-1]
                current = self.get(job_id)
                if current is None:
                    continue
                claimed_at = current.queue_claimed_at
                if (
                    current.state is not JobState.CLAIMED
                    or current.accepted_boundary_crossed
                    or current.forward_started_at is not None
                    or not current.current_attempt_id
                    or claimed_at is None
                    or float(now) - float(claimed_at) < max_age_seconds
                ):
                    continue
                try:
                    recovered.append(
                        self.requeue_claim(
                            current,
                            reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
                            terminal_at=now,
                        )
                    )
                except (ReservationStale, ContractError):
                    # A concurrent dispatcher or lifecycle update won the
                    # CAS. Re-scan on the next startup/tick; do not guess.
                    continue
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        return tuple(recovered)

    def submit(self, job: JobRecord, *, caller_scope: str) -> JobRecord:
        stored = replace(job, effective_priority=int(job.base_priority))
        keys = (
            self._idempotency_key(caller_scope, stored.idempotency_key),
            self._job_key(stored.job_id),
            self._queue_key(stored.base_priority),
        )
        try:
            result = self.client.eval(
                SUBMIT_JOB_LUA, len(keys), *keys,
                stored.request_sha256, stored.job_id,
                self._json(stored.to_dict()), stored.enqueue_sequence,
                stored.state_version,
            )
        except ResponseError as exc:
            message = str(exc)
            if "IDEMPOTENCY_KEY_REUSED" in message:
                raise ContractError(ReasonCode.IDEMPOTENCY_KEY_REUSED.value) from exc
            if "DUPLICATE_JOB_ID" in message:
                raise ContractError(f"DUPLICATE_JOB_ID:{stored.job_id}") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        status, returned_id = (self._text(item) for item in result)
        if status == "existing":
            existing = self.get(returned_id)
            if existing is None:
                raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
            return existing
        if status != "created" or returned_id != stored.job_id:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        return stored

    def get(self, job_id: str) -> JobRecord | None:
        value = self._redis(lambda: self.client.hget(self._job_key(job_id), "record"))
        if value is None:
            return None
        try:
            return JobRecord.from_dict(json.loads(value))
        except (ContractError, TypeError, ValueError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc

    def _queue_head(self, priority: Priority) -> JobRecord | None:
        rows = self._redis(lambda: self.client.zrange(self._queue_key(priority), 0, 0))
        if not rows:
            return None
        job_id = self._text(rows[0])
        job = self.get(job_id)
        if job is None:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        if job.state is not JobState.QUEUED:
            still_queued = self._redis(
                lambda: self.client.zscore(self._queue_key(priority), job_id)
            )
            if still_queued is None:
                return None
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        return job

    def peek_next(self, *, now: float) -> JobRecord | None:
        heads = [head for priority in Priority if (head := self._queue_head(priority))]
        if not heads:
            return None
        selected = min(
            heads,
            key=lambda job: (
                effective_priority(
                    job.base_priority, submitted_at=job.submitted_at,
                    now=now, aging_seconds=self.aging_seconds,
                ),
                job.enqueue_sequence,
            ),
        )
        return replace(
            selected,
            effective_priority=effective_priority(
                selected.base_priority, submitted_at=selected.submitted_at,
                now=now, aging_seconds=self.aging_seconds,
            ),
        )

    def put_member(self, member: MemberSnapshot) -> MemberSnapshot:
        # Only the runtime's view of readiness is mirrored here. Lua reservation
        # accounting in reserve_job.lua decrements `compatible_free_slots` on
        # claim and re-increments on release — those writes must not be clobbered
        # by the runtime's recompute (which would always be 4) or the broker
        # would never see its own claims and would always re-pick the same
        # first-accepting member (no spillover). But if the field is unset
        # (empty/nil) we must initialise it. The runtime's `compatible_free_slots`
        # is the runtime's *current* view which may be 0 if the backend is busy.
        # We need a fresh "full capacity" value — read it from the cached
        # member's `configured_slots` (== parallel == 4 in our case).
        #
        # T2-GREEN: write the canonical `record` JSON (the side that
        # `get_member` reads from) AND clamp `effective_version` so a stale
        # readiness refresh can never roll back a fresher Lua-emitted version.
        # Without the clamp, a refresh between get_member and put_member would
        # silently revert state_version=N-1 → state_version=N.
        member_key = self._member_key(member.name)
        current_free = self._redis(
            lambda: self.client.hget(member_key, "compatible_free_slots")
        )
        current_state_version_raw = self._redis(
            lambda: self.client.hget(member_key, "state_version")
        )
        live_leases = int(self._redis(lambda: self.client.scard(self._lease_key(member.name))))
        try:
            current_free_int = None if current_free in (None, "") else int(current_free)
        except (TypeError, ValueError):
            current_free_int = None
        try:
            current_state_version_int = (
                None
                if current_state_version_raw in (None, "")
                else int(current_state_version_raw)
            )
        except (TypeError, ValueError):
            current_state_version_int = None
        # The Lua is authoritative for state_version. A stale readiness
        # refresh must never silently roll back a fresher Lua-emitted
        # value, so the persisted version is max(redis, snapshot).
        effective_version = max(
            current_state_version_int if current_state_version_int is not None else 0,
            int(getattr(member, "state_version", 0) or 0),
        )
        # Durable state_version remains monotonic across restarts, but ordinary
        # readiness fields come from the fresh semantic probe. A newly-started
        # observer begins its local versions near zero; preserving every higher
        # Redis state here would freeze stale readiness for hours after reboot.
        # Preserve only an explicit DRAINING lifecycle fence owned by Lua.
        current_state_raw = self._redis(lambda: self.client.hget(member_key, "state"))
        if current_state_raw is not None:
            current_state_raw = self._text(current_state_raw)
        current_accepting_raw = self._redis(
            lambda: self.client.hget(member_key, "accepting")
        )
        if current_accepting_raw is not None:
            current_accepting_raw = self._text(current_accepting_raw)
        preserve_drain_fence = (
            current_state_version_int is not None
            and effective_version > int(getattr(member, "state_version", 0) or 0)
            and current_state_raw == MemberState.DRAINING.value
        )
        # FAULTED is an explicit quarantine, not a liveness observation.  A
        # fresh probe must never reopen it; only the generation-fenced rejoin
        # path may move FAULTED -> JOINING -> READY_ACCEPTING.  Unlike the
        # monotonic DRAINING check above, this must also hold when the observer
        # carries the same local state_version as Redis after a restart.
        preserve_fault_fence = (
            current_state_raw == MemberState.FAULTED.value
            and current_accepting_raw == "0"
        )
        if preserve_drain_fence or preserve_fault_fence:
            persisted_state = current_state_raw
            persisted_accepting = False
        else:
            persisted_state = member.state.value
            persisted_accepting = bool(member.accepting)
        try:
            canonical_state = MemberState(persisted_state)
        except Exception:
            # Defensive fallback in case a malformed top-level value exists.
            canonical_state = member.state
        configured = int(getattr(member, "configured_slots", 0) or 0)
        backend_total = getattr(member, "backend_slots_total", None)
        if configured <= 0 and backend_total is not None:
            configured = int(backend_total)
        backend_busy = getattr(member, "backend_slots_busy", None)
        if configured > 0 and backend_busy is not None:
            runtime_free = _derive_compatible_free_slots(
                configured_slots=configured,
                backend_slots_total=backend_total,
                backend_slots_busy=int(backend_busy),
                live_leases=live_leases,
            )
        else:
            runtime_free = max(0, int(getattr(member, "compatible_free_slots", 0) or 0))
        if live_leases == 0 and member.accepting and configured > 0 and backend_busy is not None:
            # An empty lease set is the only safe point to repair a stale
            # positive mirror from live backend evidence.
            initial_free = runtime_free
        elif current_free_int is None:
            initial_free = runtime_free
        elif current_free_int <= 0 and member.accepting and configured > 0 and backend_busy is not None:
            initial_free = runtime_free
        else:
            # Preserve in-flight Lua capacity until the terminal release
            # transaction owns the authoritative update.
            initial_free = current_free_int
        # Build the canonical record JSON using the clamped version so the
        # `get_member` reader sees the same value the Lua fences against.
        record_member = replace(
            member,
            state=canonical_state,
            accepting=bool(persisted_accepting),
            state_version=effective_version,
            dispatcher_leases=live_leases,
            compatible_free_slots=initial_free,
        )
        record_payload = self._json(record_member.to_dict())
        # The Lua refresh is atomic with the lease-set check. It repairs stale
        # capacity only when no reservation is live and cannot overwrite a
        # concurrent reservation with a larger free-slot value.
        result = self._redis(
            lambda: self.client.eval(
                REFRESH_MEMBER_MIRROR_LUA,
                2,
                self._lease_key(member.name),
                member_key,
                record_payload,
                persisted_state,
                "1" if persisted_accepting else "0",
                str(effective_version),
                str(member.generation_fence),
                str(initial_free),
            )
        )
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return replace(
                record_member,
                dispatcher_leases=int(self._text(result[0])),
                compatible_free_slots=int(self._text(result[1])),
            )
        return record_member

    def repair_member_generation_fence(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot:
        """Repair only a malformed generation fence at an empty-lease CAS point."""
        if (
            isinstance(expected_state_version, bool)
            or not isinstance(expected_state_version, int)
            or expected_state_version < 0
        ):
            raise ContractError("MEMBER_STATE_VERSION_INVALID")
        member_key = self._member_key(member_name)
        raw = self._redis(lambda: self.client.hget(member_key, "record"))
        if raw is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        raw_text = self._text(raw)
        try:
            payload = json.loads(raw_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        if not isinstance(payload, dict):
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        fence = payload.get("generation_fence")
        if isinstance(fence, int) and not isinstance(fence, bool) and fence >= 0:
            raise ContractError("MEMBER_GENERATION_FENCE_REPAIR_NOT_NEEDED")
        if payload.get("state_version") != expected_state_version:
            raise ReservationStale("RESERVATION_STATE_STALE")
        if payload.get("state") not in {
            MemberState.OFFLINE.value,
            MemberState.UNHEALTHY.value,
            MemberState.FAULTED.value,
            MemberState.DRAINING.value,
            MemberState.JOINING.value,
        } or payload.get("accepting") is not False:
            raise ContractError("MEMBER_REPAIR_STATE_INVALID")
        live_leases = int(
            self._redis(lambda: self.client.scard(self._lease_key(member_name)))
        )
        if live_leases != 0:
            raise ReservationStale("MEMBER_LEASES_REMAIN")
        repaired_payload = dict(payload)
        repaired_payload["generation_fence"] = 0
        try:
            repaired = MemberSnapshot.from_dict(repaired_payload)
        except (ContractError, TypeError, ValueError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        try:
            result = self.client.eval(
                REPAIR_MEMBER_FENCE_LUA,
                2,
                member_key,
                self._lease_key(member_name),
                str(expected_state_version),
                raw_text,
                self._json(repaired.to_dict()),
            )
        except ResponseError as exc:
            message = str(exc)
            if "UNKNOWN_MEMBER" in message:
                raise ContractError(f"UNKNOWN_MEMBER:{member_name}") from exc
            if "RESERVATION_STATE_STALE" in message:
                raise ReservationStale("RESERVATION_STATE_STALE") from exc
            if "MEMBER_LEASES_REMAIN" in message:
                raise ReservationStale("MEMBER_LEASES_REMAIN") from exc
            if "MEMBER_RECORD_CHANGED" in message:
                raise ReservationStale("MEMBER_RECORD_CHANGED") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        try:
            return MemberSnapshot.from_dict(json.loads(self._text(result)))
        except (ContractError, TypeError, ValueError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc

    def repair_malformed_member_generation_fence(self, member_name: str) -> MemberSnapshot:
        """Recover a malformed fence using the record's own CAS version."""
        raw = self._redis(lambda: self.client.hget(self._member_key(member_name), "record"))
        if raw is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        try:
            payload = json.loads(self._text(raw))
            expected_state_version = payload["state_version"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        if (
            isinstance(expected_state_version, bool)
            or not isinstance(expected_state_version, int)
            or expected_state_version < 0
        ):
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        return self.repair_member_generation_fence(
            member_name, expected_state_version=expected_state_version
        )

    def get_member(self, member_name: str) -> MemberSnapshot | None:
        value = self._redis(
            lambda: self.client.hget(self._member_key(member_name), "record")
        )
        if value is None:
            return None
        try:
            return MemberSnapshot.from_dict(json.loads(value))
        except (ContractError, TypeError, ValueError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc

    def get_reservation_member(self, member_name: str) -> MemberSnapshot | None:
        """Read the exact denormalized fields consumed by reservation Lua."""
        try:
            values = self.client.eval(
                MEMBER_RESERVATION_VIEW_LUA,
                2,
                self._member_key(member_name),
                self._lease_key(member_name),
            )
        except (RedisError, ResponseError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        if not values:
            return None
        if not isinstance(values, (list, tuple)) or len(values) != 8:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        try:
            record = MemberSnapshot.from_dict(json.loads(self._text(values[0])))
            state = MemberState(self._text(values[1]))
            accepting_raw = self._text(values[2])
            if accepting_raw not in {"0", "1"}:
                raise ValueError("MEMBER_ACCEPTING_INVALID")
            state_version = int(self._text(values[3]))
            generation_fence = int(self._text(values[4]))
            free = int(self._text(values[5]))
            configured = int(self._text(values[6]))
            leases = int(self._text(values[7]))
            if (
                record.name != member_name
                or min(state_version, generation_fence, free, configured, leases) < 0
            ):
                raise ValueError("MEMBER_RESERVATION_VIEW_INVALID")
            return replace(
                record,
                state=state,
                accepting=accepting_raw == "1",
                state_version=state_version,
                generation_fence=generation_fence,
                configured_slots=configured,
                compatible_free_slots=free,
                dispatcher_leases=leases,
            )
        except (ContractError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc

    def lease_count(self, member_name: str) -> int:
        return int(self._redis(lambda: self.client.scard(self._lease_key(member_name))))

    def add_member_lease(self, member_name: str, attempt_id: str) -> None:
        if self.get_member(member_name) is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        self._redis(lambda: self.client.sadd(self._lease_key(member_name), attempt_id))

    def _commit_member_drain(
        self,
        *,
        mode: str,
        current: MemberSnapshot,
        updated: MemberSnapshot,
        claimed_version: int = 0,
        attempt_id: str = "",
    ) -> MemberSnapshot:
        try:
            result = self.client.eval(
                MEMBER_DRAIN_LUA,
                2,
                self._member_key(current.name),
                self._lease_key(current.name),
                mode,
                current.state_version,
                claimed_version,
                attempt_id,
                updated.state_version,
                self._json(updated.to_dict()),
                updated.state.value,
                "1" if updated.accepting else "0",
                updated.compatible_free_slots or 0,
                updated.generation_fence,
                current.configured_slots,
                current.backend_slots_busy or 0,
            )
        except ResponseError as exc:
            message = str(exc)
            if "UNKNOWN_MEMBER" in message:
                raise ContractError(f"UNKNOWN_MEMBER:{current.name}") from exc
            if any(code in message for code in (
                "RESERVATION_STATE_STALE", "LEASE_NOT_OWNED", "MEMBER_LEASES_REMAIN"
            )):
                code = next(
                    code for code in (
                        "RESERVATION_STATE_STALE", "LEASE_NOT_OWNED", "MEMBER_LEASES_REMAIN"
                    ) if code in message
                )
                raise ReservationStale(code) from exc
            if "GENERATION_FENCE_STALE" in message:
                raise ContractError("GENERATION_FENCE_STALE") from exc
            if (
                "DRAIN_INVALID_STATE" in message
                or "REJOIN_INVALID_STATE" in message
                or "MEMBER_STATE_VERSION_INVALID" in message
            ):
                raise ContractError(message) from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        if mode in {"release", "release_v2"} and len(result) >= 4:
            return replace(
                updated,
                dispatcher_leases=int(self._text(result[2])),
                compatible_free_slots=int(self._text(result[3])),
            )
        return updated

    def begin_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot:
        current = self.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state_version != expected_state_version:
            raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
        # GPUManager may repeat the same drain request while the broker
        # still owns the lifecycle transition. Treat DRAINING at the
        # expected version as an idempotent no-op; the coordinator will
        # continue waiting and complete the transition to OFFLINE.
        if current.state is MemberState.DRAINING:
            return current
        validate_member_transition(current.state, MemberState.DRAINING)
        updated = replace(
            current,
            state=MemberState.DRAINING,
            accepting=False,
            compatible_free_slots=0,
            state_version=current.state_version + 1,
            blockers=tuple(dict.fromkeys((*current.blockers, "member_draining"))),
            reason_codes=tuple(dict.fromkeys((*current.reason_codes, ReasonCode.MEMBER_DRAINING))),
        )
        return self._commit_member_drain(mode="begin", current=current, updated=updated)

    def release_member_lease(
        self,
        member_name: str,
        attempt_id: str,
        *,
        claimed_member_state_version: int,
        enforce_state_version_fence: bool = False,
    ) -> MemberSnapshot:
        """Release a lease following a terminal outcome.

        Default behaviour (`enforce_state_version_fence=False`, the dispatcher
        happy-path) uses Lua's `release_v2`, which skips the state_version CAS
        check. The fence is intentionally skipped because readiness-refresh
        may have advanced the member hash between the dispatcher's Python
        `get_member()` and the Lua release commit (the original 14:17 incident
        class); pinning the lease in that window was the root cause.

        Drain/rejoin callers (`drain.py`) must opt into fence enforcement via
        `enforce_state_version_fence=True`. They route to Lua's original
        `release` branch, which keeps the strict `RESERVATION_STATE_STALE`
        check that protects operator-initiated lifecycle transitions.
        """
        current = self.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state is MemberState.DRAINING:
            updated = replace(
                current,
                dispatcher_leases=0,
                drain_disabled_slots=current.drain_disabled_slots + 1,
                compatible_free_slots=0,
                state_version=current.state_version + 1,
            )
        else:
            updated = replace(
                current,
                dispatcher_leases=0,
                compatible_free_slots=0,
                state_version=current.state_version + 1,
            )
        return self._commit_member_drain(
            mode="release_v2" if not enforce_state_version_fence else "release",
            current=current,
            updated=updated,
            claimed_version=claimed_member_state_version,
            attempt_id=attempt_id,
        )

    def complete_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot:
        current = self.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state_version != expected_state_version:
            raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
        validate_member_transition(current.state, MemberState.OFFLINE)
        updated = replace(
            current,
            state=MemberState.OFFLINE,
            accepting=False,
            compatible_free_slots=0,
            state_version=current.state_version + 1,
        )
        return self._commit_member_drain(mode="complete", current=current, updated=updated)

    def begin_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        generation_fence: int,
    ) -> MemberSnapshot:
        current = self.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state_version != expected_state_version:
            raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
        if generation_fence <= current.generation_fence:
            raise ContractError("GENERATION_FENCE_STALE")
        validate_member_transition(current.state, MemberState.JOINING)
        updated = replace(
            current,
            state=MemberState.JOINING,
            accepting=False,
            compatible_free_slots=0,
            generation_fence=generation_fence,
            state_version=current.state_version + 1,
        )
        return self._commit_member_drain(
            mode="begin_rejoin", current=current, updated=updated
        )

    def complete_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        readiness: MemberSnapshot,
    ) -> MemberSnapshot:
        current = self.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state_version != expected_state_version:
            raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
        if readiness.name != member_name:
            raise ContractError("READINESS_MEMBER_MISMATCH")
        if readiness.state is not MemberState.READY_ACCEPTING or not readiness.accepting:
            raise ContractError("MEMBER_NOT_READY")
        if readiness.blockers or readiness.reason_codes:
            raise ContractError("MEMBER_NOT_READY")
        validate_member_transition(current.state, MemberState.READY_ACCEPTING)
        updated = replace(
            readiness,
            state=MemberState.READY_ACCEPTING,
            accepting=True,
            generation_fence=current.generation_fence,
            state_version=current.state_version + 1,
            drain_disabled_slots=0,
            dispatcher_leases=0,
        )
        return self._commit_member_drain(
            mode="complete_rejoin", current=current, updated=updated
        )

    def get_attempt(self, attempt_id: str) -> AttemptRecord | None:
        value = self._redis(
            lambda: self.client.hget(self._attempt_key(attempt_id), "record")
        )
        if value is None:
            return None
        try:
            return AttemptRecord.from_dict(json.loads(value))
        except (ContractError, TypeError, ValueError) as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc

    def reserve_next(
        self,
        *,
        now: float,
        ordered_members: list[MemberSnapshot],
        requirements: RequestRequirements,
        leader_owner: str,
        fencing_token: int,
        backend_url: str | None = None,
        member_lookup: Callable[[str], MemberSnapshot] | None = None,
    ) -> Reservation | None:
        selected = self.peek_next(now=now)
        if selected is None:
            return None
        # The readiness snapshot is advisory at reservation time. Redis is
        # authoritative for each member's state/capacity inside the Lua
        # transaction. In particular, a burst can consume a member's real
        # free slots before the next readiness refresh updates this Python
        # snapshot. Do not let that stale first member strand the queue head:
        # try every ordered compatible member and let the atomic claim decide
        # which one is currently reservable.
        candidates = [
            candidate
            for candidate in ordered_members
            if candidate.accepting and evaluate(requirements, candidate).compatible
        ]
        if not candidates:
            return None
        reservation_stale_seen = False
        for member in candidates:
            attempt_id = str(uuid4())
            claimed = replace(
                selected,
                state=JobState.CLAIMED,
                selected_member=member.name,
                member_state_version=member.state_version,
                attempt_count=selected.attempt_count + 1,
                current_attempt_id=attempt_id,
                queue_claimed_at=now,
                state_version=selected.state_version + 1,
            )
            reserved_member = replace(
                member,
                dispatcher_leases=member.dispatcher_leases + 1,
                compatible_free_slots=max(0, (member.compatible_free_slots or 0) - 1),
            )
            attempt = AttemptRecord(
                attempt_id=attempt_id,
                job_id=selected.job_id,
                member_name=member.name,
                reservation_fence=fencing_token,
                backend_url=backend_url or f"member://{member.name}",
                state=AttemptState.RESERVED,
                reserved_at=now,
            )
            keys = (
                self._job_key(selected.job_id),
                self._queue_key(selected.base_priority),
                self._member_key(member.name),
                self._lease_key(member.name),
                self._leader_key(),
                self._attempt_key(attempt_id),
            )
            try:
                self.client.eval(
                    RESERVE_JOB_LUA, len(keys), *keys,
                    selected.job_id, member.name, member.state_version, now, attempt_id,
                    self._json(claimed.to_dict()), self._json(reserved_member.to_dict()),
                    leader_owner, fencing_token, selected.state_version,
                    self._json(attempt.to_dict()),
                )
            except ResponseError as exc:
                message = str(exc)
                # Candidate staleness is expected under contention. The
                # dispatch loop owns the rate-limited operator warning and
                # cumulative counter; retain candidate detail at debug level.
                import logging
                logging.getLogger("gemma_broker.redis_store").debug(
                    "RESERVE_JOB_LUA err=%s job_id=%s member=%s",
                    message, selected.job_id, member.name,
                )
                if "DISPATCHER_FENCE_STALE" in message:
                    raise LeadershipLost("DISPATCHER_FENCE_STALE") from exc
                if "RESERVATION_MEMBER_VERSION_STALE" in message:
                    # Do not spill to a slower member solely because the
                    # preferred member changed between the authoritative view
                    # and Lua. The dispatch loop's bounded stale backoff will
                    # re-read the entire ordered list on the next tick.
                    raise ReservationStale(
                        ReasonCode.RESERVATION_STATE_STALE.value
                    ) from exc
                if (
                    "RESERVATION_MEMBER_UNAVAILABLE" in message
                    or "RESERVATION_STATE_STALE" in message
                ):
                    # This member has no atomically proved capacity. It is not
                    # a queue-head failure; continue to the next ordered member.
                    reservation_stale_seen = True
                    continue
                if "QUEUE_HEAD_STALE" in message:
                    raise ReservationStale("QUEUE_HEAD_STALE") from exc
                if "JOB_STATE_VERSION_STALE" in message or "ATTEMPT_EXISTS" in message:
                    raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value) from exc
                if "JOB_NOT_QUEUED" in message or "QUEUE_CLAIM_CONFLICT" in message:
                    return None
                raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
            except RedisError as exc:
                raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
            return Reservation(job=claimed, member=reserved_member, attempt=attempt)

        if reservation_stale_seen:
            raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
        return None

    def cancel_queued(self, job_id: str, *, completed_at: float) -> JobRecord:
        current = self.get(job_id)
        if current is None:
            raise ContractError(f"UNKNOWN_JOB_ID:{job_id}")
        if current.state is not JobState.QUEUED:
            return current
        cancelled = replace(
            current,
            state=JobState.CANCELLED,
            completed_at=completed_at,
            reason_code=ReasonCode.CANCELLED_BEFORE_ACCEPTANCE,
            state_version=current.state_version + 1,
        )
        try:
            result = self.client.eval(
                CANCEL_QUEUED_LUA,
                2,
                self._job_key(job_id),
                self._queue_key(current.base_priority),
                current.state_version,
                cancelled.state_version,
                self._json(cancelled.to_dict()),
                job_id,
            )
        except ResponseError as exc:
            message = str(exc)
            if "UNKNOWN_JOB_ID" in message:
                raise ContractError(f"UNKNOWN_JOB_ID:{job_id}") from exc
            if "JOB_STATE_VERSION_STALE" in message:
                raise ReservationStale("JOB_STATE_VERSION_STALE") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        outcome = self._text(result[0])
        if outcome == "already_not_queued":
            return self.get(job_id) or current
        if outcome != "cancelled":
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        self._notify_terminal(cancelled.job_id)
        return cancelled

    def requeue_claim(
        self,
        job: JobRecord,
        *,
        reason: ReasonCode,
        terminal_at: float | None = None,
    ) -> JobRecord:
        current = self.get(job.job_id)
        if current is None:
            raise ContractError(f"UNKNOWN_JOB_ID:{job.job_id}")
        if current.state not in {JobState.CLAIMED, JobState.ACCEPTED, JobState.IN_FLIGHT}:
            raise ContractError(f"REQUEUE_INVALID_STATE:{current.state.value}")
        if current.accepted_boundary_crossed:
            raise ContractError(ReasonCode.CANCEL_UNSAFE_AFTER_ACCEPTANCE.value)
        attempt_id = current.current_attempt_id
        if not attempt_id:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        attempt = self.get_attempt(attempt_id)
        if attempt is None or attempt.accepted_boundary_crossed:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        attempt_terminal_at = float(
            terminal_at
            if terminal_at is not None
            else current.queue_claimed_at or current.submitted_at
        )
        failed_attempt = replace(
            attempt,
            state=AttemptState.FAILED,
            terminal_at=attempt_terminal_at,
            error_category=reason,
            transition_history=(*attempt.transition_history, AttemptState.FAILED.value),
        )
        queued = replace(
            current,
            state=JobState.QUEUED,
            selected_member=None,
            member_state_version=None,
            current_attempt_id=None,
            queue_claimed_at=None,
            forward_started_at=None,
            reason_code=reason,
            error=None,
            state_version=current.state_version + 1,
        )
        member_name = current.selected_member or ""
        member = self.get_member(member_name) if member_name else None
        new_member = (
            replace(
                member,
                dispatcher_leases=max(0, member.dispatcher_leases - 1),
                compatible_free_slots=min(
                    member.configured_slots, (member.compatible_free_slots or 0) + 1
                ),
                state_version=member.state_version + 1,
            )
            if member is not None
            else None
        )
        member_key = self._member_key(member_name) if member_name else self._leader_key()
        lease_key = self._lease_key(member_name) if member_name else self._leader_key()
        keys = (
            self._job_key(current.job_id),
            self._queue_key(current.base_priority),
            member_key,
            lease_key,
            self._attempt_key(attempt_id),
        )
        argv = [
            current.state_version,
            queued.state_version,
            self._json(queued.to_dict()),
            current.enqueue_sequence,
            current.job_id,
        ]
        if new_member is not None:
            argv.extend(
                [
                    member.state_version,
                    new_member.state_version,
                    self._json(new_member.to_dict()),
                    new_member.state.value,
                    "1" if new_member.accepting else "0",
                    new_member.compatible_free_slots or 0,
                    new_member.generation_fence,
                    attempt_id,
                    self._json(failed_attempt.to_dict()),
                    attempt_terminal_at,
                    reason.value,
                    self.terminal_attempt_retention_seconds,
                ]
            )
        else:
            argv.extend(
                [
                    "0",
                    "0",
                    "",
                    "",
                    "0",
                    "0",
                    "0",
                    attempt_id,
                    self._json(failed_attempt.to_dict()),
                    attempt_terminal_at,
                    reason.value,
                    self.terminal_attempt_retention_seconds,
                ]
            )
        try:
            self.client.eval(REQUEUE_CLAIM_LUA, len(keys), *keys, *argv)
        except ResponseError as exc:
            message = str(exc)
            if "UNKNOWN_JOB_ID" in message:
                raise ContractError(f"UNKNOWN_JOB_ID:{current.job_id}") from exc
            if "JOB_STATE_VERSION_STALE" in message:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value) from exc
            if "RESERVATION_STATE_STALE" in message:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value) from exc
            if "REQUEUE_INVALID_STATE" in message:
                raise ContractError(message) from exc
            if "CANCEL_UNSAFE_AFTER_ACCEPTANCE" in message:
                raise ContractError(ReasonCode.CANCEL_UNSAFE_AFTER_ACCEPTANCE.value) from exc
            if "UNKNOWN_ATTEMPT_ID" in message or "ATTEMPT_STATE_STALE" in message:
                raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        return queued

    def _attempt_for_job(self, job: JobRecord) -> AttemptRecord | None:
        if not job.current_attempt_id:
            return None
        state = {
            JobState.ACCEPTED: AttemptState.ACCEPTED,
            JobState.IN_FLIGHT: AttemptState.FORWARDING,
            JobState.COMPLETED: AttemptState.COMPLETED,
            JobState.FAILED: AttemptState.FAILED,
            JobState.CANCELLED: AttemptState.FAILED,
            JobState.OUTCOME_UNKNOWN: AttemptState.OUTCOME_UNKNOWN,
        }.get(job.state)
        if state is None:
            return None
        attempt = self.get_attempt(job.current_attempt_id)
        if attempt is None:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        terminal = job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.OUTCOME_UNKNOWN}
        return replace(attempt, state=state,
                       terminal_at=job.completed_at if terminal else attempt.terminal_at,
                       transition_history=(*attempt.transition_history, state.value))

    def replace_job(self, job: JobRecord) -> JobRecord:
        expected_version = job.state_version - 1
        if expected_version < 1:
            raise ContractError("JOB_STATE_VERSION_INVALID")
        attempt = self._attempt_for_job(job)
        attempt_key = self._attempt_key(attempt.attempt_id) if attempt else self._job_key(job.job_id)
        try:
            result = self.client.eval(
                REPLACE_JOB_LUA,
                2,
                self._job_key(job.job_id),
                attempt_key,
                expected_version,
                job.state_version,
                self._json(job.to_dict()),
                job.state.value,
                "1" if attempt else "0",
                self._json(attempt.to_dict()) if attempt else "",
                attempt.state.value if attempt else "",
                self.terminal_attempt_retention_seconds,
            )
        except ResponseError as exc:
            message = str(exc)
            if "UNKNOWN_JOB_ID" in message:
                raise ContractError(f"UNKNOWN_JOB_ID:{job.job_id}") from exc
            if "UNKNOWN_ATTEMPT_ID" in message:
                raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
            if "JOB_STATE_VERSION_STALE" in message:
                raise ReservationStale("JOB_STATE_VERSION_STALE") from exc
            if "JOB_STATE_VERSION_INVALID" in message:
                raise ContractError("JOB_STATE_VERSION_INVALID") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        if self._text(result[0]) != "updated":
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        if job.state in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.OUTCOME_UNKNOWN,
        }:
            self._notify_terminal(job.job_id)
        return job

    def reconcile_dead_job(
        self,
        job_id: str,
        *,
        repair_owner: str,
        repair_fence: int,
    ) -> JobRecord:
        if not repair_owner or repair_fence <= 0:
            raise ValueError("repair owner and positive fence are required")
        current = self.get(job_id)
        if current is None:
            raise ContractError(f"UNKNOWN_JOB_ID:{job_id}")
        if current.state in {
            JobState.QUEUED,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.OUTCOME_UNKNOWN,
        }:
            return current
        if not current.selected_member or not current.current_attempt_id:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        attempt = self.get_attempt(current.current_attempt_id)
        if attempt is None:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)

        post_acceptance = current.accepted_boundary_crossed or current.state in {
            JobState.ACCEPTED,
            JobState.IN_FLIGHT,
        }
        if post_acceptance:
            repaired = replace(
                current,
                state=JobState.OUTCOME_UNKNOWN,
                reason_code=ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN,
                error="proven_dead_after_acceptance",
                state_version=current.state_version + 1,
            )
            repaired_attempt = replace(
                attempt,
                state=AttemptState.OUTCOME_UNKNOWN,
                terminal_at=0.0,
                error_category=ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN,
                transition_history=(*attempt.transition_history, "outcome_unknown"),
            )
        else:
            repaired = replace(
                current,
                state=JobState.QUEUED,
                selected_member=None,
                member_state_version=None,
                current_attempt_id=None,
                queue_claimed_at=None,
                forward_started_at=None,
                reason_code=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
                error=None,
                state_version=current.state_version + 1,
            )
            repaired_attempt = replace(
                attempt,
                state=AttemptState.FAILED,
                terminal_at=0.0,
                error_category=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
                transition_history=(*attempt.transition_history, "failed"),
            )
        keys = (
            self._job_key(job_id),
            self._queue_key(current.base_priority),
            self._repair_key(job_id),
            self._lease_key(current.selected_member),
            self._attempt_key(attempt.attempt_id),
        )
        try:
            result = self.client.eval(
                REPAIR_DEAD_JOB_LUA,
                len(keys),
                *keys,
                repair_owner,
                repair_fence,
                current.state_version,
                repaired.state_version,
                self._json(repaired.to_dict()),
                repaired.state.value,
                repaired.enqueue_sequence,
                attempt.attempt_id,
                self._json(repaired_attempt.to_dict()),
                repaired_attempt.state.value,
                job_id,
                self.terminal_attempt_retention_seconds,
            )
        except ResponseError as exc:
            message = str(exc)
            if "UNKNOWN_JOB_ID" in message:
                raise ContractError(f"UNKNOWN_JOB_ID:{job_id}") from exc
            if "JOB_STATE_VERSION_STALE" in message:
                raise ReservationStale("JOB_STATE_VERSION_STALE") from exc
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        outcome = self._text(result[0])
        if outcome in {"already_repaired", "already_terminal"}:
            return self.get(job_id) or current
        if outcome != "repaired":
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value)
        if repaired.state in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.OUTCOME_UNKNOWN,
        }:
            self._notify_terminal(repaired.job_id)
        return repaired

    def reconcile_stale_post_acceptance_jobs(
        self,
        *,
        repair_owner: str,
        repair_fence: int,
        limit: int = 100,
    ) -> list[JobRecord]:
        """Terminalize accepted work owned by a prior dispatcher fence.

        The caller must wait at least one complete request-timeout after
        acquiring ``repair_fence``.  That startup grace is the durable proof
        that the prior process can no longer produce an authoritative result;
        persisted lifecycle timestamps are intentionally not consulted because
        older records used a process-monotonic clock that resets across reboot.

        Only job hashes are scanned.  The historical attempt namespace can be
        very large and is never traversed by this bounded repair path.
        """
        if not repair_owner or repair_fence <= 0:
            raise ValueError("repair owner and positive fence are required")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("repair limit must be a positive integer")

        recovered: list[JobRecord] = []
        job_prefix = f"{self.prefix}job:"
        try:
            keys = self.client.scan_iter(match=f"{job_prefix}*", count=100)
            for raw_key in keys:
                key = self._text(raw_key)
                if not key.startswith(job_prefix):
                    continue
                job_id = key[len(job_prefix) :]
                if not job_id:
                    continue
                current = self.get(job_id)
                if current is None or current.state not in {
                    JobState.ACCEPTED,
                    JobState.IN_FLIGHT,
                }:
                    continue
                if (
                    not current.accepted_boundary_crossed
                    or not current.selected_member
                    or not current.current_attempt_id
                ):
                    continue
                attempt = self.get_attempt(current.current_attempt_id)
                if attempt is None or attempt.state not in {
                    AttemptState.ACCEPTED,
                    AttemptState.FORWARDING,
                }:
                    continue
                if attempt.reservation_fence >= repair_fence:
                    continue
                repaired = self.reconcile_dead_job(
                    job_id,
                    repair_owner=repair_owner,
                    repair_fence=repair_fence,
                )
                if repaired.state is JobState.OUTCOME_UNKNOWN:
                    recovered.append(repaired)
                if len(recovered) >= limit:
                    break
        except (RepositoryUnavailable, ReservationStale):
            raise
        except RedisError as exc:
            raise RepositoryUnavailable(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        return recovered
