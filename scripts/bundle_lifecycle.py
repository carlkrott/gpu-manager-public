"""Bundle-level lifecycle state, recovery decisions, and durable fencing.

This module is intentionally service-agnostic. GPU Manager owns the lifecycle
of a bundle; individual services only contribute configuration and readiness
observations. The reducer remains pure, while the controller/store provide
persistence and single-owner transition fencing.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import inspect
import socket
import time
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping, Protocol


class DesiredState(StrEnum):
    STOPPED = "stopped"
    RUNNING = "running"
    IDLE = "idle"
    PINNED = "pinned"


class RecoveryPhase(StrEnum):
    DISABLED = "disabled"
    INTENTIONALLY_OFF = "intentionally_off"
    READY = "ready"
    CRASHED = "crashed"
    RECOVERING = "recovering"
    FAULTED = "faulted"
    MAINTENANCE = "maintenance"
    BLOCKED = "blocked"


class RecoveryAction(StrEnum):
    SKIP = "skip"
    PROBE = "probe"
    START = "start"
    RESTART = "restart"
    QUARANTINE = "quarantine"


@dataclass(frozen=True, slots=True)
class RecoveryState:
    phase: RecoveryPhase = RecoveryPhase.INTENTIONALLY_OFF
    transition_id: int = 0
    restart_attempts: int = 0
    retry_window_started_at: float | None = None
    last_restart_at: float | None = None
    last_healthy_at: float | None = None


@dataclass(frozen=True, slots=True)
class RecoveryObservation:
    """One reducer input sample. ``probe_ok=None`` means probe is still needed."""

    enabled: bool = True
    desired_state: DesiredState = DesiredState.IDLE
    maintenance_mode: bool = False
    planned_stop: bool = False
    coexistence_paused: bool = False
    probe_ok: bool | None = None
    now: float = 0.0
    rejoin_requested: bool = False
    max_restart_attempts: int = 3
    retry_window_seconds: float = 300.0
    min_restart_interval_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    previous: RecoveryState
    next: RecoveryState
    emit_transition_log: bool = False


def _changed(previous: RecoveryState, next_state: RecoveryState) -> bool:
    return previous.phase != next_state.phase or previous.transition_id != next_state.transition_id


def _phase(state: RecoveryState, phase: RecoveryPhase, **changes: object) -> RecoveryState:
    if phase != state.phase:
        changes.setdefault("transition_id", state.transition_id + 1)
    return replace(state, phase=phase, **changes)


def _decision(action: RecoveryAction, reason: str, previous: RecoveryState,
              next_state: RecoveryState, *, emit_transition_log: bool = False) -> RecoveryDecision:
    return RecoveryDecision(
        action=action,
        reason=reason,
        previous=previous,
        next=next_state,
        emit_transition_log=emit_transition_log or _changed(previous, next_state),
    )


def reduce_recovery(state: RecoveryState, observation: RecoveryObservation) -> RecoveryDecision:
    """Reduce one observation to exactly one safe recovery action."""
    if not observation.enabled or observation.desired_state is DesiredState.STOPPED:
        next_state = _phase(state, RecoveryPhase.DISABLED)
        return _decision(
            RecoveryAction.SKIP,
            "config_disabled" if not observation.enabled else "desired_stopped",
            state,
            next_state,
        )

    if observation.maintenance_mode:
        return _decision(RecoveryAction.SKIP, "maintenance_mode", state,
                         _phase(state, RecoveryPhase.MAINTENANCE))

    if observation.planned_stop:
        return _decision(RecoveryAction.SKIP, "planned_stop", state,
                         _phase(state, RecoveryPhase.BLOCKED))

    if observation.coexistence_paused:
        return _decision(RecoveryAction.SKIP, "coexistence_paused", state,
                         _phase(state, RecoveryPhase.BLOCKED))

    if state.phase is RecoveryPhase.FAULTED:
        if not observation.rejoin_requested:
            return _decision(RecoveryAction.SKIP, "circuit_open", state, state)
        state = replace(
            state,
            phase=RecoveryPhase.INTENTIONALLY_OFF,
            transition_id=state.transition_id + 1,
            restart_attempts=0,
            retry_window_started_at=None,
            last_restart_at=None,
        )

    if observation.probe_ok is None:
        if (
            state.phase in {RecoveryPhase.RECOVERING, RecoveryPhase.CRASHED}
            and state.last_restart_at is not None
            and observation.now - state.last_restart_at < observation.min_restart_interval_seconds
        ):
            return _decision(RecoveryAction.SKIP, "restart_cooldown", state, state)
        return _decision(RecoveryAction.PROBE, "probe_required", state, state)

    if observation.probe_ok:
        next_state = _phase(
            state,
            RecoveryPhase.READY,
            restart_attempts=0,
            retry_window_started_at=None,
            last_restart_at=None,
            last_healthy_at=observation.now,
        )
        return _decision(RecoveryAction.SKIP, "ready", state, next_state)

    if state.last_healthy_at is None:
        next_state = _phase(state, RecoveryPhase.CRASHED, last_restart_at=observation.now)
        return _decision(RecoveryAction.START, "never_healthy", state, next_state)

    window_started = state.retry_window_started_at
    attempts = state.restart_attempts
    if window_started is None or observation.now - window_started > observation.retry_window_seconds:
        window_started = observation.now
        attempts = 0

    attempts += 1
    if attempts > max(1, observation.max_restart_attempts):
        next_state = _phase(
            state,
            RecoveryPhase.FAULTED,
            restart_attempts=attempts,
            retry_window_started_at=window_started,
            last_restart_at=None,
        )
        return _decision(RecoveryAction.QUARANTINE, "restart_attempts_exhausted", state, next_state)

    next_state = replace(
        state,
        phase=RecoveryPhase.RECOVERING,
        transition_id=state.transition_id + 1,
        restart_attempts=attempts,
        retry_window_started_at=window_started,
        last_restart_at=observation.now,
    )
    return _decision(
        RecoveryAction.RESTART,
        "health_lost",
        state,
        next_state,
        emit_transition_log=state.phase is RecoveryPhase.READY,
    )


@dataclass(frozen=True, slots=True)
class BundleLifecycleRecord:
    """Durable bundle state plus the current fencing lease."""

    bundle_name: str
    desired_state: DesiredState = DesiredState.IDLE
    recovery: RecoveryState = RecoveryState()
    generation_fence: int = 0
    transition_owner: str | None = None
    transition_expires_at: float = 0.0
    last_reason: str = ""
    last_error: str | None = None
    updated_at: float = 0.0

    @property
    def phase(self) -> RecoveryPhase:
        return self.recovery.phase

    @property
    def transition_id(self) -> int:
        return self.recovery.transition_id

    def to_fields(self) -> dict[str, str]:
        def value(item: object) -> str:
            return "" if item is None else str(item)

        return {
            "schema_version": "1",
            "bundle_name": self.bundle_name,
            "desired_state": self.desired_state.value,
            "phase": self.recovery.phase.value,
            "transition_id": str(self.recovery.transition_id),
            "restart_attempts": str(self.recovery.restart_attempts),
            "retry_window_started_at": value(self.recovery.retry_window_started_at),
            "last_restart_at": value(self.recovery.last_restart_at),
            "last_healthy_at": value(self.recovery.last_healthy_at),
            "generation_fence": str(self.generation_fence),
            "transition_owner": value(self.transition_owner),
            "transition_expires_at": str(self.transition_expires_at),
            "last_reason": self.last_reason,
            "last_error": value(self.last_error),
            "updated_at": str(self.updated_at),
        }

    @classmethod
    def from_fields(cls, bundle_name: str, fields: Mapping[str, Any]) -> "BundleLifecycleRecord":
        def text(name: str, default: str = "") -> str:
            raw = fields.get(name, default)
            return default if raw is None else str(raw)

        def integer(name: str, default: int = 0) -> int:
            try:
                return int(text(name, str(default)) or default)
            except (TypeError, ValueError):
                return default

        def number_or_none(name: str) -> float | None:
            raw = text(name)
            if not raw:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        try:
            desired = DesiredState(text("desired_state", DesiredState.IDLE.value))
        except ValueError:
            desired = DesiredState.IDLE
        try:
            phase = RecoveryPhase(text("phase", RecoveryPhase.INTENTIONALLY_OFF.value))
        except ValueError:
            phase = RecoveryPhase.INTENTIONALLY_OFF

        owner = text("transition_owner") or None
        try:
            transition_expires_at = float(text("transition_expires_at", "0") or 0)
        except (TypeError, ValueError):
            transition_expires_at = 0.0
        try:
            updated_at = float(text("updated_at", "0") or 0)
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(
            bundle_name=text("bundle_name", bundle_name),
            desired_state=desired,
            recovery=RecoveryState(
                phase=phase,
                transition_id=integer("transition_id"),
                restart_attempts=integer("restart_attempts"),
                retry_window_started_at=number_or_none("retry_window_started_at"),
                last_restart_at=number_or_none("last_restart_at"),
                last_healthy_at=number_or_none("last_healthy_at"),
            ),
            generation_fence=integer("generation_fence"),
            transition_owner=owner,
            transition_expires_at=transition_expires_at,
            last_reason=text("last_reason"),
            last_error=text("last_error") or None,
            updated_at=updated_at,
        )


@dataclass(frozen=True, slots=True)
class BundleTransitionLease:
    bundle_name: str
    owner: str
    generation_fence: int
    transition_id: int
    expires_at: float


@dataclass(frozen=True, slots=True)
class TransitionAttempt:
    acquired: bool
    lease: BundleTransitionLease | None = None
    reason: str = ""
    record: BundleLifecycleRecord | None = None


@dataclass(frozen=True, slots=True)
class BundleTransitionGroupLease:
    """A single-owner lease covering several bundles acquired atomically."""

    leases: tuple[BundleTransitionLease, ...]


@dataclass(frozen=True, slots=True)
class TransitionGroupAttempt:
    acquired: bool
    lease: BundleTransitionGroupLease | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ResourceLease:
    """Fenced lease over an intersecting set of canonical resources."""

    resource_ids: tuple[str, ...]
    owner: str
    fence: int
    expires_at: float


def canonical_resource_ids(resources: Any) -> tuple[str, ...]:
    """Normalize stable resource identities for atomic lease operations.

    Callers should namespace identities (for example ``gpu:primary`` and
    ``cpu:0-7``) so unrelated resource classes cannot collide accidentally.
    Empty/blank identities are rejected rather than silently becoming a
    lease over fewer resources than the caller requested.
    """

    if isinstance(resources, (str, bytes)):
        resources = [resources]
    try:
        values = list(resources)
    except TypeError as exc:
        raise ValueError("resources must be an iterable of non-empty strings") from exc
    normalized: set[str] = set()
    for resource in values:
        if isinstance(resource, bytes):
            resource = resource.decode("utf-8", errors="strict")
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError("resource identities must be non-empty strings")
        normalized.add(resource.strip())
    return tuple(sorted(normalized))


def _normalize_group_transitions(
    transitions: list[tuple[str, DesiredState]],
) -> tuple[tuple[str, DesiredState], ...]:
    """Return deterministic, unique bundle transitions for a group operation."""
    normalized: dict[str, DesiredState] = {}
    for bundle_name, desired_state in transitions:
        name = str(bundle_name).strip()
        if not name:
            continue
        normalized[name] = (
            desired_state
            if isinstance(desired_state, DesiredState)
            else DesiredState(desired_state)
        )
    return tuple(sorted(normalized.items()))


class BundleLifecycleStore(Protocol):
    async def get(self, bundle_name: str) -> BundleLifecycleRecord: ...

    async def renew(
        self,
        lease: BundleTransitionLease,
        now: float,
        ttl_seconds: float,
    ) -> bool: ...

    async def renew_group(
        self,
        lease: BundleTransitionGroupLease,
        now: float,
        ttl_seconds: float,
    ) -> bool: ...

    async def acquire_resources(
        self,
        resource_ids: tuple[str, ...],
        owner: str,
        now: float,
        ttl_seconds: float,
        reason: str,
    ) -> ResourceLease | None: ...

    async def renew_resources(
        self,
        lease: ResourceLease,
        now: float,
        ttl_seconds: float,
    ) -> bool: ...

    async def release_resources(self, lease: ResourceLease, now: float) -> bool: ...

    async def record_recovery(
        self,
        bundle_name: str,
        desired_state: DesiredState,
        recovery: RecoveryState,
        now: float,
        reason: str,
    ) -> bool: ...

    async def begin(
        self,
        bundle_name: str,
        desired_state: DesiredState,
        owner: str,
        now: float,
        ttl_seconds: float,
        reason: str,
    ) -> TransitionAttempt: ...

    async def begin_group(
        self,
        transitions: list[tuple[str, DesiredState]],
        owner: str,
        now: float,
        ttl_seconds: float,
        reason: str,
    ) -> TransitionGroupAttempt: ...

    async def finish(
        self,
        lease: BundleTransitionLease,
        *,
        success: bool,
        now: float,
        reason: str = "",
        error: str | None = None,
    ) -> bool: ...

    async def finish_group(
        self,
        lease: BundleTransitionGroupLease,
        *,
        success: bool,
        now: float,
        reason: str = "",
        error: str | None = None,
    ) -> bool: ...


class InMemoryBundleLifecycleStore:
    """Deterministic store used by unit tests and Redis-failure fallback tests."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._records: dict[str, BundleLifecycleRecord] = {}
        self._lock = asyncio.Lock()
        self._resource_leases: dict[str, tuple[str, int, float]] = {}
        self._resource_fence = 0
        self._clock = clock

    async def acquire_resources(
        self,
        resource_ids: tuple[str, ...],
        owner: str,
        now: float,
        ttl_seconds: float,
        reason: str,
    ) -> ResourceLease | None:
        del reason
        normalized = canonical_resource_ids(resource_ids)
        if not normalized:
            return None
        expires_at = now + max(0.1, float(ttl_seconds))
        async with self._lock:
            if any(
                current is not None and current[2] > now
                for resource in normalized
                for current in (self._resource_leases.get(resource),)
            ):
                return None
            self._resource_fence += 1
            fence = self._resource_fence
            for resource in normalized:
                self._resource_leases[resource] = (owner, fence, expires_at)
            return ResourceLease(normalized, owner, fence, expires_at)

    async def renew_resources(
        self,
        lease: ResourceLease,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        normalized = canonical_resource_ids(lease.resource_ids)
        expires_at = now + max(0.1, float(ttl_seconds))
        async with self._lock:
            if any(
                (
                    (current := self._resource_leases.get(resource)) is None
                    or current[:2] != (lease.owner, lease.fence)
                    or current[2] <= now
                )
                for resource in normalized
            ):
                return False
            for resource in normalized:
                self._resource_leases[resource] = (lease.owner, lease.fence, expires_at)
            return True

    async def release_resources(self, lease: ResourceLease, now: float) -> bool:
        del now
        normalized = canonical_resource_ids(lease.resource_ids)
        async with self._lock:
            if any(
                self._resource_leases.get(resource, (None, None, None))[:2]
                != (lease.owner, lease.fence)
                for resource in normalized
            ):
                return False
            for resource in normalized:
                self._resource_leases.pop(resource, None)
            return True

    async def get(self, bundle_name: str) -> BundleLifecycleRecord:
        async with self._lock:
            return self._records.get(bundle_name) or BundleLifecycleRecord(
                bundle_name=bundle_name,
                updated_at=self._clock(),
            )

    async def begin(self, bundle_name: str, desired_state: DesiredState, owner: str,
                    now: float, ttl_seconds: float, reason: str) -> TransitionAttempt:
        async with self._lock:
            current = self._records.get(bundle_name) or BundleLifecycleRecord(
                bundle_name=bundle_name, updated_at=now,
            )
            if current.transition_owner and current.transition_expires_at > now:
                return TransitionAttempt(False, reason="transition_in_progress", record=current)
            next_recovery = replace(
                current.recovery,
                phase=RecoveryPhase.RECOVERING,
                transition_id=current.recovery.transition_id + 1,
                last_restart_at=now,
            )
            next_record = replace(
                current,
                desired_state=desired_state,
                recovery=next_recovery,
                generation_fence=current.generation_fence + 1,
                transition_owner=owner,
                transition_expires_at=now + ttl_seconds,
                last_reason=reason,
                last_error=None,
                updated_at=now,
            )
            self._records[bundle_name] = next_record
            lease = BundleTransitionLease(
                bundle_name=bundle_name,
                owner=owner,
                generation_fence=next_record.generation_fence,
                transition_id=next_record.transition_id,
                expires_at=next_record.transition_expires_at,
            )
            return TransitionAttempt(True, lease=lease, reason="acquired", record=next_record)

    async def begin_group(self, transitions: list[tuple[str, DesiredState]], owner: str,
                          now: float, ttl_seconds: float, reason: str) -> TransitionGroupAttempt:
        normalized = _normalize_group_transitions(transitions)
        if not normalized:
            return TransitionGroupAttempt(False, reason="empty_group")
        async with self._lock:
            current_records = {
                bundle_name: self._records.get(bundle_name) or BundleLifecycleRecord(
                    bundle_name=bundle_name, updated_at=now,
                )
                for bundle_name, _desired_state in normalized
            }
            if any(
                record.transition_owner and record.transition_expires_at > now
                for record in current_records.values()
            ):
                return TransitionGroupAttempt(False, reason="transition_in_progress")

            leases = []
            for bundle_name, desired_state in normalized:
                current = current_records[bundle_name]
                next_recovery = replace(
                    current.recovery,
                    phase=RecoveryPhase.RECOVERING,
                    transition_id=current.recovery.transition_id + 1,
                    last_restart_at=now,
                )
                next_record = replace(
                    current,
                    desired_state=desired_state,
                    recovery=next_recovery,
                    generation_fence=current.generation_fence + 1,
                    transition_owner=owner,
                    transition_expires_at=now + ttl_seconds,
                    last_reason=reason,
                    last_error=None,
                    updated_at=now,
                )
                self._records[bundle_name] = next_record
                leases.append(BundleTransitionLease(
                    bundle_name=bundle_name,
                    owner=owner,
                    generation_fence=next_record.generation_fence,
                    transition_id=next_record.transition_id,
                    expires_at=next_record.transition_expires_at,
                ))
            return TransitionGroupAttempt(
                True,
                lease=BundleTransitionGroupLease(tuple(leases)),
                reason="acquired",
            )

    async def renew(
        self,
        lease: BundleTransitionLease,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        """Extend a live transition only for its current fenced owner."""
        expires_at = now + max(0.1, float(ttl_seconds))
        async with self._lock:
            current = self._records.get(lease.bundle_name)
            if (
                current is None
                or current.transition_owner != lease.owner
                or current.generation_fence != lease.generation_fence
                or current.transition_id != lease.transition_id
                or current.transition_expires_at <= now
            ):
                return False
            self._records[lease.bundle_name] = replace(
                current,
                transition_expires_at=expires_at,
                updated_at=now,
            )
            return True

    async def renew_group(
        self,
        lease: BundleTransitionGroupLease,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        """Atomically extend every member of a live group transition."""
        if not lease.leases:
            return False
        expires_at = now + max(0.1, float(ttl_seconds))
        async with self._lock:
            current_records = [
                self._records.get(item.bundle_name) for item in lease.leases
            ]
            if any(
                current is None
                or current.transition_owner != item.owner
                or current.generation_fence != item.generation_fence
                or current.transition_id != item.transition_id
                or current.transition_expires_at <= now
                for current, item in zip(current_records, lease.leases)
            ):
                return False
            for current in current_records:
                assert current is not None
                self._records[current.bundle_name] = replace(
                    current,
                    transition_expires_at=expires_at,
                    updated_at=now,
                )
            return True

    async def record_recovery(self, bundle_name: str, desired_state: DesiredState,
                              recovery: RecoveryState, now: float, reason: str) -> bool:
        async with self._lock:
            current = self._records.get(bundle_name) or BundleLifecycleRecord(
                bundle_name=bundle_name, updated_at=now,
            )
            if current.transition_owner and current.transition_expires_at > now:
                return False
            self._records[bundle_name] = replace(
                current,
                desired_state=desired_state,
                recovery=recovery,
                last_reason=reason,
                updated_at=now,
            )
            return True

    async def finish(self, lease: BundleTransitionLease, *, success: bool, now: float,
                     reason: str = "", error: str | None = None) -> bool:
        async with self._lock:
            current = self._records.get(lease.bundle_name)
            if not current or current.transition_owner != lease.owner:
                return False
            if current.transition_expires_at <= now:
                return False
            if current.generation_fence != lease.generation_fence or current.transition_id != lease.transition_id:
                return False
            phase = RecoveryPhase.READY if success else RecoveryPhase.CRASHED
            healthy = now if success else current.recovery.last_healthy_at
            recovery = replace(current.recovery, phase=phase, last_healthy_at=healthy)
            self._records[lease.bundle_name] = replace(
                current,
                recovery=recovery,
                transition_owner=None,
                transition_expires_at=0.0,
                last_reason=reason or current.last_reason,
                last_error=None if success else (error or "bundle transition failed"),
                updated_at=now,
            )
            return True

    async def finish_group(self, lease: BundleTransitionGroupLease, *, success: bool,
                           now: float, reason: str = "", error: str | None = None) -> bool:
        async with self._lock:
            current_records = [self._records.get(item.bundle_name) for item in lease.leases]
            if any(
                current is None
                or current.transition_owner != item.owner
                or current.transition_expires_at <= now
                or current.generation_fence != item.generation_fence
                or current.transition_id != item.transition_id
                for current, item in zip(current_records, lease.leases)
            ):
                return False
            phase = RecoveryPhase.READY if success else RecoveryPhase.CRASHED
            for current in current_records:
                assert current is not None
                healthy = now if success else current.recovery.last_healthy_at
                recovery = replace(current.recovery, phase=phase, last_healthy_at=healthy)
                self._records[current.bundle_name] = replace(
                    current,
                    recovery=recovery,
                    transition_owner=None,
                    transition_expires_at=0.0,
                    last_reason=reason or current.last_reason,
                    last_error=None if success else (error or "bundle transition group failed"),
                    updated_at=now,
                )
            return True


class RedisBundleLifecycleStore:
    """Redis-backed implementation with atomic owner and generation fencing."""

    _RESOURCE_FENCE_SUFFIX = "resource-fence"

    _ACQUIRE_RESOURCES_SCRIPT = """
    local now = tonumber(ARGV[2])
    local owner = ARGV[1]
    local expires = tonumber(ARGV[3])
    for i = 1, (#KEYS - 1) do
      local current_owner = redis.call('HGET', KEYS[i], 'owner') or ''
      local current_expires = tonumber(redis.call('HGET', KEYS[i], 'expires_at') or '0')
      if current_owner ~= '' and current_expires > now then
        return {0, i}
      end
    end
    local fence = redis.call('INCR', KEYS[#KEYS])
    local ttl = math.max(1, math.ceil(expires - now) + 1)
    for i = 1, (#KEYS - 1) do
      redis.call('HSET', KEYS[i], 'owner', owner, 'fence', fence, 'expires_at', expires)
      redis.call('EXPIRE', KEYS[i], ttl)
    end
    return {1, fence}
    """

    _RENEW_RESOURCES_SCRIPT = """
    local now = tonumber(ARGV[3])
    local owner = ARGV[1]
    local fence = ARGV[2]
    local expires = tonumber(ARGV[4])
    for i = 1, #KEYS do
      local current_owner = redis.call('HGET', KEYS[i], 'owner') or ''
      local current_fence = redis.call('HGET', KEYS[i], 'fence') or ''
      local current_expires = tonumber(redis.call('HGET', KEYS[i], 'expires_at') or '0')
      if current_owner ~= owner or current_fence ~= fence or current_expires <= now then
        return 0
      end
    end
    local ttl = math.max(1, math.ceil(expires - now) + 1)
    for i = 1, #KEYS do
      redis.call('HSET', KEYS[i], 'expires_at', expires)
      redis.call('EXPIRE', KEYS[i], ttl)
    end
    return 1
    """

    _RELEASE_RESOURCES_SCRIPT = """
    local owner = ARGV[1]
    local fence = ARGV[2]
    for i = 1, #KEYS do
      local current_owner = redis.call('HGET', KEYS[i], 'owner') or ''
      local current_fence = redis.call('HGET', KEYS[i], 'fence') or ''
      if current_owner ~= owner or current_fence ~= fence then
        return 0
      end
    end
    for i = 1, #KEYS do
      redis.call('DEL', KEYS[i])
    end
    return 1
    """

    _RENEW_SCRIPT = """
    local owner = redis.call('HGET', KEYS[1], 'transition_owner') or ''
    local fence = tonumber(redis.call('HGET', KEYS[1], 'generation_fence') or '-1')
    local transition = tonumber(redis.call('HGET', KEYS[1], 'transition_id') or '-1')
    local expires = tonumber(redis.call('HGET', KEYS[1], 'transition_expires_at') or '0')
    local now = tonumber(ARGV[5])
    if owner ~= ARGV[1]
        or fence ~= tonumber(ARGV[2])
        or transition ~= tonumber(ARGV[3])
        or expires <= now then
      return 0
    end
    redis.call('HSET', KEYS[1],
      'transition_expires_at', ARGV[4],
      'updated_at', ARGV[5])
    redis.call('EXPIRE', KEYS[1], ARGV[6])
    return 1
    """

    _RENEW_GROUP_SCRIPT = """
    local now = tonumber(ARGV[3])
    local owner = ARGV[1]
    for i, key in ipairs(KEYS) do
      local base = 5 + ((i - 1) * 2)
      local current_owner = redis.call('HGET', key, 'transition_owner') or ''
      local fence = tonumber(redis.call('HGET', key, 'generation_fence') or '-1')
      local transition = tonumber(redis.call('HGET', key, 'transition_id') or '-1')
      local expires = tonumber(redis.call('HGET', key, 'transition_expires_at') or '0')
      if current_owner ~= owner
          or fence ~= tonumber(ARGV[base])
          or transition ~= tonumber(ARGV[base + 1])
          or expires <= now then
        return 0
      end
    end
    for i, key in ipairs(KEYS) do
      redis.call('HSET', key,
        'transition_expires_at', ARGV[2],
        'updated_at', ARGV[3])
      redis.call('EXPIRE', key, ARGV[4])
    end
    return 1
    """

    _BEGIN_SCRIPT = """
    local owner = redis.call('HGET', KEYS[1], 'transition_owner') or ''
    local expires = tonumber(redis.call('HGET', KEYS[1], 'transition_expires_at') or '0')
    local now = tonumber(ARGV[2])
    if owner ~= '' and expires > now then
      return {0, owner}
    end
    local fence = redis.call('HINCRBY', KEYS[1], 'generation_fence', 1)
    local transition = redis.call('HINCRBY', KEYS[1], 'transition_id', 1)
    redis.call('HSET', KEYS[1],
      'schema_version', '1',
      'bundle_name', ARGV[3],
      'desired_state', ARGV[4],
      'phase', 'recovering',
      'transition_owner', ARGV[1],
      'transition_expires_at', ARGV[5],
      'last_reason', ARGV[6],
      'last_error', '',
      'updated_at', ARGV[2])
    redis.call('EXPIRE', KEYS[1], ARGV[7])
    return {1, tostring(fence), tostring(transition)}
    """

    _BEGIN_GROUP_SCRIPT = """
    local now = tonumber(ARGV[2])
    local owner = ARGV[1]
    local reason = ARGV[3]
    local expires = ARGV[4]
    local ttl = ARGV[5]
    for i, key in ipairs(KEYS) do
      local current_owner = redis.call('HGET', key, 'transition_owner') or ''
      local current_expires = tonumber(redis.call('HGET', key, 'transition_expires_at') or '0')
      if current_owner ~= '' and current_expires > now then
        return {0, i, current_owner}
      end
    end
    local result = {1}
    for i, key in ipairs(KEYS) do
      local base = 6 + ((i - 1) * 2)
      local bundle_name = ARGV[base]
      local desired_state = ARGV[base + 1]
      local fence = redis.call('HINCRBY', key, 'generation_fence', 1)
      local transition = redis.call('HINCRBY', key, 'transition_id', 1)
      redis.call('HSET', key,
        'schema_version', '1',
        'bundle_name', bundle_name,
        'desired_state', desired_state,
        'phase', 'recovering',
        'transition_owner', owner,
        'transition_expires_at', expires,
        'last_reason', reason,
        'last_error', '',
        'updated_at', ARGV[2])
      redis.call('EXPIRE', key, ttl)
      table.insert(result, tostring(fence))
      table.insert(result, tostring(transition))
    end
    return result
    """

    _FINISH_SCRIPT = """
    local owner = redis.call('HGET', KEYS[1], 'transition_owner') or ''
    local fence = tonumber(redis.call('HGET', KEYS[1], 'generation_fence') or '-1')
    local transition = tonumber(redis.call('HGET', KEYS[1], 'transition_id') or '-1')
    local expires = tonumber(redis.call('HGET', KEYS[1], 'transition_expires_at') or '0')
    if owner ~= ARGV[1] or fence ~= tonumber(ARGV[2]) or transition ~= tonumber(ARGV[3]) or expires <= tonumber(ARGV[7]) then
      return 0
    end
    local phase = ARGV[4]
    local error = ARGV[6]
    redis.call('HSET', KEYS[1],
      'phase', phase,
      'transition_owner', '',
      'transition_expires_at', '0',
      'last_reason', ARGV[5],
      'last_error', error,
      'updated_at', ARGV[7])
    if ARGV[4] == 'ready' then
      redis.call('HSET', KEYS[1], 'last_healthy_at', ARGV[7], 'last_restart_at', '')
      redis.call('HSET', KEYS[1], 'restart_attempts', '0', 'retry_window_started_at', '')
    end
    redis.call('EXPIRE', KEYS[1], ARGV[8])
    return 1
    """

    _FINISH_GROUP_SCRIPT = """
    local now = tonumber(ARGV[5])
    local owner = ARGV[1]
    for i, key in ipairs(KEYS) do
      local base = 7 + ((i - 1) * 2)
      local current_owner = redis.call('HGET', key, 'transition_owner') or ''
      local fence = tonumber(redis.call('HGET', key, 'generation_fence') or '-1')
      local transition = tonumber(redis.call('HGET', key, 'transition_id') or '-1')
      local expires = tonumber(redis.call('HGET', key, 'transition_expires_at') or '0')
      if current_owner ~= owner
          or fence ~= tonumber(ARGV[base])
          or transition ~= tonumber(ARGV[base + 1])
          or expires <= now then
        return 0
      end
    end
    local phase = ARGV[2]
    local error = ARGV[4]
    for i, key in ipairs(KEYS) do
      redis.call('HSET', key,
        'phase', phase,
        'transition_owner', '',
        'transition_expires_at', '0',
        'last_reason', ARGV[3],
        'last_error', error,
        'updated_at', ARGV[5])
      if phase == 'ready' then
        redis.call('HSET', key, 'last_healthy_at', ARGV[5], 'last_restart_at', '')
        redis.call('HSET', key, 'restart_attempts', '0', 'retry_window_started_at', '')
      end
      redis.call('EXPIRE', key, ARGV[6])
    end
    return 1
    """

    _RECORD_RECOVERY_SCRIPT = """
    local owner = redis.call('HGET', KEYS[1], 'transition_owner') or ''
    local expires = tonumber(redis.call('HGET', KEYS[1], 'transition_expires_at') or '0')
    if owner ~= '' and expires > tonumber(ARGV[10]) then
      return 0
    end
    redis.call('HSET', KEYS[1],
      'schema_version', '1',
      'bundle_name', ARGV[1],
      'desired_state', ARGV[2],
      'phase', ARGV[3],
      'transition_id', ARGV[4],
      'restart_attempts', ARGV[5],
      'retry_window_started_at', ARGV[6],
      'last_restart_at', ARGV[7],
      'last_healthy_at', ARGV[8],
      'last_reason', ARGV[9],
      'updated_at', ARGV[10])
    redis.call('EXPIRE', KEYS[1], ARGV[11])
    return 1
    """

    def __init__(self, redis_client: Any, *, prefix: str = "gpu:bundle:lifecycle:"):
        self.redis = redis_client
        self.prefix = prefix

    def _resource_key(self, resource_id: str) -> str:
        digest = hashlib.sha256(resource_id.encode("utf-8")).hexdigest()
        return f"{self.prefix}{self._RESOURCE_FENCE_SUFFIX}:{digest}"

    def _resource_fence_key(self) -> str:
        return f"{self.prefix}{self._RESOURCE_FENCE_SUFFIX}:counter"

    def _key(self, bundle_name: str) -> str:
        return f"{self.prefix}{bundle_name}"

    async def acquire_resources(
        self,
        resource_ids: tuple[str, ...],
        owner: str,
        now: float,
        ttl_seconds: float,
        reason: str,
    ) -> ResourceLease | None:
        del reason
        normalized = canonical_resource_ids(resource_ids)
        if not normalized:
            return None
        expires_at = now + max(0.1, float(ttl_seconds))
        keys = [self._resource_key(resource) for resource in normalized]
        keys.append(self._resource_fence_key())
        raw = await self.redis.eval(
            self._ACQUIRE_RESOURCES_SCRIPT,
            len(keys),
            *keys,
            owner,
            repr(float(now)),
            repr(float(expires_at)),
        )
        if not raw or int(raw[0]) != 1:
            return None
        return ResourceLease(
            normalized,
            owner,
            int(raw[1]),
            expires_at,
        )

    async def renew_resources(
        self,
        lease: ResourceLease,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        normalized = canonical_resource_ids(lease.resource_ids)
        if not normalized:
            return False
        expires_at = now + max(0.1, float(ttl_seconds))
        keys = [self._resource_key(resource) for resource in normalized]
        result = await self.redis.eval(
            self._RENEW_RESOURCES_SCRIPT,
            len(keys),
            *keys,
            lease.owner,
            str(lease.fence),
            repr(float(now)),
            repr(float(expires_at)),
        )
        return bool(int(result or 0))

    async def release_resources(self, lease: ResourceLease, now: float) -> bool:
        del now
        normalized = canonical_resource_ids(lease.resource_ids)
        if not normalized:
            return False
        keys = [self._resource_key(resource) for resource in normalized]
        result = await self.redis.eval(
            self._RELEASE_RESOURCES_SCRIPT,
            len(keys),
            *keys,
            lease.owner,
            str(lease.fence),
        )
        return bool(int(result or 0))

    async def get(self, bundle_name: str) -> BundleLifecycleRecord:
        fields = await self.redis.hgetall(self._key(bundle_name))
        return BundleLifecycleRecord.from_fields(bundle_name, fields or {})

    async def begin(self, bundle_name: str, desired_state: DesiredState, owner: str,
                    now: float, ttl_seconds: float, reason: str) -> TransitionAttempt:
        ttl = max(1, int(ttl_seconds) + 1)
        raw = await self.redis.eval(
            self._BEGIN_SCRIPT,
            1,
            self._key(bundle_name),
            owner,
            repr(float(now)),
            bundle_name,
            desired_state.value,
            repr(float(now + ttl_seconds)),
            reason[:500],
            str(ttl),
        )
        if not raw or int(raw[0]) != 1:
            return TransitionAttempt(False, reason="transition_in_progress", record=await self.get(bundle_name))
        record = await self.get(bundle_name)
        lease = BundleTransitionLease(
            bundle_name=bundle_name,
            owner=owner,
            generation_fence=int(raw[1]),
            transition_id=int(raw[2]),
            expires_at=record.transition_expires_at,
        )
        return TransitionAttempt(True, lease=lease, reason="acquired", record=record)

    async def begin_group(self, transitions: list[tuple[str, DesiredState]], owner: str,
                          now: float, ttl_seconds: float, reason: str) -> TransitionGroupAttempt:
        normalized = _normalize_group_transitions(transitions)
        if not normalized:
            return TransitionGroupAttempt(False, reason="empty_group")
        ttl = max(1, int(ttl_seconds) + 1)
        keys = [self._key(bundle_name) for bundle_name, _desired_state in normalized]
        args = [
            owner,
            repr(float(now)),
            reason[:500],
            repr(float(now + ttl_seconds)),
            str(ttl),
        ]
        for bundle_name, desired_state in normalized:
            args.extend([bundle_name, desired_state.value])
        raw = await self.redis.eval(
            self._BEGIN_GROUP_SCRIPT,
            len(keys),
            *keys,
            *args,
        )
        if not raw or int(raw[0]) != 1:
            return TransitionGroupAttempt(False, reason="transition_in_progress")
        leases = tuple(
            BundleTransitionLease(
                bundle_name=bundle_name,
                owner=owner,
                generation_fence=int(raw[1 + (index * 2)]),
                transition_id=int(raw[2 + (index * 2)]),
                expires_at=now + ttl_seconds,
            )
            for index, (bundle_name, _desired_state) in enumerate(normalized)
        )
        return TransitionGroupAttempt(
            True,
            lease=BundleTransitionGroupLease(leases),
            reason="acquired",
        )

    async def renew(
        self,
        lease: BundleTransitionLease,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        ttl = max(1, int(ttl_seconds) + 1)
        expires_at = now + max(0.1, float(ttl_seconds))
        result = await self.redis.eval(
            self._RENEW_SCRIPT,
            1,
            self._key(lease.bundle_name),
            lease.owner,
            str(lease.generation_fence),
            str(lease.transition_id),
            repr(float(expires_at)),
            repr(float(now)),
            str(ttl),
        )
        return bool(int(result or 0))

    async def renew_group(
        self,
        lease: BundleTransitionGroupLease,
        now: float,
        ttl_seconds: float,
    ) -> bool:
        if not lease.leases:
            return False
        ttl = max(1, int(ttl_seconds) + 1)
        expires_at = now + max(0.1, float(ttl_seconds))
        keys = [self._key(item.bundle_name) for item in lease.leases]
        args = [
            lease.leases[0].owner,
            repr(float(expires_at)),
            repr(float(now)),
            str(ttl),
        ]
        # The Lua script uses fixed positions for the common arguments and
        # then consumes one fence/transition pair per key.
        for item in lease.leases:
            args.extend([str(item.generation_fence), str(item.transition_id)])
        result = await self.redis.eval(
            self._RENEW_GROUP_SCRIPT,
            len(keys),
            *keys,
            *args,
        )
        return bool(int(result or 0))

    async def record_recovery(self, bundle_name: str, desired_state: DesiredState,
                              recovery: RecoveryState, now: float, reason: str) -> bool:
        def value(item: object) -> str:
            return "" if item is None else str(item)

        result = await self.redis.eval(
            self._RECORD_RECOVERY_SCRIPT,
            1,
            self._key(bundle_name),
            bundle_name,
            desired_state.value,
            recovery.phase.value,
            str(recovery.transition_id),
            str(recovery.restart_attempts),
            value(recovery.retry_window_started_at),
            value(recovery.last_restart_at),
            value(recovery.last_healthy_at),
            reason[:500],
            repr(float(now)),
            "3600",
        )
        return bool(int(result or 0))

    async def finish(self, lease: BundleTransitionLease, *, success: bool, now: float,
                     reason: str = "", error: str | None = None) -> bool:
        ttl = 3600
        result = await self.redis.eval(
            self._FINISH_SCRIPT,
            1,
            self._key(lease.bundle_name),
            lease.owner,
            str(lease.generation_fence),
            str(lease.transition_id),
            RecoveryPhase.READY.value if success else RecoveryPhase.CRASHED.value,
            reason[:500],
            (error or "")[:1000],
            repr(float(now)),
            str(ttl),
        )
        return bool(int(result or 0))

    async def finish_group(self, lease: BundleTransitionGroupLease, *, success: bool,
                           now: float, reason: str = "", error: str | None = None) -> bool:
        if not lease.leases:
            return False
        ttl = 3600
        keys = [self._key(item.bundle_name) for item in lease.leases]
        args = [
            lease.leases[0].owner,
            RecoveryPhase.READY.value if success else RecoveryPhase.CRASHED.value,
            reason[:500],
            (error or "")[:1000],
            repr(float(now)),
            str(ttl),
        ]
        for item in lease.leases:
            args.extend([str(item.generation_fence), str(item.transition_id)])
        result = await self.redis.eval(
            self._FINISH_GROUP_SCRIPT,
            len(keys),
            *keys,
            *args,
        )
        return bool(int(result or 0))


class BundleLifecycleController:
    """Single lifecycle owner used by all bundle-aware backends."""

    def __init__(self, store: BundleLifecycleStore, *, owner: str | None = None,
                 transition_ttl_seconds: float = 180.0,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.owner = owner or f"gpu-manager:{socket.gethostname()}:{uuid.uuid4().hex}"
        self.transition_ttl_seconds = transition_ttl_seconds
        self.clock = clock

    async def begin_transition(self, bundle_name: str, *, desired_state: DesiredState = DesiredState.RUNNING,
                               reason: str = "") -> TransitionAttempt:
        return await self.store.begin(
            bundle_name,
            desired_state,
            self.owner,
            self.clock(),
            self.transition_ttl_seconds,
            reason,
        )

    async def acquire_resources(
        self,
        resource_ids: tuple[str, ...],
        *,
        reason: str = "",
        ttl_seconds: float | None = None,
    ) -> ResourceLease | None:
        return await self.store.acquire_resources(
            canonical_resource_ids(resource_ids),
            self.owner,
            self.clock(),
            self.transition_ttl_seconds if ttl_seconds is None else ttl_seconds,
            reason,
        )

    async def renew_resources(
        self,
        lease: ResourceLease,
        *,
        ttl_seconds: float | None = None,
    ) -> bool:
        return await self.store.renew_resources(
            lease,
            self.clock(),
            self.transition_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )

    async def release_resources(self, lease: ResourceLease) -> bool:
        return await self.store.release_resources(lease, self.clock())

    async def begin_group_transition(
        self,
        transitions: list[tuple[str, DesiredState]],
        *,
        reason: str = "",
    ) -> TransitionGroupAttempt:
        return await self.store.begin_group(
            transitions,
            self.owner,
            self.clock(),
            self.transition_ttl_seconds,
            reason,
        )

    async def renew_transition(
        self,
        lease: BundleTransitionLease,
        *,
        ttl_seconds: float | None = None,
    ) -> bool:
        """Renew a live transition lease without changing its fence."""
        return await self.store.renew(
            lease,
            self.clock(),
            self.transition_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )

    async def renew_group_transition(
        self,
        lease: BundleTransitionGroupLease,
        *,
        ttl_seconds: float | None = None,
    ) -> bool:
        """Renew every member of a group transition atomically."""
        return await self.store.renew_group(
            lease,
            self.clock(),
            self.transition_ttl_seconds if ttl_seconds is None else ttl_seconds,
        )

    async def finish_transition(self, lease: BundleTransitionLease, *, success: bool,
                                reason: str = "", error: str | None = None) -> bool:
        return await self.store.finish(
            lease,
            success=success,
            now=self.clock(),
            reason=reason,
            error=error,
        )

    async def finish_group_transition(
        self,
        lease: BundleTransitionGroupLease,
        *,
        success: bool,
        reason: str = "",
        error: str | None = None,
    ) -> bool:
        return await self.store.finish_group(
            lease,
            success=success,
            now=self.clock(),
            reason=reason,
            error=error,
        )

    async def record_recovery_state(self, bundle_name: str, recovery: RecoveryState,
                                    *, desired_state: DesiredState = DesiredState.IDLE,
                                    reason: str = "") -> bool:
        return await self.store.record_recovery(
            bundle_name,
            desired_state,
            recovery,
            self.clock(),
            reason,
        )

    async def _lease_keepalive(
        self,
        *,
        transition_lease: BundleTransitionLease | None = None,
        group_lease: BundleTransitionGroupLease | None = None,
        resource_lease: ResourceLease | None = None,
        ttl_seconds: float,
        stop: asyncio.Event,
        lost: asyncio.Event,
    ) -> None:
        """Renew transition/resource fences while a bounded operation runs.

        The loop is deliberately conservative: any renewal error is treated as
        loss of ownership.  The caller then reports a stale transition rather
        than publishing readiness after another owner may have taken over.
        """
        interval = max(0.05, min(float(ttl_seconds) / 3.0, 5.0))
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                if transition_lease is not None:
                    renewed = await self.renew_transition(
                        transition_lease, ttl_seconds=ttl_seconds
                    )
                elif group_lease is not None:
                    renewed = await self.renew_group_transition(
                        group_lease, ttl_seconds=ttl_seconds
                    )
                else:
                    renewed = True
                if resource_lease is not None:
                    renewed = bool(
                        await self.renew_resources(
                            resource_lease, ttl_seconds=ttl_seconds
                        )
                    ) and renewed
            except Exception:
                logger = globals().get("logger")
                if logger is not None:
                    logger.error("bundle lease renewal failed", exc_info=True)
                renewed = False
            if not renewed:
                lost.set()
                return

    @asynccontextmanager
    async def keep_leases_alive(
        self,
        *,
        transition_lease: BundleTransitionLease | None = None,
        group_lease: BundleTransitionGroupLease | None = None,
        resource_lease: ResourceLease | None = None,
        ttl_seconds: float | None = None,
    ):
        """Keep a manually-owned transition/resource pair fenced.

        This is used by the legacy queue admission path while it is migrated
        to :meth:`run_transition`.  The yielded event is set if ownership is
        lost; callers must treat the operation result as uncommittable.
        """
        stop = asyncio.Event()
        lost = asyncio.Event()
        keepalive = asyncio.create_task(
            self._lease_keepalive(
                transition_lease=transition_lease,
                group_lease=group_lease,
                resource_lease=resource_lease,
                ttl_seconds=(
                    self.transition_ttl_seconds
                    if ttl_seconds is None
                    else ttl_seconds
                ),
                stop=stop,
                lost=lost,
            )
        )
        try:
            yield lost
        finally:
            stop.set()
            try:
                await keepalive
            except asyncio.CancelledError:
                pass

    async def run_transition(self, bundle_name: str, operation: Callable[[], Any], *,
                             desired_state: DesiredState = DesiredState.RUNNING,
                             reason: str = "",
                             resource_lease: ResourceLease | None = None,
                             ) -> tuple[bool, Any, str]:
        """Run one bundle mutation under the durable transition fence.

        Returns ``(acquired, result, reason)``. An in-progress transition is a
        normal admission result, not an exception; callers should leave work in
        the queue and retry after the owner publishes readiness.
        """
        attempt = await self.begin_transition(
            bundle_name,
            desired_state=desired_state,
            reason=reason,
        )
        if not attempt.acquired or attempt.lease is None:
            return False, None, attempt.reason or "transition_in_progress"
        stop = asyncio.Event()
        lost = asyncio.Event()
        keepalive = asyncio.create_task(
            self._lease_keepalive(
                transition_lease=attempt.lease,
                resource_lease=resource_lease,
                ttl_seconds=self.transition_ttl_seconds,
                stop=stop,
                lost=lost,
            )
        )
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            success = bool(result)
        except asyncio.CancelledError:
            # A cancelled owner must not leave a transition blocked until TTL
            # expiry.  Persist a fail-closed state before propagating the
            # cancellation; callers still own release of any resource lease
            # passed into this runner.
            try:
                await self.finish_transition(
                    attempt.lease,
                    success=False,
                    reason=reason,
                    error="bundle operation cancelled",
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            await self.finish_transition(
                attempt.lease,
                success=False,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            stop.set()
            try:
                await keepalive
            except asyncio.CancelledError:
                pass
        if lost.is_set():
            await self.finish_transition(
                attempt.lease,
                success=False,
                reason=reason,
                error="bundle lease renewal lost",
            )
            return True, result, "stale_transition"
        finished = await self.finish_transition(
            attempt.lease,
            success=success,
            reason=reason,
            error=None if success else "bundle operation returned false",
        )
        if not finished:
            return True, result, "stale_transition"
        return True, result, "completed" if success else "failed"

    async def run_group_transition(
        self,
        transitions: list[tuple[str, DesiredState]],
        operation: Callable[[], Any],
        *,
        reason: str = "",
        resource_lease: ResourceLease | None = None,
    ) -> tuple[bool, Any, str]:
        """Run one operation while atomically fencing every member bundle."""
        attempt = await self.begin_group_transition(transitions, reason=reason)
        if not attempt.acquired or attempt.lease is None:
            return False, None, attempt.reason or "transition_in_progress"
        stop = asyncio.Event()
        lost = asyncio.Event()
        keepalive = asyncio.create_task(
            self._lease_keepalive(
                group_lease=attempt.lease,
                resource_lease=resource_lease,
                ttl_seconds=self.transition_ttl_seconds,
                stop=stop,
                lost=lost,
            )
        )
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            success = bool(result)
        except asyncio.CancelledError:
            try:
                await self.finish_group_transition(
                    attempt.lease,
                    success=False,
                    reason=reason,
                    error="bundle transition group cancelled",
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            await self.finish_group_transition(
                attempt.lease,
                success=False,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            stop.set()
            try:
                await keepalive
            except asyncio.CancelledError:
                pass
        if lost.is_set():
            await self.finish_group_transition(
                attempt.lease,
                success=False,
                reason=reason,
                error="bundle lease renewal lost",
            )
            return True, result, "stale_transition"
        finished = await self.finish_group_transition(
            attempt.lease,
            success=success,
            reason=reason,
            error=None if success else "bundle transition group returned false",
        )
        if not finished:
            return True, result, "stale_transition"
        return True, result, "completed" if success else "failed"

    async def snapshot(self, bundle_name: str) -> BundleLifecycleRecord:
        return await self.store.get(bundle_name)


__all__ = [
    "BundleLifecycleController",
    "BundleLifecycleRecord",
    "BundleLifecycleStore",
    "BundleTransitionLease",
    "BundleTransitionGroupLease",
    "ResourceLease",
    "canonical_resource_ids",
    "DesiredState",
    "InMemoryBundleLifecycleStore",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryObservation",
    "RecoveryPhase",
    "RecoveryState",
    "RedisBundleLifecycleStore",
    "TransitionAttempt",
    "TransitionGroupAttempt",
    "reduce_recovery",
]
