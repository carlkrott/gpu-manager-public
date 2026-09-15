"""Closed data contracts for the combined Gemma broker."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import IntEnum, StrEnum
import math
from typing import Any, ClassVar


class ContractError(ValueError):
    """A closed contract was violated."""


class MemberState(StrEnum):
    OFFLINE = "offline"
    JOINING = "joining"
    READY_ACCEPTING = "ready_accepting"
    READY_BUSY = "ready_busy"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    FAULTED = "faulted"


class JobState(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    ACCEPTED = "accepted"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


class AttemptState(StrEnum):
    RESERVED = "reserved"
    FORWARDING = "forwarding"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


class Priority(IntEnum):
    INTERACTIVE = 0
    NORMAL = 10
    BACKGROUND = 20


class ReasonCode(StrEnum):
    MEMBER_CONFIG_DISABLED = "MEMBER_CONFIG_DISABLED"
    MEMBER_IDLE_SERVICE_NOT_CONFIGURED = "MEMBER_IDLE_SERVICE_NOT_CONFIGURED"
    MEMBER_IDLE_SERVICE_NOT_EFFECTIVE = "MEMBER_IDLE_SERVICE_NOT_EFFECTIVE"
    MEMBER_CONFLICTING_BUNDLE = "MEMBER_CONFLICTING_BUNDLE"
    MEMBER_SYSTEMD_INACTIVE = "MEMBER_SYSTEMD_INACTIVE"
    MEMBER_HTTP_UNREADY = "MEMBER_HTTP_UNREADY"
    MEMBER_PORT_DARK = "MEMBER_PORT_DARK"
    MEMBER_PROBE_TIMEOUT = "MEMBER_PROBE_TIMEOUT"
    MEMBER_PROBE_DISCONNECTED = "MEMBER_PROBE_DISCONNECTED"
    MEMBER_UNHEALTHY_RESPONSE = "MEMBER_UNHEALTHY_RESPONSE"
    MEMBER_MODEL_NOT_RESIDENT = "MEMBER_MODEL_NOT_RESIDENT"
    MEMBER_DISPATCHER_UNREGISTERED = "MEMBER_DISPATCHER_UNREGISTERED"
    MEMBER_DRAINING = "MEMBER_DRAINING"
    MEMBER_UNHEALTHY = "MEMBER_UNHEALTHY"
    MEMBER_FAULTED = "MEMBER_FAULTED"
    MEMBER_NO_CAPACITY = "MEMBER_NO_CAPACITY"
    REQUEST_CONTEXT_INCOMPATIBLE = "REQUEST_CONTEXT_INCOMPATIBLE"
    REQUEST_CAPABILITY_INCOMPATIBLE = "REQUEST_CAPABILITY_INCOMPATIBLE"
    REDIS_STATE_UNKNOWN = "REDIS_STATE_UNKNOWN"
    TELEMETRY_STALE = "TELEMETRY_STALE"
    RESERVATION_STATE_STALE = "RESERVATION_STATE_STALE"
    PRE_ACCEPTANCE_REQUEUE = "PRE_ACCEPTANCE_REQUEUE"
    POST_ACCEPTANCE_OUTCOME_UNKNOWN = "POST_ACCEPTANCE_OUTCOME_UNKNOWN"
    CANCELLED_BEFORE_ACCEPTANCE = "CANCELLED_BEFORE_ACCEPTANCE"
    CANCELLED_AFTER_ACCEPTANCE = "cancelled_after_acceptance"
    CANCEL_UNSAFE_AFTER_ACCEPTANCE = "CANCEL_UNSAFE_AFTER_ACCEPTANCE"
    IDEMPOTENCY_KEY_REUSED = "IDEMPOTENCY_KEY_REUSED"
    MIGRATION_PEL_NOT_EMPTY = "MIGRATION_PEL_NOT_EMPTY"
    BASELINE_DRIFT = "BASELINE_DRIFT"
    NO_ELIGIBLE_MEMBER = "NO_ELIGIBLE_MEMBER"
    PRIORITY_INVALID = "PRIORITY_INVALID"
    STREAMING_NOT_SUPPORTED = "STREAMING_NOT_SUPPORTED"
    CRASH_EVIDENCE_CONFLICT = "CRASH_EVIDENCE_CONFLICT"


_ALLOWED_MEMBER_TRANSITIONS: dict[MemberState, frozenset[MemberState]] = {
    MemberState.OFFLINE: frozenset({MemberState.JOINING}),
    MemberState.JOINING: frozenset(
        {
            MemberState.READY_ACCEPTING,
            MemberState.DRAINING,
            MemberState.UNHEALTHY,
            MemberState.FAULTED,
            MemberState.OFFLINE,
        }
    ),
    MemberState.READY_ACCEPTING: frozenset(
        {
            MemberState.READY_BUSY,
            MemberState.DRAINING,
            MemberState.UNHEALTHY,
            MemberState.FAULTED,
        }
    ),
    MemberState.READY_BUSY: frozenset(
        {
            MemberState.READY_ACCEPTING,
            MemberState.DRAINING,
            MemberState.UNHEALTHY,
            MemberState.FAULTED,
        }
    ),
    MemberState.DRAINING: frozenset({MemberState.OFFLINE, MemberState.FAULTED}),
    MemberState.UNHEALTHY: frozenset(
        {MemberState.JOINING, MemberState.DRAINING, MemberState.OFFLINE, MemberState.FAULTED}
    ),
    MemberState.FAULTED: frozenset({MemberState.JOINING, MemberState.OFFLINE}),
}


def validate_member_transition(current: MemberState, target: MemberState) -> None:
    if target not in _ALLOWED_MEMBER_TRANSITIONS[current]:
        raise ContractError(f"ILLEGAL_MEMBER_TRANSITION:{current.value}->{target.value}")


def _reject_unknown(cls: type, data: dict[str, Any]) -> None:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ContractError(f"UNKNOWN_FIELDS:{cls.__name__}:{','.join(unknown)}")


@dataclass(frozen=True, slots=True)
class MemberSnapshot:
    name: str
    gpu_id: str | None
    state: MemberState
    accepting: bool
    configured_slots: int
    context_per_slot: int
    state_version: int
    observed_at: float
    generation_fence: int = 0
    backend_slots_total: int | None = None
    backend_slots_busy: int | None = None
    dispatcher_leases: int = 0
    drain_disabled_slots: int = 0
    compatible_free_slots: int | None = None
    capabilities: tuple[str, ...] = ()
    idle_service_configured: str | None = None
    idle_service_effective: bool | None = None
    systemd_active: bool = False
    main_pid: int | None = None
    semantic_health: bool = False
    model_resident: bool | None = None
    dispatcher_registered: bool = False
    snapshot_age_ms: int = 0
    blockers: tuple[str, ...] = ()
    reason_codes: tuple[ReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation_fence, bool)
            or not isinstance(self.generation_fence, int)
            or self.generation_fence < 0
        ):
            raise ContractError("MEMBER_GENERATION_FENCE_INVALID")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["capabilities"] = list(self.capabilities)
        data["blockers"] = list(self.blockers)
        data["reason_codes"] = [item.value for item in self.reason_codes]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemberSnapshot":
        _reject_unknown(cls, data)
        cooked = dict(data)
        cooked["state"] = MemberState(cooked["state"])
        for key in ("capabilities", "blockers"):
            cooked[key] = tuple(cooked.get(key, ()))
        cooked["reason_codes"] = tuple(ReasonCode(v) for v in cooked.get("reason_codes", ()))
        return cls(**cooked)


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Durable attempt identity, ownership fence, and acceptance boundary."""

    attempt_id: str
    job_id: str
    member_name: str
    reservation_fence: int
    backend_url: str
    state: AttemptState
    reserved_at: float
    forward_started_at: float | None = None
    accepted_at: float | None = None
    terminal_at: float | None = None
    accepted_boundary_crossed: bool = False
    error_category: ReasonCode | None = None
    transition_history: tuple[str, ...] = ("reserved",)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("attempt_id", "job_id", "member_name", "backend_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"ATTEMPT_{name.upper()}_INVALID")
        if (
            isinstance(self.reservation_fence, bool)
            or not isinstance(self.reservation_fence, int)
            or self.reservation_fence <= 0
        ):
            raise ContractError("ATTEMPT_RESERVATION_FENCE_INVALID")
        for name in ("reserved_at", "forward_started_at", "accepted_at", "terminal_at"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ContractError(f"ATTEMPT_{name.upper()}_INVALID")
        if not isinstance(self.accepted_boundary_crossed, bool):
            raise ContractError("ATTEMPT_ACCEPTED_BOUNDARY_INVALID")
        if not isinstance(self.transition_history, tuple) or any(
            not isinstance(item, str) or not item for item in self.transition_history
        ):
            raise ContractError("ATTEMPT_TRANSITION_HISTORY_INVALID")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["error_category"] = self.error_category.value if self.error_category else None
        data["transition_history"] = list(self.transition_history)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptRecord":
        _reject_unknown(cls, data)
        cooked = dict(data)
        cooked["state"] = AttemptState(cooked["state"])
        if cooked.get("error_category") is not None:
            cooked["error_category"] = ReasonCode(cooked["error_category"])
        cooked["transition_history"] = tuple(cooked.get("transition_history", ("reserved",)))
        return cls(**cooked)


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    idempotency_key: str
    request_sha256: str
    request_body: dict[str, Any]
    submitted_at: float
    enqueue_sequence: int
    base_priority: Priority = Priority.NORMAL
    effective_priority: int = int(Priority.NORMAL)
    state: JobState = JobState.QUEUED
    selected_member: str | None = None
    member_state_version: int | None = None
    attempt_count: int = 0
    current_attempt_id: str | None = None
    accepted_boundary_crossed: bool = False
    queue_claimed_at: float | None = None
    forward_started_at: float | None = None
    completed_at: float | None = None
    input_tokens_estimate: int | None = None
    max_output_tokens: int | None = None
    required_capabilities: tuple[str, ...] = ()
    result: dict[str, Any] | None = None
    error: str | None = None
    reason_code: ReasonCode | None = None
    cancel_requested: bool = False
    state_version: int = 1
    schema_version: int = 1

    @classmethod
    def new(cls, **kwargs: Any) -> "JobRecord":
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_priority"] = int(self.base_priority)
        data["state"] = self.state.value
        data["required_capabilities"] = list(self.required_capabilities)
        data["reason_code"] = self.reason_code.value if self.reason_code else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        _reject_unknown(cls, data)
        cooked = dict(data)
        cooked["base_priority"] = Priority(cooked.get("base_priority", int(Priority.NORMAL)))
        cooked["state"] = JobState(cooked.get("state", JobState.QUEUED.value))
        cooked["required_capabilities"] = tuple(cooked.get("required_capabilities", ()))
        if cooked.get("reason_code") is not None:
            cooked["reason_code"] = ReasonCode(cooked["reason_code"])
        return cls(**cooked)
