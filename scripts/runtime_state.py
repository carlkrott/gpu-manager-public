"""Checked runtime-instance state for GPU Manager adapters.

The controller can use this boundary to distinguish a configured service from
the process/model generation that is actually observed.  It intentionally does
not launch processes, inspect VRAM, or call systemd.  Engine adapters publish
observations only while holding the fenced transition lease returned by the
store.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
import json
import os
from pathlib import Path
import time
from collections.abc import Mapping
from typing import Any

from runtime_contracts import (
    runtime_profile_fingerprint,
    validate_runtime_observation,
    validate_runtime_profile,
)


class RuntimeStateError(ValueError):
    """A runtime state transition violates the checked control contract."""


class RuntimeConflict(RuntimeStateError):
    """Another owner currently holds the runtime transition fence."""


class RuntimeControlState(StrEnum):
    UNKNOWN = "unknown"
    STOPPED = "stopped"
    TRANSITIONING = "transitioning"
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class RuntimeDesiredState(StrEnum):
    STOPPED = "stopped"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class RuntimeTransitionLease:
    instance_id: str
    owner: str
    fence: int
    expires_at: float


@dataclass(frozen=True, slots=True)
class RuntimeInstanceRecord:
    instance_id: str
    profile_name: str
    profile_fingerprint: str
    desired_state: RuntimeDesiredState = RuntimeDesiredState.STOPPED
    state: RuntimeControlState = RuntimeControlState.UNKNOWN
    process_generation: int = 0
    process_ready: bool = False
    model_ready: bool = False
    accepting: bool = False
    transition_owner: str | None = None
    transition_fence: int = 0
    transition_expires_at: float = 0.0
    observed_at: float = 0.0
    last_error: str | None = None


def _now(value: float | None) -> float:
    return time.time() if value is None else float(value)


def _desired(value: RuntimeDesiredState | str) -> RuntimeDesiredState:
    try:
        return value if isinstance(value, RuntimeDesiredState) else RuntimeDesiredState(value)
    except ValueError as exc:
        raise RuntimeStateError(f"unsupported desired runtime state: {value!r}") from exc


def _state_for_observation(observation: Mapping[str, Any]) -> RuntimeControlState:
    state = observation.get("state")
    if state == "ready":
        return RuntimeControlState.READY
    if state == "stopped":
        return RuntimeControlState.STOPPED
    if state in {"loading", "draining"}:
        return RuntimeControlState.TRANSITIONING
    if state == "faulted":
        return RuntimeControlState.BLOCKED
    return RuntimeControlState.UNKNOWN


def _require_owner(
    record: RuntimeInstanceRecord,
    lease: RuntimeTransitionLease,
    now: float,
) -> None:
    if lease.instance_id != record.instance_id:
        raise RuntimeStateError("lease instance does not match runtime record")
    if (
        record.transition_owner != lease.owner
        or record.transition_fence != lease.fence
        or not record.transition_owner
        or record.transition_expires_at <= now
    ):
        raise RuntimeConflict("runtime transition lease is stale or expired")


class RuntimeStateStore:
    """Atomic in-memory runtime state store used by adapters and fixtures."""

    def __init__(self, *, clock=time.time):
        self._records: dict[str, RuntimeInstanceRecord] = {}
        self._lock = asyncio.Lock()
        self._clock = clock

    async def register(
        self,
        instance_id: str,
        profile_name: str,
        profile: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> RuntimeInstanceRecord:
        errors = validate_runtime_profile(profile_name, profile)
        if errors:
            raise RuntimeStateError("invalid runtime profile: " + "; ".join(errors))
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise RuntimeStateError("instance_id must be non-empty")
        fingerprint = runtime_profile_fingerprint(profile)
        observed_at = _now(now if now is not None else self._clock())
        async with self._lock:
            current = self._records.get(instance_id)
            if current is not None:
                if current.transition_owner and current.transition_expires_at > observed_at:
                    raise RuntimeConflict("cannot replace a runtime during a live transition")
                if current.state is RuntimeControlState.READY and (
                    current.profile_name != profile_name
                    or current.profile_fingerprint != fingerprint
                ):
                    raise RuntimeConflict("cannot replace an admitted runtime profile")
                record = replace(
                    current,
                    profile_name=profile_name,
                    profile_fingerprint=fingerprint,
                    state=(
                        current.state
                        if current.profile_name == profile_name
                        and current.profile_fingerprint == fingerprint
                        else RuntimeControlState.UNKNOWN
                    ),
                    process_ready=(
                        current.process_ready
                        if current.profile_name == profile_name
                        and current.profile_fingerprint == fingerprint
                        else False
                    ),
                    model_ready=(
                        current.model_ready
                        if current.profile_name == profile_name
                        and current.profile_fingerprint == fingerprint
                        else False
                    ),
                    accepting=(
                        current.accepting
                        if current.profile_name == profile_name
                        and current.profile_fingerprint == fingerprint
                        else False
                    ),
                    observed_at=observed_at,
                )
            else:
                record = RuntimeInstanceRecord(
                    instance_id=instance_id,
                    profile_name=profile_name,
                    profile_fingerprint=fingerprint,
                    observed_at=observed_at,
                )
            self._records[instance_id] = record
            return record

    async def begin_transition(
        self,
        instance_id: str,
        profile_name: str,
        profile: Mapping[str, Any],
        *,
        owner: str,
        desired_state: RuntimeDesiredState | str = RuntimeDesiredState.READY,
        ttl_seconds: float = 180.0,
        now: float | None = None,
    ) -> tuple[RuntimeInstanceRecord, RuntimeTransitionLease]:
        errors = validate_runtime_profile(profile_name, profile)
        if errors:
            raise RuntimeStateError("invalid runtime profile: " + "; ".join(errors))
        if not isinstance(owner, str) or not owner.strip():
            raise RuntimeStateError("transition owner must be non-empty")
        desired = _desired(desired_state)
        current_now = _now(now if now is not None else self._clock())
        fingerprint = runtime_profile_fingerprint(profile)
        async with self._lock:
            current = self._records.get(instance_id)
            if current is None:
                current = RuntimeInstanceRecord(
                    instance_id=instance_id,
                    profile_name=profile_name,
                    profile_fingerprint=fingerprint,
                    observed_at=current_now,
                )
            if current.transition_owner and current.transition_expires_at > current_now:
                raise RuntimeConflict("runtime transition already owned")
            fence = current.transition_fence + 1
            record = replace(
                current,
                profile_name=profile_name,
                profile_fingerprint=fingerprint,
                desired_state=desired,
                state=RuntimeControlState.TRANSITIONING,
                process_ready=False,
                model_ready=False,
                accepting=False,
                transition_owner=owner,
                transition_fence=fence,
                transition_expires_at=current_now + max(0.1, float(ttl_seconds)),
                last_error=None,
                observed_at=current_now,
            )
            self._records[instance_id] = record
            return record, RuntimeTransitionLease(
                instance_id=instance_id,
                owner=owner,
                fence=fence,
                expires_at=record.transition_expires_at,
            )

    async def renew(
        self,
        lease: RuntimeTransitionLease,
        *,
        ttl_seconds: float = 180.0,
        now: float | None = None,
    ) -> RuntimeTransitionLease:
        current_now = _now(now if now is not None else self._clock())
        async with self._lock:
            record = self._records.get(lease.instance_id)
            if record is None:
                raise RuntimeConflict("runtime instance is missing")
            _require_owner(record, lease, current_now)
            expires_at = current_now + max(0.1, float(ttl_seconds))
            self._records[lease.instance_id] = replace(
                record, transition_expires_at=expires_at, observed_at=current_now
            )
            return replace(lease, expires_at=expires_at)

    async def fail_transition(
        self,
        lease: RuntimeTransitionLease,
        *,
        error: str,
        now: float | None = None,
    ) -> RuntimeInstanceRecord:
        """Fence a failed transition and fail closed until a new retry owns it."""
        current_now = _now(now if now is not None else self._clock())
        async with self._lock:
            record = self._records.get(lease.instance_id)
            if record is None:
                raise RuntimeConflict("runtime instance is missing")
            _require_owner(record, lease, current_now)
            failed = replace(
                record,
                state=RuntimeControlState.BLOCKED,
                process_ready=False,
                model_ready=False,
                accepting=False,
                transition_owner=None,
                transition_expires_at=0.0,
                observed_at=current_now,
                last_error=str(error)[:1000] or "runtime transition failed",
            )
            self._records[lease.instance_id] = failed
            return failed

    async def publish_observation(
        self,
        lease: RuntimeTransitionLease,
        profile: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> RuntimeInstanceRecord:
        current_now = _now(now if now is not None else self._clock())
        profile_name = observation.get("profile_name")
        errors = validate_runtime_observation(str(profile_name), profile, observation)
        if errors:
            raise RuntimeStateError("invalid runtime observation: " + "; ".join(errors))
        async with self._lock:
            record = self._records.get(lease.instance_id)
            if record is None:
                raise RuntimeStateError("runtime instance is missing")
            _require_owner(record, lease, current_now)
            if record.profile_name != profile_name or record.profile_fingerprint != runtime_profile_fingerprint(profile):
                raise RuntimeStateError("observation profile does not match transition profile")
            generation = int(observation["process_generation"])
            if generation < record.process_generation:
                # A conclusively stopped unit has no PID/start-time generation
                # to report.  The host adapter therefore falls back to its
                # transition fence, which is a different (and usually lower)
                # numeric domain.  Stopping cannot resurrect an older process:
                # retain the last observed generation so the state transition
                # remains monotonic.  Ready/loading observations still fail
                # closed on a genuinely stale process generation.
                if observation.get("state") == "stopped":
                    generation = record.process_generation
                else:
                    raise RuntimeConflict("runtime observation is from an older process generation")
            state = _state_for_observation(observation)
            if state is RuntimeControlState.READY and record.desired_state is not RuntimeDesiredState.READY:
                raise RuntimeStateError(
                    "ready observation contradicts desired stopped state"
                )
            ready = state is RuntimeControlState.READY
            record = replace(
                record,
                state=state,
                process_generation=generation,
                process_ready=bool(observation["process_ready"]),
                model_ready=bool(observation["model_ready"]),
                accepting=bool(observation["accepting"]),
                transition_owner=None if state in {
                    RuntimeControlState.READY,
                    RuntimeControlState.STOPPED,
                } else record.transition_owner,
                transition_expires_at=0.0 if state in {
                    RuntimeControlState.READY,
                    RuntimeControlState.STOPPED,
                } else record.transition_expires_at,
                observed_at=current_now,
                last_error=None if state is not RuntimeControlState.BLOCKED else "adapter reported faulted",
            )
            if ready and not (record.process_ready and record.model_ready and record.accepting):
                raise RuntimeStateError("ready observation is not accepting")
            self._records[lease.instance_id] = record
            return record

    async def snapshot(self, instance_id: str) -> RuntimeInstanceRecord:
        async with self._lock:
            try:
                return self._records[instance_id]
            except KeyError as exc:
                raise RuntimeStateError(f"unknown runtime instance {instance_id!r}") from exc

    async def snapshots(self) -> tuple[RuntimeInstanceRecord, ...]:
        """Return a stable, read-only view of every tracked runtime."""
        async with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    async def admission_errors(
        self,
        instance_id: str,
        profile_name: str,
        profile: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> list[str]:
        errors = validate_runtime_profile(profile_name, profile)
        current_now = _now(now if now is not None else self._clock())
        async with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                return ["runtime instance is not registered"] + errors
            if record.profile_name != profile_name:
                errors.append("runtime profile name does not match instance")
            if record.profile_fingerprint != runtime_profile_fingerprint(profile):
                errors.append("runtime profile fingerprint does not match instance")
            if record.state is not RuntimeControlState.READY:
                errors.append(f"runtime state is {record.state.value}")
            if record.desired_state is not RuntimeDesiredState.READY:
                errors.append(f"runtime desired state is {record.desired_state.value}")
            if not record.process_ready:
                errors.append("process is not ready")
            if not record.model_ready:
                errors.append("model is not ready")
            if not record.accepting:
                errors.append("runtime is not accepting")
            if record.process_generation < 1:
                errors.append("process generation is unknown")
            if record.transition_owner and record.transition_expires_at > current_now:
                errors.append("runtime transition is active")
            return errors


class FileRuntimeStateStore(RuntimeStateStore):
    """Single-controller durable runtime fences and observations.

    The host supervisor keeps its own fence ledger. Persisting the controller's
    next fence prevents a routine controller restart from resetting it to one
    and being rejected as stale by the host.
    """

    SCHEMA_VERSION = "runtime-state-store.v1"

    def __init__(self, path: str | os.PathLike[str], *, clock=time.time):
        super().__init__(clock=clock)
        self.path = Path(path)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(document, Mapping):
                raise ValueError("state document must be an object")
            if document.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("unsupported schema version")
            raw_records = document.get("records")
            if not isinstance(raw_records, Mapping):
                raise ValueError("records must be an object")
            expected_fields = {field.name for field in fields(RuntimeInstanceRecord)}
            loaded: dict[str, RuntimeInstanceRecord] = {}
            for instance_id, raw in raw_records.items():
                if not isinstance(raw, Mapping) or set(raw) != expected_fields:
                    raise ValueError(f"invalid record shape for {instance_id!r}")
                values = dict(raw)
                values["desired_state"] = RuntimeDesiredState(values["desired_state"])
                values["state"] = RuntimeControlState(values["state"])
                record = RuntimeInstanceRecord(**values)
                if record.instance_id != instance_id or record.transition_fence < 0:
                    raise ValueError(f"invalid record identity for {instance_id!r}")
                loaded[str(instance_id)] = record
            self._records = loaded
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeStateError(
                f"cannot load durable runtime state from {self.path}: {exc}"
            ) from exc

    async def _persist(self) -> None:
        async with self._lock:
            document = {
                "schema_version": self.SCHEMA_VERSION,
                "records": {
                    instance_id: asdict(record)
                    for instance_id, record in sorted(self._records.items())
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            encoded = json.dumps(
                document, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            try:
                with temp_path.open("wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
                directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    async def register(self, *args, **kwargs):
        result = await super().register(*args, **kwargs)
        await self._persist()
        return result

    async def begin_transition(self, *args, **kwargs):
        result = await super().begin_transition(*args, **kwargs)
        await self._persist()
        return result

    async def renew(self, *args, **kwargs):
        result = await super().renew(*args, **kwargs)
        await self._persist()
        return result

    async def fail_transition(self, *args, **kwargs):
        result = await super().fail_transition(*args, **kwargs)
        await self._persist()
        return result

    async def publish_observation(self, *args, **kwargs):
        result = await super().publish_observation(*args, **kwargs)
        await self._persist()
        return result


__all__ = [
    "FileRuntimeStateStore",
    "RuntimeConflict",
    "RuntimeControlState",
    "RuntimeDesiredState",
    "RuntimeInstanceRecord",
    "RuntimeStateError",
    "RuntimeStateStore",
    "RuntimeTransitionLease",
]
