"""Engine-agnostic sequencing for checked runtime transitions.

The supervisor owns ordering and fencing only.  An adapter owns the actual
systemd/container/engine calls and must return a complete, profile-bound
observation.  This keeps ComfyUI, llama.cpp, audio.cpp, SGLang, vLLM and other
engines behind one admission contract without pretending their unload or model
residency APIs are interchangeable.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from runtime_state import (
    RuntimeControlState,
    RuntimeDesiredState,
    RuntimeInstanceRecord,
    RuntimeStateError,
    RuntimeStateStore,
    RuntimeTransitionLease,
)


class RuntimeSupervisorError(RuntimeStateError):
    """An adapter transition failed or returned unusable evidence."""


class RuntimeAdapter(Protocol):
    """Minimal adapter surface; implementations may wrap a host supervisor."""

    async def inspect(
        self, instance_id: str, profile: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    async def prepare(self, instance_id: str, profile: Mapping[str, Any]) -> None: ...

    async def load(self, instance_id: str, profile: Mapping[str, Any]) -> None: ...

    async def drain(self, instance_id: str, profile: Mapping[str, Any]) -> None: ...

    async def unload(self, instance_id: str, profile: Mapping[str, Any]) -> None: ...

    async def stop(self, instance_id: str, profile: Mapping[str, Any]) -> None: ...

    async def health(
        self, instance_id: str, profile: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    async def reconcile(
        self, instance_id: str, profile: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class TransitionPhase(StrEnum):
    INSPECT = "inspect"
    DRAIN = "drain"
    UNLOAD = "unload"
    STOP = "stop"
    PREPARE = "prepare"
    LOAD = "load"
    HEALTH = "health"
    RECONCILE = "reconcile"


async def _lease_keepalive(
    store: RuntimeStateStore,
    lease: RuntimeTransitionLease,
    *,
    ttl_seconds: float,
    stop: asyncio.Event,
    lost: asyncio.Event,
) -> None:
    """Renew a runtime fence while an adapter call may be in progress."""

    interval = max(0.05, min(float(ttl_seconds) / 3.0, 5.0))
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            renewed = await store.renew(lease, ttl_seconds=ttl_seconds)
        except Exception:
            renewed = None
        if not renewed:
            lost.set()
            return


def _ensure_lease_live(lost: asyncio.Event, phase: TransitionPhase) -> None:
    if lost.is_set():
        raise RuntimeSupervisorError(f"{phase.value} failed: runtime transition lease lost")


async def _renew_or_raise(
    store: RuntimeStateStore,
    lease: RuntimeTransitionLease,
    *,
    ttl_seconds: float,
    phase: TransitionPhase,
) -> None:
    renewed = await store.renew(lease, ttl_seconds=ttl_seconds)
    if not renewed:
        raise RuntimeSupervisorError(
            f"{phase.value} failed: runtime transition lease lost"
        )


def _observation_reached_desired(
    observation: Mapping[str, Any], desired: RuntimeDesiredState
) -> bool:
    if desired is RuntimeDesiredState.READY:
        return (
            observation.get("state") == RuntimeControlState.READY.value
            and observation.get("process_ready") is True
            and observation.get("model_ready") is True
            and observation.get("accepting") is True
        )
    return (
        observation.get("state") == RuntimeControlState.STOPPED.value
        and observation.get("process_ready") is False
        and observation.get("model_ready") is False
        and observation.get("accepting") is False
    )


async def _wait_for_desired_observation(
    adapter: RuntimeAdapter,
    *,
    instance_id: str,
    profile: Mapping[str, Any],
    desired: RuntimeDesiredState,
    lease_lost: asyncio.Event,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> Mapping[str, Any]:
    """Poll adapter evidence until the requested runtime state converges."""

    if timeout_seconds <= 0:
        raise RuntimeSupervisorError("readiness timeout must be positive")
    if poll_interval_seconds <= 0:
        raise RuntimeSupervisorError("readiness poll interval must be positive")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    last_state = "unknown"
    while True:
        evidence = await adapter.health(instance_id, profile)
        _ensure_lease_live(lease_lost, TransitionPhase.HEALTH)
        last_state = str(evidence.get("state") or "unknown")
        if _observation_reached_desired(evidence, desired):
            return evidence
        if last_state == "faulted":
            raise RuntimeSupervisorError("runtime reported a faulted state")

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RuntimeSupervisorError(
                f"runtime did not reach {desired.value!r} before readiness timeout "
                f"(last_state={last_state!r})"
            )
        try:
            await asyncio.wait_for(
                lease_lost.wait(),
                timeout=min(poll_interval_seconds, remaining),
            )
        except asyncio.TimeoutError:
            continue
        _ensure_lease_live(lease_lost, TransitionPhase.HEALTH)


async def transition_runtime(
    store: RuntimeStateStore,
    adapter: RuntimeAdapter,
    *,
    instance_id: str,
    profile_name: str,
    profile: Mapping[str, Any],
    owner: str,
    desired_state: RuntimeDesiredState | str = RuntimeDesiredState.READY,
    ttl_seconds: float = 180.0,
    readiness_timeout_seconds: float = 180.0,
    readiness_poll_interval_seconds: float = 1.0,
    now: float | None = None,
) -> RuntimeInstanceRecord:
    """Run one fenced transition and publish only adapter-backed evidence.

    An already-ready observation for the requested profile is accepted without
    a destructive reload.  Any other ready transition drains and unloads the
    owned instance before loading the requested profile.  Failures become a
    blocked record; a later caller must explicitly retry with a new fence.
    """
    record, lease = await store.begin_transition(
        instance_id,
        profile_name,
        profile,
        owner=owner,
        desired_state=desired_state,
        ttl_seconds=ttl_seconds,
        now=now,
    )
    desired = (
        desired_state
        if isinstance(desired_state, RuntimeDesiredState)
        else RuntimeDesiredState(desired_state)
    )
    phase = TransitionPhase.INSPECT
    keepalive_stop = asyncio.Event()
    lease_lost = asyncio.Event()
    keepalive = asyncio.create_task(
        _lease_keepalive(
            store,
            lease,
            ttl_seconds=ttl_seconds,
            stop=keepalive_stop,
            lost=lease_lost,
        )
    )
    try:
        bind_transition = getattr(adapter, "bind_transition", None)
        if callable(bind_transition):
            bind_transition(lease.owner, lease.fence)
        observed = await adapter.inspect(instance_id, profile)
        _ensure_lease_live(lease_lost, phase)
        observed_stopped = (
            observed.get("state") == RuntimeControlState.STOPPED.value
            and observed.get("process_ready") is False
            and observed.get("model_ready") is False
            and observed.get("accepting") is False
        )
        if (
            desired is RuntimeDesiredState.READY
            and observed.get("state") == RuntimeControlState.READY.value
            and observed.get("process_ready") is True
            and observed.get("model_ready") is True
            and observed.get("accepting") is True
        ):
            return await store.publish_observation(lease, profile, observed, now=now)
        if desired is RuntimeDesiredState.STOPPED and observed_stopped:
            return await store.publish_observation(lease, profile, observed, now=now)

        await _renew_or_raise(
            store, lease, ttl_seconds=ttl_seconds, phase=phase
        )
        _ensure_lease_live(lease_lost, phase)
        if not observed_stopped:
            phase = TransitionPhase.DRAIN
            await adapter.drain(instance_id, profile)
            _ensure_lease_live(lease_lost, phase)
            await _renew_or_raise(
                store, lease, ttl_seconds=ttl_seconds, phase=phase
            )
            phase = TransitionPhase.UNLOAD
            await adapter.unload(instance_id, profile)
            _ensure_lease_live(lease_lost, phase)
            await _renew_or_raise(
                store, lease, ttl_seconds=ttl_seconds, phase=phase
            )

        if desired is RuntimeDesiredState.STOPPED:
            phase = TransitionPhase.STOP
            await adapter.stop(instance_id, profile)
            _ensure_lease_live(lease_lost, phase)
        else:
            phase = TransitionPhase.PREPARE
            await adapter.prepare(instance_id, profile)
            _ensure_lease_live(lease_lost, phase)
            await _renew_or_raise(
                store, lease, ttl_seconds=ttl_seconds, phase=phase
            )
            phase = TransitionPhase.LOAD
            await adapter.load(instance_id, profile)
            _ensure_lease_live(lease_lost, phase)

        phase = TransitionPhase.HEALTH
        evidence = await _wait_for_desired_observation(
            adapter,
            instance_id=instance_id,
            profile=profile,
            desired=desired,
            lease_lost=lease_lost,
            timeout_seconds=float(readiness_timeout_seconds),
            poll_interval_seconds=float(readiness_poll_interval_seconds),
        )
        return await store.publish_observation(lease, profile, evidence, now=now)
    except asyncio.CancelledError:
        # Cancellation is a failed transition from the runtime's point of
        # view.  Fence it before re-raising so a cancelled task cannot leave
        # the instance looking available to a later owner.
        try:
            await adapter.reconcile(instance_id, profile)
        except Exception:
            pass
        try:
            await store.fail_transition(lease, error=f"{phase.value}: cancelled", now=now)
        except Exception:
            pass
        raise
    except Exception as exc:
        # Reconcile is diagnostic/recovery evidence only.  It must never turn a
        # failed transition into READY, and a failing reconcile must not hide the
        # original phase/error.
        try:
            await adapter.reconcile(instance_id, profile)
        except Exception:
            pass
        try:
            await store.fail_transition(
                lease, error=f"{phase.value}: {exc}", now=now
            )
        except Exception as fence_error:
            raise RuntimeSupervisorError(
                f"{phase.value} failed and transition fence was lost: {fence_error}"
            ) from exc
        raise RuntimeSupervisorError(f"{phase.value} failed: {exc}") from exc
    finally:
        keepalive_stop.set()
        try:
            await keepalive
        except asyncio.CancelledError:
            pass
        clear_transition = getattr(adapter, "clear_transition", None)
        if callable(clear_transition):
            clear_transition()


__all__ = [
    "RuntimeAdapter",
    "RuntimeSupervisorError",
    "TransitionPhase",
    "transition_runtime",
]
