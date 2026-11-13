"""Pure decision rules for durable-queue recovery, lease bounds and stale flags.

Nothing here performs I/O. The boot reconciler, the lease reapers and the
eviction watchdog in ``gpu-manager.py`` apply these rules, and the unit tests
prove them without Redis, a GPU or a running service.

Process-liveness rule
---------------------
A Redis PEL entry (or a slot) is *owned by this process* only if it was
delivered (acquired) after this process started. Anything older was held by a
previous incarnation whose workers no longer exist, so it is an orphan. This
lets the reconciler sweep repeatedly without ever requeueing work that a live
worker of this process is still executing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Collection, Mapping, Sequence

# ── In-flight lease vs. generation poll deadline ─────────────────────────────
# Every ComfyUI workflow poll loop clamps ``workflow_timeout_s`` into this
# range. The lease that protects an in-flight job (slot lease and pending-entry
# lease) must outlive the longest poll, otherwise a reaper can fail or requeue a
# job that is still legitimately running and cause a duplicate submission.
GENERATION_POLL_DEADLINE_DEFAULT_S = 900
GENERATION_POLL_DEADLINE_MIN_S = 60
GENERATION_POLL_DEADLINE_MAX_S = 7200
LEASE_SAFETY_MARGIN_S = 60

# ── Bounded restart requeues ─────────────────────────────────────────────────
# A job that is still orphaned after this many restart requeues is poison (it
# keeps taking the manager down or never completes). It is moved to the
# dead-letter stream instead of looping across restarts forever.
RESTART_REQUEUE_MAX = 4
# The pending-entry idle clock is the backoff timer: an orphaned entry becomes
# eligible only after it has been idle for the backoff of its next attempt. The
# first requeue keeps the historical 30 s guard.
RESTART_REQUEUE_BASE_BACKOFF_MS = 30_000
RESTART_REQUEUE_MAX_BACKOFF_MS = 300_000

# ── Reconcile sweep cadence ──────────────────────────────────────────────────
# Entries preserved by the first pass (still inside their backoff window) are
# revisited on this cadence for a bounded window after boot. Past the window the
# lease reapers own any remaining entry.
RECONCILE_SWEEP_INTERVAL_S = 30
RECONCILE_SWEEP_WINDOW_S = 900

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_truthy(value: Any) -> bool:
    """True for the string/boolean spellings Redis hashes use for flags."""
    return str(value).strip().lower() in _TRUTHY


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ── Lease bounds ─────────────────────────────────────────────────────────────

def generation_poll_deadline_s(metadata: Mapping[str, Any] | None) -> int:
    """Return the clamped poll deadline for one ComfyUI workflow job."""
    requested = _as_int(
        (metadata or {}).get("workflow_timeout_s"),
        GENERATION_POLL_DEADLINE_DEFAULT_S,
    )
    return max(
        GENERATION_POLL_DEADLINE_MIN_S,
        min(GENERATION_POLL_DEADLINE_MAX_S, requested),
    )


def lease_floor_s() -> int:
    """Smallest in-flight lease that safely outlives any generation poll."""
    return GENERATION_POLL_DEADLINE_MAX_S + LEASE_SAFETY_MARGIN_S


def bounded_lease_s(configured_s: int, *, polls_workflows: bool) -> int:
    """Raise a configured lease to the poll-deadline floor when it must hold."""
    configured = max(0, _as_int(configured_s))
    return max(configured, lease_floor_s()) if polls_workflows else configured


def service_polls_workflows(service_config: Mapping[str, Any] | None) -> bool:
    """True when the service runs ComfyUI workflow jobs that poll /history."""
    service_config = service_config or {}
    return (
        str(service_config.get("type") or "") == "generation_backend"
        and str(service_config.get("input_body_format") or "") == "comfyui_workflow"
    )


# ── Restart requeue bound + backoff ──────────────────────────────────────────

def restart_requeue_backoff_ms(requeues_so_far: int) -> int:
    """Idle time an orphaned entry needs before its next restart requeue."""
    attempts = max(0, _as_int(requeues_so_far))
    return min(
        RESTART_REQUEUE_MAX_BACKOFF_MS,
        RESTART_REQUEUE_BASE_BACKOFF_MS * (2 ** min(attempts, 8)),
    )


class PelAction(StrEnum):
    ACK_TERMINAL = "ack_terminal"
    ACK_CANCELLED = "ack_cancelled"
    RECOVER_FORWARDED = "recover_forwarded"
    PRESERVE_ACTIVE = "preserve_active"
    PRESERVE_BACKOFF = "preserve_backoff"
    REQUEUE = "requeue"
    DEAD_LETTER = "dead_letter"


@dataclass(frozen=True)
class PelDecision:
    action: PelAction
    reason: str
    wait_ms: int = 0


def classify_pel_entry(
    *,
    job: Mapping[str, Any] | None,
    idle_ms: int,
    uptime_ms: int,
    requeues_so_far: int,
    recovery_active: bool = False,
) -> PelDecision:
    """Decide what the reconciler does with one pending-entry lease.

    ``idle_ms`` is the PEL idle clock, ``uptime_ms`` this process's age and
    ``requeues_so_far`` the restart requeues already charged to the entry.
    ``recovery_active`` means a retained recovery task of this process already
    owns the job's upstream handle.
    """
    job = job or {}
    idle_ms = max(0, _as_int(idle_ms))
    uptime_ms = max(0, _as_int(uptime_ms))
    status = str(job.get("status") or "").lower()

    # Delivered after this process started: a live worker of this process owns
    # it. Never touch it; its lease is governed by the lease reapers.
    if idle_ms <= uptime_ms:
        return PelDecision(PelAction.PRESERVE_ACTIVE, "delivered_by_current_process")

    # A real terminal outcome is never rewritten, even if a cancel was requested
    # afterwards: the entry is simply released.
    if status in TERMINAL_JOB_STATUSES:
        return PelDecision(PelAction.ACK_TERMINAL, f"job_{status}")
    if is_truthy(job.get("cancellation_intent")):
        return PelDecision(PelAction.ACK_CANCELLED, "cancellation_intent")

    if recovery_active:
        return PelDecision(PelAction.PRESERVE_ACTIVE, "recovery_task_active")

    # A persisted upstream handle proves the worker crossed the side-effect
    # boundary: never requeue the original payload (duplicate submission).
    if str(job.get("comfyui_prompt_id") or "").strip():
        return PelDecision(PelAction.RECOVER_FORWARDED, "upstream_handle_persisted")

    need_ms = restart_requeue_backoff_ms(requeues_so_far)
    if idle_ms < need_ms:
        action = (
            PelAction.PRESERVE_ACTIVE
            if _as_int(requeues_so_far) == 0
            else PelAction.PRESERVE_BACKOFF
        )
        return PelDecision(action, "inside_backoff_window", wait_ms=need_ms - idle_ms)

    if _as_int(requeues_so_far) >= RESTART_REQUEUE_MAX:
        return PelDecision(PelAction.DEAD_LETTER, "restart_requeue_exhausted")
    return PelDecision(PelAction.REQUEUE, "orphaned_after_restart")


# ── Slot leases at boot ──────────────────────────────────────────────────────

class SlotAction(StrEnum):
    KEEP = "keep"
    RELEASE = "release"


@dataclass(frozen=True)
class SlotDecision:
    action: SlotAction
    reason: str


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def classify_boot_slot(
    *,
    slot: Mapping[str, Any],
    job: Mapping[str, Any] | None,
    process_started_at: datetime,
    recovery_job_ids: Collection[str] = (),
) -> SlotDecision:
    """Decide whether a busy slot lease still has a live owner.

    A slot acquired before this process started was held by a previous
    incarnation. It stays only if a retained recovery task of this process owns
    the job's upstream handle; otherwise it would block new work until the
    time-based reaper (forward_timeout + margin, up to hours) expired it.
    """
    if str(slot.get("status") or "") != "busy":
        return SlotDecision(SlotAction.KEEP, "not_busy")
    job_id = str(slot.get("job_id") or "")
    if not job_id:
        return SlotDecision(SlotAction.RELEASE, "busy_without_job")
    acquired = _parse_iso(slot.get("started_at"))
    if acquired is not None and acquired >= process_started_at:
        return SlotDecision(SlotAction.KEEP, "acquired_by_current_process")
    if job_id in recovery_job_ids:
        return SlotDecision(SlotAction.KEEP, "live_recovery_task")
    if job is None:
        return SlotDecision(SlotAction.RELEASE, "job_record_missing")
    status = str(job.get("status") or "").lower()
    if status in TERMINAL_JOB_STATUSES:
        return SlotDecision(SlotAction.RELEASE, f"job_{status}")
    return SlotDecision(SlotAction.RELEASE, "orphaned_after_restart")


# ── Stale evicting flag ──────────────────────────────────────────────────────

def evicting_stale_after_s(
    *,
    drain_timeout_s: float,
    restore_timeout_s: float,
    idle_restore_interval_s: float,
    floor_s: float,
) -> float:
    """Bound for an ownerless ``evicting`` flag, from observed operation deadlines.

    Once no generation owner remains, the idle loop must notice (one interval),
    settle (2 s) and restore within ``restore_timeout_s``; a drain that was
    still pending may take ``drain_timeout_s`` first. A bare 90 s (the restore
    timeout alone) would clear a restore that is legitimately using its whole
    deadline, so the bound is the longest observed cycle, never below the
    existing OOM stuck-eviction floor.
    """
    cycle = (
        float(drain_timeout_s)
        + float(restore_timeout_s)
        + float(idle_restore_interval_s)
        + 2.0
    )
    return max(float(floor_s), cycle)


@dataclass(frozen=True)
class EvictingDecision:
    stale: bool
    reason: str
    age_s: float = 0.0


def evicting_decision(
    *,
    evicting: bool,
    evicting_since: float,
    last_activity_at: float,
    now: float,
    worker_in_flight: int,
    pending_requests: int,
    loading: bool,
    draining: bool,
    transition_active: bool,
    generation_blockers: Sequence[str],
    stale_after_s: float,
) -> EvictingDecision:
    """Is this GPU's ``evicting`` flag stale (set, ownerless and old enough)?

    The flag is legitimately held for the whole duration of a generation job,
    so age alone is never enough: any live owner keeps it, and the clock only
    runs from the last sign of activity.
    """
    if not evicting:
        return EvictingDecision(False, "not_evicting")
    owners = [
        name
        for name, active in (
            ("worker_in_flight", _as_int(worker_in_flight) > 0),
            ("pending_requests", _as_int(pending_requests) > 0),
            ("loading", bool(loading)),
            ("draining", bool(draining)),
            ("bundle_transition", bool(transition_active)),
            ("generation_blockers", bool(generation_blockers)),
        )
        if active
    ]
    if owners:
        return EvictingDecision(False, "owner_live:" + ",".join(owners))
    if float(evicting_since) <= 0.0:
        return EvictingDecision(False, "unstamped")
    quiet_since = max(float(evicting_since), float(last_activity_at or 0.0))
    age = max(0.0, float(now) - quiet_since)
    if age >= float(stale_after_s):
        return EvictingDecision(True, f"no_owner_for_{age:.0f}s", age)
    return EvictingDecision(False, "within_bound", age)


# ── Queue reason ─────────────────────────────────────────────────────────────

class QueueReason(StrEnum):
    WAITING_FOR_WORKER = "waiting_for_worker"
    GPU_HELD_BY_GENERATION = "gpu_held_by_generation"
    REQUEUED_AFTER_RESTART = "requeued_after_restart"


def queue_reason(
    job: Mapping[str, Any] | None,
    *,
    gpu_transitioning: bool = False,
) -> str:
    """Why a queued job has not started yet, as a stable machine-readable code."""
    job = job or {}
    if is_truthy(job.get("reconciled_after_restart")):
        return QueueReason.REQUEUED_AFTER_RESTART.value
    if gpu_transitioning:
        return QueueReason.GPU_HELD_BY_GENERATION.value
    return QueueReason.WAITING_FOR_WORKER.value
