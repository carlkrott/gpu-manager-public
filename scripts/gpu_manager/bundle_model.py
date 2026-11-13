"""Facade-bound GPU state and bundle coordinator model classes.

Class templates live here; each facade bind creates isolated classes whose
methods resolve collaborators from that facade's globals at call time.
``GpuState`` is an Enum and is rebuilt through ``EnumType.__prepare__``.
"""
from __future__ import annotations

import enum
import types
from enum import Enum


NAMES = ("GpuState", "GpuBundleState", "BundleCoordinator")


class GpuState(Enum):
    """GPU state machine states.

    LLM_LOADED    → Qwen3.6 on GPU, serving requests (default idle state)
    DRAINING      → Waiting for active LLM requests to finish before evicting
    GPU_WORK      → ComfyUI or ACE-Step using GPU
    LLM_RELIEF    → Temporary LLM restore to clear queued request backlog
    RESTORING     → Qwen3.6 reloading to GPU after GPU work complete
    """
    LLM_LOADED = "llm_loaded"
    DRAINING = "draining"
    GPU_WORK = "gpu_work"
    LLM_RELIEF = "llm_relief"
    RESTORING = "restoring"


class GpuBundleState:
    """Per-GPU bundle state tracker.

    Tracks what bundle is loaded on this GPU, its priority, whether it's
    draining, and the priority queue of pending requests.
    """

    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id
        self.loaded_bundle: str | None = None  # Currently loaded bundle name
        self.bundle_priority: int = 0          # Priority of the loaded bundle
        self.draining: bool = False            # True when draining pending requests
        self.drain_started_at: float | None = None  # When drain began
        self._drain_event = asyncio.Event()    # Signals drain completion
        self.pending_requests: int = 0         # Count of in-flight requests
        self.state: str = "idle"               # idle | loaded | draining | evicting | loading
        # Priority queue: list of (priority, timestamp, bundle_name, request_data) tuples
        # Sorted by -priority (highest first), then +timestamp (FIFO within same priority)
        self.priority_queue: list[dict] = []
        # ── Phase 0: fields formerly in scheduler._gpu_state[gpu_id] ────
        # GpuBundleState is now the SINGLE SOURCE OF TRUTH for these. Setters
        # below mirror changes back to scheduler._gpu_state for /health.
        self.loaded_bundles: list[str] = []    # Mirror of scheduler._gpu_state["loaded_bundles"]
        self.evicting: bool = False            # Mirror of scheduler._gpu_state["evicting"]
        # time.time() when `evicting` last turned True (0.0 while False). The
        # stale-flag watchdog measures ownerless age from here; additive and
        # defaulted on load, so it needs no schema-version bump.
        self.evicting_since: float = 0.0
        self.current_priority: int = 0         # Mirror of scheduler._gpu_state["current_priority"]
        self.loading: bool = False             # Mirror of scheduler._gpu_state["loading"]
        # Audit fields (setters populate)
        self.last_change_ts: float | None = None
        self.last_event: str | None = None
        self.events: list[dict] = []
        # ── Phase 4g: idle-bundle auto-eviction activity timestamp ─────
        self.last_activity_at: float = 0.0
        # Phase 5.2.36: count of workers currently executing a job in
        # _execute_job for this bundle. _auto_evict_idle_bundles skips
        # evictions when this is > 0, preventing the auto-evict from
        # killing comfyui mid-inference (Bug C race).
        self.worker_in_flight: int = 0
        # ── Phase 5.2.4: post-load grace deadline (used by _auto_evict) ─
        # Set in _load_bundle after port-ready check. Auto-evict skips
        # bundles where now < this value, so backends that need warmup
        # (e.g. ComfyUI's ComfyRegistry fetch) are not freed mid-grace.
        self.bundle_post_load_grace_at: float = 0.0
        # ── Phase 4h.3 S5: backoff for repeated unload failures ────────
        self.last_evict_attempt_at: float = 0.0
        # ── Phase 5.1.5 P0-1: bundle-load grace (auto-evict skip) ────
        # Set in _load_bundle after port-ready check; auto-evict skips
        # bundles within _BUNDLE_LOAD_GRACE_SECONDS of being loaded.
        self.bundle_loaded_at: float = 0.0
        # ── Phase 1: OOM Recovery State Machine fields ────────────────
        # See OOM_RECOVERY_STATE_MACHINE_DESIGN.md §A.2 for the per-GPU
        # oom sub-state machine. All OOM state lives here, no parallel
        # dict (Phase 0 unified the state owner). Values are written
        # through set_oom_state() / record_oom_event() to keep the audit
        # ring buffer consistent.
        self.oom_event_count: int = 0                  # Lifetime OOM events on this GPU
        self.oom_recovery_count: int = 0               # Successful recoveries
        self.oom_permanent_failure_count: int = 0      # Permanent failures
        self.oom_state: str = "idle"                   # idle | detected | draining | verifying | recovering | restored | oom_failed | permanent_failed
        self.oom_state_since: float = 0.0              # time.time() when entered current state
        self.oom_last_event_ts: float = 0.0            # time.time() of last OOM event
        self.oom_last_reason: str = ""                 # Free-form reason string
        self.oom_active_recovery_job_id: str = ""      # Job being recovered, or ""
        self.oom_retry_attempted: bool = False         # True if an auto-retry was already issued
        self.oom_recent_failure_times: deque = deque(maxlen=10)  # timestamps for permanent-failure window
        # ── Phase 5: OOM-implicated service tracker ────────────────────
        # Set by WorkerPool._worker_loop right before _handle_oom_event so
        # _resolve_service_for_gpu can fall back to it when the GPU has
        # no loaded bundles (typical race: OOM kills the worker before the
        # bundle is registered as loaded).
        self.oom_last_service_name: str | None = None
        # ── Phase 7: oom_state transition timestamp ────────────────────
        # Mirror of oom_state_since; named differently so the supervisor's
        # TTL gate has its own field (clearer intent, future-proof if
        # oom_state_since semantics drift). Updated by set_oom_state().
        self.oom_state_changed_at: float = 0.0
        # State schema version — bumped whenever GpuBundleState adds required
        # fields. from_dict() reads this and runs migrations as needed.
        self.state_version: int = 4

    def _record_event(self, label: str, **fields) -> None:
        """Record a state change. Keeps a bounded 50-entry ring buffer."""
        self.last_change_ts = time.time()
        self.last_event = label
        evt = {"ts": self.last_change_ts, "event": label, **fields}
        self.events.append(evt)
        if len(self.events) > 50:
            self.events = self.events[-50:]

    def set_evicting(self, value: bool, source: str = "unknown") -> None:
        """Set evicting flag atomically; records the change for audit.

        `source` identifies the caller (e.g. "evict_llm", "restore_llm",
        "_load_bundle") so divergence debugging has a trail.
        """
        old = self.evicting
        self.evicting = bool(value)
        self._record_event(
            f"evicting={self.evicting}",
            old=old, new=self.evicting, source=source,
        )
        if old != self.evicting:
            logger.info(
                f"[gpu-bundle:{self.gpu_id}] evicting: {old} -> {self.evicting} "
                f"(source={source})"
            )
            self.evicting_since = time.time() if self.evicting else 0.0
            if self.evicting:
                self.state = "evicting"
            elif self.state == "evicting":
                self.state = "idle" if not self.loaded_bundles else "loaded"
        _sync_legacy_gpu_state(self.gpu_id, self)

    def set_draining(self, value: bool, source: str = "unknown") -> None:
        """Set draining flag atomically; syncs to legacy dict."""
        old = self.draining
        self.draining = bool(value)
        self._record_event(
            f"draining={self.draining}",
            old=old, new=self.draining, source=source,
        )
        if old != self.draining:
            logger.debug(
                f"[gpu-bundle:{self.gpu_id}] draining: {old} -> {self.draining} "
                f"(source={source})"
            )
        _sync_legacy_gpu_state(self.gpu_id, self)

    def set_loaded_bundles(self, bundles, source: str = "unknown") -> None:
        """Set loaded_bundles list atomically; syncs to legacy dict."""
        old = list(self.loaded_bundles)
        self.loaded_bundles = list(bundles) if bundles else []
        self._record_event(
            f"loaded_bundles={self.loaded_bundles}",
            old=old, new=self.loaded_bundles, source=source,
        )
        if old != self.loaded_bundles:
            logger.debug(
                f"[gpu-bundle:{self.gpu_id}] loaded_bundles: {old} -> {self.loaded_bundles} "
                f"(source={source})"
            )
        _sync_legacy_gpu_state(self.gpu_id, self)

    def set_current_priority(self, value: int, source: str = "unknown") -> None:
        """Set current_priority atomically; syncs to legacy dict."""
        old = self.current_priority
        self.current_priority = int(value)
        self._record_event(
            f"current_priority={self.current_priority}",
            old=old, new=self.current_priority, source=source,
        )
        _sync_legacy_gpu_state(self.gpu_id, self)

    def set_loading(self, value: bool, source: str = "unknown") -> None:
        """Set loading flag atomically; syncs to legacy dict."""
        old = self.loading
        self.loading = bool(value)
        self._record_event(
            f"loading={self.loading}",
            old=old, new=self.loading, source=source,
        )
        _sync_legacy_gpu_state(self.gpu_id, self)

    def touch_activity(self, source: str = "unknown") -> None:
        """Stamp last_activity_at for idle-bundle auto-eviction decisions.

        Deliberately does NOT call _record_event: at 10 jobs/s this floods
        the audit ring (was 99% of entries). The timestamp itself is the
        signal — see Phase 4h.3 S3.

        Call sites: _load_bundle start (M3), _load_bundle success,
        WorkerPool._execute_job completion (S4).
        """
        self.last_activity_at = time.time()

    def clear_eviction(self, source: str) -> None:
        """Atomically clear evicting + loading + draining after a load/restore.

        Used at every successful bundle load/restore path so evicting can
        never get stuck True (the original Phase 0 bug).
        """
        self.set_evicting(False, source=source)
        self.set_loading(False, source=source)
        self.set_draining(False, source=source)

    # ── Phase 1: OOM Recovery state setters ────────────────────────────
    # The OOM sub-state machine lives entirely on GpuBundleState. Allowed
    # values for oom_state: idle | detected | draining | verifying |
    # recovering | restored | oom_failed | permanent_failed. Setters validate and
    # record the change in the audit ring buffer.
    _VALID_OOM_STATES = frozenset({
        "idle", "detected", "draining", "verifying",
        "recovering", "restored", "oom_failed", "permanent_failed",
    })

    def set_oom_state(self, value: str, source: str = "unknown", reason: str = "") -> None:
        """Set oom_state atomically; records the change for audit.

        Unknown states are coerced to "idle" so a bad transition cannot
        wedge the state machine (worst case: we lose the bad state and
        treat it as no-recovery-needed).
        """
        if value not in self._VALID_OOM_STATES:
            logger.warning(
                f"[gpu-bundle:{self.gpu_id}] set_oom_state: unknown state "
                f"'{value}' (source={source}); coercing to 'idle'"
            )
            value = "idle"
        old = self.oom_state
        self.oom_state = value
        self.oom_state_since = time.time()
        # Phase 7: also stamp the dedicated supervisor-readable field.
        # Only stamp on actual transitions (old != new) — repeats of
        # the same state shouldn't reset the supervisor's TTL clock.
        if old != value:
            self.oom_state_changed_at = self.oom_state_since
        if reason:
            self.oom_last_reason = reason
        self._record_event(
            f"oom_state={self.oom_state}",
            old=old, new=self.oom_state, source=source, reason=reason,
        )
        if old != self.oom_state:
            logger.info(
                f"[gpu-bundle:{self.gpu_id}] oom_state: {old} -> {self.oom_state} "
                f"(source={source}, reason={reason!r})"
            )

    def record_oom_event(self, reason: str, job_id: str = "") -> None:
        """Record an OOM event on this GPU. Bumps counters and timestamps.

        Does NOT change oom_state — the caller decides whether this is a
        new detection (→ "detected"), a recovery-in-progress marker, etc.
        The only side effect on oom_state is: if we're in a "happy" state
        (idle or restored) we leave it alone; if the caller wants to move
        to "detected" they should call set_oom_state("detected", ...) right
        after.
        """
        self.oom_event_count += 1
        self.oom_last_event_ts = time.time()
        if reason:
            self.oom_last_reason = reason
        if job_id:
            self.oom_active_recovery_job_id = job_id
        # Track the failure in the bounded window for permanent-failure detection
        self.oom_recent_failure_times.append(time.time())
        self._record_event(
            "oom_event",
            reason=reason, job_id=job_id,
            count=self.oom_event_count,
        )
        logger.info(
            f"[gpu-bundle:{self.gpu_id}] OOM event recorded "
            f"(count={self.oom_event_count}, reason={reason!r}, job_id={job_id!r})"
        )

    def enqueue(self, bundle_name: str, priority: int, request_data: dict) -> None:
        """Add a request to this GPU's priority queue, sorted by priority."""
        import bisect
        entry = {
            "bundle_name": bundle_name,
            "priority": priority,
            "enqueued_at": time.time(),
            "request_data": request_data,
        }
        # Insert in priority order: higher priority first, FIFO within same priority
        # Sort key: (-priority, +timestamp)
        self.priority_queue.append(entry)
        self.priority_queue.sort(key=lambda e: (-e["priority"], e["enqueued_at"]))
        logger.debug(
            f"GPU[{self.gpu_id}]: enqueued {bundle_name} priority={priority} "
            f"(queue depth: {len(self.priority_queue)})"
        )

    def peek_next(self) -> dict | None:
        """Look at the highest-priority item without removing it."""
        if self.priority_queue:
            return self.priority_queue[0]
        return None

    def dequeue_next(self) -> dict | None:
        """Remove and return the highest-priority item."""
        if self.priority_queue:
            entry = self.priority_queue.pop(0)
            logger.debug(
                f"GPU[{self.gpu_id}]: dequeued {entry['bundle_name']} "
                f"priority={entry['priority']}"
            )
            return entry
        return None

    def start_drain(self) -> None:
        """Mark GPU as draining — stop accepting new requests for current bundle."""
        self.draining = True
        self.drain_started_at = time.time()
        self._drain_event.clear()
        self.state = "draining"
        logger.info(f"GPU[{self.gpu_id}]: drain started (pending={self.pending_requests})")

    async def wait_drain(self, timeout: float = 60) -> bool:
        """Wait for all pending requests to complete, or timeout.

        Returns True if drained successfully, False on timeout.
        """
        if self.pending_requests == 0:
            logger.info(f"GPU[{self.gpu_id}]: no pending requests — drain immediate")
            self.draining = False
            self.drain_started_at = None
            self._drain_event.set()
            return True

        logger.info(
            f"GPU[{self.gpu_id}]: waiting for {self.pending_requests} "
            f"pending request(s) to complete (timeout={timeout}s)"
        )
        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=timeout)
            logger.info(f"GPU[{self.gpu_id}]: drain complete")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"GPU[{self.gpu_id}]: drain timeout after {timeout}s "
                f"({self.pending_requests} still pending)"
            )
            return False

    def request_started(self) -> None:
        """Called when a request begins processing on this GPU."""
        self.pending_requests += 1

    def request_finished(self) -> None:
        """Called when a request finishes processing on this GPU."""
        self.pending_requests = max(0, self.pending_requests - 1)
        if self.pending_requests == 0 and self.draining:
            self._drain_event.set()

    def to_dict(self) -> dict:
        """Serialize for state persistence."""
        return {
            "gpu_id": self.gpu_id,
            "loaded_bundle": self.loaded_bundle,
            "bundle_priority": self.bundle_priority,
            "draining": self.draining,
            "drain_started_at": self.drain_started_at,
            "pending_requests": self.pending_requests,
            "state": self.state,
            "queue_depth": len(self.priority_queue),
            "queue_items": [
                {"bundle": e["bundle_name"], "priority": e["priority"],
                 "enqueued_at": e["enqueued_at"]}
                for e in self.priority_queue
            ],
            # ── Phase 0: unified fields ──────────────────────────────
            "loaded_bundles": list(self.loaded_bundles),
            "evicting": self.evicting,
            "evicting_since": self.evicting_since,
            "current_priority": self.current_priority,
            "loading": self.loading,
            # ── Phase 4g: idle-bundle auto-eviction activity timestamp ───
            "last_activity_at": self.last_activity_at,
            # ── Phase 5.2.36: in-flight worker counter (state_version bump → 5) ─
            "worker_in_flight": self.worker_in_flight,
            # ── Phase 1: OOM fields (state_version bump → 2) ──────────
            "state_version": BUNDLE_STATE_SCHEMA_VERSION,
            "oom_event_count": self.oom_event_count,
            "oom_recovery_count": self.oom_recovery_count,
            "oom_permanent_failure_count": self.oom_permanent_failure_count,
            "oom_state": self.oom_state,
            "oom_state_since": self.oom_state_since,
            "oom_last_event_ts": self.oom_last_event_ts,
            "oom_last_reason": self.oom_last_reason,
            "oom_active_recovery_job_id": self.oom_active_recovery_job_id,
            "oom_retry_attempted": self.oom_retry_attempted,
            "oom_recent_failure_times": list(self.oom_recent_failure_times),
            # ── Phase 5/7: OOM-implicated service + state-change timestamp ──
            # state_version bump → 3. oom_last_service_name is set by the
            # WorkerPool right before _handle_oom_event so _resolve_service_for_gpu
            # can recover the offending service even when the GPU has no
            # loaded_bundle recorded yet. oom_state_changed_at mirrors
            # oom_state_since and is read by the supervisor's TTL gate.
            "oom_last_service_name": self.oom_last_service_name,
            "oom_state_changed_at": self.oom_state_changed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GpuBundleState":
        """Deserialize from persisted state.

        Reads ``state_version`` and runs migration when an older shape is
        detected.

        Migration history:
          v1 (Phase 0 baseline, no OOM fields): OOM fields stay at __init__
              defaults; logged once per GPU at load.
          v2 (Phase 1, full OOM state machine): same as v3 except no
              ``oom_last_service_name`` / ``oom_state_changed_at`` keys.
              Initialized to defaults on load.
          v3 (Phase 5/7, OOM-implicated service tracker + state-change
              timestamp): the current shape written by to_dict().

        Newer-than-current versions are accepted as-is and logged.
        """
        gpu = cls(data.get("gpu_id", "unknown"))
        gpu.loaded_bundle = data.get("loaded_bundle")
        gpu.bundle_priority = data.get("bundle_priority", 0)
        gpu.draining = data.get("draining", False)
        gpu.drain_started_at = data.get("drain_started_at")
        gpu.pending_requests = data.get("pending_requests", 0)
        gpu.state = data.get("state", "idle")
        # ── Phase 0: backward-compat defaults for new fields ────────
        gpu.loaded_bundles = list(data.get("loaded_bundles") or [])
        gpu.evicting = bool(data.get("evicting", False))
        gpu.evicting_since = float(data.get("evicting_since", 0.0) or 0.0)
        gpu.current_priority = int(data.get("current_priority", 0))
        gpu.loading = bool(data.get("loading", False))
        gpu.last_activity_at = float(data.get("last_activity_at", 0.0))
        gpu.worker_in_flight = int(data.get("worker_in_flight", 0) or 0)
        # ── State-shape migration gate ────────────────────────────
        # v1 → v2: OOM fields added. v2 → v3: oom_last_service_name +
        # oom_state_changed_at added. v3 → v4: last_activity_at added.
        # Newer-than-4 is accepted as-is and logged.
        saved_version = int(data.get("state_version", 1))
        if saved_version < 2:
            logger.info(
                f"[gpu-bundle:{gpu.gpu_id}] migrating state.json v{saved_version} → v2 "
                f"(OOM fields initialized to defaults)"
            )
        if saved_version < 3:
            # v3 adds oom_last_service_name + oom_state_changed_at. v2 files
            # never had them; default them so downstream consumers (the
            # supervisor's TTL gate, _resolve_service_for_gpu) see sane values.
            logger.info(
                f"[gpu-bundle:{gpu.gpu_id}] migrating state.json v{saved_version} → v3 "
                f"(oom_last_service_name + oom_state_changed_at initialized to defaults)"
            )
        if saved_version < 4:
            # v4 adds last_activity_at. v3 files never had it; default to 0.0
            # so auto-evict treats the bundle as needing an activity stamp.
            logger.info(
                f"[gpu-bundle:{gpu.gpu_id}] migrating state.json v{saved_version} → v4 "
                f"(last_activity_at initialized to 0.0)"
            )
        if saved_version < BUNDLE_STATE_SCHEMA_VERSION:
            # v5 adds worker_in_flight. v4 files never had it; default to 0
            # so auto-evict treats the bundle as not having an in-flight
            # worker (the safe behaviour — eviction proceeds normally).
            logger.info(
                f"[gpu-bundle:{gpu.gpu_id}] migrating state.json v{saved_version} → v5 "
                f"(worker_in_flight initialized to 0)"
            )
        elif saved_version > BUNDLE_STATE_SCHEMA_VERSION:
            # Plan 04 §4: this nested per-GPU bundle state schema is v5. An
            # old binary on a newer file must say "up to v5", not "up to v4"
            # (that copy-paste bug is what the plan flagged as a confirmed
            # contradiction against the top-level manager state which is
            # also a v-series schema with its own version).
            logger.warning(
                f"[gpu-bundle:{gpu.gpu_id}] state.json is v{saved_version}, this binary "
                f"understands up to v{BUNDLE_STATE_SCHEMA_VERSION} — proceeding with best-effort load"
            )
        gpu.state_version = max(saved_version, BUNDLE_STATE_SCHEMA_VERSION)
        gpu.oom_event_count = int(data.get("oom_event_count", 0))
        gpu.oom_recovery_count = int(data.get("oom_recovery_count", 0))
        gpu.oom_permanent_failure_count = int(data.get("oom_permanent_failure_count", 0))
        gpu.oom_state = data.get("oom_state", "idle")
        # Validate: bad persisted state can't wedge us
        if gpu.oom_state not in cls._VALID_OOM_STATES:
            logger.warning(
                f"[gpu-bundle:{gpu.gpu_id}] persisted oom_state={gpu.oom_state!r} "
                f"is not in valid set; coercing to 'idle'"
            )
            gpu.oom_state = "idle"
        gpu.oom_state_since = float(data.get("oom_state_since", 0.0))
        gpu.oom_last_event_ts = float(data.get("oom_last_event_ts", 0.0))
        gpu.oom_last_reason = data.get("oom_last_reason", "")
        gpu.oom_active_recovery_job_id = data.get("oom_active_recovery_job_id", "")
        gpu.oom_retry_attempted = bool(data.get("oom_retry_attempted", False))
        # The deque maxlen is set in __init__; .extend() will respect it.
        saved_failures = data.get("oom_recent_failure_times") or []
        gpu.oom_recent_failure_times.extend(float(t) for t in saved_failures)
        # ── Phase 5/7: OOM-implicated service + state-change timestamp ──
        # Use .get() with explicit defaults so a v2 file (no key at all)
        # initializes cleanly. None / 0.0 are the documented defaults.
        gpu.oom_last_service_name = data.get("oom_last_service_name", None)
        gpu.oom_state_changed_at = float(data.get("oom_state_changed_at", 0.0))
        # Note: queue items are not restored — they're ephemeral
        return gpu


class BundleCoordinator:
    """Tracks per-bundle-job GPU readiness for cross-GPU coordination.

    When a bundle spans multiple GPUs (e.g. sulphur-diffusion on gpu_primary
    linked to sulphur-encoder on gpu_secondary), all GPUs must load their components
    before the bundle can start.  This coordinator tracks which GPUs have
    finished loading and resolves the shared Future when all are ready.
    """

    def __init__(self):
        self._active_jobs: dict[str, dict] = {}  # bundle_job_id -> info
        self._lock = asyncio.Lock()

    async def register_job(
        self,
        bundle_job_id: str,
        bundle_name: str,
        gpu_ids: list[str],
        priority: int,
        request_data: dict,
    ) -> asyncio.Future:
        """Register a new bundle job spanning multiple GPUs.

        Returns a Future that resolves when ALL GPUs are ready.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._active_jobs[bundle_job_id] = {
                "bundle_name": bundle_name,
                "gpu_ids": set(gpu_ids),
                "ready_gpus": set(),
                "priority": priority,
                "request_data": request_data,
                "future": future,
                "created_at": time.time(),
            }
            logger.info(
                f"BundleCoordinator: registered job {bundle_job_id} "
                f"for '{bundle_name}' across GPUs {gpu_ids}"
            )
            return future

    async def mark_gpu_ready(self, bundle_job_id: str, gpu_id: str) -> bool:
        """Mark a GPU as ready for a bundle job.

        Returns True if ALL GPUs are now ready (bundle can start).
        """
        async with self._lock:
            job = self._active_jobs.get(bundle_job_id)
            if not job:
                logger.warning(
                    f"BundleCoordinator: mark_gpu_ready for unknown job {bundle_job_id}"
                )
                return False
            job["ready_gpus"].add(gpu_id)
            all_ready = job["ready_gpus"] >= job["gpu_ids"]
            logger.info(
                f"BundleCoordinator: job {bundle_job_id} GPU {gpu_id} ready "
                f"({len(job['ready_gpus'])}/{len(job['gpu_ids'])}) — all_ready={all_ready}"
            )
            return all_ready

    async def complete_job(self, bundle_job_id: str, success: bool = True) -> None:
        """Resolve the Future and clean up the job."""
        async with self._lock:
            job = self._active_jobs.pop(bundle_job_id, None)
            if job and not job["future"].done():
                if success:
                    job["future"].set_result(True)
                else:
                    job["future"].set_exception(
                        RuntimeError(f"Bundle job {bundle_job_id} failed")
                    )
            logger.info(
                f"BundleCoordinator: job {bundle_job_id} completed (success={success})"
            )

    def is_gpu_ready(self, bundle_job_id: str, gpu_id: str) -> bool:
        """Check if a specific GPU is ready for a bundle job."""
        job = self._active_jobs.get(bundle_job_id)
        if not job:
            return False
        return gpu_id in job["ready_gpus"]

    def get_all_ready(self, bundle_job_id: str) -> set:
        """Return set of ready GPUs for a bundle job."""
        job = self._active_jobs.get(bundle_job_id)
        if not job:
            return set()
        return job["ready_gpus"].copy()

    def get_job_info(self, bundle_job_id: str) -> dict | None:
        """Get full job info for a bundle job."""
        job = self._active_jobs.get(bundle_job_id)
        if not job:
            return None
        return {
            "bundle_name": job["bundle_name"],
            "gpu_ids": list(job["gpu_ids"]),
            "ready_gpus": list(job["ready_gpus"]),
            "priority": job["priority"],
            "created_at": job["created_at"],
        }

    def cleanup_stale(self, max_age: float = 600.0) -> int:
        """Remove jobs older than max_age seconds. Returns count removed."""
        now = time.time()
        stale = [
            bid for bid, job in self._active_jobs.items()
            if (now - job["created_at"]) > max_age
        ]
        for bid in stale:
            job = self._active_jobs.pop(bid, None)
            if job and not job["future"].done():
                job["future"].set_exception(
                    RuntimeError(f"Bundle job {bid} timed out after {max_age}s")
                )
            logger.warning(f"BundleCoordinator: cleaned up stale job {bid}")
        return len(stale)


class _Probe(Enum):
    MEMBER = 1


# Entries EnumType synthesises on every enum class. The set differs between
# Python versions (3.14 adds _hashable_values_ and others), so it is measured
# from a probe enum instead of listed by hand. The class docstring and the
# compiler-provided entries are copied, and the default _generate_next_value_
# is handled separately because an enum may override it.
_ENUM_INTERNAL = frozenset(
    set(vars(_Probe))
    - {"MEMBER", "__module__", "__doc__", "__firstlineno__", "__static_attributes__",
       "_generate_next_value_"}
)


def bind(facade_globals):
    """Build the three classes against one facade namespace."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    module_name = facade_globals.get("__name__", __name__)

    def bind_class(template):
        class_cell = types.CellType()

        def rebuild(function, *, drain_default=False):
            closure = function.__closure__
            if closure and "__class__" in function.__code__.co_freevars:
                closure = tuple(
                    class_cell if name == "__class__" else cell
                    for name, cell in zip(function.__code__.co_freevars, closure)
                )
            defaults = function.__defaults__
            if drain_default:
                defaults = (facade_globals["DRAIN_TIMEOUT"],)
            rebuilt = types.FunctionType(
                function.__code__, facade_globals, function.__name__, defaults, closure
            )
            rebuilt.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
            rebuilt.__annotations__ = function.__annotations__.copy()
            rebuilt.__doc__ = function.__doc__
            rebuilt.__qualname__ = function.__qualname__
            rebuilt.__module__ = module_name
            rebuilt.__dict__.update(function.__dict__)
            return rebuilt

        def convert(key, value):
            if isinstance(value, types.FunctionType):
                return rebuild(value, drain_default=(
                    template.__name__ == "GpuBundleState" and key == "wait_drain"
                ))
            if isinstance(value, (staticmethod, classmethod)):
                return type(value)(rebuild(value.__func__))
            if type(value) is property:
                return property(
                    rebuild(value.fget) if value.fget else None,
                    rebuild(value.fset) if value.fset else None,
                    rebuild(value.fdel) if value.fdel else None,
                    value.__doc__,
                )
            return value

        meta = type(template)
        if isinstance(template, enum.EnumMeta):
            # EnumType needs the _EnumDict it prepares itself; a plain dict fails.
            namespace = meta.__prepare__(template.__name__, template.__bases__)
            members = template._member_map_
            default_next = enum.Enum._generate_next_value_.__code__
            for key, value in vars(template).items():
                if key in _ENUM_INTERNAL or key in members:
                    continue
                # Python 3.11 stores the default as a function, 3.14 as a staticmethod.
                if key == "_generate_next_value_" and getattr(
                    value, "__func__", value
                ).__code__ is default_next:
                    continue
                namespace[key] = convert(key, value)
            namespace["__qualname__"] = template.__qualname__
            namespace["__module__"] = module_name
            # Members last, in definition order; an alias re-states its canonical value.
            for name, member in members.items():
                namespace[name] = member._value_
        else:
            namespace = {}
            for key, value in vars(template).items():
                if key in ("__dict__", "__weakref__"):
                    continue
                namespace[key] = convert(key, value)
            namespace["__module__"] = module_name
        bound = meta(template.__name__, template.__bases__, namespace)
        class_cell.cell_contents = bound
        return bound

    return tuple(bind_class(globals()[name]) for name in NAMES)
