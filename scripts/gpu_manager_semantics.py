"""Canonical GPU Manager runtime semantics reducer.

Pure functions consume configuration intent + observed runtime facts and
produce a consistent ``runtime_semantics`` envelope.  Every health/state
surface in gpu-manager (/health, /v1/health/all, /vram, websocket full
state, persisted state.json, MCP get_health) must surface the SAME
fields so dashboards don't need decoders per endpoint.

The seven runtime modes:

  * ``cpu_only``             — no GPU service is configured or all GPU
                               services are disabled; only CPU LLM/QC
                               route is available.
  * ``gpu_idle_llm``         — the configured idle GPU service is loaded
                               and healthy; no generation work in flight.
  * ``generation_loaded``    — at least one generation bundle is loaded.
  * ``transitioning``        — a load/drain/restore/evict transition is in
                               progress (draining/loading/restoring/evicting).
  * ``maintenance``          — maintenance_mode is True.
  * ``degraded``             — observed reality contradicts configured intent
                               in a way that cannot be safely reconciled.
  * ``unknown``              — observation is empty/stale and intent cannot
                               be classified.

Output envelope (always present):

  {
    "runtime_mode":            str,
    "configured_intent":       { ... },     # what we want loaded
    "observed_runtime":        { ... },     # what is actually loaded
    "route_availability":      { ... },     # what we can serve right now
    "blocked_reasons":         [str, ...],
    "contradictions":          [str, ...],
    "intentionally_idle":      [str, ...],  # service names configured off
    "observed_at":             float,       # unix ts of last observation
  }

This module MUST NOT call into gpu-manager globals — only receives a
``Snapshot`` built by the caller.  All live state lookup happens in
gpu-manager.py; this module only reduces it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ── Schema namespace (Plan 04 §4) ────────────────────────────────────────────
# Top-level manager state ("state.json") is v4.  Nested per-GPU bundle state
# (GpuBundleState inside ``gpu_states``) is v5.  The two namespaces evolve
# independently — never bump one to "fix" the other.  Consumers (dashboard,
# MCP, harness scripts) must read the version they actually care about.
MANAGER_STATE_SCHEMA_VERSION = 4
BUNDLE_STATE_SCHEMA_VERSION = 5
RESPONSE_SCHEMA_VERSION = 1  # HTTP response envelope for /health, /v1/health/all, etc.
PROMPT_EVIDENCE_SCHEMA_VERSION = 1  # Prompt evidence rows (separate namespace)

# Schema pair assertion — fails loudly if someone bumps one without the other.
assert BUNDLE_STATE_SCHEMA_VERSION > MANAGER_STATE_SCHEMA_VERSION, (
    "Plan 04 §4: nested per-GPU bundle state must remain a higher version "
    "than the top-level manager state. If you're adding fields to the top "
    "level, bump MANAGER_STATE_SCHEMA_VERSION; if you're adding fields to "
    "GpuBundleState, bump BUNDLE_STATE_SCHEMA_VERSION. Never edit one to "
    "match the other — they are different schemas."
)

# ── Runtime mode constants ────────────────────────────────────────────────
RUNTIME_MODE_CPU_ONLY = "cpu_only"
RUNTIME_MODE_GPU_IDLE_LLM = "gpu_idle_llm"
RUNTIME_MODE_GENERATION_LOADED = "generation_loaded"
RUNTIME_MODE_TRANSITIONING = "transitioning"
RUNTIME_MODE_MAINTENANCE = "maintenance"
RUNTIME_MODE_DEGRADED = "degraded"
RUNTIME_MODE_UNKNOWN = "unknown"

ALL_RUNTIME_MODES = frozenset({
    RUNTIME_MODE_CPU_ONLY,
    RUNTIME_MODE_GPU_IDLE_LLM,
    RUNTIME_MODE_GENERATION_LOADED,
    RUNTIME_MODE_TRANSITIONING,
    RUNTIME_MODE_MAINTENANCE,
    RUNTIME_MODE_DEGRADED,
    RUNTIME_MODE_UNKNOWN,
})


# ── Transition states ─────────────────────────────────────────────────────
# Names that indicate a load/drain/restore/evict transition is in flight.
# gpu-manager uses different shapes (scheduler state strings + VRAM state
# enum values).  We accept both so the reducer is reusable.
TRANSITION_STATES = frozenset({
    "draining",
    # ``gpu_work`` is the stable generation-residency state, not a transition.
    "llm_relief",
    "loading",
    "restoring",
    "evicting",
    "loading_bundle",
    "draining_bundle",
    "restoring_idle",
})


# ── Data shape ────────────────────────────────────────────────────────────
@dataclass
class SemanticSnapshot:
    """All facts the reducer needs to derive the runtime envelope.

    The caller (gpu-manager handlers / websocket / _save_state) builds
    this from the live state and passes it to ``reduce_semantics``.  No
    live I/O happens here.
    """

    # Configuration intent ─────────────────────────────────────────────
    configured_idle_service: str = ""
    configured_idle_services_by_gpu: dict[str, str] = field(default_factory=dict)
    pinned_service: str = ""
    maintenance_mode: bool = False
    enabled_service_names: list[str] = field(default_factory=list)
    disabled_service_names: list[str] = field(default_factory=list)

    # Observed runtime ─────────────────────────────────────────────────
    loaded_runtime_services: list[str] = field(default_factory=list)
    loaded_runtime_bundles: list[dict] = field(default_factory=list)
    gpu_state_value: str = ""
    llm_evicted: bool = False
    queue_active_jobs: int = 0
    queue_pending_jobs: int = 0
    any_bundle_draining: bool = False
    any_bundle_loading: bool = False
    any_bundle_restoring: bool = False
    any_bundle_evicting: bool = False
    restore_operation_active: bool = False

    # Service health (per service, optional) ────────────────────────────
    service_health: dict[str, dict] = field(default_factory=dict)

    # Route availability (caller-supplied, e.g. _llm_readiness_snapshot) ─
    llm_route_available: bool = False
    cpu_fallback_available: bool = False
    generation_route_available: bool = False

    # Staleness / freshness ────────────────────────────────────────────
    observed_at: float = field(default_factory=time.time)


# ── Reducer ───────────────────────────────────────────────────────────────
def reduce_semantics(snap: SemanticSnapshot) -> dict[str, Any]:
    """Derive the canonical runtime envelope from a snapshot.

    Returns a dict suitable to merge into any health/state payload.

    Priority (first match wins):
      1. ``maintenance``   — maintenance_mode is on.
      2. ``unknown``       — observation is empty/stale.
      3. ``cpu_only``      — no GPU services configured/loaded; CPU route OK.
      4. ``transitioning`` — load/drain/restore/evict in flight.
      5. ``generation_loaded`` — at least one generation bundle is loaded.
      6. ``gpu_idle_llm``  — configured idle GPU LLM is loaded.
      7. ``degraded``      — observation contradicts intent (and not idle).
      8. ``gpu_idle_llm``  — configured idle declared but unloaded → idle.
      9. ``unknown``       — fallback.
    """
    observed_at = float(snap.observed_at or time.time())

    # ── 1. Maintenance short-circuit ────────────────────────────────────
    if snap.maintenance_mode:
        return _envelope(
            mode=RUNTIME_MODE_MAINTENANCE,
            snap=snap,
            observed_at=observed_at,
            blocked_reasons=["maintenance_mode_enabled"],
            contradictions=[],
            intentionally_idle=_classify_intentionally_idle(snap),
        )

    # ── 2. Stale / empty observation ───────────────────────────────────
    if not snap.observed_at or snap.observed_at <= 0:
        return _envelope(
            mode=RUNTIME_MODE_UNKNOWN,
            snap=snap,
            observed_at=observed_at,
            blocked_reasons=["no_observation_available"],
            contradictions=[],
            intentionally_idle=[],
        )

    # ── 3. CPU-only: no GPU service configured or loaded ───────────────
    # "GPU service" = anything not marked cpu_only AND has a gpu_id OR is
    # not in the cpu fallback list.  We approximate: if there are zero
    # loaded runtime services AND zero configured GPU LLM idle services
    # AND the LLM route is the CPU fallback, we are in cpu_only.
    configured_idle_disabled = (
        bool(snap.configured_idle_service)
        and snap.configured_idle_service in set(snap.disabled_service_names)
    )
    if not snap.loaded_runtime_services and (
        not snap.configured_idle_service or configured_idle_disabled
    ):
        if snap.llm_route_available or snap.cpu_fallback_available:
            return _envelope(
                mode=RUNTIME_MODE_CPU_ONLY,
                snap=snap,
                observed_at=observed_at,
                blocked_reasons=(
                    [] if (snap.llm_route_available or snap.cpu_fallback_available)
                    else ["no_llm_route"]
                ),
                contradictions=[],
                intentionally_idle=_classify_intentionally_idle(snap),
            )

    # ── 4. Transition state ────────────────────────────────────────────
    transition_reason = _detect_transition(snap)
    if transition_reason:
        return _envelope(
            mode=RUNTIME_MODE_TRANSITIONING,
            snap=snap,
            observed_at=observed_at,
            blocked_reasons=[f"transition_in_progress:{transition_reason}"],
            contradictions=[],
            intentionally_idle=_classify_intentionally_idle(snap),
        )

    # ── 5. Generation-loaded path ──────────────────────────────────────
    if _generation_bundles(snap):
        contradictions = _detect_contradictions(snap)
        return _envelope(
            mode=(
                RUNTIME_MODE_DEGRADED
                if contradictions
                else RUNTIME_MODE_GENERATION_LOADED
            ),
            snap=snap,
            observed_at=observed_at,
            blocked_reasons=[],
            contradictions=contradictions,
            intentionally_idle=_classify_intentionally_idle(snap),
        )

    # ── 6. GPU idle LLM path ───────────────────────────────────────────
    if snap.configured_idle_service and snap.loaded_runtime_services:
        contradictions = _detect_contradictions(snap)
        return _envelope(
            mode=(
                RUNTIME_MODE_DEGRADED
                if contradictions
                else RUNTIME_MODE_GPU_IDLE_LLM
            ),
            snap=snap,
            observed_at=observed_at,
            blocked_reasons=(
                [] if snap.llm_route_available else ["idle_llm_unhealthy"]
            ),
            contradictions=contradictions,
            intentionally_idle=_classify_intentionally_idle(snap),
        )

    # ── 7. Detect contradictions for the catch-all path ───────────────
    contradictions = _detect_contradictions(snap)
    blocked_reasons: list[str] = []
    if not snap.llm_route_available and not snap.cpu_fallback_available:
        blocked_reasons.append("no_llm_route")

    # ── 8. Configured idle declared but nothing loaded ─────────────────
    # This is the legacy `llm_service=<configured_idle> but
    # loaded_runtime_services=[]` contradiction.  We do NOT silently
    # report gpu_idle_llm because that would lie; degraded is honest.
    if (
        snap.configured_idle_service
        and not configured_idle_disabled
        and not snap.loaded_runtime_services
    ):
        if "configured_idle_service_not_loaded" not in contradictions:
            contradictions.append("configured_idle_service_not_loaded")
        return _envelope(
            mode=RUNTIME_MODE_DEGRADED,
            snap=snap,
            observed_at=observed_at,
            blocked_reasons=blocked_reasons,
            contradictions=contradictions,
            intentionally_idle=_classify_intentionally_idle(snap),
        )

    # ── 9. Catch-all ───────────────────────────────────────────────────
    if contradictions:
        mode = RUNTIME_MODE_DEGRADED
    elif snap.llm_route_available or snap.cpu_fallback_available:
        mode = RUNTIME_MODE_CPU_ONLY
    else:
        mode = RUNTIME_MODE_UNKNOWN

    return _envelope(
        mode=mode,
        snap=snap,
        observed_at=observed_at,
        blocked_reasons=blocked_reasons,
        contradictions=contradictions,
        intentionally_idle=_classify_intentionally_idle(snap),
    )


def _envelope(
    *,
    mode: str,
    snap: SemanticSnapshot,
    observed_at: float,
    blocked_reasons: list[str],
    contradictions: list[str],
    intentionally_idle: list[str],
) -> dict[str, Any]:
    return {
        "runtime_mode": mode,
        "configured_intent": _build_configured_intent(snap),
        "observed_runtime": _build_observed_runtime(snap),
        "route_availability": _build_route_availability(snap, mode),
        "intentionally_idle": intentionally_idle,
        "blocked_reasons": _dedupe(blocked_reasons),
        "contradictions": _dedupe(contradictions),
        "observed_at": observed_at,
    }


def _build_configured_intent(snap: SemanticSnapshot) -> dict[str, Any]:
    return {
        "configured_idle_service": snap.configured_idle_service,
        "configured_idle_services_by_gpu": dict(snap.configured_idle_services_by_gpu),
        "pinned_service": snap.pinned_service,
        "maintenance_mode": bool(snap.maintenance_mode),
        # Registry overlays can repeat the same systemd unit under aliases;
        # expose one ordered identity so health does not suggest duplicate
        # processes or owners.
        "enabled_service_names": _dedupe(
            [str(name) for name in snap.enabled_service_names if name]
        ),
        "disabled_service_names": _dedupe(
            [str(name) for name in snap.disabled_service_names if name]
        ),
    }


def _build_observed_runtime(snap: SemanticSnapshot) -> dict[str, Any]:
    return {
        "gpu_state": snap.gpu_state_value,
        "llm_evicted": bool(snap.llm_evicted),
        "loaded_runtime_services": _dedupe(
            [str(name) for name in snap.loaded_runtime_services if name]
        ),
        "loaded_runtime_bundles": list(snap.loaded_runtime_bundles),
        "queue_active_jobs": int(snap.queue_active_jobs),
        "queue_pending_jobs": int(snap.queue_pending_jobs),
    }


def _build_route_availability(snap: SemanticSnapshot, mode: str) -> dict[str, Any]:
    active_route = ""
    if snap.llm_route_available:
        active_route = "pinned" if snap.pinned_service else "configured_idle"
    elif snap.cpu_fallback_available:
        active_route = "cpu_fallback"
    return {
        "active_llm_route": active_route,
        "llm_route_available": bool(snap.llm_route_available),
        "cpu_fallback_available": bool(snap.cpu_fallback_available),
        "generation_route_available": bool(snap.generation_route_available),
        "mode": mode,
    }


# ── Helpers ───────────────────────────────────────────────────────────────
def _classify_intentionally_idle(snap: SemanticSnapshot) -> list[str]:
    """Service names that are configured off / loaded-by-default but absent.

    These are NOT problems — they are intentional.  /v1/health/all must
    surface this so dashboards don't mark them "down".
    """
    loaded = set(snap.loaded_runtime_services)
    idle_targets = set(snap.configured_idle_services_by_gpu.values())
    if snap.configured_idle_service:
        idle_targets.add(snap.configured_idle_service)

    # If a service is enabled, not the configured idle, and not in the
    # loaded set, it is *intentionally idle*.
    intentionally_idle: list[str] = [
        name for name in snap.disabled_service_names if name not in loaded
    ]
    for name in snap.enabled_service_names:
        if name in idle_targets:
            continue  # configured idle — must be loaded, not "idle"
        if name in loaded:
            continue
        intentionally_idle.append(name)
    return _dedupe(intentionally_idle)


def _detect_transition(snap: SemanticSnapshot) -> str:
    if snap.any_bundle_draining:
        return "draining"
    if snap.any_bundle_loading:
        return "loading"
    if snap.any_bundle_restoring:
        return "restoring"
    if snap.any_bundle_evicting:
        return "evicting"
    gpu_state = (snap.gpu_state_value or "").lower()
    # The legacy VRAM enum can remain at RESTORING after the restore
    # coroutine has exited.  Only the operation-active fact or a per-bundle
    # restoring flag is authoritative for that transition.  This prevents a
    # stable restored idle route from remaining permanently "transitioning".
    if gpu_state == "restoring":
        return "restoring" if snap.restore_operation_active else ""
    if gpu_state in TRANSITION_STATES:
        return gpu_state
    return ""


def _bundle_service_identifiers(entry) -> set[str]:
    """Return the set of service identifiers carried by one bundle service entry.

    Service entries may be plain strings (a systemd unit / service name) or
    dictionaries with ``name`` and/or ``systemd_unit`` keys.  Both kinds of
    identifier are extracted so the configured-idle comparison sees the same
    surface that ``_get_loaded_runtime_snapshot()`` populates.  Empty/falsy
    entries contribute nothing.
    """
    identifiers: set[str] = set()
    if isinstance(entry, dict):
        name = entry.get("name")
        if name:
            identifiers.add(str(name))
        unit = entry.get("systemd_unit")
        if unit:
            identifiers.add(str(unit))
        return identifiers
    if entry:
        identifiers.add(str(entry))
    return identifiers


def _generation_bundles(snap: SemanticSnapshot) -> list[dict]:
    """Return loaded bundles that are not solely configured idle services."""
    idle_units = {
        str(unit)
        for unit in (
            snap.configured_idle_service,
            *snap.configured_idle_services_by_gpu.values(),
        )
        if unit
    }
    generation: list[dict] = []
    for bundle in snap.loaded_runtime_bundles:
        services: set[str] = set()
        for entry in bundle.get("services", []) or []:
            services.update(_bundle_service_identifiers(entry))
        # Unknown/legacy bundle shapes (empty service list, or services we
        # could not reduce to any identifier) remain conservatively classified
        # as generation residency.  Only a fully identified idle-only bundle
        # is excluded.
        if not services or not services.issubset(idle_units):
            generation.append(bundle)
    return generation


def _detect_contradictions(snap: SemanticSnapshot) -> list[str]:
    """Pure contradiction detector.

    Plan §6 cases:
      * state says loaded but no bundle/process exists
      * service process exists while configured disabled and no maintenance
      * bundle ready but linked member absent
      * queue job active without worker/lease
      * port ready without expected cgroup/GPU tenant
      * transition owner/fence is stale
    """
    contradictions: list[str] = []

    # GPU state says LLM loaded but no runtime service is reported.
    if snap.gpu_state_value == "llm_loaded" and not snap.loaded_runtime_services:
        contradictions.append("gpu_state_llm_loaded_but_no_runtime_services")

    # llm_evicted=True but state still says llm_loaded.
    if snap.llm_evicted and snap.gpu_state_value == "llm_loaded":
        contradictions.append("llm_evicted_with_llm_loaded_state")

    # Configured idle service is reported as loaded_runtime but its unit
    # is not in loaded_runtime_services (i.e. configuration lies about
    # presence).
    if (
        snap.configured_idle_service
        and snap.configured_idle_service not in set(snap.disabled_service_names)
        and not snap.loaded_runtime_bundles
    ):
        # During generation residency the idle LLM is intentionally displaced;
        # absence only contradicts intent when no generation bundle owns GPUs.
        # are systemd unit names too.  A configured idle unit not present
        # in loaded services = configuration/observation contradiction.
        if snap.configured_idle_service not in snap.loaded_runtime_services:
            contradictions.append("configured_idle_service_not_observed_loaded")

    # Pinned service is configured but no route is available.
    if snap.pinned_service and not snap.llm_route_available and not snap.cpu_fallback_available:
        contradictions.append("pinned_service_with_no_route")

    # Maintenance mode is off but we have no observation at all (pure
    # stale) — not really a contradiction; the reducer routes to
    # RUNTIME_MODE_UNKNOWN via the empty-observation branch.

    # Bundle is loaded but its declared gpu_id is unknown.
    for entry in snap.loaded_runtime_bundles:
        if isinstance(entry, dict) and not entry.get("gpu_id"):
            contradictions.append("loaded_bundle_missing_gpu_id")

    # Queue says active jobs but no LLM route and no CPU fallback.
    if snap.queue_active_jobs > 0 and not snap.llm_route_available and not snap.cpu_fallback_available:
        contradictions.append("active_jobs_without_route")

    return contradictions


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ── Convenience builder for gpu-manager integration ──────────────────────
def snapshot_from_manager(
    *,
    configured_idle_service: str,
    configured_idle_services_by_gpu: dict[str, str] | None = None,
    pinned_service: str = "",
    maintenance_mode: bool = False,
    enabled_service_names: list[str] | None = None,
    disabled_service_names: list[str] | None = None,
    loaded_runtime_services: list[str] | None = None,
    loaded_runtime_bundles: list[dict] | None = None,
    gpu_state_value: str = "",
    llm_evicted: bool = False,
    queue_active_jobs: int = 0,
    queue_pending_jobs: int = 0,
    any_bundle_draining: bool = False,
    any_bundle_loading: bool = False,
    any_bundle_restoring: bool = False,
    any_bundle_evicting: bool = False,
    restore_operation_active: bool = False,
    service_health: dict[str, dict] | None = None,
    llm_route_available: bool = False,
    cpu_fallback_available: bool = False,
    generation_route_available: bool = False,
    observed_at: float | None = None,
) -> SemanticSnapshot:
    """Construct a SemanticSnapshot from individual fields.

    gpu-manager handlers call this with their locally-resolved facts and
    then call ``reduce_semantics(...)``.
    """
    return SemanticSnapshot(
        configured_idle_service=configured_idle_service or "",
        configured_idle_services_by_gpu=dict(configured_idle_services_by_gpu or {}),
        pinned_service=pinned_service or "",
        maintenance_mode=bool(maintenance_mode),
        enabled_service_names=list(enabled_service_names or []),
        disabled_service_names=list(disabled_service_names or []),
        loaded_runtime_services=list(loaded_runtime_services or []),
        loaded_runtime_bundles=list(loaded_runtime_bundles or []),
        gpu_state_value=gpu_state_value or "",
        llm_evicted=bool(llm_evicted),
        queue_active_jobs=int(queue_active_jobs or 0),
        queue_pending_jobs=int(queue_pending_jobs or 0),
        any_bundle_draining=bool(any_bundle_draining),
        any_bundle_loading=bool(any_bundle_loading),
        any_bundle_restoring=bool(any_bundle_restoring),
        any_bundle_evicting=bool(any_bundle_evicting),
        restore_operation_active=bool(restore_operation_active),
        service_health=dict(service_health or {}),
        llm_route_available=bool(llm_route_available),
        cpu_fallback_available=bool(cpu_fallback_available),
        generation_route_available=bool(generation_route_available),
        observed_at=float(observed_at if observed_at is not None else time.time()),
    )
