"""Single-owner placement, atomic model sets and idle holds for GPU Manager v2.

Implements:
1. ModelPlacementLease: TTL-bound placement lease owned by GPUScheduler.
2. IdleHold: TTL-bound idle hold with explicit release, preventing auto-restore/preemption.
3. ModelGroupPlacement: Fitting and pinning model sets across GPUs, checking EVERY
   participating GPU before evicting any (zero partial eviction).
4. Pins known-bad GPU/model pairs (e.g. BF16 models on gfx1030/gfx906, over-budget VRAM on desktop display GPU).
5. Preserves bundle lifecycle fences, CPU Gemma P2 baseline, linked bundles, and controller-loss semantics.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from admission_affirmation import AdmissionController, AdmissionDecision, AdmissionStatus
from bundle_lifecycle import BundleLifecycleController, DesiredState, ResourceLease
from hardware_recognition import ARCH_CAPABILITIES, DeviceStatus, HardwareProfile


@dataclass
class IdleHold:
    hold_id: str
    gpu_id: str
    owner: str
    reason: str
    created_at: float
    ttl_seconds: float
    released: bool = False

    def check_expired(self, now: float) -> bool:
        if self.released:
            return True
        if self.ttl_seconds <= 0:
            return False
        return now > (self.created_at + self.ttl_seconds)


@dataclass
class ModelPlacementLease:
    lease_id: str
    model_set_id: str
    allocations: dict[str, int]  # gpu_id -> vram_mib
    owner: str
    created_at: float
    ttl_seconds: float
    resource_lease: Optional[ResourceLease] = None
    released: bool = False

    @property
    def is_expired(self) -> bool:
        if self.released:
            return True
        if self.ttl_seconds <= 0:
            return False
        return time.time() > (self.created_at + self.ttl_seconds)


class PlacementDecisionStatus(str, Enum):
    PLACED = "placed"
    QUEUED = "queued"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PlacementDecision:
    status: PlacementDecisionStatus
    lease: Optional[ModelPlacementLease] = None
    position: Optional[int] = None
    reason: Optional[str] = None
    nearest_alternative: Optional[dict[str, Any]] = None
    evictions_required: list[str] = field(default_factory=list)


class PlacementOrchestrator:
    """Sole placement and lease owner for GPUScheduler."""

    def __init__(
        self,
        hardware_registry: Mapping[str, HardwareProfile],
        lifecycle_controller: Optional[BundleLifecycleController] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.hardware_registry = dict(hardware_registry)
        self.lifecycle_controller = lifecycle_controller
        self.clock = clock
        self._idle_holds: dict[str, IdleHold] = {}  # hold_id -> IdleHold
        self._active_leases: dict[str, ModelPlacementLease] = {}  # lease_id -> Lease
        self._pinned_pairs: set[tuple[str, str]] = set()  # (device, model/bundle) known-bad pins

    # ── Idle holds management ──────────────────────────────────────────

    def acquire_idle_hold(
        self,
        hold_id: str,
        gpu_id: str,
        *,
        owner: str = "gpu-manager",
        reason: str = "caller_hold",
        ttl_seconds: float = 60.0,
    ) -> IdleHold:
        """Acquire a TTL-bound idle hold preventing idle restoration or preemption on gpu_id."""
        self._prune_expired_holds()
        hold = IdleHold(
            hold_id=hold_id,
            gpu_id=gpu_id,
            owner=owner,
            reason=reason,
            created_at=self.clock(),
            ttl_seconds=ttl_seconds,
            released=False,
        )
        self._idle_holds[hold_id] = hold
        return hold

    def release_idle_hold(self, hold_id: str) -> bool:
        """Explicit release of an idle hold."""
        hold = self._idle_holds.get(hold_id)
        if hold and not hold.released:
            hold.released = True
            return True
        return False

    def is_idle_held(self, gpu_id: str) -> bool:
        """Check if gpu_id is held by an active unexpired idle hold."""
        self._prune_expired_holds()
        now = self.clock()
        for hold in self._idle_holds.values():
            if hold.gpu_id == gpu_id and not hold.check_expired(now):
                return True
        return False

    def _prune_expired_holds(self) -> None:
        now = self.clock()
        expired = [
            hid for hid, h in self._idle_holds.items()
            if h.released or (h.ttl_seconds > 0 and now > (h.created_at + h.ttl_seconds))
        ]
        for hid in expired:
            self._idle_holds.pop(hid, None)

    # ── Known-bad pinning ──────────────────────────────────────────────

    def pin_known_bad_pair(self, device: str, model_id: str) -> None:
        """Explicitly pin a known-bad device/model combination so it is always rejected or re-routed."""
        self._pinned_pairs.add((device, model_id))

    def is_known_bad(self, device: str, model_id: str) -> bool:
        return (device, model_id) in self._pinned_pairs

    # ── Atomic Model-Set Placement ────────────────────────────────────

    def place_model_set(
        self,
        model_set: Mapping[str, Any],
        gpu_states: Mapping[str, Any],
        *,
        owner: str = "gpu-scheduler",
        lease_ttl_seconds: float = 120.0,
        queue_depth: int = 0,
    ) -> PlacementDecision:
        """Fit and place a model set atomically across all required GPUs.
        
        Contract:
        1. Check every participating GPU before evicting any.
        2. If any GPU fails hardware compatibility, precision support, desktop ceiling,
           is pinned as known-bad, or is held by an idle hold:
           REJECT or QUEUE without performing ANY partial eviction.
        3. All-or-nothing: zero partial evictions.
        """
        self._prune_expired_holds()
        model_set_id = model_set.get("id") or model_set.get("name") or "unnamed-set"
        components = model_set.get("components", [])

        # Validate components
        participating_gpus: list[str] = []
        gpu_vram_needed: dict[str, int] = {}
        for comp in components:
            dev = comp.get("device", "")
            if not dev:
                continue
            if dev not in ("cpu", "CPU"):
                participating_gpus.append(dev)
                req_vram = int(comp.get("min_vram_mib") or comp.get("vram_mib") or 0)
                gpu_vram_needed[dev] = gpu_vram_needed.get(dev, 0) + req_vram

        # Explicit budgets check
        budgets = model_set.get("memory_budget_mib", {})
        if isinstance(budgets, Mapping):
            for dev, budget in budgets.items():
                if dev.lower() != "cpu" and budget is not None:
                    if dev not in participating_gpus:
                        participating_gpus.append(dev)
                    gpu_vram_needed[dev] = max(gpu_vram_needed.get(dev, 0), int(budget))

        # Check known-bad pins & hardware capability across ALL participating GPUs FIRST
        for comp in components:
            dev = comp.get("device", "")
            model_id = comp.get("model") or comp.get("id") or model_set_id
            precision = (comp.get("precision") or "").lower()

            if self.is_known_bad(dev, model_id) or self.is_known_bad(dev, model_set_id):
                return PlacementDecision(
                    status=PlacementDecisionStatus.REJECTED,
                    reason=f"Device/model pair ({dev}, {model_id}) is pinned as known-bad",
                    nearest_alternative={"device": "cpu"},
                )

            if dev in ("cpu", "CPU"):
                continue

            hw = self.hardware_registry.get(dev)
            if not hw:
                return PlacementDecision(
                    status=PlacementDecisionStatus.REJECTED,
                    reason=f"Unknown or unobserved device '{dev}' in model set",
                    nearest_alternative={"device": "cpu"},
                )

            # Hardware status
            if hw.status != DeviceStatus.OBSERVED:
                return PlacementDecision(
                    status=PlacementDecisionStatus.REJECTED,
                    reason=f"Device '{dev}' is not observed (status: {hw.status.value})",
                    nearest_alternative={"device": "cpu"},
                )

            # Precision check (e.g. BF16 on gfx1030/gfx906)
            if precision in hw.unsupported_precisions or (precision == "bf16" and not hw.native_bf16):
                return PlacementDecision(
                    status=PlacementDecisionStatus.REJECTED,
                    reason=f"Component '{model_id}' requires precision '{precision}' not supported on {dev} ({hw.gfx_arch})",
                    nearest_alternative={"device": "cpu", "precision": precision},
                )

        # Check capacity, ceiling, idle-holds, and eviction viability across EVERY GPU
        evictions_needed: list[str] = []
        for dev in participating_gpus:
            hw = self.hardware_registry.get(dev)
            needed = gpu_vram_needed.get(dev, 0)
            effective_ceiling = hw.vram_ceiling_mib if (hw and hw.vram_ceiling_mib) else (hw.vram_total_mib if hw else 0)

            # Desktop ceiling enforce
            if dev == "rx6950xt":
                effective_ceiling = min(effective_ceiling, 14832)

            if needed > effective_ceiling:
                return PlacementDecision(
                    status=PlacementDecisionStatus.REJECTED,
                    reason=f"Requested {needed} MiB exceeds effective device ceiling {effective_ceiling} MiB on '{dev}'",
                    nearest_alternative={"device": "cpu"},
                )

            # Check if held by idle hold
            if self.is_idle_held(dev):
                return PlacementDecision(
                    status=PlacementDecisionStatus.QUEUED,
                    position=queue_depth + 1,
                    reason=f"GPU '{dev}' is held by an active TTL idle hold; cannot evict or place",
                )

            # Check GPU current state
            gs = gpu_states.get(dev)
            if gs is not None:
                # Check if higher priority bundle running
                loaded_bundle = getattr(gs, "loaded_bundle", None) or getattr(gs, "current_bundle", None)
                if isinstance(gs, dict):
                    loaded_bundle = gs.get("loaded_bundle")
                    bundle_prio = gs.get("bundle_priority", 0)
                    state_str = gs.get("state", "idle")
                else:
                    bundle_prio = getattr(gs, "bundle_priority", 0)
                    state_str = getattr(gs, "state", "idle")

                req_prio = int(model_set.get("priority", 5))
                if loaded_bundle and loaded_bundle != model_set_id:
                    if bundle_prio > req_prio:
                        # Refuse to evict higher priority bundle
                        return PlacementDecision(
                            status=PlacementDecisionStatus.QUEUED,
                            position=queue_depth + 1,
                            reason=f"GPU '{dev}' running higher-priority bundle '{loaded_bundle}' (prio {bundle_prio} > {req_prio})",
                        )
                    if state_str in ("loading", "evicting", "draining"):
                        return PlacementDecision(
                            status=PlacementDecisionStatus.QUEUED,
                            position=queue_depth + 1,
                            reason=f"GPU '{dev}' is transitioning ({state_str})",
                        )
                    evictions_needed.append(dev)

        # All participating GPUs passed verification! Zero partial eviction guaranteed.
        import uuid
        lease = ModelPlacementLease(
            lease_id=f"lease-{uuid.uuid4().hex[:8]}",
            model_set_id=model_set_id,
            allocations=gpu_vram_needed,
            owner=owner,
            created_at=self.clock(),
            ttl_seconds=lease_ttl_seconds,
        )
        self._active_leases[lease.lease_id] = lease

        return PlacementDecision(
            status=PlacementDecisionStatus.PLACED,
            lease=lease,
            evictions_required=evictions_needed,
        )

    def release_lease(self, lease_id: str) -> bool:
        """Release a model placement lease."""
        lease = self._active_leases.get(lease_id)
        if lease and not lease.released:
            lease.released = True
            return True
        return False
