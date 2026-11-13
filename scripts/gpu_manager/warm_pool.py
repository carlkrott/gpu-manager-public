"""Warm-pool pre-load (catalog plan 5e): queue look-ahead preloads the next bundle.

Flag ``GPU_MANAGER_WARM_POOL_PRELOAD`` (default off). This module holds the pure
per-GPU rule and the per-job / per-bundle memory; the facade glue
``_warm_pool_preload_tick`` supplies live state and does the load through the
canonical ``_evict_then_load_bundle`` path.

Safety model: a preload only ever replaces a *finished generation bundle* (or
fills a GPU that has no idle member configured). An empty GPU whose idle member
is configured counts as the idle member's slot, so a preload can never race
the idle restore after an auto-evict (the 10-14 thrash shape).
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict

FLAG = "GPU_MANAGER_WARM_POOL_PRELOAD"
# One preload per bundle per window, even for a new head job.
BUNDLE_COOLDOWN_S = 300.0
_MAX_REMEMBERED = 256


def enabled(env=None) -> bool:
    value = (os.environ if env is None else env).get(FLAG, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def loaded_names(gs) -> list[str]:
    names: list[str] = []
    for name in [getattr(gs, "loaded_bundle", None), *list(getattr(gs, "loaded_bundles", None) or [])]:
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def gpu_blocker(gs, *, bundles, gpu_has_idle_member: bool, pending_bundles) -> str | None:
    """Why this GPU may not take a warm preload, or None when it may."""
    if gs is None:
        return "gpu_state_unreadable"
    if getattr(gs, "loading", False) or getattr(gs, "evicting", False) or getattr(gs, "draining", False):
        return "gpu_in_transition"
    if int(getattr(gs, "pending_requests", 0) or 0) > 0 or int(getattr(gs, "worker_in_flight", 0) or 0) > 0:
        return "gpu_running_job"
    names = loaded_names(gs)
    if not names:
        return "idle_member_slot" if gpu_has_idle_member else None
    for name in names:
        if name.startswith("llm:"):
            return "llm_resident"
        cfg = bundles.get(name)
        if not isinstance(cfg, dict):
            return "unknown_occupant"
        if cfg.get("idle"):
            return "idle_member_resident"
        if cfg.get("evict_only_when_idle"):
            return "protected_occupant"
        if name in pending_bundles:
            return "occupant_has_demand"
    return None


class WarmPoolMemory:
    """Hysteresis + once-per-job logging + single in-flight preload."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.preloaded_jobs: OrderedDict[str, str] = OrderedDict()
        self.bundle_last: dict[str, float] = {}
        self.logged: OrderedDict[tuple, None] = OrderedDict()
        self.task = None

    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def hysteresis(self, job_id: str, bundle: str) -> str | None:
        if job_id in self.preloaded_jobs:
            return "already_preloaded_for_job"
        last = self.bundle_last.get(bundle)
        if last is not None and self.clock() - last < BUNDLE_COOLDOWN_S:
            return "bundle_cooldown"
        return None

    def record(self, job_id: str, bundle: str) -> None:
        self.preloaded_jobs[job_id] = bundle
        self.bundle_last[bundle] = self.clock()
        while len(self.preloaded_jobs) > _MAX_REMEMBERED:
            self.preloaded_jobs.popitem(last=False)

    def log_once(self, log, job_id: str, decision: str, msg: str, *args) -> bool:
        key = (job_id, decision)
        if key in self.logged:
            return False
        self.logged[key] = None
        while len(self.logged) > _MAX_REMEMBERED:
            self.logged.popitem(last=False)
        log.info(msg, *args)
        return True
