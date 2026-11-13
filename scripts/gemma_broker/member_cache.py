"""Broker-owned cache bridging asynchronous evidence and synchronous dispatch.

The dispatch loop is intentionally synchronous when it requests current member
snapshots.  GPUManager refreshes evidence asynchronously and writes it through
this cache; until a complete probe arrives every member is represented as
unhealthy and non-accepting.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import time
from typing import Any

from .contracts import MemberSnapshot, MemberState
from .runtime import map_member_ready


#: The refresh loop re-observes every member every ~2 s plus its probe sweep. A
#: snapshot older than this means the loop or that member's probe is wedged, so the
#: last accepting view must not keep taking work.
STALE_AFTER_SECONDS = 30.0


class StrictMemberSnapshotCache:
    """Ordered, fail-closed snapshot holder for configured broker members."""

    def __init__(
        self,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        freshness_ms: int = 5_000,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._configs = {name: dict(config) for name, config in configs.items()}
        self._monotonic = monotonic
        self._updated_mono: dict[str, float] = {}
        self._freshness_ms = int(freshness_ms)
        self._snapshots: dict[str, MemberSnapshot] = {
            name: self._initial_snapshot(name, config)
            for name, config in self._configs.items()
        }

    @staticmethod
    def _initial_snapshot(name: str, config: dict[str, Any]) -> MemberSnapshot:
        return MemberSnapshot(
            name=name,
            gpu_id=config.get("gpu_id"),
            state=MemberState.UNHEALTHY,
            accepting=False,
            configured_slots=int(config.get("parallel", config.get("worker_parallel", 0)) or 0),
            # broker_context_limit: admission ceiling below the real n_ctx (slow member).
            context_per_slot=int(config.get("broker_context_limit") or config.get("context_per_slot", 0) or 0),
            state_version=1,
            observed_at=0.0,
            idle_service_configured=config.get("idle_service_configured"),
            idle_service_effective=None if config.get("cpu_only") or not config.get("gpu_id") else False,
            systemd_active=False,
            semantic_health=False,
            model_resident=None if config.get("cpu_only") or not config.get("gpu_id") else False,
            dispatcher_registered=False,
            compatible_free_slots=0,
            blockers=(
                (f"self-disabled at broker build: {config['self_disabled_reason']}",)
                if config.get("self_disabled_reason")
                else ("no semantic readiness snapshot observed",)
            ),
        )

    def reconfigure(self, name: str, config: Mapping[str, Any]) -> None:
        """Hot-apply a changed member config; fail closed until the next probe."""
        if name not in self._configs:
            raise KeyError(f"UNKNOWN_MEMBER:{name}")
        self._configs[name] = dict(config)
        prev = self._snapshots[name]  # versions only move forward (fenced mirror)
        self._snapshots[name] = replace(
            self._initial_snapshot(name, self._configs[name]),
            state_version=prev.state_version + 1, generation_fence=prev.generation_fence)
        self._updated_mono.pop(name, None)

    def snapshots(self, *, now: float | None = None) -> list[MemberSnapshot]:
        """Return snapshots in configured order; this method does no I/O.

        A member whose snapshot has not been refreshed by update() within
        STALE_AFTER_SECONDS (monotonic, independent of probe clocks) is served
        as unhealthy; never-updated boot snapshots are already unhealthy.
        """
        mono = self._monotonic()
        out = []
        for name in self._configs:
            snap = self._snapshots[name]
            updated = self._updated_mono.get(name)
            if updated is not None and snap.accepting and mono - updated > STALE_AFTER_SECONDS:
                snap = replace(
                    snap,
                    state=MemberState.UNHEALTHY,
                    accepting=False,
                    compatible_free_slots=0,
                    blockers=(*snap.blockers, f"readiness snapshot stale ({mono - updated:.0f}s)"),
                )
            out.append(snap)
        return out

    def update(self, name: str, probes: Mapping[str, Any], *, now: float) -> MemberSnapshot:
        if name not in self._configs:
            raise KeyError(f"UNKNOWN_MEMBER:{name}")
        previous = self._snapshots[name]
        snapshot = map_member_ready(
            name=name,
            config=self._configs[name],
            probes=dict(probes),
            now=float(now),
            freshness_ms=self._freshness_ms,
            previous=previous,
        )
        self._snapshots[name] = snapshot
        self._updated_mono[name] = self._monotonic()
        return snapshot


__all__ = ["StrictMemberSnapshotCache"]
