"""Broker-owned cache bridging asynchronous evidence and synchronous dispatch.

The dispatch loop is intentionally synchronous when it requests current member
snapshots.  GPUManager refreshes evidence asynchronously and writes it through
this cache; until a complete probe arrives every member is represented as
unhealthy and non-accepting.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import MemberSnapshot, MemberState
from .runtime import map_member_ready


class StrictMemberSnapshotCache:
    """Ordered, fail-closed snapshot holder for configured broker members."""

    def __init__(self, configs: Mapping[str, Mapping[str, Any]], *, freshness_ms: int = 5_000) -> None:
        self._configs = {name: dict(config) for name, config in configs.items()}
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
            context_per_slot=int(config.get("context_per_slot", 0) or 0),
            state_version=1,
            observed_at=0.0,
            idle_service_configured=config.get("idle_service_configured"),
            idle_service_effective=None if config.get("cpu_only") or not config.get("gpu_id") else False,
            systemd_active=False,
            semantic_health=False,
            model_resident=None if config.get("cpu_only") or not config.get("gpu_id") else False,
            dispatcher_registered=False,
            compatible_free_slots=0,
            blockers=("no semantic readiness snapshot observed",),
        )

    def snapshots(self, *, now: float | None = None) -> list[MemberSnapshot]:
        """Return snapshots in configured order; this method does no I/O."""
        return [self._snapshots[name] for name in self._configs]

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
        return snapshot


__all__ = ["StrictMemberSnapshotCache"]
