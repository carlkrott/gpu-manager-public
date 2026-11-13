"""Idle baseline target selection with bounded backoff and state-change logging.

Pure logic (no I/O, injected clock/rng) so the controller and the tests share it.
An idle decision never carries worker claims: it only names which configured idle
service each GPU should hold warm.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Mapping

BACKOFF_BASE_S = 5.0
BACKOFF_MAX_S = 300.0
JITTER_FRACTION = 0.2


def service_loadable(service: Any) -> bool:
    """A service may be loaded only when enabled and it has a unit and a bounded port."""
    if not isinstance(service, Mapping) or service.get("enabled") is not True:
        return False
    port = service.get("port")
    return bool(service.get("systemd_unit")) and type(port) is int and 1 <= port <= 65535


def _fallback_for(scheduling: Mapping[str, Any], gpu_id: str) -> str:
    per_gpu = scheduling.get("idle_fallback_services") or {}
    if isinstance(per_gpu, Mapping) and per_gpu.get(gpu_id):
        return str(per_gpu[gpu_id])
    resident = scheduling.get("idle_fallback_service")
    return str(resident) if isinstance(resident, str) and resident else ""


class IdleTargetPlanner:
    """Chooses per-GPU idle targets: preferred, else configured resident, else none."""

    def __init__(
        self,
        log: Callable[[str], None],
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._log = log
        self._rng = rng
        self._state: dict[str, dict[str, Any]] = {}

    def _signature(self, config: Mapping[str, Any], gpu_id: str) -> tuple:
        scheduling = config.get("scheduling") or {}
        services = config.get("services") or {}
        preferred = str((scheduling.get("idle_services") or {}).get(gpu_id) or "")
        fallback = _fallback_for(scheduling, gpu_id)
        return (
            preferred, service_loadable(services.get(preferred)),
            fallback, service_loadable(services.get(fallback)),
        )

    def _transition(self, gpu_id: str, st: dict, state: str, detail: str) -> None:
        if st.get("state") != state or st.get("detail") != detail:
            self._log(f"idle target {gpu_id}: {state} {detail}".rstrip())
        st["state"], st["detail"] = state, detail

    def plan(self, config: Mapping[str, Any], now: float) -> dict[str, dict[str, Any]]:
        """Return {gpu_id: {service, mode, ready}}; ``ready`` False while backing off."""
        scheduling = config.get("scheduling") or {}
        services = config.get("services") or {}
        gpus = sorted(str(g) for g in (scheduling.get("idle_services") or {}))
        out: dict[str, dict[str, Any]] = {}
        for gpu_id in gpus:
            st = self._state.setdefault(gpu_id, {"attempts": 0, "retry_at": 0.0})
            sig = self._signature(config, gpu_id)
            if st.get("sig") != sig:
                # Configuration/availability changed: retry state restarts.
                st.update(sig=sig, attempts=0, retry_at=0.0)
            preferred, pref_ok, fallback, fb_ok = sig
            if pref_ok:
                service, mode = preferred, "preferred"
            elif fb_ok:
                service, mode = fallback, "fallback"
            else:
                service, mode = "", "unavailable"
            if mode == "unavailable":
                self._transition(
                    gpu_id, st, "unavailable",
                    f"(preferred={preferred or '-'} fallback={fallback or '-'} not loadable)",
                )
                out[gpu_id] = {"service": None, "mode": mode, "ready": False}
                continue
            detail = f"-> {service}" + (f" (preferred {preferred} unavailable)" if mode == "fallback" else "")
            ready = now >= st["retry_at"]
            self._transition(gpu_id, st, mode, detail)
            out[gpu_id] = {"service": service, "mode": mode, "ready": ready}
        return out

    def report_result(self, gpu_id: str, restored: bool, now: float) -> None:
        st = self._state.setdefault(gpu_id, {"attempts": 0, "retry_at": 0.0})
        if restored:
            st.update(attempts=0, retry_at=0.0)
            return
        st["attempts"] += 1
        delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** (st["attempts"] - 1)))
        delay *= 1.0 + JITTER_FRACTION * (2.0 * self._rng() - 1.0)
        st["retry_at"] = now + min(delay, BACKOFF_MAX_S)

    def retry_at(self, gpu_id: str) -> float:
        return float(self._state.get(gpu_id, {}).get("retry_at", 0.0))


class DisabledMemberReporter:
    """Reports each disabled/absent group member once per state, never per tick."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log
        self._seen: dict[str, str] = {}

    def probeable(self, name: str, service: Any) -> bool:
        state = "enabled" if isinstance(service, Mapping) and service.get("enabled") is not False and service else "disabled"
        if self._seen.get(name) != state:
            if state == "disabled":
                self._log(f"member {name}: disabled/absent; health probing skipped")
            elif name in self._seen:
                self._log(f"member {name}: enabled; health probing resumed")
            self._seen[name] = state
        return state == "enabled"
