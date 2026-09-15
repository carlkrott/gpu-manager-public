"""Pure, fail-closed readiness reduction for combined-Gemma members."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .contracts import MemberSnapshot, MemberState, ReasonCode


def _semantic(snapshot: MemberSnapshot) -> dict[str, Any]:
    data = snapshot.to_dict()
    for key in ("observed_at", "snapshot_age_ms", "state_version"):
        data.pop(key, None)
    return data


def reduce_member_readiness(
    name: str,
    config: dict[str, Any],
    probes: dict[str, Any],
    *,
    now: float,
    freshness_ms: int = 5000,
    previous: MemberSnapshot | None = None,
) -> MemberSnapshot:
    gpu_id = config.get("gpu_id")
    cpu_only = bool(config.get("cpu_only")) or not gpu_id
    configured_slots = int(config.get("parallel", config.get("worker_parallel", 0)) or 0)
    observed_at = float(probes.get("observed_at", 0.0))
    snapshot_age_ms = max(0, int((now - observed_at) * 1000))
    stale = snapshot_age_ms > freshness_ms
    reasons: list[ReasonCode] = []
    blockers: list[str] = []

    def block(reason: ReasonCode, detail: str) -> None:
        if reason not in reasons:
            reasons.append(reason)
            blockers.append(detail)

    if not bool(config.get("enabled")):
        block(ReasonCode.MEMBER_CONFIG_DISABLED, "service disabled in config")
    if stale:
        block(ReasonCode.TELEMETRY_STALE, f"snapshot age {snapshot_age_ms}ms exceeds {freshness_ms}ms")

    idle_configured = config.get("idle_service_configured")
    idle_effective = probes.get("idle_service_effective")
    model_resident = probes.get("model_resident")
    if not cpu_only:
        if not idle_configured:
            block(ReasonCode.MEMBER_IDLE_SERVICE_NOT_CONFIGURED, "idle service not configured")
        if idle_effective is not True:
            block(ReasonCode.MEMBER_IDLE_SERVICE_NOT_EFFECTIVE, "configured idle service is not effective")
        if bool(probes.get("conflicting_bundle")):
            block(ReasonCode.MEMBER_CONFLICTING_BUNDLE, "conflicting bundle is active")
        if model_resident is not True:
            block(ReasonCode.MEMBER_MODEL_NOT_RESIDENT, "model residency not proved")
    else:
        idle_effective = None
        model_resident = None

    systemd_active = bool(probes.get("systemd_active"))
    main_pid = probes.get("main_pid")
    if not systemd_active or not main_pid:
        block(ReasonCode.MEMBER_SYSTEMD_INACTIVE, "systemd unit or MainPID is inactive")
    semantic_health = bool(probes.get("semantic_health"))
    probe_error = probes.get("probe_error")
    if probe_error in {"connection_refused", "connection_error"}:
        block(ReasonCode.MEMBER_PORT_DARK, "port unreachable (connection refused)")
    elif probe_error == "probe_timeout":
        block(ReasonCode.MEMBER_PROBE_TIMEOUT, "probe timed out")
    elif probe_error == "server_disconnected":
        block(ReasonCode.MEMBER_PROBE_DISCONNECTED, "server disconnected")
    elif isinstance(probe_error, str) and probe_error.startswith("http_"):
        status = probe_error.removeprefix("http_")
        block(ReasonCode.MEMBER_UNHEALTHY_RESPONSE, f"backend responded {status}")
    elif probe_error == "bad_json":
        block(ReasonCode.MEMBER_UNHEALTHY_RESPONSE, "backend returned invalid JSON")
    elif probe_error == "systemd_error":
        # The canonical blocker is emitted above from systemd_active/MainPID;
        # do not misreport this control-plane failure as an HTTP health failure.
        pass
    elif probe_error == "probe_unknown":
        block(ReasonCode.MEMBER_HTTP_UNREADY, "backend probe result unavailable")
    elif not semantic_health:
        block(ReasonCode.MEMBER_HTTP_UNREADY, "semantic HTTP health failed")
    dispatcher_registered = bool(probes.get("dispatcher_registered"))
    if not dispatcher_registered:
        block(ReasonCode.MEMBER_DISPATCHER_UNREGISTERED, "dispatcher registration is absent")

    backend_total = probes.get("backend_slots_total")
    backend_busy = probes.get("backend_slots_busy")
    dispatcher_leases = max(0, int(probes.get("dispatcher_leases", 0) or 0))
    free: int | None
    if stale or backend_total is None or backend_busy is None:
        free = None
    else:
        backend_free = max(0, int(backend_total) - int(backend_busy))
        dispatcher_free = max(0, configured_slots - dispatcher_leases)
        free = min(configured_slots, backend_free, dispatcher_free)
        if free <= 0:
            block(ReasonCode.MEMBER_NO_CAPACITY, "no proved immediate slot")

    hard_unhealthy = any(
        reason in reasons
        for reason in (
            ReasonCode.MEMBER_SYSTEMD_INACTIVE,
            ReasonCode.MEMBER_HTTP_UNREADY,
            ReasonCode.MEMBER_PORT_DARK,
            ReasonCode.MEMBER_PROBE_TIMEOUT,
            ReasonCode.MEMBER_PROBE_DISCONNECTED,
            ReasonCode.MEMBER_UNHEALTHY_RESPONSE,
            ReasonCode.MEMBER_MODEL_NOT_RESIDENT,
            ReasonCode.TELEMETRY_STALE,
        )
    )
    accepting = not reasons and free is not None and free > 0
    if accepting:
        state = MemberState.READY_ACCEPTING
    elif set(reasons) == {ReasonCode.MEMBER_NO_CAPACITY}:
        # Exhausted capacity is normal steady-state load, not a lifecycle
        # transition. Publishing JOINING here lets observation overwrite an
        # active DRAINING fence as leases finish and breaks drain completion.
        state = MemberState.READY_BUSY
    elif hard_unhealthy:
        state = MemberState.UNHEALTHY
    else:
        state = MemberState.JOINING

    candidate = MemberSnapshot(
        name=name,
        gpu_id=gpu_id,
        state=state,
        accepting=accepting,
        configured_slots=configured_slots,
        context_per_slot=int(config.get("context_per_slot", 0) or 0),
        state_version=1,
        observed_at=observed_at,
        generation_fence=(previous.generation_fence if previous is not None else 0),
        backend_slots_total=int(backend_total) if backend_total is not None else None,
        backend_slots_busy=int(backend_busy) if backend_busy is not None else None,
        dispatcher_leases=dispatcher_leases,
        drain_disabled_slots=max(0, int(probes.get("drain_disabled_slots", 0) or 0)),
        compatible_free_slots=free,
        capabilities=tuple(config.get("capabilities", ())),
        idle_service_configured=idle_configured,
        idle_service_effective=idle_effective,
        systemd_active=systemd_active,
        main_pid=int(main_pid) if main_pid else None,
        semantic_health=semantic_health,
        model_resident=model_resident,
        dispatcher_registered=dispatcher_registered,
        snapshot_age_ms=snapshot_age_ms,
        blockers=tuple(blockers),
        reason_codes=tuple(reasons),
    )
    if previous is None:
        return candidate
    version = previous.state_version + (1 if _semantic(previous) != _semantic(candidate) else 0)
    return replace(candidate, state_version=version)
