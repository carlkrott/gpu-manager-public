"""Fail-closed adapter from llama-server payloads to broker readiness probes.

The adapter is deliberately pure: GPUManager owns the HTTP/systemd collection
and passes the observed JSON plus manager-owned facts here.  Missing, malformed,
or contradictory evidence never becomes a positive readiness claim.
"""
from __future__ import annotations

from math import isfinite
from pathlib import PurePath
from typing import Any, Mapping


def _classify_probe_error(error: Any) -> str | None:
    if error is None:
        return None
    if isinstance(error, TimeoutError):
        return "probe_timeout"
    if isinstance(error, ConnectionRefusedError):
        return "connection_refused"

    if isinstance(error, Mapping):
        # GPUManager and the standalone observer record independent probe
        # outcomes. Readiness needs one deterministic scalar for its state
        # machine, while diagnostics need the complete map (preserved by
        # member_probes_from_payloads). Keep systemd first because its
        # inactive state is independently represented by systemd_active/MainPID;
        # then prefer the HTTP health probe over capacity/model diagnostics.
        for key in ("systemd", "health", "slots", "models", "dispatcher"):
            classified = _classify_probe_error(error.get(key))
            if classified is not None:
                return classified
        for value in error.values():
            classified = _classify_probe_error(value)
            if classified is not None:
                return classified
        return "probe_unknown"

    name = type(error).__name__
    if name == "ClientConnectorError":
        os_error = getattr(error, "os_error", None)
        if isinstance(os_error, ConnectionRefusedError) or "Connection refused" in str(error):
            return "connection_refused"
        return "connection_error"
    if name == "ServerDisconnectedError":
        return "server_disconnected"
    if isinstance(error, str):
        normalized = error.strip().lower()
        if normalized == "connection_refused" or normalized == "connection_error":
            return normalized
        if normalized in {"probe_timeout", "server_disconnected", "systemd_error", "bad_json"}:
            return normalized
        if normalized == "unknown" or normalized.startswith("unknown:"):
            return "probe_unknown"
        if normalized == "probe_unknown":
            return normalized
        if normalized.startswith("http_") and normalized[5:].isdigit():
            return normalized
    return f"unknown:{name}"


def select_probe_error(probe_errors: Mapping[str, Any] | None) -> str | None:
    """Choose one stable scalar while retaining per-probe diagnostics."""
    if not isinstance(probe_errors, Mapping):
        return None
    for key in ("health", "slots", "models"):
        if key in probe_errors:
            return _classify_probe_error(probe_errors[key])
    for value in probe_errors.values():
        classified = _classify_probe_error(value)
        if classified is not None:
            return classified
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _model_matches(
    config: dict[str, Any],
    models: Any,
    *,
    observed_context: int | None = None,
) -> bool:
    expected_path = config.get("model_path")
    expected_context = _positive_int(config.get("context_per_slot"))
    if not isinstance(expected_path, str) or not expected_path or expected_context is None:
        return False
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        return False
    expected_name = PurePath(expected_path).name
    required_context = observed_context if config.get("cpu_only") is True else expected_context
    for model in models["data"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        meta = model.get("meta")
        reported_context = meta.get("n_ctx") if isinstance(meta, dict) else None
        if reported_context is None:
            reported_context = model.get("max_model_len")
        if (
            isinstance(model_id, str)
            and PurePath(model_id).name == expected_name
            and reported_context == required_context
        ):
            return True
    return False


def _slot_counts(
    config: dict[str, Any], slots: Any, health: Any
) -> tuple[int | None, int | None, int | None]:
    expected_context = _positive_int(config.get("context_per_slot"))
    configured_slots = _positive_int(config.get("parallel", config.get("worker_parallel")))
    cpu_only = config.get("cpu_only") is True
    if expected_context is None or configured_slots is None or not isinstance(slots, list):
        return None, None, None
    slots_to_check = slots if cpu_only else slots[:configured_slots]
    if not slots_to_check or (not cpu_only and len(slots) < configured_slots):
        return None, None, None
    contexts = [_positive_int(slot.get("n_ctx")) if isinstance(slot, dict) else None for slot in slots_to_check]
    if any(context is None for context in contexts):
        return None, None, None
    first_context = contexts[0]
    if first_context is None:
        return None, None, None
    observed_context = first_context
    if any(context != observed_context for context in contexts):
        return None, None, None
    if not cpu_only and observed_context != expected_context:
        return None, None, None

    processing = [slot.get("is_processing") if isinstance(slot, dict) else None for slot in slots_to_check]
    if all(type(value) is bool for value in processing):
        busy = sum(int(value) for value in processing)
    elif cpu_only and isinstance(health, dict):
        idle = health.get("slots_idle")
        busy_value = health.get("slots_processing")
        if (
            isinstance(idle, int)
            and not isinstance(idle, bool)
            and idle >= 0
            and isinstance(busy_value, int)
            and not isinstance(busy_value, bool)
            and busy_value >= 0
            and idle + busy_value == len(slots_to_check)
        ):
            busy = busy_value
        else:
            return None, None, None
    else:
        return None, None, None
    return len(slots_to_check), busy, observed_context


def _health_payload_ok(health: Any) -> bool:
    if not isinstance(health, dict):
        return False
    if health.get("status") == "ok":
        return True
    idle = health.get("slots_idle")
    processing = health.get("slots_processing")
    return (
        health.get("status") == "no slot available"
        and idle == 0
        and isinstance(processing, int)
        and not isinstance(processing, bool)
        and processing > 0
    )


def member_probes_from_payloads(
    *,
    config: dict[str, Any],
    payloads: dict[str, Any],
    manager_facts: dict[str, Any],
    observed_at: float,
) -> dict[str, Any]:
    """Return strict readiness probes from already-observed backend evidence.

    ``semantic_health`` requires all of: an explicit ``{"status": "ok"}``
    health payload, complete matching slot context evidence, and a matching
    loaded model/context entry. GPU residency requires that same model proof;
    CPU-only members keep the existing ``None`` residency convention.
    """
    if not isinstance(config, dict):
        config = {}
    if not isinstance(payloads, dict):
        payloads = {}
    if not isinstance(manager_facts, dict):
        manager_facts = {}

    health = payloads.get("health")
    slots = payloads.get("slots")
    models = payloads.get("models")
    total, busy, observed_context = _slot_counts(config, slots, health)
    model_match = _model_matches(
        config,
        models,
        observed_context=observed_context,
    )
    health_ok = _health_payload_ok(health)
    semantic_health = health_ok and model_match and total is not None
    raw_probe_error = manager_facts.get("probe_error")
    raw_probe_errors = manager_facts.get("probe_errors")
    probe_error = _classify_probe_error(raw_probe_error)
    if probe_error is None and isinstance(raw_probe_errors, Mapping):
        probe_error = select_probe_error(raw_probe_errors)
    cpu_only = config.get("cpu_only") is True or not config.get("gpu_id")

    main_pid = _positive_int(manager_facts.get("main_pid"))
    observed = float(observed_at) if isinstance(observed_at, (int, float)) and not isinstance(observed_at, bool) and isfinite(float(observed_at)) else 0.0
    result = {
        "semantic_health": semantic_health,
        "health_ok": health_ok,
        "probe_error": probe_error,
        "model_resident": None if cpu_only else model_match,
        "backend_slots_total": total,
        "backend_slots_busy": busy,
        "systemd_active": manager_facts.get("systemd_active") is True,
        "main_pid": main_pid,
        "idle_service_effective": None if cpu_only else manager_facts.get("idle_service_effective") is True,
        "dispatcher_registered": manager_facts.get("dispatcher_registered") is True,
        "dispatcher_leases": _nonnegative_int(manager_facts.get("dispatcher_leases")),
        "observed_at": observed,
    }
    if isinstance(raw_probe_error, Mapping):
        result["probe_errors"] = dict(raw_probe_error)
    elif isinstance(raw_probe_errors, Mapping):
        result["probe_errors"] = dict(raw_probe_errors)
    return result
