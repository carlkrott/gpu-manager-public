"""Facade-bound OOM telemetry policy helpers.

Function bodies are templates: external dependencies resolve in the facade
dictionary at call time. Binding creates independent functions for each facade
and preserves the facade's eager PSI threshold snapshot.
"""
from __future__ import annotations

from types import FunctionType

# Template default only; bind() replaces it with each facade snapshot.
_TASK8A_PSI_FULL_ABORT_PCT = 5.0

NAMES = (
    '_junction_crit_aborted',
    '_parse_psi_avg10',
    '_psi_full_stall_pct',
    '_parse_cgroup_memory_events',
    '_cgroup_events_oom',
    '_journal_timestamp',
    '_parse_prior_boot_thrash',
    '_prior_boot_oom_rate_per_minute',
    '_prior_boot_is_thrash',
    '_oom_counters_snapshot',
    '_watchdog_should_escalate',
    '_should_admit_generation',
    '_watchdog_ladder_snapshot',
)


def _rebuild(function, facade_globals):
    rebuilt = FunctionType(
        function.__code__, facade_globals, function.__name__,
        function.__defaults__, function.__closure__,
    )
    rebuilt.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    rebuilt.__annotations__ = function.__annotations__.copy()
    rebuilt.__doc__ = function.__doc__
    rebuilt.__qualname__ = function.__qualname__
    rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt


def _junction_crit_aborted(per_gpu) -> bool | tuple[str, str]:
    """Classify per-GPU junction evidence as critical, safe, or unknown."""
    if not isinstance(per_gpu, Mapping) or not per_gpu:
        return "unknown", "no_telemetry"
    telemetry_rows = [row for row in per_gpu.values() if isinstance(row, Mapping)]
    for telemetry in telemetry_rows:
        if is_junction_critical(telemetry):
            return True
    if not telemetry_rows:
        return "unknown", "no_telemetry"
    for telemetry in telemetry_rows:
        junction = telemetry.get("temp_junction_c")
        critical = telemetry.get("temp_junction_crit_c")
        if junction is None:
            return "unknown", "junction_channel_unreadable"
        if critical is None:
            return "unknown", "threshold_unknown"
        try:
            float(junction)
            float(critical)
        except (TypeError, ValueError):
            return "unknown", "junction_channel_unreadable"
    return False
def _parse_psi_avg10(psi_text: str) -> tuple[float | None, float | None]:
    some = None
    full = None
    if not isinstance(psi_text, str):
        return some, full
    for raw_line in psi_text.splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        target = fields[0]
        for field in fields[1:]:
            if not field.startswith("avg10="):
                continue
            try:
                value = float(field.split("=", 1)[1])
            except (TypeError, ValueError):
                value = None
            if target == "some":
                some = value
            elif target == "full":
                full = value
    return some, full
def _psi_full_stall_pct(
    psi_path: str,
    *,
    some_avg10=None,
    full_avg10=None,
    psi_text: str | None = None,
    abort_threshold_pct: float = _TASK8A_PSI_FULL_ABORT_PCT,
) -> dict:
    """Classify injected PSI memory evidence; never opens ``psi_path``."""
    del psi_path  # path is provenance only; I/O belongs to an authorized wrapper.
    if psi_text is not None:
        parsed_some, parsed_full = _parse_psi_avg10(psi_text)
        if some_avg10 is None:
            some_avg10 = parsed_some
        if full_avg10 is None:
            full_avg10 = parsed_full
    try:
        some = None if some_avg10 is None else float(some_avg10)
        full = None if full_avg10 is None else float(full_avg10)
    except (TypeError, ValueError):
        some, full = None, None
    result = {
        "some_pct": some,
        "full_pct": full,
        "abort_threshold_pct": float(abort_threshold_pct),
        "abort": full is None or full >= float(abort_threshold_pct),
    }
    if full is None:
        result["reason"] = "psi_full_unreadable"
    return result
def _parse_cgroup_memory_events(events_text: str) -> dict[str, int] | None:
    if not isinstance(events_text, str) or not events_text.strip():
        return None
    result: dict[str, int] = {}
    try:
        for line in events_text.splitlines():
            key, raw_value = line.split()
            value = int(raw_value)
            if value < 0:
                return None
            result[key] = value
    except (TypeError, ValueError):
        return None
    return result or None
def _cgroup_events_oom(per_unit) -> dict:
    """Return nonzero memory.events OOM/high deltas, unknown fail-closed."""
    if not isinstance(per_unit, Mapping):
        return {"any_oom": True, "delta": {"_input": {"unknown": 1}}}
    delta: dict[str, dict[str, int]] = {}
    for unit, events in per_unit.items():
        if not isinstance(events, Mapping):
            delta[str(unit)] = {"unknown": 1}
            continue
        nonzero = {
            key: int(events.get(key, 0) or 0)
            for key in ("high", "max", "oom", "oom_kill")
            if int(events.get(key, 0) or 0) > 0
        }
        if nonzero:
            delta[str(unit)] = nonzero
    return {"any_oom": bool(delta), "delta": delta}
def _journal_timestamp(line: str) -> str | None:
    match = re.match(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\b", line)
    return match.group(1) if match else None
def _parse_prior_boot_thrash(journal_block: str) -> dict:
    """Parse a caller-supplied prior-boot kernel journal block (no I/O)."""
    lines = journal_block.splitlines() if isinstance(journal_block, str) else []
    oom_markers: list[str] = []
    memcgs: list[str] = []
    page_alloc_failures = 0
    amdgpu_userptr = 0
    for line_number, line in enumerate(lines):
        lower = line.lower()
        is_oom = (
            "invoked oom-killer" in lower
            or "oom-kill:" in lower
            or "out of memory: killed process" in lower
        )
        if is_oom:
            timestamp = _journal_timestamp(line)
            # Timestamp is provenance only.  The line number is part of the
            # event key so same-second bursts remain distinct.
            oom_markers.append(timestamp or f"line:{line_number}")
        if "page allocation failure" in lower:
            page_alloc_failures += 1
        if "amdgpu_amdkfd_restore_userptr_worker" in lower:
            amdgpu_userptr += 1
        match = re.search(r"task_memcg=([^,\s]+)", line)
        if match:
            memcgs.append(match.group(1))
    counts = {memcg: memcgs.count(memcg) for memcg in set(memcgs)}
    dominant = max(counts, key=lambda key: (counts[key], key)) if counts else None
    first = oom_markers[0] if oom_markers else None
    last = oom_markers[-1] if oom_markers else None
    # Synthetic no-timestamp markers are useful for counting but are not valid
    # timestamp provenance.
    first = None if isinstance(first, str) and first.startswith("line:") else first
    last = None if isinstance(last, str) and last.startswith("line:") else last
    return {
        "oom_kills": len(oom_markers),
        "killed_processes": len(oom_markers),
        "page_alloc_failures": page_alloc_failures,
        "amdgpu_userptr": amdgpu_userptr,
        "first_oom_ts": first,
        "last_oom_ts": last,
        "distinct_memcgs": len(counts),
        "dominant_memcg": dominant,
    }
def _prior_boot_oom_rate_per_minute(parsed: Mapping) -> float:
    first = parsed.get("first_oom_ts")
    last = parsed.get("last_oom_ts")
    if not isinstance(first, str) or not isinstance(last, str):
        return 0.0
    try:
        from datetime import datetime
        start = datetime.strptime(f"2000 {first}", "%Y %b %d %H:%M:%S")
        end = datetime.strptime(f"2000 {last}", "%Y %b %d %H:%M:%S")
    except (TypeError, ValueError):
        return 0.0
    elapsed_minutes = (end - start).total_seconds() / 60.0
    if elapsed_minutes <= 0:
        return 0.0
    return float(parsed.get("oom_kills", 0) or 0) / elapsed_minutes
def _prior_boot_is_thrash(parsed) -> bool:
    if not isinstance(parsed, Mapping) or not parsed:
        return True
    try:
        return (
            int(parsed.get("oom_kills", 0) or 0) > 10
            or int(parsed.get("page_alloc_failures", 0) or 0) > 0
            or int(parsed.get("distinct_memcgs", 0) or 0) > 3
            or _prior_boot_oom_rate_per_minute(parsed) > 0.5
        )
    except (TypeError, ValueError):
        return True
def _oom_counters_snapshot(
    restart_signals_fn=None,
    *,
    _read_restart_signals_fn=None,
    now: float | None = None,
) -> dict:
    """Join existing OOM scopes with injected systemd restart evidence."""
    sample_ts = time.time() if now is None else float(now)
    reader = restart_signals_fn or _read_restart_signals_fn
    health = _oom_health_section(now=sample_ts)
    restart_signals: dict = {
        "scope": "systemd",
        "sample_ts": sample_ts,
        "complete": reader is not None,
        "read_errors": {},
    }
    for unit in _TASK8A_RESTART_UNITS:
        normalized = _normalized_unit_name(unit)
        data = None
        if reader is not None:
            try:
                candidate = reader(normalized)
                if isinstance(candidate, Mapping):
                    data = candidate
                else:
                    raise ValueError("restart reader returned non-mapping")
            except Exception as exc:
                restart_signals["read_errors"][normalized] = str(exc)
                restart_signals["complete"] = False
        else:
            restart_signals["read_errors"][normalized] = "reader_not_injected"
        raw_restarts = data.get("NRestarts") if data is not None else None
        restart_usec = data.get("RestartUSec") if data is not None else None
        if data is None:
            count = None
        elif isinstance(raw_restarts, bool) or raw_restarts is None:
            count = None
            restart_signals["read_errors"][normalized] = "invalid_NRestarts"
            restart_signals["complete"] = False
        else:
            try:
                count = int(raw_restarts)
            except (TypeError, ValueError):
                count = None
                restart_signals["read_errors"][normalized] = "invalid_NRestarts"
                restart_signals["complete"] = False
        restart_signals[f"nrestarts_{normalized}"] = count
        restart_signals[f"restart_usec_{normalized}"] = (
            restart_usec if count is not None else None
        )
    return {
        "process_events": health["process_events"],
        "dead_letter_retained": health["dead_letter_retained"],
        "watchdog": health["watchdog"],
        "restart_signals": restart_signals,
    }
def _watchdog_should_escalate(
    counters,
    prior_boot,
    junction,
    psi,
    cgroup,
) -> tuple[bool, str]:
    if not isinstance(counters, Mapping) or not isinstance(
        counters.get("process_events"), Mapping
    ):
        return True, "counters_unknown"
    process = counters["process_events"]
    if not isinstance(process.get("total_events"), (int, float)) or not isinstance(
        process.get("active_recoveries"), Mapping
    ):
        return True, "counters_unknown"
    active = process["active_recoveries"]
    active_state = next(
        (
            state
            for state in active.values()
            if state in {"detected", "draining", "verifying", "recovering"}
        ),
        None,
    )
    if int(process.get("total_events", 0) or 0) > 0 and active_state:
        return True, "oom_state_detected" if active_state == "detected" else "oom_recovery_active"
    if prior_boot is None:
        return True, "prior_boot_unknown"
    prior_thrash = (
        prior_boot
        if isinstance(prior_boot, bool)
        else _prior_boot_is_thrash(prior_boot)
    )
    if prior_thrash:
        return True, "prior_boot_thrash"
    junction_verdict = _junction_crit_aborted(
        {"watchdog": junction} if isinstance(junction, Mapping) else {}
    )
    if junction_verdict is True:
        return True, "junction_critical"
    if junction_verdict is not False:
        return True, "junction_unknown"
    try:
        full_pct = psi.get("full_pct")
        if full_pct is None:
            return True, "psi_unknown"
        if float(full_pct) >= _TASK8A_PSI_FULL_ABORT_PCT:
            return True, "psi_full_stall"
    except (AttributeError, TypeError, ValueError):
        return True, "psi_unknown"
    if not isinstance(cgroup, Mapping) or not isinstance(cgroup.get("any_oom"), bool):
        return True, "cgroup_unknown"
    if cgroup.get("any_oom") is True:
        return True, "cgroup_oom_increment"
    return False, "clear"
def _should_admit_generation(cfg, *, junction, psi, prior_boot) -> bool:
    """Pure fail-closed generation admission gate for Task 8A evidence."""
    if not isinstance(cfg, Mapping):
        return False
    oom = cfg.get("oom")
    if not isinstance(oom, Mapping) or not isinstance(oom.get("any_oom"), bool):
        return False
    if oom.get("any_oom"):
        return False
    junction_verdict = _junction_crit_aborted(
        {"admission": junction} if isinstance(junction, Mapping) else {}
    )
    if junction_verdict is not False:
        return False
    if not isinstance(psi, Mapping) or psi.get("full_pct") is None:
        return False
    try:
        if float(psi["full_pct"]) >= _TASK8A_PSI_FULL_ABORT_PCT:
            return False
    except (TypeError, ValueError):
        return False
    if prior_boot is not False:
        return False
    return True
def _watchdog_ladder_snapshot(
    gpu_id: str,
    *,
    restart_ts: float | None = None,
    state_timestamps: Mapping | None = None,
) -> list[tuple[str, float]]:
    """Return a replayable monotonic escalation ladder from injected times."""
    del gpu_id  # identity is carried by the caller; this helper has no I/O.
    anchor = _scheduler_started_at if restart_ts is None else float(restart_ts)
    supplied = state_timestamps if isinstance(state_timestamps, Mapping) else {}
    states = (
        "idle", "detected", "draining", "verifying", "recovering",
        "restored", "permanent_failed",
    )
    result: list[tuple[str, float]] = []
    last = anchor
    for state in states:
        candidate = supplied.get(state, anchor if state == "idle" else last)
        try:
            since = max(last, float(candidate))
        except (TypeError, ValueError):
            since = last
        result.append((state, since))
        last = since
    result[0] = ("idle", anchor)
    return result



def bind(facade_globals):
    """Return the OOM telemetry functions bound to one facade namespace."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    bound = []
    for name in NAMES:
        function = _rebuild(namespace[name], facade_globals)
        if name == "_psi_full_stall_pct":
            # Its eager default belonged to the facade at the original definition
            # boundary; snapshot it once for this bound function.
            function.__kwdefaults__["abort_threshold_pct"] = facade_globals[
                "_TASK8A_PSI_FULL_ABORT_PCT"
            ]
        bound.append(function)
    return tuple(bound)
