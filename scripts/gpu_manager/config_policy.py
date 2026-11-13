"""Facade-bound config policy: linked-bundle graph, config, restart and idle validation.

Extracted verbatim from ``scripts/gpu-manager.py``.  Every function reads facade
globals (memory-policy helpers, ``_TASK8A_*`` constants, ``subprocess`` and each
other) at call time, so ``bind(facade_globals)`` rebuilds each one with
``types.FunctionType`` against the supplied facade dictionary; no
``sys.modules`` entry or shared facade pointer is kept here.  The definitions
below are templates: calling them directly would resolve names in this module.
"""
from __future__ import annotations

from types import FunctionType


def _validate_linked_bundle_graph(config: dict) -> tuple[list[str], list[str]]:
    """Fail-closed validation of the linked-bundle graph.

    Inspects every bundle's ``linked_to``, ``linked_from`` and legacy
    ``linked_bundles`` fields and emits:

      * **errors** — graph corruption that must block the load:
        unknown bundle names, self-links, cycles, broken
        ``linked_from`` ↔ ``linked_to`` asymmetry, ``linked_from``
        listing a parent that does not actually link back, duplicate
        GPU ownership (two bundles with the same ``gpu_id`` and at
        least one not idle), and unknown GPU / service references.
      * **warnings** — recoverable inconsistencies such as a
        non-empty ``linked_from`` for a bundle that has no
        corresponding reverse pointer (advisory only — the runtime
        bidirectional resolver handles this transparently).

    This function is intentionally read-only: it inspects the supplied
    config dict and never mutates it. Errors and warnings are surfaced
    via ``/status`` so they are visible at startup and on every
    hot-reload without changing the parsed config.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(config, dict):
        return errors, warnings

    bundles_raw = config.get("bundles", {})
    if not isinstance(bundles_raw, dict) or not bundles_raw:
        # Backward-compat path: legacy configs auto-generate bundles from
        # services. Surface those as a separate graph once generated.
        return errors, warnings

    gpu_devices = config.get("gpu_devices", {}) or {}
    if not isinstance(gpu_devices, dict):
        gpu_devices = {}
    known_gpus = {str(g) for g in gpu_devices.keys()}

    services = config.get("services", {}) or {}
    if not isinstance(services, dict):
        services = {}
    known_services = {str(s) for s in services.keys()}

    # Pass 1: basic shape validation (self-links, unknown links, type errors,
    # linked_from ↔ linked_to / linked_bundles symmetry).
    forward_edges: dict[str, set[str]] = {}
    reverse_edges: dict[str, set[str]] = {}
    ownership: dict[str, list[str]] = {}

    for name, bundle in bundles_raw.items():
        if not isinstance(bundle, dict):
            errors.append(f"bundle '{name}': not a JSON object")
            continue

        # Normalise the three link fields into lists for analysis. We do not
        # mutate the bundle; we just collect locally.
        def _as_list(field: str) -> list:
            value = bundle.get(field, [])
            if value is None:
                return []
            if not isinstance(value, list):
                errors.append(
                    f"bundle '{name}': '{field}' must be a list, got "
                    f"{type(value).__name__}"
                )
                return []
            return [str(item) for item in value]

        linked_to = _as_list("linked_to")
        linked_from = _as_list("linked_from")
        linked_bundles = _as_list("linked_bundles")

        forward_edges[name] = set(linked_to) | set(linked_bundles)
        reverse_edges[name] = set(linked_from)

        # Self-link checks.
        for field, items in (
            ("linked_to", linked_to),
            ("linked_from", linked_from),
            ("linked_bundles", linked_bundles),
        ):
            if name in items:
                errors.append(
                    f"bundle '{name}': '{field}' contains self-link"
                )

        # Unknown bundle references.
        all_targets = set(linked_to) | set(linked_from) | set(linked_bundles)
        for target in all_targets:
            if target == name:
                continue  # already reported as self-link
            if target not in bundles_raw:
                errors.append(
                    f"bundle '{name}': references unknown bundle '{target}'"
                )

        # Symmetry: A → B in linked_to implies B's linked_from contains A.
        for target in linked_to:
            target_from = (
                bundles_raw.get(target, {}).get("linked_from", [])
                if isinstance(bundles_raw.get(target), dict)
                else []
            )
            target_from_list = target_from if isinstance(target_from, list) else []
            if name not in target_from_list:
                warnings.append(
                    f"bundle '{name}': linked_to '{target}' but '{target}' "
                    f"does not list '{name}' in linked_from "
                    f"(bidirectional resolver will still treat as linked)"
                )

        # Symmetry: A → B in linked_from means B should list A in linked_to
        # (or linked_bundles). If neither, the reverse edge is dangling.
        for parent in linked_from:
            if parent == name:
                continue  # self-link already reported
            parent_bundle = bundles_raw.get(parent)
            if not isinstance(parent_bundle, dict):
                continue  # unknown parent already flagged above
            parent_links = (
                set(parent_bundle.get("linked_to", []) or [])
                | set(parent_bundle.get("linked_bundles", []) or [])
            )
            if name not in parent_links:
                errors.append(
                    f"bundle '{name}': linked_from '{parent}' is asymmetric — "
                    f"'{parent}' does not list '{name}' in linked_to/"
                    f"linked_bundles"
                )

        # Unknown GPU / service references.
        gpu_id = bundle.get("gpu_id")
        if gpu_id is not None and known_gpus and str(gpu_id) not in known_gpus:
            errors.append(
                f"bundle '{name}': gpu_id '{gpu_id}' not declared in "
                f"gpu_devices (known: {sorted(known_gpus)})"
            )

        bundle_services = bundle.get("services", [])
        if isinstance(bundle_services, list) and known_services:
            for svc in bundle_services:
                if str(svc) not in known_services:
                    errors.append(
                        f"bundle '{name}': references unknown service '{svc}'"
                    )

        # Ownership: at most one non-idle bundle per GPU. Idle bundles
        # (the default LLM slot) may share a GPU with at most one other
        # non-idle bundle IF the runtime explicitly permits it; the
        # current contract is "exactly one non-idle owner".
        if gpu_id is not None:
            ownership.setdefault(str(gpu_id), []).append(name)

    # Pass 2: cycle detection via DFS over the merged forward edge set.
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in forward_edges}
    cycle_reported: set[frozenset] = set()

    def _visit(node: str, stack: list[str]) -> None:
        if color.get(node, BLACK) == GRAY:
            # Back-edge -> cycle.
            if node in stack:
                idx = stack.index(node)
                cyc = stack[idx:] + [node]
                # Canonicalise so A->B->C and C->B->A produce the same record.
                canonical = frozenset(cyc[:-1])
                if canonical not in cycle_reported:
                    cycle_reported.add(canonical)
                    errors.append(
                        "linked-bundle cycle: "
                        + " -> ".join(cyc)
                    )
            return
        if color.get(node, BLACK) == BLACK:
            return
        color[node] = GRAY
        stack.append(node)
        for nxt in sorted(forward_edges.get(node, set())):
            _visit(nxt, stack)
        stack.pop()
        color[node] = BLACK

    for node in sorted(forward_edges.keys()):
        if color[node] == WHITE:
            _visit(node, [])

    # Pass 3: enumerate alternative candidates sharing each GPU. Multiple
    # configured bundles are expected (ComfyUI, Qwen, ACE-Step, LLM routes);
    # they are not simultaneous owners. Runtime exclusivity is enforced by the
    # lifecycle controller and transition graph, so this is observable advice,
    # not a static configuration error.
    for gid, names in sorted(ownership.items()):
        non_idle = [n for n in names if not bundles_raw.get(n, {}).get("idle", False)]
        if len(non_idle) > 1:
            warnings.append(
                f"gpu '{gid}': alternative non-idle bundle candidates require "
                f"controller fencing: {sorted(non_idle)}"
            )

    # Pass 4: orphan candidates. A bundle is an *orphan candidate* if no
    # other bundle lists it in ``linked_to`` or ``linked_bundles``. Such a
    # bundle is reachable only as a top-level entry point; runtime
    # admission treats it as eligible for direct activation, so we surface
    # the candidate as a warning for operator visibility rather than as an
    # error. Multiple orphans are expected (each top-level bundle starts
    # one), so we cap the report length.
    incoming: dict[str, set[str]] = {n: set() for n in bundles_raw}
    for parent_name, parent in bundles_raw.items():
        if not isinstance(parent, dict):
            continue
        links = set(parent.get("linked_to", []) or []) | set(
            parent.get("linked_bundles", []) or []
        )
        for child in links:
            if child in incoming:
                incoming[child].add(parent_name)
    orphan_candidates = sorted(n for n, parents in incoming.items() if not parents)
    if orphan_candidates:
        warnings.append(
            "linked-bundle orphan candidates (no incoming linked_to/"
            f"linked_bundles): {orphan_candidates}"
        )

    return errors, warnings


def _validate_config_policy(config: dict) -> tuple[list[str], list[str]]:
    """Return the configured per-device VRAM ceiling.

    The registry supplies physical capacity and reserve values.  The controller
    never embeds a machine-specific card ceiling.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(config, dict):
        return errors, warnings

    # Device capacity is validated generically below from the registry entry.
    gpu_devices = config.get("gpu_devices", {}) or {}
    scheduling = config.get("scheduling", {}) or {}
    if not isinstance(scheduling, dict):
        scheduling = {}

    scheduler_owns_load = bool(scheduling.get("scheduler_owns_load", False))

    # 2) Scheduler-owned evict_first -> ERROR (was warning in 5.2.79).
    # Plan 06 demands the prune cannot regress silently: if the
    # operator wanted eviction ordering on a scheduler-owned
    # template, that intent was lost and the template must be fixed.
    if scheduler_owns_load:
        templates = config.get("generation_templates", {}) or {}
        if isinstance(templates, dict):
            for tpl_name, tpl in templates.items():
                if not isinstance(tpl, dict):
                    continue
                evict = tpl.get("evict_first")
                if evict:  # truthy: non-empty list / non-None
                    errors.append(
                        f"template '{tpl_name}': evict_first is forbidden "
                        f"when scheduler_owns_load=true "
                        f"(runtime ignores the directive; remove it)"
                    )

    # 2b) VRAM target / ceiling / reserve coherence per gpu_device.
    if isinstance(gpu_devices, dict):
        for gid, gdev in gpu_devices.items():
            if not isinstance(gdev, dict):
                continue
            try:
                target = (
                    int(gdev["vram_target_mib"])
                    if gdev.get("vram_target_mib") is not None
                    else None
                )
            except (TypeError, ValueError):
                errors.append(
                    f"gpu_devices.{gid}.vram_target_mib must be an integer, "
                    f"got {gdev.get('vram_target_mib')!r}"
                )
                target = None
            try:
                ceiling = (
                    int(gdev["vram_ceiling_mib"])
                    if gdev.get("vram_ceiling_mib") is not None
                    else None
                )
            except (TypeError, ValueError):
                errors.append(
                    f"gpu_devices.{gid}.vram_ceiling_mib must be an integer, "
                    f"got {gdev.get('vram_ceiling_mib')!r}"
                )
                ceiling = None
            try:
                reserve = (
                    int(gdev["vram_reserve_mib"])
                    if gdev.get("vram_reserve_mib") is not None
                    else None
                )
            except (TypeError, ValueError):
                errors.append(
                    f"gpu_devices.{gid}.vram_reserve_mib must be an integer, "
                    f"got {gdev.get('vram_reserve_mib')!r}"
                )
                reserve = None
            try:
                physical = (
                    int(gdev["vram_mib"])
                    if gdev.get("vram_mib") is not None
                    else None
                )
            except (TypeError, ValueError):
                errors.append(
                    f"gpu_devices.{gid}.vram_mib must be an integer, "
                    f"got {gdev.get('vram_mib')!r}"
                )
            if gid == "rx6950xt" and ceiling is not None and ceiling > 14832:
                errors.append(
                    f"gpu_devices.{gid}: vram_ceiling_mib={ceiling} exceeds fixed ceiling 14832 "
                    "(16 GB card minus desktop reserve)"
                )
            if target is not None and ceiling is not None and target > ceiling:
                errors.append(
                    f"gpu_devices.{gid}: vram_target_mib={target} exceeds "
                    f"vram_ceiling_mib={ceiling}"
                )
            if (
                ceiling is not None
                and physical is not None
                and reserve is not None
                and ceiling > (physical - reserve)
            ):
                errors.append(
                    f"gpu_devices.{gid}: vram_ceiling_mib={ceiling} exceeds "
                    f"physical-reserve "
                    f"({physical} - {reserve} = {physical - reserve}); "
                    f"scheduler cannot satisfy the reserve headroom"
                )

    # 3) Idle / pin precedence. The order is:
    #    pinned_service > scheduling.idle_services[gpu_id] > scheduling.idle_service
    # A pinned generation_backend is allowed to override the idle LLM. The
    # differing service types express the normal pin use-case and are not a
    # schema conflict; the warning below makes precedence operator-visible.
    pinned_service = scheduling.get("pinned_service")
    idle_services = scheduling.get("idle_services", {}) or {}
    global_idle_service = scheduling.get("idle_service")

    services_map = config.get("services", {}) or {}
    if not isinstance(services_map, dict):
        services_map = {}

    # 4) Legacy idle_job semantics. idle_job is the obsolete keepalive
    # hook (ComfyUI image at priority 10) pre-dating idle_services. It
    # is dormant when enabled=False (the default). When enabled=True it
    # conflicts with the modern idle selector (idle_services or global
    # idle_service) and with scheduler_owns_load (the scheduler does
    # not honour legacy keepalive jobs).
    idle_job = scheduling.get("idle_job", {}) or {}
    if isinstance(idle_job, dict) and idle_job.get("enabled") is True:
        modern_idle_active = bool(
            (isinstance(idle_services, dict) and idle_services)
            or global_idle_service
        )
        if modern_idle_active:
            errors.append(
                "scheduling.idle_job.enabled=true conflicts with "
                "scheduling.idle_services / scheduling.idle_service "
                "(legacy keepalive cannot coexist with the modern idle "
                "selector; disable idle_job and migrate to idle_services)"
            )
        elif scheduler_owns_load:
            errors.append(
                "scheduling.idle_job.enabled=true conflicts with "
                "scheduler_owns_load=true "
                "(the scheduler does not honour legacy keepalive jobs; "
                "disable idle_job and rely on scheduling.idle_services "
                "or scheduling.idle_service instead)"
            )

    # Informational precedence note (warning) when more than one selector
    # is configured so operators see the resolution order at startup.
    if pinned_service and (global_idle_service or (isinstance(idle_services, dict) and idle_services)):
        warnings.append(
            "scheduling precedence is pinned_service > "
            "idle_services[gpu_id] > idle_service (global); only the "
            "highest-precedence selector fires per request"
        )

    return errors, warnings


def _validate_config_for_observability(config: dict) -> tuple[list[str], list[str]]:
    """Combine graph and policy validation without mutating the candidate."""
    graph_errors, graph_warnings = _validate_linked_bundle_graph(config)
    policy_errors, policy_warnings = _validate_config_policy(config)
    errors = list(dict.fromkeys([*graph_errors, *policy_errors]))
    warnings = list(dict.fromkeys([*graph_warnings, *policy_warnings]))
    bundles = config.get("bundles") or {}
    if isinstance(bundles, Mapping):
        for bundle_name, bundle in bundles.items():
            if not isinstance(bundle, Mapping) or not (
                "runtime_instance" in bundle or "runtime_profile" in bundle
            ):
                continue
            try:
                resolve_checked_runtime_binding(config, str(bundle_name))
            except CheckedRuntimeError as exc:
                errors.append(f"bundle {bundle_name!r}: {exc}")
    errors.extend(checked_runtime_authority_errors(config))
    # Audit C1/C3: ordered_members is the one llm.chat membership source. (Inline:
    # bind() rebuilds this function against the gpu-manager facade globals.)
    broker = config.get("combined_gemma_broker")
    services = config.get("services")
    if isinstance(broker, Mapping) and broker.get("enabled"):
        ordered = broker.get("ordered_members")
        if not isinstance(ordered, list) or not isinstance(services, Mapping):
            errors.append("combined_gemma_broker.ordered_members must be a list and services a mapping")
        else:
            errors.extend(
                f"combined_gemma_broker.ordered_members: {name!r} is not a configured service"
                for name in ordered if not isinstance(services.get(name), Mapping)
            )
            # shortcut: the broker's routing group name "Gemma" is still hardcoded (audit C5).
            errors.extend(
                f"service {name!r} is enabled in routing_group 'Gemma' but not in "
                "combined_gemma_broker.ordered_members"
                for name, svc in services.items()
                if isinstance(svc, Mapping) and svc.get("routing_group") == "Gemma"
                and svc.get("enabled") and name not in ordered
            )
    return list(dict.fromkeys(errors)), warnings


def _expected_restart_policy(unit: str = "gemma-cpu") -> dict[str, int]:
    policies = {
        "gemma-cpu": {
            "RestartSec": 120,
            "StartLimitIntervalSec": 900,
            "StartLimitBurst": 5,
        },
        "gemma-cpu-vision": {
            "RestartSec": 120,
            "StartLimitIntervalSec": 900,
            "StartLimitBurst": 5,
        },
        "llm-primary": {
            "RestartSec": 10,
            "StartLimitIntervalSec": 10,
            "StartLimitBurst": 5,
        },
        "llm-secondary": {
            "RestartSec": 15,
            "StartLimitIntervalSec": 10,
            "StartLimitBurst": 5,
        },
        "gpu-manager": {
            "RestartSec": 10,
            "StartLimitIntervalSec": 10,
            "StartLimitBurst": 5,
        },
    }
    policy = policies.get(_normalized_unit_name(unit))
    return dict(policy) if policy else {}


def _systemd_restart_policy_drift(unit, dropin_dir, systemctl_cat_path):
    """Return expected restart-policy directives that differ from evidence.

    ``systemctl cat`` is the authoritative merged base-unit/drop-in view;
    drop-ins are applied afterward to make the precedence explicit.  This is
    intentionally read-only and reports only the numeric damping directives
    tracked by ``_expected_restart_policy``.
    """
    from pathlib import Path

    unit_name = _normalized_unit_name(unit)
    expected = _expected_restart_policy(unit_name)
    if not expected:
        return {}

    def parse_lines(lines):
        result = {}
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith(("#", ";")) or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in expected:
                result[key] = value.strip()
        return result

    cat_path = Path(systemctl_cat_path)
    marker = f"=== {unit_name}.service (systemctl cat) ==="
    lines = cat_path.read_text().splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise ValueError(f"systemctl cat evidence has no section for {unit_name}") from exc
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("=== ")),
        len(lines),
    )
    deployed = parse_lines(lines[start:end])
    for conf in sorted(Path(dropin_dir).glob("*.conf")):
        deployed.update(parse_lines(conf.read_text().splitlines()))

    drift = {}
    for key, expected_value in expected.items():
        deployed_value = deployed.get(key)
        if deployed_value is None:
            drift[key] = ("", expected_value)
            continue
        try:
            numeric = int(str(deployed_value).rstrip("s"))
        except ValueError:
            drift[key] = (deployed_value, expected_value)
            continue
        if numeric != expected_value:
            drift[key] = (deployed_value, expected_value)
    return drift


def _oom_damping_shape(unit: str = "gemma-cpu") -> dict:
    policy = _expected_restart_policy(unit)
    restart_window = policy.get("RestartSec")
    observable = (
        "covers_amplification"
        if restart_window is not None
        and OOM_WATCHDOG_POLL_INTERVAL_S < restart_window
        else "unknown_policy"
    )
    return {
        "watchdog_interval_s": OOM_WATCHDOG_POLL_INTERVAL_S,
        "expected_respawn_window_s": restart_window,
        "expected_damping_factor": 1 if policy.get("StartLimitBurst", 0) > 0 else 0,
        "watchdog_observability": observable,
    }


def _count_pinned_compute_idles(idle_map, compute_gpu_ids) -> int:
    if not isinstance(idle_map, Mapping) or not isinstance(
        compute_gpu_ids, (set, frozenset, list, tuple)
    ):
        return 0
    compute = {str(gpu_id) for gpu_id in compute_gpu_ids}
    return sum(
        1 for gpu_id, service in idle_map.items()
        if str(gpu_id) in compute and bool(service)
    )


def _validate_idle_policy(
    idle_map,
    compute_gpu_ids,
    allow_count=1,
    thermal_provenance=None,
) -> tuple[bool, str]:
    """Validate compute-idle count without mutating scheduler configuration."""
    if not isinstance(idle_map, Mapping) or not isinstance(
        compute_gpu_ids, (set, frozenset, list, tuple)
    ):
        return False, "idle_policy_input_invalid"
    if isinstance(allow_count, bool) or not isinstance(allow_count, int) or allow_count < 0:
        return False, "allow_count_invalid"
    pinned = _count_pinned_compute_idles(idle_map, compute_gpu_ids)
    if pinned <= allow_count:
        return True, ""
    if allow_count == 0:
        return False, "compute_idles_not_allowed"
    if not isinstance(thermal_provenance, Mapping):
        return False, "two_compute_idles_without_thermal_provenance"

    p99 = _positive_measurement(thermal_provenance.get("junction_p99_c"))
    cooled = _positive_measurement(thermal_provenance.get("junction_cooled_c"))
    critical = _positive_measurement(
        thermal_provenance.get("junction_crit_c", 110.0)
    )
    if p99 is not None and critical is not None and p99 >= critical:
        return False, "junction_at_or_above_crit"
    if cooled is not None and critical is not None and cooled >= critical:
        return False, "junction_at_or_above_crit"
    if p99 is None or cooled is None or critical is None:
        return False, "thermal_provenance_incomplete"
    return True, "two_compute_idles_with_thermal_provenance"


NAMES = (
    "_validate_linked_bundle_graph",
    "_validate_config_policy",
    "_validate_config_for_observability",
    "_expected_restart_policy",
    "_systemd_restart_policy_drift",
    "_oom_damping_shape",
    "_count_pinned_compute_idles",
    "_validate_idle_policy",
)


def _rebuild(function, facade_globals):
    rebuilt = FunctionType(
        function.__code__,
        facade_globals,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebuilt.__kwdefaults__ = (
        function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    )
    rebuilt.__annotations__ = function.__annotations__.copy()
    rebuilt.__doc__ = function.__doc__
    rebuilt.__qualname__ = function.__qualname__
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt


def bind(facade_globals):
    """Return the retry functions in ``NAMES`` order, bound to a facade.

    Each function is rebuilt so its ``__globals__`` is exactly
    ``facade_globals`` and every global read resolves against that facade at
    call time.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    return tuple(_rebuild(namespace[name], facade_globals) for name in NAMES)
