"""Facade-bound scheduler helpers: head-job prechecks, target resolution, load evaluation and dispatch.

Extracted verbatim from ``scripts/gpu-manager.py``.  Every function reads facade
globals (``logger``, ``_load_history``, ``_services_config``, the scheduler-state
helpers from ``gpu_manager.scheduler_state`` and each other) at call time, so
``bind(facade_globals)`` rebuilds each one with ``types.FunctionType`` against
the supplied facade dictionary.  No ``sys.modules`` entry or shared facade pointer
is kept here, and all mutable state stays facade-owned.  The scheduler-state
names are therefore not imported: they resolve through the facade dictionary
like every other global.  The definitions below are templates: calling them
directly would resolve names in this module.
"""
from __future__ import annotations

from types import FunctionType


def _record_load_history(decision):
    _load_history.insert(0, decision)
    if len(_load_history) > 50:
        _load_history.pop()


def _record_head_job_precheck(record):
    with _head_job_pre_checks_lock:
        _head_job_pre_checks.insert(0, record)
        if len(_head_job_pre_checks) > 20:
            _head_job_pre_checks.pop()


def _group_circuit_state(group, now, config):
    return _group_failure_state.get(group, {'state': 'closed'})


def _active_group_still_draining(target_group: str) -> bool:
    """Dry-run check: is any other active group still draining queue lag?

    When the scheduler sees demand:<target_group>, but a different active
    routing group still reports queue_lag > 0, we hold the load so the
    active group can drain its accepted/cutoff queue before swap.
    """
    for rg, cutoff in _drain_cutoffs.copy().items():
        if rg == target_group:
            continue
        try:
            if (time.time() - float(cutoff)) <= 60:
                return True
        except ValueError:
            return False
    return False


def _target_gpus_hold_warm_bundle(target_group: str) -> bool:
    """Advisory telemetry only.

    Returns True only when at least one target GPU for ``target_group`` is
    holding a bundle that is NOT the target bundle and NOT a valid linked
    component of it. Returns False when:

      * no target GPUs can be derived from a reservation or template/bundle
        topology, or
      * every target GPU is either idle or already holds the target bundle
        (including any valid linked bundle component).

    ``requires_eviction`` is **advisory, factual decision telemetry**: it tells
    operators whether a swap will be required for the upcoming load. It is NOT
    a second control path in the dispatcher. ``_evict_then_load_bundle`` is the
    authoritative atomic state-transition / readiness path and runs its own
    priority-aware precheck. See ``_dispatch_scheduler_decision``.

    NOTE: this function deliberately swallows exceptions from the GPU state
    read and converts unreadable state into a "warm, unknown" signal so the
    caller can choose to fail closed. The fail-closed path is implemented in
    ``_scheduler_evaluate_load`` (which switches to a non-dispatch action when
    state cannot be inspected) rather than here, so this helper remains a
    pure telemetry predicate.
    """
    grp_res = {}
    try:
        grp_res = _group_reservations.get(target_group, {}) or {}
    except Exception:
        grp_res = {}
    target_gpus: list[str] = []
    if isinstance(grp_res, dict) and grp_res.get('requires_gpus'):
        for gid in grp_res.get('requires_gpus') or []:
            if gid and gid not in target_gpus:
                target_gpus.append(gid)
    if not target_gpus:
        # Fall back to a config-driven derivation rather than hard-coded names:
        # the proactive loop must not silently default to "warm, no eviction"
        # when it can't see reservation state.
        target_gpus = list(_resolve_group_target_gpus(target_group) or [])
    else:
        # Reservation slice is non-empty — UNION with topology so a partial or
        # stale reservation can NEVER hide a linked gpu that needs warm-state
        # inspection. Without this, a partial reservation that omits the linked
        # gpu would silently mask an evictor resident on that gpu.
        for gid in list(_resolve_group_target_gpus(target_group) or []):
            if gid and gid not in target_gpus:
                target_gpus.append(gid)
    if not target_gpus:
        return False

    target_bundle, linked_bundles = _resolve_target_bundle_components(target_group)
    accepted_names = set()
    if target_bundle:
        accepted_names.add(target_bundle)
        accepted_names.update(linked_bundles or [])

    for gid in target_gpus:
        try:
            gs = _get_gpu_state(gid)
        except Exception:
            # Fail-closed signal: the caller will see this via
            # _scheduler_evaluate_load's state-unreadable branch and emit a
            # non-dispatch action. Returning False here would falsely advertise
            # "cold, no eviction" for a GPU whose state we cannot inspect.
            raise _TargetStateUnreadable(target_group, gid)
        warm = getattr(gs, 'loaded_bundle', None) or getattr(gs, 'current_bundle', None)
        if not warm:
            continue
        if accepted_names and warm in accepted_names:
            # Same target bundle (or a valid linked component) — not an eviction.
            continue
        return True
    return False


def _resolve_target_bundle_components(target_group: str) -> tuple[str | None, list[str]]:
    """Resolve (target_bundle_name, [linked_bundle_names]) for a routing group.

    Reads ``_services_config.generation_templates`` for the template that maps
    to ``target_group`` and pulls the bundle's ``linked_to`` chain from the
    in-memory ``_services_config.bundles``.

    If the in-memory ``_services_config.bundles`` section is absent, falls back
    to the cached/disk ``_resolve_bundles()`` resolver — same safe-fallback
    policy used by ``_resolve_group_target_gpus``. The fallback MUST NOT
    mutate the caller-provided ``_services_config`` (monkeypatched in tests);
    it only refreshes the bundle cache that ``_resolve_bundles`` owns.

    Returns ``(None, [])`` if the group has no template/bundle or if the
    bundle cannot be resolved via either source.
    """
    templates = ((_services_config or {}).get('generation_templates') or {})
    bundle_name: str | None = None
    for tmpl in templates.values():
        if not isinstance(tmpl, dict):
            continue
        if tmpl.get('routing_group') == target_group:
            cand = tmpl.get('bundle')
            if cand and bundle_name is None:
                bundle_name = str(cand)
            elif cand and bundle_name != str(cand):
                # Ambiguous mapping for a single group → refuse to derive.
                return None, []
    if not bundle_name:
        return None, []

    # Walk in-memory bundles first; never trigger _resolve_bundles() if we
    # have the topology locally. _resolve_bundles() may reach disk via
    # _load_services_config() which CAN clobber a monkeypatched _services_config.
    in_memory_bundles = ((_services_config or {}).get('bundles') or {})
    if in_memory_bundles:
        linked: list[str] = []
        try:
            for ln in _get_linked_bundles(bundle_name, bundles=in_memory_bundles) or []:
                if ln and ln not in linked:
                    linked.append(str(ln))
        except Exception:
            pass
        return bundle_name, linked

    # In-memory bundles section is missing — fall back to the cached/disk
    # resolver. Safe-fallback policy matches _resolve_group_target_gpus:
    # snapshot/restore _services_config so _resolve_bundles() → _load_services_config()
    # cannot clobber a caller-provided (monkeypatched) value.
    snapshot = _services_config
    try:
        resolved = _resolve_bundles()
    except Exception:
        return bundle_name, []
    finally:
        try:
            globals()['_services_config'] = snapshot
        except Exception:
            pass
    if not resolved:
        return bundle_name, []
    linked: list[str] = []
    try:
        for ln in _get_linked_bundles(bundle_name, bundles=resolved) or []:
            if ln and ln not in linked:
                linked.append(str(ln))
    except Exception:
        pass
    return bundle_name, linked


def _resolve_group_target_gpus(target_group: str) -> list[str]:
    """Derive the full set of target GPUs for ``target_group`` from current config.

    Order of precedence:
      1. ``_group_reservations[group].requires_gpus`` (HTTP path; pure in-memory).
      2. The target bundle's primary GPU + every linked GPU via
         ``_collect_linked_gpus`` (config-driven; works on a fresh start).
         This step needs ``_resolve_bundles()``, which may read disk if
         ``_services_config`` does not yet carry a ``bundles`` section.

    The two sources are UNIONed (deduped, order preserved) so that a partial
    or stale reservation can NEVER hide the topology-derived linked GPUs.
    A partial reservation that only mentions the primary gpu must still
    surface the linked gpu through step 2 — otherwise the proactive loop
    would inspect only a subset of the topology and incorrectly conclude
    a valid linked component is an evictor.

    Returns an empty list iff neither source resolves — fail closed so the
    caller can choose to advertise a non-dispatch action.

    Safe-fallback policy: when the topology step needs ``_resolve_bundles()``
    and that resolver reaches disk via ``_load_services_config()``, the call
    MAY overwrite the module-level ``_services_config``. To preserve a
    caller-provided (e.g. monkeypatched) ``_services_config``, we snapshot
    and restore around the resolver call. The snapshot/restore is a no-op
    when ``_resolve_bundles`` only refreshes its own cache.
    """
    union: list[str] = []

    # 1) Reservation-driven path — pure memory. Hot path for the steady state.
    try:
        grp_res = _group_reservations.get(target_group, {}) or {}
    except Exception:
        grp_res = {}
    if isinstance(grp_res, dict) and grp_res.get('requires_gpus'):
        for gid in grp_res.get('requires_gpus') or []:
            if gid and gid not in union:
                union.append(gid)

    # 2) Config-driven path. Find the template's bundle in the in-memory
    # _services_config first to avoid triggering a disk load (which would
    # overwrite the caller's monkeypatched _services_config).
    templates = ((_services_config or {}).get('generation_templates') or {})
    bundle_name: str | None = None
    for tmpl in templates.values():
        if not isinstance(tmpl, dict):
            continue
        if tmpl.get('routing_group') == target_group:
            cand = tmpl.get('bundle')
            if cand and bundle_name is None:
                bundle_name = str(cand)
            elif cand and bundle_name != str(cand):
                # Ambiguous mapping: still return what we have so far
                # (the reservation-derived subset is better than nothing).
                return list(dict.fromkeys(union))
    if not bundle_name:
        return list(dict.fromkeys(union))

    # Look the bundle up in-memory first.
    in_memory_bundles = ((_services_config or {}).get('bundles') or {})
    bundle_cfg = in_memory_bundles.get(bundle_name)
    if bundle_cfg is None:
        # Snapshot _services_config so a disk-backed _resolve_bundles() does
        # not clobber a caller-provided (monkeypatched) value. _load_services_config()
        # inside _resolve_bundles() may promote a freshly-read config to the
        # module global; we restore it on the way out.
        snapshot = _services_config
        try:
            bundles = _resolve_bundles()
        except Exception:
            return list(dict.fromkeys(union))
        finally:
            # _resolve_bundles() may have mutated _services_config via
            # _load_services_config(); restore the caller's snapshot.
            try:
                # pylint: disable=global-statement
                globals()['_services_config'] = snapshot
            except Exception:
                pass
        bundle_cfg = bundles.get(bundle_name) or {}
        bundles_for_walk = bundles
    else:
        bundles_for_walk = in_memory_bundles
    try:
        for gid in _collect_linked_gpus(
            bundle_cfg, bundles_for_walk, bundle_name=bundle_name
        ):
            if gid and gid not in union:
                union.append(gid)
    except Exception:
        pass
    return list(dict.fromkeys(union))


def _scheduler_precheck_head_jobs(group: str) -> dict:
    if 'queue_engine' not in globals() or not queue_engine or not getattr(queue_engine, 'r', None):
        return {'group': group, 'decision': 'unknown', 'skipped_reason': 'no_queue_engine'}
    if 'job_tracker' not in globals() or not job_tracker:
        return {'group': group, 'decision': 'unknown', 'skipped_reason': 'no_job_tracker'}

    try:
        # Inspect the same logical stream set as the queue worker.  Lane mode
        # is opt-in, but the scheduler precheck must not silently ignore a
        # ready interactive job when it is enabled; the legacy stream remains
        # last for drain/replay compatibility.
        specs = (
            queue_engine._stream_specs(group, routing_group=group)
            if getattr(queue_engine, 'priority_lanes_enabled', False)
            else [(queue_engine._stream(group, routing_group=group), None)]
        )
        entries = []
        for stream, _lane in specs:
            entries.extend(queue_engine.r.xrange(stream, count=10))
    except Exception as e:
        return {'group': group, 'decision': 'unknown', 'skipped_reason': f'redis_error_or_missing: {e}'}

    if not entries:
        return {'group': group, 'decision': 'skip', 'skipped_reason': 'empty'}

    queued_job_ids = []
    saw_cancelled = False
    for entry_id, data in entries:
        job_id = data.get('job_id')
        if not job_id:
            continue

        job = job_tracker.get_job(job_id)
        if not job:
            continue

        # JobTracker.get_job may return either a Job-shaped object with a
        # `.status` attribute (legacy / live JobTracker) OR a plain mapping
        # (rehydrated payloads, see handle_get_job). A stream entry alone is
        # not dispatch authority: only a durable *queued* job may cause a
        # scheduler-owned bundle transition. Claimed/in-flight entries remain
        # in Redis Streams until terminal ACK and must not trigger a second
        # swap after the worker has already taken ownership.
        if isinstance(job, Mapping):
            status = job.get('status')
        else:
            status = getattr(job, 'status', None)

        if status == 'queued':
            queued_job_ids.append(str(job_id))
        elif status == 'cancelled':
            saw_cancelled = True

    if not queued_job_ids:
        reason = 'all_cancelled' if saw_cancelled else 'no_durable_queued_job'
        return {
            'group': group,
            'decision': 'skip',
            'skipped_reason': reason,
            'head_job_ids': [],
        }

    return {
        'group': group,
        'decision': 'proceed',
        'skipped_reason': 'none',
        'head_job_ids': queued_job_ids,
    }


def _scheduler_run_precheck(group: str) -> dict:
    """Run precheck and record it in the head_job_pre_checks ring buffer."""
    record = _scheduler_precheck_head_jobs(group)
    try:
        record['recorded_at'] = time.time()
    except Exception:
        pass
    _record_head_job_precheck(record)
    return record


def _refresh_phase3_scheduler_state() -> dict:
    """Refresh phase-3 demand/grace state without an observer request."""
    all_gpu_states = _get_all_gpu_states()
    generation_templates = (_services_config.get("generation_templates", {}) or {})
    groups = {}
    for service_name, service_cfg in (_services_config.get("services", {}) or {}).items():
        routing_group = service_cfg.get("routing_group") or service_name
        group = groups.setdefault(
            routing_group,
            {
                "type": "unknown",
                "priority": service_cfg.get("priority", 50),
                "queue_lag": 0,
            },
        )
        if service_cfg.get("type") == "llm_backend":
            group["type"] = "llm"
        elif service_cfg.get("type") == "generation_backend" or service_cfg.get("bundle"):
            group["type"] = "generation"

    if queue_engine is not None:
        for routing_group, group in groups.items():
            try:
                observation = _queue_depth_observation(
                    queue_engine, routing_group, routing_group
                )
                group["queue_lag"] = (
                    observation["depth"] if observation["known"] else 0
                )
                group["queue_lag_observed"] = observation["depth"]
                group["queue_lag_known"] = observation["known"]
                group["queue_lag_reason"] = observation["reason"]
            except Exception as exc:
                logger.debug("phase3 refresh queue-depth failed for %s: %s", routing_group, exc)

    return _get_phase3_scheduler_state(groups, all_gpu_states, generation_templates)


def _scheduler_evaluate_load(now=None) -> dict:
    global _last_decision
    if now is None:
        now = time.time()

    config = _phase3_scheduling_config()
    dry_run_mode = config.get('scheduler_dry_run_mode', True)
    decision = {
        'action': 'noop',
        'reason': 'default',
        'target_gpus': [],
        'head_job_ids': [],
        'decision_id': str(uuid.uuid4()),
        'decided_at': now,
        'group': None,
        'dry_run': dry_run_mode,
        'dispatch_performed': False
    }

    if not config.get('scheduler_owns_load'):
        decision['reason'] = 'scheduler_owns_load_false'
        _last_decision = decision.copy()
        _record_load_history(decision)
        return decision

    last_next = _last_next_decision.get('next_decision')

    if last_next == 'awaiting_grace':
        decision['action'] = 'awaiting_grace'
        decision['reason'] = 'awaiting_grace phase 3 state'
    elif 'queue_engine' not in globals() or not queue_engine or not _services_config:
        decision['action'] = 'awaiting_judging'
        decision['reason'] = 'no queue_engine or service-groups info'
    elif isinstance(last_next, str) and last_next.startswith('demand:'):
        group = last_next.split(':', 1)[1]
        decision['group'] = group
        precheck = _scheduler_run_precheck(group)
        head_job_ids = precheck.get('head_job_ids') or []
        if isinstance(head_job_ids, (list, tuple)):
            decision['head_job_ids'] = [
                str(job_id) for job_id in head_job_ids
                if isinstance(job_id, str) and job_id
            ]
        if (
            precheck.get('skipped_reason') in (
                'empty', 'all_cancelled', 'no_durable_queued_job'
            )
            or not decision['head_job_ids']
        ):
            decision['action'] = 'pre_load_failed_skip'
            decision['reason'] = (
                'precheck has no durable queued head job: '
                f"{precheck.get('skipped_reason', 'missing_head_job_ids')}"
            )
        elif _active_group_still_draining(group):
            decision['action'] = 'hold_drain'
            decision['reason'] = 'active group draining; defer load'
        else:
            target_gpus = list(_resolve_group_target_gpus(group) or [])
            # If we couldn't derive ANY target GPUs from config, fail closed:
            # do not advertise a demand_load the dispatcher can't actually act
            # on. (Advisory telemetry only — the authoritative atomic path is
            # still _evict_then_load_bundle, which validates targets itself.)
            if not target_gpus:
                decision['action'] = 'insufficient_visibility'
                decision['reason'] = (
                    f"unreadable target_gpus for group {group!r}: "
                    "no template/bundle resolution available"
                )
                decision['requires_eviction'] = False
            elif group in {'comfyui-gen', 'image-gen'}:
                decision['action'] = 'dual_gpu_load'
                decision['target_gpus'] = target_gpus
                decision['reason'] = 'demand load for dual gpu'
                try:
                    decision['requires_eviction'] = _target_gpus_hold_warm_bundle(group)
                except _TargetStateUnreadable as exc:
                    decision['action'] = 'insufficient_visibility'
                    decision['reason'] = (
                        f"unreadable target state for group {group!r}: "
                        f"gpu {exc.gpu_id!r}"
                    )
                    decision['requires_eviction'] = False
            else:
                decision['action'] = 'demand_load'
                decision['target_gpus'] = target_gpus
                decision['reason'] = 'demand load'
                try:
                    decision['requires_eviction'] = _target_gpus_hold_warm_bundle(group)
                except _TargetStateUnreadable as exc:
                    decision['action'] = 'insufficient_visibility'
                    decision['reason'] = (
                        f"unreadable target state for group {group!r}: "
                        f"gpu {exc.gpu_id!r}"
                    )
                    decision['requires_eviction'] = False
    elif last_next == 'idle':
        idle_preferred = _last_next_decision.get('idle_preferred', False)
        if idle_preferred:
            # Idle residency is not worker reservation: the decision names the
            # idle targets to keep warm and never carries head jobs or a claim.
            # Unavailable preferred targets fall back to the configured
            # resident; GPUs backing off or with no loadable target are left out.
            plan = _IDLE_PLANNER.plan(_services_config or {}, now)
            targets = sorted(g for g, t in plan.items() if t["service"] and t["ready"])
            if targets:
                decision['action'] = 'idle_load'
                decision['reason'] = 'idle_preferred'
                decision['target_gpus'] = targets
            else:
                decision['action'] = 'noop'
                decision['reason'] = 'idle targets unavailable or backing off'
            decision['claims_worker'] = False
        else:
            decision['action'] = 'noop'
            decision['reason'] = 'idle noop'

    if _is_repeat_idle_decision(decision):
        _last_decision = decision.copy()
        return decision
    logger.info(f"evaluator.decision: {json.dumps(decision)}")
    _last_decision = decision.copy()
    _record_load_history(decision)
    return decision


def _resolve_scheduler_bundle_for_group(group: str) -> tuple[str | None, int | None, str | None]:
    """Resolve exactly one configured generation bundle for a selected group."""
    templates = (_services_config or {}).get("generation_templates", {}) or {}
    bundles = []
    for template in templates.values():
        if not isinstance(template, dict) or template.get("routing_group") != group:
            continue
        bundle = template.get("bundle")
        if bundle and bundle not in bundles:
            bundles.append(str(bundle))
    if len(bundles) != 1:
        if not bundles:
            return None, None, "no_configured_bundle_for_group"
        return None, None, "ambiguous_configured_bundles_for_group"
    return bundles[0], _service_group_priority(group, {}, templates), None


def _is_repeat_idle_decision(decision: dict) -> bool:
    """True for an idle/noop decision identical to the previous one (state unchanged)."""
    global _idle_decision_signature
    if decision.get('action') not in {'idle_load', 'noop'} or decision.get('head_job_ids'):
        _idle_decision_signature = None
        return False
    sig = (decision.get('action'), decision.get('reason'), tuple(decision.get('target_gpus') or ()))
    if sig == _idle_decision_signature:
        return True
    _idle_decision_signature = sig
    return False


async def _dispatch_scheduler_decision(decision: dict) -> dict:
    """Execute one demand decision only behind three independent live gates."""
    result = dict(decision)
    result["dispatch_performed"] = False
    if result.get("action") not in {"demand_load", "dual_gpu_load"}:
        result["dispatch_reason"] = "non_dispatch_action"
        return result

    config = _phase3_scheduling_config()
    if not config.get("scheduler_owns_load"):
        result["dispatch_reason"] = "scheduler_owns_load_false"
        return result
    if config.get("scheduler_dry_run_mode", True):
        result["dispatch_reason"] = "scheduler_dry_run_mode"
        return result
    if not config.get("proactive_scheduling_enabled", False):
        result["dispatch_reason"] = "proactive_scheduling_disabled"
        return result

    head_job_ids = result.get("head_job_ids")
    if (
        not isinstance(head_job_ids, list)
        or not any(isinstance(job_id, str) and job_id for job_id in head_job_ids)
    ):
        result["dispatch_reason"] = "no_durable_head_job"
        return result

    group = result.get("group")
    if not isinstance(group, str) or not group:
        result["dispatch_reason"] = "missing_selected_group"
        return result
    bundle, priority, resolution_error = _resolve_scheduler_bundle_for_group(group)
    if resolution_error or bundle is None or priority is None:
        result["dispatch_reason"] = resolution_error or "invalid_bundle_resolution"
        return result

    result["bundle"] = bundle
    try:
        loaded = await _evict_then_load_bundle(
            bundle,
            reason=f"scheduler:{group}",
            priority=priority,
        )
    except Exception as exc:
        result["dispatch_reason"] = f"bundle_load_raised:{type(exc).__name__}"
        result["dispatch_error"] = str(exc)[:300]
        return result
    if not loaded:
        result["dispatch_reason"] = "bundle_load_failed"
        return result

    result["dispatch_performed"] = True
    result["dispatch_reason"] = "bundle_loaded"
    return result


async def _proactive_dispatch_loop():
    """Evaluate demand continuously and dispatch only behind the live triple gate.

    In dry-run mode this records would-have decisions. A real bundle transition
    requires scheduler ownership, non-dry-run configuration, and explicit
    proactive scheduling enablement.
    """
    global _last_decision
    logger.info(
        "Proactive dispatch startup grace: waiting %.1fs before first decision",
        GPU_PROBE_STARTUP_GRACE_S,
    )
    await asyncio.sleep(GPU_PROBE_STARTUP_GRACE_S)
    try:
        import time as _time
        _proactive_dispatch_state["last_iteration_at"] = _time.time()
    except Exception as e:
        logger.warning(f"proactive_dispatch_loop import failed: {e}")
        return
    while True:
        try:
            _proactive_dispatch_state["iteration_count"] += 1
            _proactive_dispatch_state["last_iteration_at"] = _time.time()
            # Read phase 3 scheduling config (returns the env-var-aware dict)
            config = _phase3_scheduling_config()
            live_dispatch_enabled = bool(
                config.get("scheduler_owns_load")
                and not config.get("scheduler_dry_run_mode", True)
                and config.get("proactive_scheduling_enabled", False)
            )
            _proactive_dispatch_state["dry_run"] = not live_dispatch_enabled
            # Skip if scheduler is not the owner
            if not config.get("scheduler_owns_load"):
                await asyncio.sleep(_PROACTIVE_LOOP_INTERVAL_SEC)
                continue
            # Get a fresh decision
            now = _time.time()
            _refresh_phase3_scheduler_state()
            decision = _scheduler_evaluate_load(now)
            if not isinstance(decision, dict):
                await asyncio.sleep(_PROACTIVE_LOOP_INTERVAL_SEC)
                continue
            action = decision.get("action", "noop")
            # Classify and update counters
            _proactive_dispatch_state["last_decision"] = decision
            # The 10-second same-action spacing suppresses **dry-run telemetry
            # only**. Live dispatch fires every iteration regardless of the
            # previous decision, because _evict_then_load_bundle is the
            # authoritative atomic path and idempotency is enforced there
            # (priority-aware precheck + readiness probe). See
            # _should_suppress_proactive_spacing.
            global _PROACTIVE_LAST_DECISION
            last_recorded_at = _proactive_dispatch_state.get("last_action_recorded_at", 0.0)
            last_decision_action = (
                _PROACTIVE_LAST_DECISION.get("action")
                if isinstance(_PROACTIVE_LAST_DECISION, dict)
                else None
            )
            suppress_telemetry = _should_suppress_proactive_spacing(
                now=now,
                action=action,
                last_action_recorded_at=last_recorded_at,
                last_decision_action=last_decision_action,
                dry_run=not live_dispatch_enabled,
            )
            if suppress_telemetry:
                _proactive_dispatch_state["spaced_actions"] += 1
            else:
                # Record the would-have-done
                if action in ("demand_load", "dual_gpu_load", "idle_load"):
                    _proactive_dispatch_state["would_have_loaded"] += 1
                elif action == "evict_only":
                    _proactive_dispatch_state["would_have_evicted"] += 1
                elif action in ("hold_drain", "awaiting_grace"):
                    _proactive_dispatch_state["would_have_held"] += 1
                _PROACTIVE_LAST_DECISION = decision
                _proactive_dispatch_state["last_action_recorded_at"] = now
            if live_dispatch_enabled:
                dispatched = await _dispatch_scheduler_decision(decision)
                _proactive_dispatch_state["last_decision"] = dispatched
                _last_decision = dispatched.copy()
                _record_load_history(dispatched)
                if dispatched.get("dispatch_performed"):
                    _proactive_dispatch_state["dispatched"] += 1
                elif dispatched.get("action") in {"demand_load", "dual_gpu_load"}:
                    _proactive_dispatch_state["dispatch_failures"] += 1
                # Catalog 5e warm pool: flag-off means this line is the only change.
                if _warm_pool.enabled() and not _maintenance_mode_enabled(config):
                    await _warm_pool_preload_tick()
        except Exception as e:
            _proactive_dispatch_state["errors"].append({
                "at": _time.time(),
                "error": str(e)[:200]
            })
            # Cap errors list at 10
            if len(_proactive_dispatch_state["errors"]) > 10:
                _proactive_dispatch_state["errors"] = _proactive_dispatch_state["errors"][-10:]
            logger.warning(f"proactive_dispatch_loop error: {e}")
        await asyncio.sleep(_PROACTIVE_LOOP_INTERVAL_SEC)


def _should_suppress_proactive_spacing(
    *,
    now: float,
    action: str,
    last_action_recorded_at: float,
    last_decision_action: str | None,
    dry_run: bool,
) -> bool:
    """Decide whether the proactive loop should suppress dry-run telemetry.

    Returns True iff the previous decision was the same action AND it was
    recorded less than ``_PROACTIVE_SPACING_WINDOW_SEC`` seconds ago AND we
    are in dry-run mode. Live dispatch is NEVER suppressed by this rule —
    see the docstring on ``_proactive_dispatch_loop`` for why.
    """
    if not dry_run:
        return False
    if last_decision_action is None:
        return False
    if action != last_decision_action:
        return False
    if (now - last_action_recorded_at) >= _PROACTIVE_SPACING_WINDOW_SEC:
        return False
    return True


NAMES = (
    "_record_load_history",
    "_record_head_job_precheck",
    "_group_circuit_state",
    "_active_group_still_draining",
    "_target_gpus_hold_warm_bundle",
    "_resolve_target_bundle_components",
    "_resolve_group_target_gpus",
    "_scheduler_precheck_head_jobs",
    "_scheduler_run_precheck",
    "_refresh_phase3_scheduler_state",
    "_scheduler_evaluate_load",
    "_resolve_scheduler_bundle_for_group",
    "_is_repeat_idle_decision",
    "_dispatch_scheduler_decision",
    "_proactive_dispatch_loop",
    "_should_suppress_proactive_spacing",
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
    rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt


def bind(facade_globals):
    """Return the scheduler-helper functions in ``NAMES`` order, bound to a facade.

    Each function is rebuilt so its ``__globals__`` is exactly ``facade_globals``.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    return tuple(_rebuild(namespace[name], facade_globals) for name in NAMES)
