"""Facade-bound GPU reservation helpers.

Function bodies are templates: external dependencies and mutable state resolve
in the facade dictionary at call time. ``bind`` rebuilds each function against
the supplied facade without retaining a facade pointer or importing runtime
subsystems.
"""
from __future__ import annotations

from types import FunctionType

NAMES = (
    '_phase5_reservation_config',
    '_derive_group_reservations',
    '_validate_reservation_config',
    '_record_reservation_change',
    'reserve_gpus',
    'release_gpus',
    'expire_reservations',
    '_reservation_persistence_path',
    '_persist_reservations',
    '_record_ingress_blocked_by_reservation',
)


def _phase5_reservation_config() -> dict:
    root = _services_config or {}
    res_cfg = root.get('reservations', {}) or {}
    explicit_path = root.get('gpu_reservations_path') or res_cfg.get('gpu_reservations_path')
    path = explicit_path or '/run/gpu-manager/reservations.json'
    defaults = res_cfg.get('reservation_defaults', {}) or {}
    persistence_enabled = bool(explicit_path) or bool(res_cfg.get('enabled') or res_cfg.get('persistence_enabled'))
    return {
        'gpu_reservations_path': path,
        'reservation_defaults': defaults,
        'persistence_enabled': persistence_enabled
    }

def _derive_group_reservations(groups=None, generation_templates=None) -> dict:
    global _reservation_config_warnings
    warnings = []
    derived = {}
    if groups is None:
        groups = {}
    if generation_templates is None:
        generation_templates = {}
    # Qualification/draft templates remain visible in the endpoint's raw
    # inventory, but they must not create operational GPU reservations or
    # warnings that look like active queue ownership.  Services are filtered
    # separately by _is_queue_consumer_service().
    generation_templates = {
        name: template
        for name, template in generation_templates.items()
        if isinstance(template, Mapping)
        and _service_is_publicly_qualified(template)
        and not is_retired_minimax_comfyui_template(template)
    }

    routing_groups_config = {}
    try:
        routing_groups_config = _services_config.get('routing_groups', {}) or {}
    except Exception:
        routing_groups_config = {}

    all_groups = set(list(groups.keys()) + list(generation_templates.keys()) + list(routing_groups_config.keys()))
    for t in generation_templates.values():
        if t.get('routing_group'):
            all_groups.add(t['routing_group'])

    krea_like_groups = {'comfyui-gen', 'image', 'image-edit-krea2'}
    for grp_name in sorted(all_groups):
        explicit = routing_groups_config.get(grp_name, {}) or {}
        requires_gpus = list(dict.fromkeys(explicit.get('requires_gpus', []) or []))
        mode = explicit.get('gpu_reservation_mode', explicit.get('mode', 'exclusive'))
        hold_s = explicit.get('gpu_reservation_hold_s', explicit.get('hold_s', 120))
        priority = _service_group_priority(grp_name, groups, generation_templates)
        owner = explicit.get('owner')
        description = explicit.get('description')

        if 'priority' in explicit:
            priority = _coerce_group_priority(explicit.get('priority'), allow_cpu_fallback=True)

        if not requires_gpus:
            # infer from template bundle
            tmpl = generation_templates.get(grp_name)
            if not tmpl:
                for t in generation_templates.values():
                    if t.get('routing_group') == grp_name:
                        tmpl = t
                        break
            if not tmpl:
                tmpl = {}
            bundle = tmpl.get('bundle')
            if bundle:
                try:
                    bundle_map = _resolve_bundles()
                    res = bundle_map.get(bundle)
                    if res and res.get('gpu_id'):
                        requires_gpus.append(res['gpu_id'])
                        try:
                            linked = _collect_linked_gpus(res, bundle_map)
                            for lg in linked:
                                if lg not in requires_gpus:
                                    requires_gpus.append(lg)
                        except Exception:
                            pass
                except Exception:
                    pass
            basename = str(tmpl.get('workflow_template', '')).split('/')[-1]
            if ('krea2_multigpu' in basename or 'multigpu' in basename) and len(requires_gpus) <= 1:
                warnings.append(f"group '{grp_name}': inferred single GPU but template suggests multigpu")
            if 'image-edit-krea2' in str(tmpl.get('workflow_template', '')) and len(requires_gpus) <= 1:
                warnings.append(f"group '{grp_name}': ambiguous/missing routing_group config; image-edit-krea2 requires both gpu_primary and gpu_secondary but inferred only {requires_gpus}")

        if grp_name in krea_like_groups:
            if grp_name not in routing_groups_config:
                warnings.append(f"group '{grp_name}': missing routing_groups[{grp_name}] config; inferred {requires_gpus or 'no GPUs'}")
            if len(requires_gpus) <= 1:
                warnings.append(f"group '{grp_name}': configured for only {requires_gpus[0] if requires_gpus else 'no GPU'}, but known Krea path needs both gpu_primary and gpu_secondary")

        if grp_name in routing_groups_config and 'priority' not in explicit:
            warnings.append(f"group '{grp_name}': routing_groups entry missing priority; derived ownership priority {priority}")

        derived[grp_name] = {
            'requires_gpus': requires_gpus,
            'mode': mode,
            'hold_s': hold_s,
            'priority': priority,
        }
        if owner is not None:
            derived[grp_name]['owner'] = owner
        if description is not None:
            derived[grp_name]['description'] = description

    _reservation_config_warnings = warnings
    return derived

def _validate_reservation_config(group_reservations) -> list:
    warnings = []
    known_gpus = []
    try:
        known_gpus = list(_services_config.get('gpu_devices', {}).keys())
    except Exception:
        pass

    for grp, res in group_reservations.items():
        if len(res.get('requires_gpus', [])) > 1:
            try:
                tmpl = _services_config.get('generation_templates', {}).get(grp, {})
                bundle = tmpl.get('bundle')
                if bundle:
                    b_info = _resolve_bundles().get(bundle, {})
                    b_gpu = b_info.get('gpu_id')
                    try:
                        linked = _collect_linked_gpus(b_info, _resolve_bundles()) if b_info else []
                    except Exception:
                        linked = []
                    evidence = [b_gpu] + list(linked)
                    if not any(g in evidence for g in res['requires_gpus']):
                        warnings.append(f"group '{grp}': requires_gpus > 1 but no bundle/link evidence inferred")
            except Exception:
                pass

        if grp in {'comfyui-gen', 'image', 'image-edit-krea2'} and set(res.get('requires_gpus', [])) <= {'gpu_primary'}:
            warnings.append(f"group '{grp}': configured for only gpu_primary, but known Krea path needs both gpu_primary and gpu_secondary")

        if known_gpus:
            for g in res.get('requires_gpus', []):
                if g not in known_gpus:
                    warnings.append(f"group '{grp}': unknown gpu_id '{g}' in requires_gpus")

    try:
        config = _phase3_scheduling_config()
        if config.get('scheduler_owns_load'):
            templates = _services_config.get('generation_templates', {})
            for name, tmpl in templates.items():
                if tmpl.get('evict_first'):
                    warnings.append(f"template '{name}': evict_first is forbidden when scheduler_owns_load=true (runtime ignores it; remove from config)")
    except Exception:
        pass

    return warnings

def _record_reservation_change(action, group, token, gpu_ids, **kwargs):
    global _last_reservation_change
    now = time.time()
    record = {
        'timestamp': now,
        'action': action,
        'group': group,
        'token': token,
        'gpu_ids': gpu_ids
    }
    record.update(kwargs)
    _reservation_history.insert(0, record)
    while len(_reservation_history) > 50:
        _reservation_history.pop()

    _last_reservation_change = {
        'group': group,
        'gpu_ids': gpu_ids,
        'action': action,
        'token': token,
        'at': now
    }
    if 'reason' in kwargs:
        _last_reservation_change['reason'] = kwargs['reason']
    if 'mode' in kwargs:
        _last_reservation_change['mode'] = kwargs['mode']
    if 'hold_s' in kwargs:
        _last_reservation_change['hold_s'] = kwargs['hold_s']

def reserve_gpus(group, gpu_ids, token=None, mode='exclusive', hold_s=120) -> tuple:
    if not token:
        token = str(uuid.uuid4())
    now = time.time()

    # check conflict
    for gid in gpu_ids:
        if gid in _gpu_reservations:
            res = _gpu_reservations[gid]
            if res['group'] != group or res['token'] != token:
                if res['hold_expires_at'] > now:
                    return False, {'conflict_gpu': gid, 'reserved_by': res['group']}

    # success
    for gid in gpu_ids:
        _gpu_reservations[gid] = {
            'group': group,
            'token': token,
            'reserved_at': now,
            'hold_expires_at': now + hold_s,
            'mode': mode
        }

    _record_reservation_change('reserve', group, token, gpu_ids, mode=mode, hold_s=hold_s)
    _persist_reservations()
    return True, {'token': token, 'expires_at': now + hold_s}

def release_gpus(group, token, reason='release') -> tuple:
    released = []
    for gid, res in list(_gpu_reservations.items()):
        if res['group'] == group and res['token'] == token:
            del _gpu_reservations[gid]
            released.append(gid)

    if released:
        _record_reservation_change(reason, group, token, released)
        _persist_reservations()
        return True, {'released': released}
    return False, {'reason': 'not_found_or_mismatch'}

def expire_reservations(now=None) -> list:
    if now is None:
        now = time.time()
    expired = []
    for gid, res in list(_gpu_reservations.items()):
        if res['hold_expires_at'] <= now:
            expired.append((gid, res))
            del _gpu_reservations[gid]

    if expired:
        for gid, res in expired:
            _record_reservation_change('expire', res['group'], res['token'], [gid])
        _persist_reservations()
    return [e[0] for e in expired]

def _reservation_persistence_path() -> str:
    return _phase5_reservation_config().get('gpu_reservations_path', '/run/gpu-manager/reservations.json')

def _persist_reservations() -> dict:
    path = _reservation_persistence_path()
    config = _phase5_reservation_config()
    # Explicitly check if persistence is configured
    if not config.get('persistence_enabled'):
        return {'ok': False, 'reason': 'not_enabled', 'path': path}

    try:
        import tempfile
        import os
        import json

        data = {
            'reservations': _gpu_reservations,
            'last_change': _last_reservation_change,
            'timestamp': time.time()
        }

        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix='reservations_', suffix='.json.tmp')
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())

        if os.path.exists(path):
            try:
                os.replace(path, path + '.bak')
            except Exception:
                pass
        os.replace(temp_path, path)
        return {'ok': True, 'path': path}
    except Exception as e:
        return {'ok': False, 'reason': 'error', 'details': str(e)}

def _record_ingress_blocked_by_reservation(group):
    global _ingress_blocked_by_reservation
    _ingress_blocked_by_reservation += 1
    _ingress_blocked_by_reservation_by_group[group] = _ingress_blocked_by_reservation_by_group.get(group, 0) + 1

def bind(facade_globals):
    """Return reservation helpers bound to one facade namespace."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    bound = []
    for name in NAMES:
        function = namespace[name]
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
        bound.append(rebuilt)
    return tuple(bound)
