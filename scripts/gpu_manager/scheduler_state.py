"""Facade-bound scheduler state: queue observation, quiescence and admission guard.

Extracted verbatim from ``scripts/gpu-manager.py``.  Every function reads facade
globals (``logger``, ``queue_engine`` helpers, ``_can_consume_last_log``,
``_dual_swap_lock`` and each other) at call time, so ``bind(facade_globals)``
rebuilds each one with ``types.FunctionType`` against the supplied facade
dictionary and builds a fresh ``_RestoreAdmissionGuard`` class whose methods are
rebuilt the same way.  No ``sys.modules`` entry or shared facade pointer is kept
here, and the mutable state (``_dual_swap_lock``, ``_can_consume_last_log``)
stays facade-owned.  The definitions below are templates: calling them directly
would resolve names in this module.
"""
from __future__ import annotations

from types import CellType, FunctionType


def _queue_depth_observation(
    queue,
    service_name: str,
    routing_group: str | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Normalize queue depth without turning missing evidence into demand."""

    fallback = {
        "depth": 0,
        "known": False,
        "reason": "queue_engine_unavailable",
        "pending": 0,
    }
    if queue is None:
        return fallback
    observer = getattr(queue, "get_queue_depth_observation", None)
    if callable(observer):
        try:
            result = observer(service_name, routing_group=routing_group)
            if isinstance(result, Mapping):
                return {
                    "depth": max(0, int(result.get("depth", 0) or 0)),
                    "known": result.get("known") is True,
                    "reason": str(result.get("reason") or "unknown"),
                    "pending": max(0, int(result.get("pending", 0) or 0)),
                    "stream": result.get("stream"),
                    "error": str(result.get("error") or "")[:240],
                }
        except Exception as exc:
            if strict:
                raise
            return {**fallback, "reason": f"observation_error:{type(exc).__name__}"}
    try:
        try:
            depth = queue.get_queue_depth(
                service_name, routing_group=routing_group, strict=strict
            )
        except TypeError as exc:
            if "strict" not in str(exc):
                raise
            # Older queue adapters did not expose the strict keyword.  Retry
            # the bounded integer API instead of turning a non-strict health
            # observation into an avoidable unknown-depth result.
            depth = queue.get_queue_depth(
                service_name, routing_group=routing_group
            )
        return {
            "depth": max(0, int(depth or 0)),
            "known": True,
            "reason": "legacy_integer_api",
            "pending": 0,
        }
    except Exception as exc:
        if strict:
            raise
        return {**fallback, "reason": f"depth_error:{type(exc).__name__}"}


def _logical_group_quiescent(routing_group: str) -> tuple[bool, dict]:
    """Return whether a logical routing group is fully drained.

    Queue lag alone is insufficient: Redis PEL entries may already be
    assigned to workers, and a backend can still be executing a request.
    This helper is intentionally group-wide so a GPU service can drain
    without blocking eligible CPU/other-GPU members of the same group.
    """
    snapshot = {
        'routing_group': routing_group,
        'queue_depth': 0,
        'queue_depth_known': False,
        'queue_depth_reason': 'queue_engine_unavailable',
        'pending': 0,
        'active_workers': 0,
        'job_demand': None,
    }
    try:
        if queue_engine is not None:
            observation = _queue_depth_observation(
                queue_engine, routing_group, routing_group, strict=True
            )
            snapshot['queue_depth'] = observation['depth']
            snapshot['queue_depth_known'] = observation['known']
            snapshot['queue_depth_reason'] = observation['reason']
            pending_reader = queue_engine.get_pending
            try:
                snapshot['pending'] = len(pending_reader(
                    routing_group, routing_group=routing_group, strict=True
                ) or [])
            except TypeError as exc:
                # Compatibility fakes/older queue adapters may not expose the
                # strict keyword yet. Keep their established behavior, while
                # the production RedisQueue takes the fail-closed path above.
                if "strict" not in str(exc):
                    raise
                snapshot['pending'] = len(pending_reader(
                    routing_group, routing_group=routing_group
                ) or [])
            if (
                snapshot['queue_depth_known'] is not True
                and snapshot['queue_depth_reason'] != 'external_queue_authority'
            ):
                snapshot['error'] = (
                    'queue depth unavailable: '
                    f"{snapshot['queue_depth_reason']}"
                )
                return False, snapshot
    except Exception as exc:
        snapshot['error'] = f'queue inspection failed: {exc}'
        return False, snapshot
    try:
        if worker_supervisor is not None:
            for pool in worker_supervisor._pools.values():
                if routing_group in getattr(pool, 'routing_groups', ()):
                    snapshot['active_workers'] += int(
                        getattr(pool, 'active_workers', 0) or 0
                    )
    except Exception as exc:
        snapshot['error'] = f'worker inspection failed: {exc}'
        return False, snapshot

    # Redis stream depth/P.E.L. and worker counters can briefly disagree with
    # the durable tracker. Include claimed/loading/in-flight records; an
    # absent or unavailable tracker is not safe evidence that a shared runtime
    # can be unloaded, so the transition fails closed.
    tracker = globals().get('job_tracker')
    if tracker is None or not hasattr(tracker, 'demand_snapshot'):
        snapshot['error'] = 'job demand tracker unavailable'
        return False, snapshot
    try:
        demand = tracker.demand_snapshot(routing_group=routing_group)
        snapshot['job_demand'] = demand
        if demand.get('status') != 'available':
            snapshot['error'] = 'job demand unavailable'
            return False, snapshot
    except Exception as exc:
        snapshot['error'] = f'job demand inspection failed: {exc}'
        return False, snapshot

    durable_active = (
        int((snapshot.get('job_demand') or {}).get('active_demand') or 0)
        if snapshot.get('job_demand') is not None
        else 0
    )
    return (
        snapshot['queue_depth'] == 0
        and snapshot['pending'] == 0
        and snapshot['active_workers'] == 0
        and durable_active == 0,
        snapshot,
    )


async def _wait_for_logical_group_quiescence(
    routing_group: str, timeout: float = 1800.0, settle_seconds: float = 0.5
) -> bool:
    """Wait until a logical group is empty and stays empty briefly.

    New arrivals remain valid work and are consumed by any healthy eligible
    member.  The settle window closes the check/enqueue race immediately
    before a higher-priority GPU bundle is admitted.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        empty, snapshot = _logical_group_quiescent(routing_group)
        if empty:
            await asyncio.sleep(settle_seconds)
            empty_again, snapshot_again = _logical_group_quiescent(routing_group)
            if empty_again:
                logger.info(
                    f'Logical group {routing_group} quiescent before swap: '
                    f'{snapshot_again}'
                )
                return True
            snapshot = snapshot_again
        logger.info(
            f'Logical group {routing_group} still draining before swap: '
            f'{snapshot}'
        )
        await asyncio.sleep(0.5)
    logger.warning(
        f'Logical group {routing_group} did not quiesce within {timeout}s; '
        'refusing the bundle swap'
    )
    return False


def _log_can_consume_pause(service_name, routing_group, reason, worker_id):
    now = time.time()
    key = (service_name, routing_group, reason)
    last_log = _can_consume_last_log.get(key, 0)
    if now - last_log >= 10:
        logger.debug(f"Worker {worker_id} for {service_name} paused on {routing_group}: {reason}")
        _can_consume_last_log[key] = now


def _get_dual_swap_lock():
    global _dual_swap_lock
    if _dual_swap_lock is None:
        import asyncio
        _dual_swap_lock = asyncio.Lock()
    return _dual_swap_lock


class _RestoreAdmissionGuard:
    """Acquire bundle admission before the per-manager restore lock.

    The ordering matches the bundle-load transaction (physical admission
    first, lifecycle operation second), so a restore cannot observe an empty
    generation state and then race a new admission into the same GPUs.
    """

    def __init__(self, manager):
        self._manager = manager
        self._admission_lock = _get_dual_swap_lock()

    async def __aenter__(self):
        await self._admission_lock.acquire()
        try:
            await self._manager._evict_lock.acquire()
        except Exception:
            self._admission_lock.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._manager._evict_lock.release()
        self._admission_lock.release()
        return False


NAMES = (
    "_queue_depth_observation",
    "_logical_group_quiescent",
    "_wait_for_logical_group_quiescence",
    "_log_can_consume_pause",
    "_get_dual_swap_lock",
    "_RestoreAdmissionGuard",
)


def _rebuild(function, facade_globals, class_cell=None):
    closure = function.__closure__
    if closure is not None and class_cell is not None:
        closure = tuple(
            class_cell if name == "__class__" else cell
            for name, cell in zip(function.__code__.co_freevars, closure)
        )
    rebuilt = FunctionType(
        function.__code__,
        facade_globals,
        function.__name__,
        function.__defaults__,
        closure,
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


def _rebuild_class(template, facade_globals):
    """Return a fresh class equivalent to ``template`` bound to ``facade_globals``.

    Same mechanism as the class-binding proof in
    ``tests/test_memory_policy_binding.py``: every function in the class
    dictionary is rebuilt against the facade globals and a ``__class__`` cell
    (zero-argument ``super()``) is re-pointed at the new class.
    """
    class_cell = CellType()

    def rebuild(function):
        return _rebuild(function, facade_globals, class_cell)

    namespace = {}
    for key, value in vars(template).items():
        if key in ("__dict__", "__weakref__"):
            continue
        if isinstance(value, FunctionType):
            value = rebuild(value)
        elif isinstance(value, (staticmethod, classmethod)):
            value = type(value)(rebuild(value.__func__))
        elif isinstance(value, property):
            value = property(
                rebuild(value.fget) if value.fget else None,
                rebuild(value.fset) if value.fset else None,
                rebuild(value.fdel) if value.fdel else None,
                value.__doc__,
            )
        namespace[key] = value
    namespace["__module__"] = facade_globals.get("__name__", template.__module__)
    bound = type(template)(template.__name__, template.__bases__, namespace)
    class_cell.cell_contents = bound
    return bound


def bind(facade_globals):
    """Return the scheduler-state symbols in ``NAMES`` order, bound to a facade.

    Each function is rebuilt so its ``__globals__`` is exactly
    ``facade_globals``; ``_RestoreAdmissionGuard`` is a new class per call, so
    the facade must call ``bind`` once and keep the result.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    bound = []
    for name in NAMES:
        template = namespace[name]
        if isinstance(template, type):
            bound.append(_rebuild_class(template, facade_globals))
        else:
            bound.append(_rebuild(template, facade_globals))
    return tuple(bound)
