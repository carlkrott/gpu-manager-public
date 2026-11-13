"""Facade-bound OOM detection, resource locks and dead-letter helpers."""
from __future__ import annotations

from types import CellType, FunctionType

NAMES = ('_detect_oom_in_history', '_verify_oom_via_vram', '_detect_silent_oom_via_dmesg', '_get_force_recover_lock', '_get_bundle_load_lock', '_get_bundle_admission_lock', '_PhysicalResourceSetLock', '_get_physical_resource_lock', '_reset_physical_resource_locks_for_tests', '_bundle_physical_gpu_set', '_recent_oom_events', '_init_oom_redis', '_close_oom_redis', '_dead_letter_identity', '_matching_dead_letter_members', '_add_to_dead_letter', '_get_dead_letters', '_refresh_dead_letter_cache')

def _detect_oom_in_history(
    status_str: str = "",
    messages: list | None = None,
    response_body: str = "",
    status_code: int = 200,
    model_size_hint_gb: float = 0.0,
    vram_free_mib: int = 0,
    vram_total_mib: int = 0,
) -> dict | None:
    """Detect OOM from a ComfyUI / ACE-Step / generic generation response.

    Returns a dict with detection details if OOM is suspected, None otherwise.

    Sources (any one triggers):
    1. status_str == 'error' AND any message matches OOM_DETECTION_REGEX
    2. response_body matches OOM_DETECTION_REGEX
    3. status_code is 5xx AND vram_free_mib < OOM_VRAM_PRESSURE_THRESHOLD_MIB
    4. model_size_hint_gb * 1024 > vram_total_mib * 0.9 (model too big to fit)

    Per adversarial check #7: do NOT use bare 'OOM' substring. The regex
    OOM_DETECTION_REGEX requires the OOM marker to be preceded by cuda/HIP/
    RuntimeError/torch.cuda to avoid false positives.
    """
    messages = messages or []

    # Source 1: status_str=error AND messages match OOM regex
    if status_str == "error":
        for msg in messages:
            msg_text = str(msg)
            if _OOM_PATTERN.search(msg_text):
                return {
                    "source": "history_messages",
                    "reason": "status_str=error, message matched OOM regex",
                    "details": msg_text[:500],
                }

    # Source 2: response body matches OOM regex
    if response_body:
        match = _OOM_PATTERN.search(response_body)
        if match:
            return {
                "source": "response_body",
                "reason": "response body matched OOM regex",
                "details": match.group(0)[:200],
            }

    # Source 3: 5xx + low VRAM (corroborates a backend OOM)
    if status_code >= 500 and vram_free_mib > 0 and vram_free_mib < OOM_VRAM_PRESSURE_THRESHOLD_MIB:
        return {
            "source": "5xx_low_vram",
            "reason": (
                f"HTTP {status_code} with VRAM {vram_free_mib} MiB free "
                f"(< {OOM_VRAM_PRESSURE_THRESHOLD_MIB})"
            ),
            "details": f"status={status_code}, free={vram_free_mib} MiB",
        }

    # Source 4: model size too big to fit
    if model_size_hint_gb > 0 and vram_total_mib > 0:
        required_mib = model_size_hint_gb * 1024
        if required_mib > vram_total_mib * 0.9:
            return {
                "source": "model_too_large",
                "reason": (
                    f"Model {model_size_hint_gb:.1f} GB > 90% of VRAM "
                    f"{vram_total_mib/1024:.1f} GB"
                ),
                "details": f"required={required_mib:.0f} MiB, available={vram_total_mib*0.9:.0f} MiB",
            }

    # A transport/server failure alone is not evidence of GPU memory pressure.
    # Reporting it as OOM triggers destructive recovery of otherwise healthy
    # model owners. Require one of the corroborated sources above.
    return None


def _verify_oom_via_vram(gpu_id: str) -> dict:
    """Confirm OOM by reading the cached VRAM snapshot for a GPU.

    Returns:
        {gpu_id, available, free_mib, used_mib, total_mib, pressure, critical}
        where `pressure` is True if free < OOM_VRAM_PRESSURE_THRESHOLD_MIB and
        `critical` is True if free < OOM_VRAM_CRITICAL_THRESHOLD_MIB.

        If the snapshot cannot be obtained the function returns
        {gpu_id, available: False} (or {gpu_id, available: False, error: str}).
    """
    try:
        snapshot = None
        vram_mod = globals().get("vram")
        if vram_mod is not None and hasattr(vram_mod, "get_snapshot"):
            try:
                snapshot = vram_mod.get_snapshot(gpu_id)
            except Exception as e:
                logger.debug(f"[oom-detect] vram.get_snapshot({gpu_id}) raised: {e}")
        # Fallback: read from the cached poll result
        if snapshot is None:
            cache = globals().get("_vram_cache") or {}
            for gpu in cache.get("gpus") or []:
                # Match by gpu_id string or card name; the polled cache
                # uses "card" + "pci_address" rather than "gpu_id".
                if gpu.get("gpu_id") == gpu_id or gpu.get("card") == gpu_id:
                    snapshot = gpu
                    break
        if snapshot is None:
            return {"gpu_id": gpu_id, "available": False}
        free = int(snapshot.get("free_mib") or snapshot.get("free_vram_mib") or 0)
        total = int(snapshot.get("total_mib") or snapshot.get("total_vram_mib") or 0)
        used = int(snapshot.get("used_mib") or snapshot.get("vram_used_mib") or (total - free if total else 0))
        return {
            "gpu_id": gpu_id,
            "available": True,
            "free_mib": free,
            "used_mib": used,
            "total_mib": total,
            "pressure": free > 0 and free < OOM_VRAM_PRESSURE_THRESHOLD_MIB,
            "critical": free > 0 and free < OOM_VRAM_CRITICAL_THRESHOLD_MIB,
        }
    except Exception as e:
        logger.debug(f"[oom-detect] vram verify failed for {gpu_id}: {e}")
        return {"gpu_id": gpu_id, "available": False, "error": str(e)}


def _detect_silent_oom_via_dmesg(since_ts: float = 0.0) -> list[dict]:
    """Scan dmesg for OOM-killer events on Linux.

    Returns a list of dicts with keys: ts, pid, name, gpu_hint, raw_line.
    Called by the watchdog to catch process kills that did not surface to
    the worker (e.g. the kernel reaped a backend process before its 5xx
    reached the worker).

    Recognised patterns:
      - 'Out of memory: Killed process <pid> (<name>)' (Linux kernel OOM killer)
      - 'memory cgroup out of memory: Killed process <pid> (<name>)' (cgroup OOM)
      - 'oom-kill:constraint=CONSTRAINT_MEMCG ... Killed process <pid> (<name>)'
        (constraint-based OOM)

    The kernel OOM-killer message format changed in 5.14+ and may include
    additional fields; we only require pid+name for our purposes.
    """
    events: list[dict] = []
    try:
        # dmesg --since only works on journald; fall back to last 200 lines
        args = None
        if since_ts and since_ts > 0:
            try:
                ago = max(1, int(time.time() - since_ts))
                args = ["dmesg", "--since", f"{ago}s ago"]
            except Exception:
                args = None
        if args is None:
            args = ["dmesg", "-T", "-l", "emerg,alert,crit,err,warn"]
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not (result.stdout or "").strip():
            # Fallback: just get last 200 lines
            result = subprocess.run(
                ["dmesg", "-T"], capture_output=True, text=True, timeout=5,
            )
            lines = (result.stdout or "").splitlines()[-200:]
        else:
            lines = (result.stdout or "").splitlines()
        oom_patterns = [
            re.compile(r"Out of memory: Killed process (\d+) \((\S+)\)", re.IGNORECASE),
            re.compile(r"memory cgroup out of memory: Killed process (\d+) \((\S+)\)", re.IGNORECASE),
            re.compile(r"oom-kill:constraint=CONSTRAINT_MEMCG.*?Killed process (\d+) \((\S+)\)", re.IGNORECASE),
        ]
        for line in lines:
            for pat in oom_patterns:
                m = pat.search(line)
                if m:
                    try:
                        pid = int(m.group(1))
                    except (TypeError, ValueError):
                        pid = 0
                    name = m.group(2) or ""
                    gpu_hint = "cpu"
                    if any(x in name.lower() for x in ("python", "comfyui", "main.py", "sd-server", "llama", "ideogram", "sulphur", "acestep", "gemma", "qwen")):
                        gpu_hint = "gpu_likely"
                    events.append({
                        "ts": time.time(),
                        "pid": pid,
                        "name": name,
                        "gpu_hint": gpu_hint,
                        "raw_line": line[:300],
                    })
                    break
    except Exception as e:
        logger.debug(f"[oom-detect] dmesg scan failed: {e}")
    return events


def _get_force_recover_lock(service_name: str) -> "asyncio.Lock":
    """Per-service lock for _force_recover_failed_service to prevent reentrancy.

    Per adversarial #5: returning the same asyncio.Lock instance for a given
    service_name means concurrent callers (pipeline + watchdog + worker) will
    queue rather than race. Creates the lock on first use so we don't need a
    running event loop at import time.
    """
    lock = _force_recover_locks.get(service_name)
    if lock is None:
        lock = asyncio.Lock()
        _force_recover_locks[service_name] = lock
    return lock


def _get_bundle_load_lock(bundle_name: str) -> "asyncio.Lock":
    """Per-bundle lock for _load_bundle to prevent concurrent load races."""
    lock = _bundle_load_locks.get(bundle_name)
    if lock is None:
        lock = asyncio.Lock()
        _bundle_load_locks[bundle_name] = lock
    return lock


def _get_bundle_admission_lock(bundle_name: str) -> "asyncio.Lock":
    """Serialize every admission path for a bundle, including eviction."""
    lock = _bundle_admission_locks.get(bundle_name)
    if lock is None:
        lock = asyncio.Lock()
        _bundle_admission_locks[bundle_name] = lock
    return lock


class _PhysicalResourceSetLock:
    """Acquire all canonical resource locks in a deterministic order."""

    def __init__(self, resource_ids: tuple[str, ...], atoms: tuple["asyncio.Lock", ...]):
        self.resource_ids = resource_ids
        self._atoms = atoms
        self._acquired = False

    def locked(self) -> bool:
        return self._acquired or any(lock.locked() for lock in self._atoms)

    async def __aenter__(self):
        acquired: list["asyncio.Lock"] = []
        try:
            for lock in self._atoms:
                await lock.acquire()
                acquired.append(lock)
        except BaseException:
            for lock in reversed(acquired):
                lock.release()
            raise
        self._acquired = True
        return self

    async def __aexit__(self, *_exc):
        for lock in reversed(self._atoms):
            if lock.locked():
                lock.release()
        self._acquired = False


def _get_physical_resource_lock(
    gpu_ids: "list[str] | tuple[str, ...] | set[str]",
) -> _PhysicalResourceSetLock:
    """Return a composite lock for the SORTED physical resource ID set.

    Two bundles that touch the same physical GPU set resolve to the SAME
    composite regardless of ordering or bundle name. Intersecting sets share
    their per-device atom locks, so a dual-GPU request conflicts with either
    single-GPU request that touches one of its devices. A disjoint set has no
    shared atoms and can proceed concurrently.
    """
    normalized = tuple(sorted(
        {str(g) for g in gpu_ids if g}
    ))
    if not normalized:
        raise ValueError(
            "_get_physical_resource_lock requires at least one GPU id"
        )
    lock = _physical_resource_locks.get(normalized)
    if lock is None:
        atoms = tuple(
            _physical_resource_atom_locks.setdefault(resource_id, asyncio.Lock())
            for resource_id in normalized
        )
        lock = _PhysicalResourceSetLock(normalized, atoms)
        _physical_resource_locks[normalized] = lock
    return lock


def _reset_physical_resource_locks_for_tests() -> None:
    """Test helper: clear the physical-resource lock cache."""
    _physical_resource_locks.clear()
    _physical_resource_atom_locks.clear()


def _bundle_physical_gpu_set(
    bundle_name: str, bundles: dict | None = None
) -> tuple[str, ...]:
    """Return the sorted physical GPU set touched by *bundle_name*.

    Walks ``linked_to`` and ``linked_from`` so the result spans every
    physical GPU the bundle needs to own. Used to derive the physical
    admission lock key.
    """
    if bundles is None:
        bundles = _resolve_bundles()
    bundle = bundles.get(bundle_name) or {}
    primary = bundle.get("gpu_id")
    gpus: list[str] = []
    if primary:
        gpus.append(str(primary))
    for linked_name in _get_linked_bundles(bundle_name, bundles=bundles):
        lgid = (bundles.get(linked_name) or {}).get("gpu_id")
        if lgid and lgid not in gpus:
            gpus.append(str(lgid))
    return tuple(sorted(gpus))


def _recent_oom_events(limit: int = 100) -> list[dict]:
    """Return recent OOM events for the /health endpoint."""
    return list(_recent_oom_events_log)[-limit:]


async def _init_oom_redis():
    """Initialize the OOM Redis pool. Called from create_app on_startup.

    Per adversarial #11: dead-letter list lives in Redis (sorted set,
    capped, TTL). We prefer to reuse the existing _queue_redis client
    (synchronous redis-py) when available — that keeps the lifecycle tied
    to a single, already-tested connection. If _queue_redis is not set
    (Redis unavailable at create_app time), we open a dedicated client.
    """
    global _oom_redis_pool
    try:
        existing = globals().get("_queue_redis") or globals().get("redis_client")
        if existing is not None:
            _oom_redis_pool = existing
            logger.info("[oom-redis] Reusing existing _queue_redis client for dead-letter store")
            return
        # Fallback: open a dedicated client using the same credential source
        # as the queue engine; this must not drift from the shared client.
        _oom_redis_pool = _redis_mod.Redis(**_redis_connection_kwargs())
        await asyncio.to_thread(_oom_redis_pool.ping)
        logger.info("[oom-redis] Connected to dedicated redis://localhost:6379/0 client")
    except Exception as e:
        logger.warning(f"[oom-redis] Failed to initialize: {e}")
        _oom_redis_pool = None


async def _close_oom_redis():
    """Close the OOM Redis pool. Called on app shutdown.

    No-op if we reused the shared _queue_redis client (which is owned by
    the queue engine and closed elsewhere).
    """
    global _oom_redis_pool
    if _oom_redis_pool is None:
        return
    shared = globals().get("_queue_redis")
    if _oom_redis_pool is shared:
        # Shared client — don't close it here, owner handles lifecycle.
        _oom_redis_pool = None
        return
    try:
        await asyncio.to_thread(_oom_redis_pool.close)
    except Exception:
        pass
    _oom_redis_pool = None


def _dead_letter_identity(entry: dict) -> tuple[str, str] | None:
    """Stable dead-letter identity; blank job IDs are intentionally unkeyed."""
    if not isinstance(entry, dict):
        return None
    job_id = str(entry.get("job_id", "")).strip()
    if not job_id:
        return None
    return job_id, str(entry.get("gpu_id", "")).strip()


def _matching_dead_letter_members(raw_members: list, job_id: str, gpu_id: str) -> list:
    target = (str(job_id).strip(), str(gpu_id).strip())
    matches = []
    if not target[0]:
        return matches
    for raw in raw_members:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if _dead_letter_identity(json.loads(text)) == target:
                matches.append(raw)
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return matches


async def _add_to_dead_letter(job_id: str, gpu_id: str, reason: str, last_state: str) -> bool:
    """Add a job to the dead-letter list (Redis sorted set, capped, TTL).

    Per adversarial #11: sorted set keyed by timestamp (newest = highest
    score), trimmed to the most recent _OOM_DEAD_LETTER_CAP entries, with
    a whole-set TTL of _OOM_DEAD_LETTER_TTL_S so abandoned lists expire
    automatically. Best-effort: returns False if Redis is unreachable.
    """
    if _oom_redis_pool is None:
        return False
    try:
        score = time.time()
        entry = {
            "job_id": job_id,
            "gpu_id": gpu_id,
            "reason": reason,
            "last_state": last_state,
            "ts": score,
        }
        member = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        # Replace prior records for the same job/GPU identity before inserting
        # the latest terminal state. This prevents one recovery from inflating
        # retained history when multiple failure branches report it.
        raw_members = await asyncio.to_thread(
            _oom_redis_pool.zrange, OOM_REDIS_DEAD_LETTER, 0, -1
        )
        duplicates = _matching_dead_letter_members(raw_members, job_id, gpu_id)
        if duplicates:
            await asyncio.to_thread(
                _oom_redis_pool.zrem, OOM_REDIS_DEAD_LETTER, *duplicates
            )
        # ZADD with score=ts; trim oldest entries if over cap (rank 0..-(CAP+1))
        await asyncio.to_thread(
            _oom_redis_pool.zadd, OOM_REDIS_DEAD_LETTER, {member: score}
        )
        await asyncio.to_thread(
            _oom_redis_pool.zremrangebyrank,
            OOM_REDIS_DEAD_LETTER, 0, -(_OOM_DEAD_LETTER_CAP + 1),
        )
        await asyncio.to_thread(
            _oom_redis_pool.expire, OOM_REDIS_DEAD_LETTER, _OOM_DEAD_LETTER_TTL_S
        )
        return True
    except Exception as e:
        logger.debug(f"[oom-dead-letter] Add failed: {e}")
        return False


async def _get_dead_letters(limit: int = 100) -> list[dict]:
    """Return the most recent dead-letter entries (newest first).

    Sorted set is keyed by score=ts; ZREVRANGE returns newest first.

    Returns an empty list if Redis is unreachable, but raises
    RuntimeError on a hard Redis error so callers can distinguish
    "no entries" from "Redis went away". The watchdog's
    ``_refresh_dead_letter_cache`` uses the raise to decide whether
    to stamp the refresh timestamp.
    """
    if _oom_redis_pool is None:
        return []
    try:
        raw = await asyncio.to_thread(
            _oom_redis_pool.zrevrange, OOM_REDIS_DEAD_LETTER, 0, limit - 1
        )
        return [json.loads(r) for r in raw]
    except Exception as e:
        logger.debug(f"[oom-dead-letter] Get failed: {e}")
        raise


async def _refresh_dead_letter_cache() -> int:
    """Refresh _dead_letter_cache from Redis. Returns count.

    Called by the watchdog every ~30s. /health reads len(_dead_letter_cache)
    so the sync handler doesn't have to await Redis.

    On a SUCCESSFUL refresh we also stamp ``_dead_letter_cache_refresh_ts``
    so /health and the websocket payload can compute cache_age_seconds
    without issuing a synchronous Redis call. On failure the timestamp is
    NOT advanced — operators can then see the data is stale.
    """
    global _dead_letter_cache, _dead_letter_cache_refresh_ts
    try:
        rows = await _get_dead_letters(limit=_OOM_DEAD_LETTER_CAP)
    except Exception as e:
        logger.debug(f"[oom-dead-letter] Cache refresh failed: {e}")
        return len(_dead_letter_cache)
    _dead_letter_cache = rows
    _dead_letter_cache_refresh_ts = time.time()
    return len(_dead_letter_cache)


def _rebuild(function, facade_globals, class_cell=None):
    closure = function.__closure__
    if closure is not None and class_cell is not None:
        closure = tuple(class_cell if name == "__class__" else cell for name, cell in zip(function.__code__.co_freevars, closure))
    rebuilt = FunctionType(function.__code__, facade_globals, function.__name__, function.__defaults__, closure)
    rebuilt.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    rebuilt.__annotations__ = function.__annotations__.copy()
    rebuilt.__doc__ = function.__doc__
    rebuilt.__qualname__ = function.__qualname__
    rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt

def _rebuild_class(template, facade_globals):
    class_cell = CellType()
    namespace = {}
    for key, value in vars(template).items():
        if key in ("__dict__", "__weakref__"): continue
        if isinstance(value, FunctionType): value = _rebuild(value, facade_globals, class_cell)
        elif isinstance(value, (staticmethod, classmethod)): value = type(value)(_rebuild(value.__func__, facade_globals, class_cell))
        elif isinstance(value, property): value = property(_rebuild(value.fget, facade_globals, class_cell) if value.fget else None, _rebuild(value.fset, facade_globals, class_cell) if value.fset else None, _rebuild(value.fdel, facade_globals, class_cell) if value.fdel else None, value.__doc__)
        namespace[key] = value
    namespace["__module__"] = facade_globals.get("__name__", template.__module__)
    bound = type(template)(template.__name__, template.__bases__, namespace)
    class_cell.cell_contents = bound
    return bound

def bind(facade_globals):
    if not isinstance(facade_globals, dict): raise TypeError("facade must be its module globals dictionary")
    ns = globals()
    return tuple(_rebuild_class(ns[name], facade_globals) if isinstance(ns[name], type) else _rebuild(ns[name], facade_globals) for name in NAMES)
