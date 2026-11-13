"""Facade-bound service events and persistence/recovery helpers."""
from __future__ import annotations

from types import CellType, FunctionType

NAMES = ("_queue_depth_for_group", "ServiceEventBus", "_force_recover_failed_service", "_persist_audio_cpp_result", "_persist_generated_image", "_required_native_helper_token_file")

def _queue_depth_for_group(routing_group: str) -> int:
    """Sum queue depths of all services in the routing group.

    Reuses RoutingEngine.get_group_members(), which already populates
    'queue_depth' per member via _get_queue_depth. Falls back to 0 on
    any exception (e.g. routing_engine not yet initialised or Redis
    hiccup).
    """
    try:
        if not routing_engine:
            return 0
        members = routing_engine.get_group_members(routing_group)
        return sum(int(m.get("queue_depth", 0)) for m in members)
    except Exception:
        logger.debug("queue_depth_for_group(%s) failed", routing_group,
                     exc_info=True)
        return 0

class ServiceEventBus:
    """In-process event bus for service lifecycle changes.

    Publishers: GPU Manager lifecycle code (service start/stop/swap)
    Subscribers: WorkerSupervisor (to spawn/kill workers)

    No Redis, no network — pure asyncio.Event primitives.
    Follows the existing pattern of _idle_check_event, _drain_event, _llm_ready_event.
    """

    def __init__(self):
        # Service name → event (set when state changes)
        self._events: dict[str, asyncio.Event] = {}
        # Service name → last known state
        self._states: dict[str, str] = {}  # "healthy" | "unhealthy" | "starting" | "stopping"
        # Global event — set when ANY service changes (for supervisor wake)
        self._any_change = asyncio.Event()
        # Phase 5.1.5: side-table of services we have *intentionally* asked
        # systemd to stop (via evict_first, evict_llm, _unload_bundle, etc.).
        # The Supervisor consults this before reacting to "unhealthy" /
        # "stopping" events so a planned evict does not tear down an
        # unrelated pool whose service briefly looked unhealthy during the
        # shutdown cascade. TTL-bound so a missed clear never permanently
        # freezes the supervisor.
        self._planned_stops: dict[str, float] = {}
        self._PLANNED_STOP_GRACE_S: float = 600.0  # 10 min backstop; the periodic re-stamp keeps it alive in practice

    def mark_planned_stop(self, service_name: str) -> None:
        """Stamp `service_name` as intentionally being stopped.

        Callers SHOULD invoke this immediately before issuing a
        ``sudo systemctl stop`` they initiated themselves (evict_first,
        evict_llm, _unload_bundle, evict_for_vram). The Supervisor will
        then defer pool-shutdown reactions for `_PLANNED_STOP_GRACE_S`
        seconds — long enough to absorb transient "unhealthy" probes
        during the shutdown cascade.
        """
        import time as _t
        self._planned_stops[service_name] = _t.time()
        logger.debug(f"EventBus: marked planned_stop on {service_name}")

    def clear_planned_stop(self, service_name: str) -> None:
        """Forget a previously-stamped planned stop (idempotent)."""
        self._planned_stops.pop(service_name, None)

    def is_planned_stop(self, service_name: str) -> bool:
        """True iff `mark_planned_stop` was called within the grace window.

        Stale entries (>_PLANNED_STOP_GRACE_S seconds old) are dropped
        on read so the dict cannot grow unboundedly.
        """
        import time as _t
        ts = self._planned_stops.get(service_name)
        if ts is None:
            return False
        if _t.time() - ts > self._PLANNED_STOP_GRACE_S:
            # Stale — drop and report not-planned so the supervisor can
            # proceed with a normal reaction if one is pending.
            self._planned_stops.pop(service_name, None)
            return False
        return True

    def emit(self, service_name: str, new_state: str):
        """Fire a state change event. Called by lifecycle code."""
        old = self._states.get(service_name)
        self._states[service_name] = new_state

        if service_name not in self._events:
            self._events[service_name] = asyncio.Event()
        self._events[service_name].set()
        self._any_change.set()

        logger.info(f"EventBus: {service_name} {old or '?'} -> {new_state}")

    async def wait_for_change(self, timeout: float = 5.0):
        """Wait until any service changes state, or timeout."""
        try:
            await asyncio.wait_for(self._any_change.wait(), timeout=timeout)
            self._any_change.clear()
        except asyncio.TimeoutError:
            pass

    def get_changed_services(self) -> list[tuple[str, str]]:
        """Return (service, state) for all services with pending events."""
        changed = []
        for svc, event in self._events.items():
            if event.is_set():
                event.clear()
                changed.append((svc, self._states.get(svc, "unknown")))
        return changed

    def get_state(self, service_name: str) -> str:
        return self._states.get(service_name, "unknown")

async def _force_recover_failed_service(service_name: str, job_id: str, error: str) -> dict:
    """Force-evict a service that failed (OOM, crash, 5xx) and reset GPU state.

    Does NOT restart the service. The watchdog will re-load the idle LLM
    on its next tick. Desktop processes (Xorg, kwin, etc.) are NOT affected
    because we only target the service's specific systemd_unit.

    Returns a status dict describing what was done.
    """
    config = _load_services_config()
    services = config.get("services", {})
    svc = services.get(service_name, {})
    if not svc:
        return {"ok": False, "error": f"Unknown service: {service_name}"}
    # Only recover generation_backend services
    if svc.get("type") != "generation_backend":
        return {"ok": False, "error": f"Service {service_name} is not a generation backend, skipping recovery"}
    gpu_id = svc.get("gpu_id")
    systemd_unit = svc.get("systemd_unit", "")
    if not systemd_unit:
        return {"ok": False, "error": f"Service {service_name} has no systemd_unit, cannot recover"}
    result = {
        "ok": True,
        "service": service_name,
        "gpu_id": gpu_id,
        "systemd_unit": systemd_unit,
        "actions": [],
    }
    # Find the bundle that contains this service
    bundles = _resolve_bundles()
    matching_bundle = None
    for bname, bcfg in bundles.items():
        if service_name in (bcfg.get("services") or []):
            matching_bundle = bname
            break
    admission_lock = (
        _get_bundle_admission_lock(matching_bundle)
        if matching_bundle else None
    )
    if admission_lock is not None:
        await admission_lock.acquire()
    recovery_stop_ok = True
    try:
        if matching_bundle:
            result["bundle"] = matching_bundle
            # Bundle recovery owns one linked-graph mutation under the durable
            # controller fence. A rejected/stale transition must not clear the
            # authoritative GPU state behind the still-running bundle.
            try:
                unload_ok = await _teardown_bundle_graph(
                    matching_bundle,
                    reason=f"failure-recovery:{service_name}:{job_id}",
                )
                result["actions"].append(
                    f"teardown_bundle({matching_bundle}) -> {unload_ok}"
                )
                recovery_stop_ok = bool(unload_ok)
            except Exception as e:
                recovery_stop_ok = False
                result["actions"].append(
                    f"teardown_bundle({matching_bundle}) failed: {e}"
                )
        else:
            # Not in a bundle — just stop the systemd unit directly
            try:
                await vram._systemctl("stop", systemd_unit)
                result["actions"].append(f"systemctl stop {systemd_unit} (no bundle)")
            except Exception as e:
                recovery_stop_ok = False
                result["actions"].append(f"systemctl stop {systemd_unit} failed: {e}")

        if not recovery_stop_ok:
            result["ok"] = False
            result["error"] = (
                f"failed recovery teardown deferred for {service_name}; "
                "authoritative GPU state retained"
            )

        # Clear GPU state only after the stop/teardown actually succeeded.
        if gpu_id and recovery_stop_ok:
            gs = _get_gpu_state(gpu_id)
            if gs is not None:
                old_bundle = gs.loaded_bundle
                if matching_bundle:
                    _transition_linked_bundle_state(
                        matching_bundle, [], source="_force_recover_failed_service"
                    )
                else:
                    gs.loaded_bundle = None
                    gs.bundle_priority = 0
                    gs.state = "idle"
                    gs.set_loaded_bundles([], source="_force_recover_failed_service")
                    gs.set_current_priority(0, source="_force_recover_failed_service")
                result["actions"].append(
                    f"cleared GpuBundleState for {gpu_id} (was {old_bundle})"
                )
                if matching_bundle:
                    result["actions"].append(
                        f"cleared linked scheduler state for {matching_bundle}"
                    )
    finally:
        if admission_lock is not None and admission_lock.locked():
            admission_lock.release()
    logger.warning(
        f"force_recover: {service_name} (job {job_id}) — {error[:200]}. "
        f"Actions: {result['actions']}"
    )
    return result

def _persist_audio_cpp_result(job_id: str, response_text: str) -> str:
    """Persist native audio.cpp PCM and return a bounded artifact receipt JSON.

    audio.cpp returns the completed WAV inline.  Keeping that base64 in the
    Redis job hash is both wasteful and unusable by the coordinator's bounded
    leaf contract, so the worker commits the file before terminalising the job.
    """
    import base64
    import wave

    result = json.loads(response_text)
    if not isinstance(result, dict):
        raise ValueError("audio.cpp result is not an object")
    encoded = result.pop("audio", None)
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("audio.cpp completed response has no audio")
    audio = base64.b64decode(encoded, validate=True)
    output_root = Path(
        os.environ.get(
            "GPU_MANAGER_GENERATION_OUTPUT_DIR",
            str(PROJECT_ROOT / "output" / "gpu-manager"),
        )
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{job_id}-minimax-music3.wav"
    tmp = path.with_suffix(".wav.tmp")
    tmp.write_bytes(audio)
    with wave.open(str(tmp), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
    if channels < 1 or sample_rate < 8000 or sample_width != 2 or frames <= 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("audio.cpp returned an invalid 16-bit PCM WAV")
    tmp.replace(path)
    digest = hashlib.sha256(audio).hexdigest()
    artifact = {
        "path_or_url": str(path),
        "sha256": digest,
        "byte_size": len(audio),
        "media_type": "audio/wav",
        "duration_seconds": frames / sample_rate,
        "channels": channels,
        "sample_rate": sample_rate,
    }
    result["artifact"] = artifact
    result["audio"] = artifact
    result.setdefault("generation_receipt", {})
    if isinstance(result["generation_receipt"], dict):
        result["generation_receipt"].update({
            "job_id": job_id,
            "model_id": result.get("model") or "minimax-music3-audiocpp-q8-bf16",
            "artifact_sha256": digest,
        })
    return json.dumps(result, sort_keys=True, separators=(",", ":"))

def _persist_generated_image(job_id: str, image: Mapping[str, Any], index: int) -> dict:
    import base64
    from pathlib import Path
    from native_task_host import atomic_write

    raw = base64.b64decode(image["data_b64"], validate=True)
    _verified_image_bytes(raw)
    root = Path(os.environ.get("GPU_MANAGER_GENERATION_OUTPUT_DIR",
                              str(PROJECT_ROOT / "output" / "gpu-manager"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(image.get("filename") or "image.png")).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise RuntimeError("unsupported generated image artifact extension")
    path = root / f"{job_id}-image-{index}{suffix}"
    atomic_write(path, raw)
    return {"path_or_url": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_size": len(raw), "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

def _required_native_helper_token_file() -> str:
    """Resolve the dedicated native-task helper credential before network I/O."""
    from runtime_host_client import (
        HELPER_TOKEN_FILE_ENV,
        NATIVE_TASK_TOKEN_FILE_ENV,
        resolve_required_service_credential,
    )

    token_file = os.environ.get(NATIVE_TASK_TOKEN_FILE_ENV)
    if not token_file:
        if os.environ.get(HELPER_TOKEN_FILE_ENV):
            logger.warning(
                "%s is deprecated and ignored; configure %s for native-task authentication",
                HELPER_TOKEN_FILE_ENV,
                NATIVE_TASK_TOKEN_FILE_ENV,
            )
        raise RuntimeError("native task helper credential is required")
    try:
        resolve_required_service_credential(service_token_file=token_file)
    except Exception as exc:
        raise RuntimeError("native task helper credential is unavailable") from exc
    return token_file


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
