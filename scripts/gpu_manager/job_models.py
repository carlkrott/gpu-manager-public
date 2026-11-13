"""Facade-bound plain job model classes.

Class templates live here; each facade bind creates isolated classes whose
methods resolve collaborators from that facade's globals at call time.
"""
from __future__ import annotations

import types


NAMES = ("Job", "AceStepJob", "LLMProxy")


class Job:
    """A queued ComfyUI workflow submission."""

    def __init__(self, workflow: dict, callback_url: str | None = None,
                 priority: int = 0, metadata: dict | None = None):
        self.id = str(uuid.uuid4())[:8]
        self.backend = "comfyui"
        self.workflow = workflow
        self.model_key = extract_model_key(workflow)
        self.fingerprint = extract_model_fingerprint(workflow)
        self.callback_url = callback_url
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.status: str = "queued"  # queued | running | completed | failed | cancelled
        self.result: dict | None = None
        self.comfyui_prompt_id: str | None = None
        self.preempted: bool = False  # flag for preempted-re-queued jobs

    @property
    def priority_tier(self) -> PriorityTier:
        return priority_to_tier(self.priority)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backend": self.backend,
            "model_key": self.model_key,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "priority": self.priority,
            "priority_tier": self.priority_tier.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "comfyui_prompt_id": self.comfyui_prompt_id,
            "metadata": self.metadata,
            "result": self.result,
            "preempted": self.preempted,
        }


class AceStepJob:
    """A queued ACE-Step music generation job."""

    def __init__(self, payload: dict, callback_url: str | None = None,
                 priority: int = 0, metadata: dict | None = None,
                 service_name: str = "acestep", bundle_name: str | None = None,
                 job_id: str | None = None):
        service = _resolve_acestep_service(service_name)
        self.id = job_id or str(uuid.uuid4())[:8]
        self.backend = "acestep"
        self.payload = payload
        self.callback_url = callback_url
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.status: str = "queued"  # queued | running | completed | failed | cancelled
        self.result: dict | None = None
        self.acestep_task_id: str | None = None
        self.preempted: bool = False  # flag for preempted-re-queued jobs
        self.service_name = service["service_name"]
        self.endpoint = service["endpoint"]
        self.audio_dir = service["audio_dir"]
        self.bundle_name = bundle_name or service.get("bundle_name")

    @property
    def priority_tier(self) -> PriorityTier:
        return priority_to_tier(self.priority)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backend": self.backend,
            "status": self.status,
            "priority": self.priority,
            "priority_tier": self.priority_tier.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "acestep_task_id": self.acestep_task_id,
            "service_name": self.service_name,
            "endpoint": self.endpoint,
            "bundle_name": self.bundle_name,
            "metadata": self.metadata,
            "preempted": self.preempted,
            "result": {k: v for k, v in (self.result or {}).items()
                       if k != "lyrics"},  # Don't include lyrics in status
        }


class LLMProxy:
    """Lightweight proxy with health monitoring and request queueing."""

    def __init__(self):
        self.active_requests = 0
        self.queued_requests = 0
        self.total_requests = 0
        self.total_served = 0
        self.total_queued = 0
        self.total_timeouts = 0
        # Explicit process-scoped accounting. Legacy total_* fields remain for
        # compatibility but no longer obscure wait/queue/forward semantics.
        self.wait_attempts = 0
        self.wait_timeouts = 0
        self.durable_queue_admissions = 0
        self.forward_attempts = 0
        self.backend_responses = 0
        self.backend_successes = 0
        self.backend_failures = 0
        self._llm_healthy = False
        self.session: ClientSession | None = None
        self.logger = logging.getLogger("gpu-manager")
        self._llm_ready_event = asyncio.Event()
        self._llm_ready_event.set()

    async def check_health(self) -> bool:
        target = _configured_llm_health_target()
        if target is None:
            self.logger.debug("LLM health check skipped: no configured registry idle service")
            return False
        if self.session is None or self.session.closed:
            self.logger.debug("LLM health check skipped: HTTP session unavailable")
            return False
        url, service_name = target
        try:
            async with self.session.get(
                url,
                timeout=ClientTimeout(total=5, connect=3),
            ) as resp:
                ok = resp.status == 200
                if ok:
                    body = await resp.json()
                    ok = body.get("status") == "ok"
                self.logger.debug(
                    "LLM health check service=%s status=%s ok=%s",
                    service_name,
                    resp.status,
                    ok,
                )
                return ok
        except Exception as e:
            self.logger.warning(f"LLM health check failed: {e}")
            return False

    async def wait_for_llm(self, timeout: float = 120) -> bool:
        if self._llm_healthy:
            return True
        try:
            await asyncio.wait_for(self._llm_ready_event.wait(), timeout=timeout)
            return self._llm_healthy
        except asyncio.TimeoutError:
            return False

    def update_health(self, healthy: bool):
        prev = self._llm_healthy
        self._llm_healthy = healthy
        if healthy and not prev:
            if self.queued_requests > 0:
                self.logger.info(f"LLM healthy — waking {self.queued_requests} queued request(s)")
            self._llm_ready_event.set()
        elif not healthy and prev:
            self.logger.info("LLM offline — future requests will queue")
            self._llm_ready_event.clear()

    _PERSISTED_COUNTERS = (
        "total_requests", "total_served", "total_queued", "total_timeouts",
        "wait_attempts", "wait_timeouts", "durable_queue_admissions", "forward_attempts",
        "backend_responses", "backend_successes", "backend_failures",
    )

    def counters(self) -> dict:
        return {name: getattr(self, name) for name in self._PERSISTED_COUNTERS}

    def restore_counters(self, saved) -> None:
        """Add counters saved in state.json so totals survive a restart."""
        for name in self._PERSISTED_COUNTERS:
            value = saved.get(name) if isinstance(saved, dict) else None
            if type(value) is int and value > 0:
                setattr(self, name, getattr(self, name) + value)

    def status(self) -> dict:
        return {
            "llm_healthy": self._llm_healthy,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_requests": self.total_requests,
            "total_served": self.total_served,
            "total_queued": self.total_queued,
            "total_timeouts": self.total_timeouts,
            "request_metrics": {
                "scope": "state_file",  # saved with state.json (~10 s), restored at start
                "resets_on_restart": False,
                "requests_received": self.total_requests,
                "wait_attempts": self.wait_attempts,
                "wait_timeouts": self.wait_timeouts,
                "durable_queue_admissions": self.durable_queue_admissions,
                "forward_attempts": self.forward_attempts,
                "backend_responses": self.backend_responses,
                "backend_successes": self.backend_successes,
                "backend_failures": self.backend_failures,
            },
        }


def bind(facade_globals):
    """Build the three classes against one facade namespace."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    class_cell = types.CellType()

    def rebuild(function, *, timeout_default=False):
        closure = function.__closure__
        if closure and "__class__" in function.__code__.co_freevars:
            closure = tuple(
                class_cell if name == "__class__" else cell
                for name, cell in zip(function.__code__.co_freevars, closure)
            )
        defaults = function.__defaults__
        if timeout_default:
            defaults = (facade_globals["LLM_WAIT_TIMEOUT"],)
        rebuilt = types.FunctionType(
            function.__code__, facade_globals, function.__name__, defaults, closure
        )
        rebuilt.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
        rebuilt.__annotations__ = function.__annotations__.copy()
        rebuilt.__doc__ = function.__doc__
        rebuilt.__qualname__ = function.__qualname__
        rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
        rebuilt.__dict__.update(function.__dict__)
        return rebuilt

    def bind_class(template):
        class_cell = types.CellType()
        namespace = {}
        for key, value in vars(template).items():
            if key in ("__dict__", "__weakref__"):
                continue
            if isinstance(value, types.FunctionType):
                value = rebuild(value, timeout_default=(
                    template.__name__ == "LLMProxy" and key == "wait_for_llm"
                ))
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

    return tuple(bind_class(globals()[name]) for name in NAMES)
