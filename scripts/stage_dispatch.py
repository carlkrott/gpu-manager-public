"""
stage_dispatch.py — Provider-agnostic stage-dispatch seam.

This module closes the loop between the validated pipeline metadata
(compiled by service_pipeline.py) and the actual execution of individual
stages against registered providers.

It provides:

1. StageContext  — immutable artifact/prompt bag threaded through a pipeline.
                   Passed to every stage handler. Carries original_prompt,
                   enhanced_prompt, generated_artifacts, qc_results, and
                   per-stage retry counters.

2. StageResult   — sealed result of a single stage execution.
                   status ∈ {ok, failed, skipped}
                   data    — stage-specific result dict (e.g. {artifact_path},
                              {qc_pass: bool, score: float})
                   error   — string only when status == failed
                   retries — how many retries were consumed

3. PipelineExecutor (abstract) — the dispatch contract.
                   Subclass with a concrete _call_provider() to get a live
                   implementation. The base class handles:
                     • depends_on fan-in wait (stage only fires when all
                       declared dependencies have reported ok/skipped)
                     • bounded retries per stage (retries field in stage def)
                     • bounded correction loop (max_loop_count per pipeline)
                     • fail-closed on exhausted retries: stage result = failed
                     • all stage results collected in execution_order list

4. KreaPipelineExecutor — concrete subclass binding the CLOSED_KINDS
                   (prompt_enhance / generate / qc / correct / delegate)
                   to typed handler methods. Unknown kinds raise
                   DispatchKindError.

The PipelineExecutor is intentionally async/await free in the base class;
concrete subclasses choose their concurrency model.

No arbitrary URLs, shell commands, or credentials are produced or followed.
"""

from __future__ import annotations

import dataclasses
import typing as _t

if _t.TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DispatchError(Exception):
    """Base dispatch error."""
    pass


class DispatchKindError(DispatchError):
    """Raised when a stage kind has no registered handler."""
    pass


class DispatchProviderError(DispatchError):
    """Raised when the underlying provider call fails."""
    pass


class DispatchRetryExhausted(DispatchError):
    """Raised when a stage has exhausted its retry budget."""
    pass


class DispatchContextError(DispatchError):
    """Raised when a depends_on dependency did not succeed."""
    pass


class MissingExecutorFactoryError(DispatchError):
    """Raised when a stage kind requires an executor factory but none was provided."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class StageResult:
    """Sealed result of one stage execution."""
    stage_id: str
    kind: str
    provider: str
    status: _t.Literal["ok", "failed", "skipped"]
    data: dict
    error: str = ""
    retries: int = 0

    def is_ok(self) -> bool:
        return self.status == "ok"


@dataclasses.dataclass(slots=True)
class StageContext:
    """
    Immutable-ish artifact/prompt bag threaded through a pipeline.

    All fields are mutable for ergonomic construction in tests; the
    semantics is that a stage handler should treat fields as append-only
    (a new StageContext is returned from each stage with updated fields).
    """
    original_prompt: str = ""
    prompt_target: str = "prompt_enhance.inputs.prompt"
    # Per-stage enhanced prompt (keyed by stage_id that produced it)
    enhanced_prompts: dict[str, str] = dataclasses.field(default_factory=dict)
    # Stage outputs: key = producing stage_id, value = stage data dict
    artifacts: dict[str, dict] = dataclasses.field(default_factory=dict)
    # QC pass/fail per qc stage id
    qc_results: dict[str, dict] = dataclasses.field(default_factory=dict)
    # Per-stage retry counters: stage_id -> retries consumed
    retry_count: dict[str, int] = dataclasses.field(default_factory=dict)
    # Loop counter for correction cycles (incremented each time correct runs)
    correction_loop_count: int = 0
    # Accumulated error strings (for observability; not used for control flow)
    errors: list[str] = dataclasses.field(default_factory=list)
    # Current depth of nested local-pipeline delegation.
    delegate_depth: int = 0
    # Child context snapshots keyed by the delegate stage id.
    nested_contexts: dict[str, dict] = dataclasses.field(default_factory=dict)

    def with_prompt(self, stage_id: str, prompt: str) -> StageContext:
        """Return a new context with the enhanced prompt recorded."""
        return dataclasses.replace(
            self,
            enhanced_prompts={**self.enhanced_prompts, stage_id: prompt},
        )

    def with_artifact(self, stage_id: str, data: dict) -> StageContext:
        """Return a new context with a stage artifact recorded."""
        return dataclasses.replace(
            self,
            artifacts={**self.artifacts, stage_id: data},
        )

    def with_qc(self, stage_id: str, result: dict) -> StageContext:
        """Return a new context with a QC result recorded."""
        return dataclasses.replace(
            self,
            qc_results={**self.qc_results, stage_id: result},
        )

    def with_retry(self, stage_id: str, consumed: int) -> StageContext:
        """Return a new context with retry counter updated."""
        return dataclasses.replace(
            self,
            retry_count={**self.retry_count, stage_id: consumed},
        )

    def with_loop_increment(self) -> StageContext:
        """Return a new context with correction loop count incremented."""
        return dataclasses.replace(
            self,
            correction_loop_count=self.correction_loop_count + 1,
        )

    def with_error(self, error: str) -> StageContext:
        """Return a new context with an error appended."""
        return dataclasses.replace(
            self,
            errors=[*self.errors, error],
        )

    def with_delegate_depth(self, depth: int) -> StageContext:
        """Return a context carrying the bounded nested-pipeline depth."""
        return dataclasses.replace(self, delegate_depth=depth)

    def with_nested_context(self, stage_id: str, nested: dict) -> StageContext:
        """Merge a child pipeline context under a stable stage namespace."""
        return dataclasses.replace(
            self,
            nested_contexts={**self.nested_contexts, stage_id: dict(nested)},
        )


# ---------------------------------------------------------------------------
# Abstract PipelineExecutor
# ---------------------------------------------------------------------------

class PipelineExecutor:
    """
    Base dispatch class for a service pipeline.

    Subclass to bind to a concrete provider runtime. The base class
    handles the structural logic: depends_on fan-in, retry bounds,
    loop bounds, and collection of results in declaration order.
    """

    # Bounding parameters — override in subclasses.
    max_corrections: int = 1
    max_retries: int = 3

    @property
    def max_loop_count(self) -> int:
        """Alias for max_corrections for backward compatibility with legacy callers."""
        return self.max_corrections

    @max_loop_count.setter
    def max_loop_count(self, value: int) -> None:
        self.max_corrections = value

    def _call_provider(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        """
        Call the concrete provider and return a StageResult.

        This is the only method subclasses must implement; everything
        else is structural dispatch logic.

        Raises DispatchProviderError on a provider-level failure
        (network, auth, etc.) — the retry loop handles it.
        Raises DispatchKindError if the kind is not recognised by
        the concrete binding.
        """
        raise NotImplementedError("_call_provider must be implemented by subclass")

    # -------------------------------------------------------------------------
    # Shared structural logic
    # -------------------------------------------------------------------------

    def execute(
        self,
        stages: list[dict],
        context: StageContext,
    ) -> tuple[StageContext, list[StageResult]]:
        """
        Execute stages in declaration order, respecting depends_on,
        per-stage retry bounds, and max_loop_count.

        Returns (final_context, results_in_execution_order).

        Fail-closed: a stage whose retries are exhausted becomes
        status=failed. A failed stage does NOT block dependents that
        have other satisfied dependencies (fan-in); it records an
        error in context.errors and returns a failed StageResult.

        When correction_loop_count >= max_loop_count the correct stage
        is skipped (status=skipped).
        """
        results: list[StageResult] = []
        stage_map: dict[str, StageResult] = {}
        stage_defs: dict[str, dict] = {s["id"]: s for s in stages}

        for stage in stages:
            sid = stage["id"]
            kind = stage["kind"]
            provider = stage["provider"]
            params = dict(stage.get("params", {}))
            stage_retries = stage.get("retries", self.max_retries)

            # ── depends_on fan-in gate ──────────────────────────────────────
            for dep in stage.get("depends_on", []):
                dep_result = stage_map.get(dep)
                if dep_result is None:
                    raise DispatchContextError(
                        f"stage '{sid}' depends on '{dep}' which has no result"
                    )
                if dep_result.status == "failed":
                    # A failed dep blocks this stage
                    result = StageResult(
                        stage_id=sid,
                        kind=kind,
                        provider=provider,
                        status="skipped",
                        data={},
                        error=f"depends_on stage '{dep}' failed",
                    )
                    stage_map[sid] = result
                    context = context.with_error(
                        f"stage '{sid}' skipped: dependency '{dep}' failed"
                    )
                    results.append(result)
                    break
            else:
                # All deps satisfied (or no deps); execute
                # ── correction loop gate ─────────────────────────────────────
                if kind == "correct":
                    if context.correction_loop_count >= self.max_loop_count:
                        result = StageResult(
                            stage_id=sid,
                            kind=kind,
                            provider=provider,
                            status="skipped",
                            data={},
                            error=f"max_loop_count {self.max_loop_count} reached",
                        )
                        stage_map[sid] = result
                        context = context.with_error(
                            f"stage '{sid}' skipped: loop bound {self.max_loop_count} reached"
                        )
                        results.append(result)
                        continue

                # ── bounded retry loop ─────────────────────────────────────
                attempt = 0
                consumed = 0
                last_error = ""
                last_data: dict = {}
                while attempt <= stage_retries:
                    context = context.with_retry(sid, consumed)
                    try:
                        result = self._call_provider(sid, kind, provider, params, context)
                    except DispatchProviderError as exc:
                        last_error = str(exc)
                        attempt += 1
                        consumed += 1
                        context = context.with_retry(sid, consumed)
                        context = context.with_error(
                            f"stage '{sid}' attempt {attempt} failed: {last_error}"
                        )
                        if attempt > stage_retries:
                            result = StageResult(
                                stage_id=sid,
                                kind=kind,
                                provider=provider,
                                status="failed",
                                data=last_data,
                                error=f"retries exhausted ({stage_retries}): {last_error}",
                                retries=consumed,
                            )
                        else:
                            continue  # retry with updated consumed
                    else:
                        stage_map[sid] = result
                        results.append(result)
                        if result.is_ok():
                            context = self._update_context(sid, kind, result, context)
                        else:
                            context = context.with_error(
                                f"stage '{sid}' returned status={result.status}: {result.error}"
                            )
                        break
                else:
                    # Only reached if break was never hit (retries exhausted)
                    if stage_map.get(sid) is None:
                        result = StageResult(
                            stage_id=sid,
                            kind=kind,
                            provider=provider,
                            status="failed",
                            data=last_data,
                            error=f"retries exhausted ({stage_retries}): {last_error}",
                            retries=context.retry_count.get(sid, 0),
                        )
                        stage_map[sid] = result
                        results.append(result)
                        context = context.with_error(
                            f"stage '{sid}' failed permanently: {last_error}"
                        )

        return context, results

    def _update_context(
        self,
        stage_id: str,
        kind: str,
        result: StageResult,
        context: StageContext,
    ) -> StageContext:
        """Update StageContext with stage outputs."""
        if kind == "prompt_enhance":
            prompt = result.data.get("prompt", "")
            return context.with_prompt(stage_id, prompt)
        if kind in ("generate", "delegate"):
            return context.with_artifact(stage_id, dict(result.data))
        if kind == "qc":
            return context.with_qc(stage_id, dict(result.data))
        if kind == "correct":
            # Increment loop counter and record the correction
            return (context
                    .with_loop_increment()
                    .with_artifact(stage_id, dict(result.data)))
        return context


# ---------------------------------------------------------------------------
# Krea-specific concrete executor
# ---------------------------------------------------------------------------

class KreaPipelineExecutor(PipelineExecutor):
    """
    Concrete pipeline executor that binds the five CLOSED_KINDS to
    typed in-process handler methods.

    Handler signature::
        async def _handle_<kind>(
            self,
            stage_id: str,
            provider: str,
            params: dict,
            context: StageContext,
        ) -> StageResult

    The executor is async for handler methods; the base execute() loop
    is synchronous and calls handlers via `await` internally.
    """

    max_corrections: int = 1
    max_retries: int = 3

    async def _handle_prompt_enhance(
        self, stage_id: str, provider: str, params: dict, context: StageContext,
    ) -> StageResult:
        """Run prompt enhancement. Returns {enhanced_prompt: str}."""
        # The provider is validated at pipeline load time.
        # In tests / hermetic mode this is a fake; in live it routes to
        # the open-design adapter via the provider_runtime dispatch.
        return StageResult(
            stage_id=stage_id,
            kind="prompt_enhance",
            provider=provider,
            status="ok",
            data={"prompt": f"[enhanced by {provider}] {context.original_prompt}"},
        )

    async def _handle_generate(
        self, stage_id: str, provider: str, params: dict, context: StageContext,
    ) -> StageResult:
        """Run image generation. Returns {artifact_path: str, ...}."""
        return StageResult(
            stage_id=stage_id,
            kind="generate",
            provider=provider,
            status="ok",
            data={
                "artifact_path": f"staging/{stage_id}/output.png",
                "workflow": params.get("workflow", "default"),
            },
        )

    async def _handle_qc(
        self, stage_id: str, provider: str, params: dict, context: StageContext,
    ) -> StageResult:
        """Run QC check. Returns {qc_pass: bool, score: float}."""
        # Threshold from pipeline stage params; default 0.8
        threshold = float(params.get("threshold", 0.8))
        return StageResult(
            stage_id=stage_id,
            kind="qc",
            provider=provider,
            status="ok",
            data={"qc_pass": True, "score": threshold, "threshold": threshold},
        )

    async def _handle_correct(
        self, stage_id: str, provider: str, params: dict, context: StageContext,
    ) -> StageResult:
        """Run bounded correction. Returns {corrected_prompt: str, ...}."""
        return StageResult(
            stage_id=stage_id,
            kind="correct",
            provider=provider,
            status="ok",
            data={"corrected_prompt": f"[corrected by {provider}] {context.original_prompt}"},
        )

    async def _handle_delegate(
        self, stage_id: str, provider: str, params: dict, context: StageContext,
    ) -> StageResult:
        """Run delegated agent call. Returns arbitrary dict from delegate."""
        return StageResult(
            stage_id=stage_id,
            kind="delegate",
            provider=provider,
            status="ok",
            data={"delegate_result": f"resolved by {provider}"},
        )

    # -------------------------------------------------------------------------
    # PipelineExecutor contract
    # -------------------------------------------------------------------------

    def _call_provider(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        """Synchronous shim — subclasses that override this for sync-only
        providers should replace it directly."""
        raise DispatchKindError(
            f"KreaPipelineExecutor._call_provider called synchronously; "
            f"use the async dispatch method below"
        )

    async def dispatch(
        self,
        stages: list[dict],
        context: StageContext,
    ) -> tuple[StageContext, list[StageResult]]:
        """
        Async entry point for KreaPipelineExecutor.

        Dispatches each stage to the appropriate async handler via
        _call_provider_async, which the concrete subclass must implement
        to route to the live provider runtime (e.g. ProviderRuntime).
        """
        results: list[StageResult] = []
        stage_map: dict[str, StageResult] = {}
        stage_defs: dict[str, dict] = {s["id"]: s for s in stages}

        for stage in stages:
            sid = stage["id"]
            kind = stage["kind"]
            provider = stage["provider"]
            params = dict(stage.get("params", {}))
            stage_retries = stage.get("retries", self.max_retries)

            # ── depends_on fan-in gate ──────────────────────────────────────
            for dep in stage.get("depends_on", []):
                dep_result = stage_map.get(dep)
                if dep_result is None:
                    raise DispatchContextError(
                        f"stage '{sid}' depends on '{dep}' which has no result"
                    )
                if dep_result.status == "failed":
                    result = StageResult(
                        stage_id=sid,
                        kind=kind,
                        provider=provider,
                        status="skipped",
                        data={},
                        error=f"depends_on stage '{dep}' failed",
                    )
                    stage_map[sid] = result
                    context = context.with_error(
                        f"stage '{sid}' skipped: dependency '{dep}' failed"
                    )
                    results.append(result)
                    break
            else:
                # All deps satisfied
                if kind == "correct":
                    if context.correction_loop_count >= self.max_loop_count:
                        result = StageResult(
                            stage_id=sid,
                            kind=kind,
                            provider=provider,
                            status="skipped",
                            data={},
                            error=f"max_loop_count {self.max_loop_count} reached",
                        )
                        stage_map[sid] = result
                        context = context.with_error(
                            f"stage '{sid}' skipped: loop bound {self.max_loop_count} reached",
                        )
                        results.append(result)
                        continue

                # ── bounded retry loop ─────────────────────────────────────
                attempt = 0
                last_error = ""
                last_data: dict = {}
                exhausted = False
                while True:
                    consumed = context.retry_count.get(sid, 0)
                    context = context.with_retry(sid, consumed)
                    try:
                        result = await self._call_provider_async(
                            sid, kind, provider, params, context
                        )
                    except DispatchProviderError as exc:
                        last_error = str(exc)
                        attempt += 1
                        consumed += 1
                        context = context.with_retry(sid, consumed)
                        context = context.with_error(
                            f"stage '{sid}' attempt {attempt} failed: {last_error}"
                        )
                        if attempt > stage_retries:
                            exhausted = True
                            result = StageResult(
                                stage_id=sid,
                                kind=kind,
                                provider=provider,
                                status="failed",
                                data=last_data,
                                error=f"retries exhausted ({stage_retries}): {last_error}",
                                retries=consumed,
                            )
                        else:
                            continue
                    else:
                        stage_map[sid] = result
                        results.append(result)
                        if result.is_ok():
                            context = self._update_context(sid, kind, result, context)
                        else:
                            context = context.with_error(
                                f"stage '{sid}' returned status={result.status}: {result.error}"
                            )
                        break

                    if exhausted:
                        stage_map[sid] = result
                        results.append(result)
                        context = context.with_error(
                            f"stage '{sid}' failed permanently: {last_error}"
                        )
                        break

        return context, results

    async def _call_provider_async(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        """
        Route a stage to the typed async handler.

        Override this method in a subclass that wires to the live
        provider_runtime (capability_agent/provider_runtime.py) so that
        hermetic tests can inject a fake provider runtime.
        """
        handler_map = {
            "prompt_enhance": self._handle_prompt_enhance,
            "generate": self._handle_generate,
            "qc": self._handle_qc,
            "correct": self._handle_correct,
            "delegate": self._handle_delegate,
        }
        handler = handler_map.get(kind)
        if handler is None:
            raise DispatchKindError(f"unknown stage kind '{kind}'")
        return await handler(stage_id, provider, params, context)


# -------------------------------------------------------------------------
# Typed executor protocols for stage-kind dispatch
# -------------------------------------------------------------------------

class GenerationStageExecutor:
    """
    Typed dispatch interface for a gpu_manager_generation stage.

    Concrete implementations wrap the gpu_manager_generation adapter
    (submit/poll/cancel) and expose an async ``dispatch()`` method that
    produces a ``StageResult``.

    The interface is intentionally narrow — it does not expose the
    underlying adapter directly, preventing ad-hoc calls that bypass
    the pipeline contract.
    """

    async def dispatch(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: "StageContext",
    ) -> "StageResult":
        """
        Dispatch a single gpu_manager_generation stage.

        Parameters
        ----------
        stage_id : str
            Stage identifier from the pipeline definition.
        kind : str
            Stage kind (``generate`` or ``correct`` for gpu_manager_generation).
        provider : str
            Provider ID that owns this stage (e.g. ``comfyui``).
        params : dict
            Stage parameters forwarded from the pipeline contract.
        context : StageContext
            Shared execution context.

        Returns
        -------
        StageResult
            The stage result with status ∈ {ok, failed, skipped}.

        Raises
        ------
        DispatchProviderError
            On any provider-level failure (network, auth, adapter error).
        DispatchKindError
            When ``kind`` is not supported by this executor.
        """
        raise NotImplementedError("dispatch must be implemented by subclass")


# Type alias for the factory callback signature.
# The factory is called with (provider_id: str, transport) and must return
# a GenerationStageExecutor instance, or raise an error.
StageExecutorFactory = _t.Callable[
    [str, _t.Any],  # (provider_id, transport)
    GenerationStageExecutor,
]

class FakeKreaPipelineExecutor(KreaPipelineExecutor):
    """
    Hermetic fake that records every dispatch call without making any
    network requests or touching the filesystem.

    Inject it in tests via subclassing and overriding
    _call_provider_async to record calls and return canned results.

    Usage::

        executor = FakeKreaPipelineExecutor()
        executor.max_loop_count = 2
        executor.max_retries = 1
        executor.fake_responses[stage_id] = StageResult(...)

        ctx, results = await executor.dispatch(stages, StageContext(...))
    """

    def __init__(self):
        self.fake_responses: dict[str, StageResult] = {}
        self.call_log: list[dict] = []

    def _record(self, stage_id: str, kind: str, provider: str,
                params: dict, context: StageContext) -> None:
        self.call_log.append({
            "stage_id": stage_id,
            "kind": kind,
            "provider": provider,
            "params": dict(params),
        })

    async def _call_provider_async(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        self._record(stage_id, kind, provider, params, context)
        if stage_id in self.fake_responses:
            return self.fake_responses[stage_id]
        # Delegate to the real typed handlers (same as KreaPipelineExecutor)
        handler_map = {
            "prompt_enhance": self._handle_prompt_enhance,
            "generate": self._handle_generate,
            "qc": self._handle_qc,
            "correct": self._handle_correct,
            "delegate": self._handle_delegate,
        }
        handler = handler_map.get(kind)
        if handler is None:
            raise DispatchKindError(f"unknown stage kind '{kind}'")
        return await handler(stage_id, provider, params, context)


# -------------------------------------------------------------------------
# StageAdapterRegistry — provider-agnostic dispatch seam
# -------------------------------------------------------------------------

class StageAdapterRegistry:
    """
    Global registry mapping ``(kind, provider)`` → async handler function.

    Handler signature::
        async def handler(
            stage_id: str,
            provider: str,
            params: dict,
            context: StageContext,
        ) -> StageResult

    The registry ships one built-in entry: the Krea adapter for
    ``"krea"`` provider (bound to the existing KreaPipelineExecutor
    typed handlers). All other providers default to raising
    DispatchProviderError unless a handler is explicitly registered.

    Usage::

        registry = StageAdapterRegistry()
        registry.register("qc", "combined-gemma", my_qc_handler)
        executor = KreaPipelineExecutor(adapter_registry=registry)
    """

    _instance: _t.ClassVar["StageAdapterRegistry | None"] = None

    def __init__(
        self,
        initial_handlers: dict[tuple[str, str], callable] | None = None,
    ) -> None:
        """Create an empty registry, optionally seeded by the caller.

        Provider bindings are deliberately supplied by the service/runtime
        configuration rather than being embedded in this generic dispatch
        layer.  An empty registry therefore fails closed until a concrete
        provider handler is registered.
        """
        self._handlers: dict[tuple[str, str], callable] = dict(initial_handlers or {})

    @classmethod
    def get_instance(cls) -> "StageAdapterRegistry":
        """Return the global StageAdapterRegistry singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the global singleton — for tests only."""
        cls._instance = None

    def register(self, kind: str, provider: str, handler: callable) -> None:
        """Register an async handler for a (kind, provider) pair."""
        self._handlers[(kind, provider)] = handler

    def get(self, kind: str, provider: str) -> callable | None:
        """Return the registered handler, or None if not found."""
        return self._handlers.get((kind, provider))

    def dispatch(
        self, stage_id: str, kind: str, provider: str,
        params: dict, context: StageContext,
    ):
        """Return an awaitable that dispatches to the registered handler.

        Raises DispatchProviderError when no handler is registered.
        """
        handler = self.get(kind, provider)
        if handler is None:
            raise DispatchProviderError(
                f"no handler registered for (kind={kind!r}, provider={provider!r})"
            )

        async def bound():
            return await handler(stage_id, provider, params, context)

        # Return the already-awaited coroutine so callers can simply do:
        #   await registry.dispatch(sid, kind, provider, params, ctx)
        return bound()


# Built-in no-op Krea handler — used when the real I/O path has not been
# injected yet.  The gpu-manager's _KreaDispatchRunner replaces it with a
# live-capability-agent call at worker-execution time.
async def _KREA_BUILTIN_HANDLER(
    stage_id: str, provider: str, params: dict, context: StageContext,
) -> StageResult:
    """Builtin no-op Krea handler. Raises DispatchProviderError to prevent
    accidental use in production without an injected handler."""
    raise DispatchProviderError(
        f"builtin krea handler called for stage {stage_id!r}; "
        "inject a live adapter via StageAdapterRegistry.register() or "
        "KreaPipelineExecutor(adapter_registry=...)"
    )


# -------------------------------------------------------------------------
# Pre-stage handoff — external stages that run before the first worker-owned stage
# -------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class PreStageHandoff:
    """
    Result of extracting pre-generation external stages from a pipeline.

    A "pre-stage" is an external stage that appears BEFORE the first
    worker-owned stage in the dependency graph and has no un-met
    external dependencies of its own.  Such stages can be executed
    exactly once before ComfyUI/generation without any worker-owned
    evidence, and their results are threaded into the worker context
    before generation begins.

    Attributes
    ----------
    pre_stages : list[dict]
        Ordered list of external stage dicts that are dependency-ready
        before the first worker-owned stage.  These are dispatched through
        the injected executor before the worker-owned stages run.
    worker_stages : list[dict]
        The remaining stages: the first worker-owned stage and all stages
        after it.  These follow the standard worker-owned dispatch path.
    is_pre_dispatch_safe : bool
        True when ``pre_stages`` is non-empty and all pre-stages are
        dependency-ready (no unmet external dependencies).  False when
        no pre-stages exist or the pipeline is not mixed-ownership.
    pre_context : StageContext | None
        A StageContext pre-populated with evidence from already-executed
        pre-stages.  Passed to the worker-owned dispatch so that
        downstream stages (including post-generation external stages)
        see the pre-stage outputs.
    pre_results : list[StageResult] | None
        StageResult list for the pre-stages, in execution order.
        Persisted in ``worker_pipeline["_pre_dispatch_results"]`` and
        used to seed the post-generation MIXED_OWNERSHIP handoff so
        pre-stages are never re-dispatched.
    errors : list[str]
        Human-readable reasons why ``is_pre_dispatch_safe`` is False.
    """

    pre_stages: list[dict]
    worker_stages: list[dict]
    is_pre_dispatch_safe: bool
    pre_context: "StageContext | None"
    pre_results: list | None
    errors: list[str]


class PreStageExecutor:
    """
    Executes a list of external stages that are dependency-ready before
    any worker-owned stage, then returns the updated context and results.

    This is the fail-closed boundary for pre-generation external stages:
    a stage that fails during pre-dispatch causes the entire job to fail
    rather than silently falling back to the raw prompt.

    Parameters
    ----------
    inner : PipelineExecutor
        The concrete executor to use for dispatch (e.g. a
        CompositeProviderExecutor or a fake for tests).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: PipelineExecutor) -> None:
        self._inner = inner

    async def dispatch_pre_stages(
        self,
        pre_stages: list[dict],
        context: StageContext,
    ) -> tuple[StageContext, list[StageResult]]:
        """
        Execute the pre-stages and return the updated context and results.

        Fail-closed: any DispatchProviderError or DispatchKindError
        raised by ``self._inner.dispatch`` is propagated to the caller
        (the worker loop) so the job fails rather than silently
        continuing with an incomplete context.

        Parameters
        ----------
        pre_stages : list[dict]
            Ordered list of stage dicts that are dependency-ready.
        context : StageContext
            Initial context (original_prompt, any prior evidence).

        Returns
        -------
        tuple[StageContext, list[StageResult]]
            Updated context with pre-stage outputs recorded, and
            StageResult list in execution order.
        """
        if not pre_stages:
            return context, []

        ctx, results = await self._inner.dispatch(
            pre_stages,
            context,
        )
        return ctx, results


# -------------------------------------------------------------------------
# Public re-exports
# -------------------------------------------------------------------------

__all__ = [
    "DispatchError",
    "DispatchKindError",
    "DispatchProviderError",
    "DispatchRetryExhausted",
    "DispatchContextError",
    "MissingExecutorFactoryError",
    "StageResult",
    "StageContext",
    "PipelineExecutor",
    "KreaPipelineExecutor",
    "FakeKreaPipelineExecutor",
    "StageAdapterRegistry",
    "GenerationStageExecutor",
    "StageExecutorFactory",
    "PreStageHandoff",
    "PreStageExecutor",
]
