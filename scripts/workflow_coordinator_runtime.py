"""Controller-owned coordinator runtime seams.

This module is intentionally small.  ``pipeline_coordinator`` owns durable
parent state and this module supplies the two things it must not discover from
user input: a reviewed provider/adapter map and a bounded consumer loop.

The map is exact on ``(provider, adapter)``.  A provider being present in a
registry is not enough to make it executable; a missing or conflicting binding
fails closed.  The consumer uses the coordinator's existing parent stream and
does not create a second queue or state store.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pipeline_coordinator import (
    CoordinatorStore,
    ExternalExecutor,
    LeafGenerationClient,
    PipelineCoordinator,
    PipelineRunState,
    QCVerdict,
    StageAuthorityResolver,
    _deep_sanitize,
    _now_iso,
)
from gpu_manager_pipeline_runtime import ControllerPipelineRuntime


logger = logging.getLogger(__name__)


class AdapterBindingError(RuntimeError):
    """Raised when a provider/adapter has no reviewed execution binding."""


@dataclass(frozen=True, slots=True)
class AdapterBinding:
    provider_id: str
    adapter_id: str
    authority: str


class ReviewedAdapterRegistry(StageAuthorityResolver):
    """Exact provider/adapter registry used by the durable coordinator.

    ``bindings`` accepts either ``{(provider, adapter): authority}`` or
    ``{"provider": {"adapter": authority}}``.  The latter is convenient for
    a checked host overlay; neither form is accepted from a request.  The
    ``from_config`` helper only consumes server-owned ``pipeline_providers``
    entries and requires an explicit ``managed`` value unless the caller
    supplies reviewed adapter sets.
    """

    def __init__(
        self,
        bindings: Mapping[Any, Any] | None = None,
    ) -> None:
        self._bindings: dict[tuple[str, str], AdapterBinding] = {}
        for raw_key, raw_value in (bindings or {}).items():
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                provider, adapter = raw_key
                authority = raw_value
            elif isinstance(raw_key, str) and isinstance(raw_value, Mapping):
                provider = raw_key
                for adapter, authority in raw_value.items():
                    self.register(provider, adapter, authority)
                continue
            else:
                raise AdapterBindingError(
                    "adapter bindings must use (provider, adapter) keys"
                )
            self.register(provider, adapter, authority)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        operation_catalogs: Mapping[Any, Any] | None = None,
        leaf_adapters: set[str] | frozenset[str] = frozenset(),
        external_adapters: set[str] | frozenset[str] = frozenset(),
    ) -> "ReviewedAdapterRegistry":
        """Build bindings from server-owned ``pipeline_providers`` metadata.

        A provider must either declare ``managed: worker|external`` or use an
        adapter in one of the explicitly reviewed sets.  Unknown values are
        rejected instead of silently becoming HTTP/external work.  When
        operation catalogs are supplied, each catalog operation's adapter is
        also registered for its declared provider references.  This matters
        when a provider's transport adapter (for example
        ``http_openai_compatible``) differs from the stage operation adapter
        (for example ``llm`` or ``minimax.preparation``).
        """

        providers = config.get("pipeline_providers", {})
        if not isinstance(providers, Mapping):
            raise AdapterBindingError("pipeline_providers must be an object")
        registry = cls()
        provider_authority: dict[str, str] = {}
        for provider_id, provider_config in providers.items():
            if not isinstance(provider_config, Mapping):
                raise AdapterBindingError(
                    f"provider {provider_id!r} configuration must be an object"
                )
            adapter = provider_config.get("adapter")
            if not isinstance(adapter, str) or not adapter.strip():
                raise AdapterBindingError(
                    f"provider {provider_id!r} has no reviewed adapter"
                )
            adapter = adapter.strip()
            managed = provider_config.get("managed")
            if managed == "worker":
                authority = "leaf"
            elif managed == "external":
                authority = "external"
            elif adapter in leaf_adapters:
                authority = "leaf"
            elif adapter in external_adapters:
                authority = "external"
            else:
                raise AdapterBindingError(
                    f"provider {provider_id!r} adapter {adapter!r} has no "
                    "reviewed managed authority"
                )
            registry.register(provider_id, adapter, authority)
            provider_authority[str(provider_id).strip()] = authority

        catalogs = operation_catalogs
        if catalogs is None:
            catalogs = config.get("operation_catalogs", {})
        if not isinstance(catalogs, Mapping):
            raise AdapterBindingError("operation_catalogs must be an object")
        for catalog_id, catalog in catalogs.items():
            if not isinstance(catalog, Mapping):
                raise AdapterBindingError(
                    f"operation catalog {catalog_id!r} must be an object"
                )
            operations = catalog.get("operations", {})
            if not isinstance(operations, Mapping):
                raise AdapterBindingError(
                    f"operation catalog {catalog_id!r} operations must be an object"
                )
            for operation_id, operation in operations.items():
                if not isinstance(operation, Mapping):
                    raise AdapterBindingError(
                        f"operation {operation_id!r} in catalog {catalog_id!r} "
                        "must be an object"
                    )
                stage_adapter = operation.get("adapter")
                if not isinstance(stage_adapter, str) or not stage_adapter.strip():
                    raise AdapterBindingError(
                        f"operation {operation_id!r} in catalog {catalog_id!r} "
                        "has no adapter"
                    )
                provider_refs = operation.get("provider_refs", [])
                if not isinstance(provider_refs, list):
                    raise AdapterBindingError(
                        f"operation {operation_id!r} in catalog {catalog_id!r} "
                        "provider_refs must be a list"
                    )
                for provider_id in provider_refs:
                    provider_key = str(provider_id).strip()
                    if provider_key not in provider_authority:
                        raise AdapterBindingError(
                            f"operation {operation_id!r} references provider "
                            f"{provider_key!r} without a reviewed provider binding"
                        )
                    registry.register(
                        provider_key,
                        stage_adapter,
                        provider_authority[provider_key],
                    )
        return registry

    def register(self, provider_id: Any, adapter_id: Any, authority: Any) -> None:
        provider = str(provider_id).strip()
        adapter = str(adapter_id).strip()
        owner = str(authority).strip().lower()
        if not provider or not adapter:
            raise AdapterBindingError("provider and adapter IDs must be non-empty")
        if owner not in {"leaf", "external"}:
            raise AdapterBindingError(
                f"authority for {provider!r}/{adapter!r} must be leaf or external"
            )
        key = (provider, adapter)
        existing = self._bindings.get(key)
        if existing is not None and existing.authority != owner:
            raise AdapterBindingError(
                f"conflicting authority for provider={provider!r}, adapter={adapter!r}"
            )
        self._bindings[key] = AdapterBinding(provider, adapter, owner)

    def binding_for(self, stage: Mapping[str, Any]) -> AdapterBinding:
        provider = str(stage.get("provider") or "").strip()
        adapter = str(stage.get("adapter") or "").strip()
        binding = self._bindings.get((provider, adapter))
        if binding is None:
            raise AdapterBindingError(
                f"no reviewed adapter binding for provider={provider!r}, "
                f"adapter={adapter!r}"
            )
        return binding

    def resolve(self, stage: dict) -> str:
        """Implement ``StageAuthorityResolver`` with exact matching."""

        return self.binding_for(stage).authority

    def summary(self) -> list[dict[str, str]]:
        """Return safe bindings for diagnostics (no endpoints or credentials)."""

        return [
            {
                "provider": binding.provider_id,
                "adapter": binding.adapter_id,
                "authority": binding.authority,
            }
            for binding in sorted(
                self._bindings.values(),
                key=lambda item: (item.provider_id, item.adapter_id),
            )
        ]


def _normalise_qc(value: Any, stage_id: str) -> QCVerdict | None:
    if isinstance(value, QCVerdict):
        return value
    if not isinstance(value, Mapping) or "passed" not in value:
        return None
    return QCVerdict(
        stage_id=str(value.get("stage_id") or stage_id),
        passed=bool(value.get("passed")),
        report=str(value.get("report") or "")[:10_000],
        raw_data=(
            str(value.get("raw_data"))[:10_000]
            if value.get("raw_data") is not None
            else None
        ),
        scored_at=str(value.get("scored_at") or _now_iso()),
    )


class CoordinatorExternalExecutor(ExternalExecutor):
    """Dispatch external stages through server-owned reviewed callbacks.

    ``handlers`` is a code-owned map keyed by the same exact
    ``(provider, adapter)`` identity used by ``ReviewedAdapterRegistry``. A
    handler receives ``(stage, state, idempotency_key)`` and returns a small
    result mapping. It may not return arbitrary binary data because the
    coordinator persists its output. Cancellation uses one controller-owned
    callback because the coordinator cancellation protocol carries only a
    parent/stage identity, not a provider-specific handle.
    """

    # Providers may either perform a bounded submit/poll inside ``execute`` or
    # return an opaque handle and use the optional ``poll_handlers`` map.  The
    # latter is what makes an accepted external operation restart-safe.
    _STATUSES = frozenset({"completed", "failed", "queued", "in_flight", "pending"})

    def __init__(
        self,
        registry: ReviewedAdapterRegistry,
        handlers: Mapping[tuple[str, str], Callable[[dict, PipelineRunState, str], Mapping[str, Any]]],
        poll_handlers: Mapping[
            tuple[str, str],
            Callable[[dict, PipelineRunState, str, str], Mapping[str, Any]],
        ] | None = None,
        cancel_handler: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._registry = registry
        self._handlers = dict(handlers)
        self._poll_handlers = dict(poll_handlers or {})
        self._cancel_handler = cancel_handler

    def _binding(self, stage: dict) -> AdapterBinding:
        binding = self._registry.binding_for(stage)
        if binding.authority != "external":
            raise AdapterBindingError(
                f"stage {stage.get('id')!r} is not externally owned"
            )
        return binding

    def _normalise(self, stage: dict, raw: Mapping[str, Any]) -> dict:
        if not isinstance(raw, Mapping):
            raise AdapterBindingError("external adapter result must be an object")
        status = str(raw.get("status") or "failed").lower()
        if status not in self._STATUSES:
            raise AdapterBindingError(
                f"external adapter returned unsupported status {status!r}"
            )
        output = raw.get("output") or {}
        if not isinstance(output, Mapping):
            raise AdapterBindingError("external adapter output must be an object")
        result: dict[str, Any] = {
            "status": status,
            "output": _deep_sanitize(copy.deepcopy(dict(output))),
            "error": str(raw.get("error"))[:10_000]
            if raw.get("error") is not None
            else None,
        }
        if raw.get("provider_handle") is not None:
            result["provider_handle"] = str(raw.get("provider_handle"))
        verdict = _normalise_qc(raw.get("qc_verdict"), str(stage.get("id") or ""))
        if verdict is not None:
            result["qc_verdict"] = verdict
        if raw.get("artifact") is not None:
            artifact = raw.get("artifact")
            if not isinstance(artifact, Mapping):
                raise AdapterBindingError("external artifact must be an object")
            result["artifact"] = _deep_sanitize(copy.deepcopy(dict(artifact)))
        return result

    def execute(
        self,
        stage: dict,
        state: PipelineRunState,
        idempotency_key: str,
    ) -> dict:
        binding = self._binding(stage)
        handler = self._handlers.get((binding.provider_id, binding.adapter_id))
        if handler is None:
            raise AdapterBindingError(
                f"no external executor for provider={binding.provider_id!r}, "
                f"adapter={binding.adapter_id!r}"
            )
        raw = handler(copy.deepcopy(stage), state, idempotency_key)
        return self._normalise(stage, raw)

    def poll(
        self,
        stage: dict,
        state: PipelineRunState,
        idempotency_key: str,
        provider_handle: str,
    ) -> dict:
        """Poll one previously accepted operation through its reviewed map."""
        binding = self._binding(stage)
        handler = self._poll_handlers.get((binding.provider_id, binding.adapter_id))
        if handler is None:
            return {
                "status": "failed",
                "error": "unsafe_external_resume: no reviewed provider poll handler",
            }
        raw = handler(
            copy.deepcopy(stage), state, idempotency_key, provider_handle,
        )
        return self._normalise(stage, raw)

    def cancel(self, run_id: str, stage_id: str) -> bool:
        # The coordinator protocol carries only run_id/stage_id at cancellation
        # time, so one controller-owned callback is safer than guessing among
        # several provider handlers.  A missing callback is a truthful no-op;
        # cancellation intent remains durable and the parent is not told that
        # an external provider definitely stopped.
        if self._cancel_handler is None:
            return False
        try:
            return bool(self._cancel_handler(run_id, stage_id))
        except Exception:
            logger.exception("external adapter cancellation failed")
            return False


class ReviewedLeafGenerationClient(LeafGenerationClient):
    """Guard a trusted leaf client with the exact adapter registry."""

    def __init__(self, registry: ReviewedAdapterRegistry, delegate: LeafGenerationClient):
        self._registry = registry
        self._delegate = delegate

    def submit(
        self,
        stage: dict,
        parent_run_id: str,
        parent_state: PipelineRunState,
        idempotency_key: str,
        trusted_config: dict,
    ) -> str:
        binding = self._registry.binding_for(stage)
        if binding.authority != "leaf":
            raise AdapterBindingError(
                f"stage {stage.get('id')!r} is not leaf-owned"
            )
        return self._delegate.submit(
            stage,
            parent_run_id,
            parent_state,
            idempotency_key,
            trusted_config,
        )

    def poll(self, child_job_id: str) -> dict:
        return self._delegate.poll(child_job_id)

    def cancel(self, child_job_id: str) -> bool:
        return self._delegate.cancel(child_job_id)


class CoordinatorAdmissionError(RuntimeError):
    """A queued workflow entry cannot be handed to the coordinator safely."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class CoordinatorParentAdmission:
    """Translate one controller queue entry into one durable parent run.

    This is the explicit handoff from the existing GPU Manager queue worker to
    the coordinator's parent stream.  It never executes a provider itself and
    never creates another queue.  The source entry remains the priority owner;
    once the worker has admitted it, the coordinator owns all subsequent stage
    receipts and child recovery.
    """

    def __init__(self, runtime: ControllerPipelineRuntime):
        if not callable(getattr(runtime, "submit_parent", None)):
            raise TypeError("coordinator runtime must expose submit_parent()")
        if not callable(getattr(runtime, "get", None)):
            raise TypeError("coordinator runtime must expose get()")
        self.runtime = runtime

    @staticmethod
    def _pipeline_from_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
        pipeline = entry.get("workflow_pipeline")
        if isinstance(pipeline, Mapping):
            return pipeline
        try:
            metadata = json.loads(str(entry.get("metadata") or "{}"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise CoordinatorAdmissionError(
                f"workflow metadata is invalid: {exc}"
            ) from exc
        # Internal workflow entries use ``workflow_pipeline``.  A mixed
        # ComfyUI ingress entry carries the same server-compiled graph under
        # ``worker_pipeline`` because that metadata is also consumed by the
        # legacy worker boundary.  Accept that one server-owned alias here so
        # the coordinator handoff does not reject a valid mixed graph before
        # admission.  The caller has already classified the graph and this
        # helper still requires a mapping; it never discovers executable
        # provider policy from arbitrary request fields.
        pipeline = None
        if isinstance(metadata, Mapping):
            pipeline = metadata.get("workflow_pipeline")
            if not isinstance(pipeline, Mapping):
                pipeline = metadata.get("worker_pipeline")
        if not isinstance(pipeline, Mapping):
            raise CoordinatorAdmissionError(
                "workflow entry has no compiled workflow_pipeline"
            )
        return pipeline

    @staticmethod
    def _request_from_entry(
        entry: Mapping[str, Any],
        pipeline: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded = entry.get("body_b64")
        if not isinstance(encoded, str) or not encoded:
            raise CoordinatorAdmissionError("workflow entry has no body_b64 request")
        try:
            raw = base64.b64decode(encoded, validate=True)
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CoordinatorAdmissionError(
                f"workflow request body is invalid: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise CoordinatorAdmissionError("workflow request body must be an object")

        # ComfyUI-backed declarative image entries carry a rendered graph in
        # ``body_b64``.  The coordinator needs the original user prompt for
        # prompt-enhancement/QC bindings, not the graph object itself.  The
        # prompt is already frozen into the server-compiled pipeline, so use it
        # as a bounded request snapshot rather than duplicating binary image
        # inputs into the coordinator stream.
        original_prompt = (
            pipeline.get("original_prompt") if isinstance(pipeline, Mapping) else None
        )
        if (
            isinstance(original_prompt, str)
            and original_prompt
            and isinstance(value.get("prompt"), Mapping)
        ):
            return {"prompt": original_prompt}
        return copy.deepcopy(dict(value))

    @staticmethod
    def _run_id(entry: Mapping[str, Any], pipeline: Mapping[str, Any]) -> str:
        parent_job_id = str(entry.get("job_id") or "").strip()
        fingerprint = str(
            pipeline.get("workflow_fingerprint")
            or pipeline.get("revision_hash")
            or ""
        ).strip().lower()
        if not parent_job_id:
            raise CoordinatorAdmissionError("workflow entry has no parent job_id")
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise CoordinatorAdmissionError("compiled workflow identity is invalid")
        # Keep the run identity deterministic for an ACK-loss redelivery while
        # avoiding caller-controlled request content in the durable key.
        suffix = hashlib.sha256(fingerprint.encode("ascii")).hexdigest()[:16]
        return f"gm-coordinator:{parent_job_id}:{suffix}"

    def admit_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Admit an internal-workflow queue entry exactly once.

        Returns a bounded receipt with ``accepted``, ``run_id`` and
        ``already_admitted``.  A store outage is marked retryable so the source
        queue worker can leave/requeue its claim; malformed or conflicting
        identity is permanent and must be terminalized without running a leaf.
        """

        if not isinstance(entry, Mapping):
            raise CoordinatorAdmissionError("workflow entry must be an object")
        pipeline = self._pipeline_from_entry(entry)
        request_params = self._request_from_entry(entry, pipeline)
        run_id = self._run_id(entry, pipeline)
        parent_job_id = str(entry.get("job_id") or "").strip()
        revision = str(
            entry.get("workflow_fingerprint")
            or pipeline.get("workflow_fingerprint")
            or pipeline.get("revision_hash")
            or ""
        ).strip().lower()
        existing = self.runtime.get(run_id)
        if existing is not None:
            if (
                str(getattr(existing, "parent_job_id", "")) != parent_job_id
                or str(getattr(existing, "revision_hash", "")) != revision
            ):
                raise CoordinatorAdmissionError(
                    f"coordinator run identity conflict for parent {parent_job_id!r}"
                )
            return {
                "accepted": True,
                "already_admitted": True,
                "run_id": run_id,
                "pipeline_id": str(pipeline.get("pipeline_id") or ""),
            }

        try:
            accepted = bool(
                self.runtime.submit_parent(
                    compiled_pipeline=dict(pipeline),
                    original_request=request_params,
                    revision=revision,
                    parent_job_id=parent_job_id,
                    run_id=run_id,
                )
            )
        except Exception as exc:
            raise CoordinatorAdmissionError(
                f"coordinator parent admission failed: {str(exc)[:500]}",
                retryable=True,
            ) from exc
        if not accepted:
            # A transactional store returns False without leaving a runnable
            # parent.  The worker must retain/requeue the source claim rather
            # than acknowledging it as a terminal provider failure.
            return {
                "accepted": False,
                "already_admitted": False,
                "retryable": True,
                "run_id": run_id,
                "pipeline_id": str(pipeline.get("pipeline_id") or ""),
            }
        return {
            "accepted": True,
            "already_admitted": False,
            "run_id": run_id,
            "pipeline_id": str(pipeline.get("pipeline_id") or ""),
        }


class CoordinatorJobMirror:
    """Mirror terminal parent evidence into the existing GPU job hash."""

    def __init__(self, update_status: Callable[..., Any]):
        if not callable(update_status):
            raise TypeError("job status mirror requires update_status()")
        self._update_status = update_status

    @staticmethod
    def _result_payload(state: PipelineRunState) -> str:
        stages: list[dict[str, Any]] = []
        for stage_id, attempts in sorted(state.stage_attempts.items()):
            if not attempts:
                continue
            latest = attempts[-1]
            stages.append(
                {
                    "stage_id": stage_id,
                    "attempt": latest.attempt,
                    "status": latest.status.value,
                    "error": latest.error,
                    "output": _deep_sanitize(copy.deepcopy(latest.output or {})),
                    "qc_verdict": (
                        latest.qc_verdict.to_dict()
                        if latest.qc_verdict is not None
                        else None
                    ),
                }
            )
        payload = {
            "coordinator_run_id": state.run_id,
            "pipeline_id": state.pipeline_id,
            "revision_hash": state.revision_hash,
            "status": state.status,
            "terminal_code": state.terminal_code.value if state.terminal_code else None,
            "terminal_reason": state.terminal_reason,
            "artifact": state.artifact.to_dict() if state.artifact else None,
            "initial_artifact": (
                state.initial_artifact.to_dict()
                if state.initial_artifact else None
            ),
            "last_good_artifact": (
                state.last_good_artifact.to_dict()
                if state.last_good_artifact else None
            ),
            "stages": stages,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def __call__(self, state: PipelineRunState) -> bool:
        if not isinstance(state, PipelineRunState):
            return False
        kwargs: dict[str, Any] = {
            "coordinator_run_id": state.run_id,
            "terminal_code": (
                state.terminal_code.value if state.terminal_code else ""
            ),
            "terminal_reason": state.terminal_reason or "",
            "result": self._result_payload(state),
        }
        if state.status == "failed":
            kwargs["error"] = state.terminal_reason or "coordinator pipeline failed"
        elif state.status == "cancelled":
            kwargs["error"] = state.terminal_reason or "coordinator pipeline cancelled"
        return bool(self._update_status(state.parent_job_id, state.status, **kwargs))


@dataclass(frozen=True, slots=True)
class CoordinatorCallbackMap:
    """One reviewed callback set for a controller-owned coordinator.

    The registry remains the authority for stage ownership.  This bundle only
    wires already-reviewed callbacks to that registry; it does not discover
    endpoints, construct a queue, or accept executable policy from a service
    definition.  Keeping the map as an explicit object makes a staging
    bootstrap auditable and prevents a second ad-hoc provider dispatch path.
    """

    registry: ReviewedAdapterRegistry
    external: CoordinatorExternalExecutor
    leaf: ReviewedLeafGenerationClient

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        external_handlers: Mapping[
            tuple[str, str],
            Callable[[dict, PipelineRunState, str], Mapping[str, Any]],
        ],
        external_poll_handlers: Mapping[
            tuple[str, str],
            Callable[[dict, PipelineRunState, str, str], Mapping[str, Any]],
        ] | None = None,
        leaf_client: LeafGenerationClient,
        operation_catalogs: Mapping[Any, Any] | None = None,
        leaf_adapters: set[str] | frozenset[str] = frozenset(),
        external_adapters: set[str] | frozenset[str] = frozenset(),
        cancel_handler: Callable[[str, str], bool] | None = None,
    ) -> "CoordinatorCallbackMap":
        """Build an exact callback map from server-owned reviewed metadata.

        Every externally-owned binding must have exactly one callback.  Extra
        callbacks are rejected as well: a typo or stale provider identity must
        not quietly become an unreachable second execution surface.
        """

        registry = ReviewedAdapterRegistry.from_config(
            config,
            operation_catalogs=operation_catalogs,
            leaf_adapters=leaf_adapters,
            external_adapters=external_adapters,
        )
        bindings = {
            (item["provider"], item["adapter"]): item["authority"]
            for item in registry.summary()
        }
        expected_external = {
            key for key, authority in bindings.items() if authority == "external"
        }
        supplied = set(external_handlers)
        missing = sorted(expected_external - supplied)
        extra = sorted(supplied - set(bindings))
        poll_supplied = set(external_poll_handlers or {})
        poll_extra = sorted(poll_supplied - expected_external)
        if missing or extra or poll_extra:
            details: list[str] = []
            if missing:
                details.append(f"missing external callbacks: {missing}")
            if extra:
                details.append(f"callbacks have no reviewed binding: {extra}")
            if poll_extra:
                details.append(
                    f"poll callbacks have no reviewed external binding: {poll_extra}"
                )
            raise AdapterBindingError("; ".join(details))

        external = CoordinatorExternalExecutor(
            registry,
            external_handlers,
            poll_handlers=external_poll_handlers,
            cancel_handler=cancel_handler,
        )
        return cls(
            registry=registry,
            external=external,
            leaf=ReviewedLeafGenerationClient(registry, leaf_client),
        )


class CoordinatorConsumer:
    """Bounded async loop over one ``ControllerPipelineRuntime`` consumer."""

    def __init__(
        self,
        runtime: ControllerPipelineRuntime,
        *,
        idle_delay_seconds: float = 0.5,
        error_delay_seconds: float = 2.0,
        on_terminal: Callable[[PipelineRunState], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.idle_delay_seconds = min(max(float(idle_delay_seconds), 0.05), 30.0)
        self.error_delay_seconds = min(max(float(error_delay_seconds), 0.1), 60.0)
        self.on_terminal = on_terminal

    async def process_one(self) -> PipelineRunState | None:
        """Run one synchronous coordinator tick without blocking aiohttp."""

        state = await asyncio.to_thread(self.runtime.process_one)
        if (
            state is not None
            and state.status in {"completed", "failed", "cancelled"}
            and self.on_terminal is not None
        ):
            try:
                mirrored = self.on_terminal(state)
                if inspect.isawaitable(mirrored):
                    mirrored = await mirrored
                if mirrored is False:
                    logger.warning(
                        "workflow coordinator terminal mirror returned false for %s",
                        state.run_id,
                    )
            except Exception:
                # The coordinator state has already been durably terminalized
                # and ACKed.  Keep the consumer alive and leave the parent
                # evidence authoritative rather than retrying provider work.
                logger.exception(
                    "workflow coordinator terminal mirror failed for %s",
                    state.run_id,
                )
        return state

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Consume until stopped, with bounded idle/error backoff."""

        stop = stop_event or asyncio.Event()
        while not stop.is_set():
            try:
                state = await self.process_one()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "workflow coordinator tick failed: %s",
                    str(exc)[:500],
                )
                await self._wait_or_stop(stop, self.error_delay_seconds)
                continue
            if state is None:
                await self._wait_or_stop(stop, self.idle_delay_seconds)

    @staticmethod
    async def _wait_or_stop(stop: asyncio.Event, delay: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return


def build_controller_runtime(
    *,
    coordinator: Any,
    store: CoordinatorStore,
    consumer_name: str,
) -> ControllerPipelineRuntime:
    """Small named factory used by controller wiring and isolated staging."""

    return ControllerPipelineRuntime(
        coordinator=coordinator,
        store=store,
        consumer_name=consumer_name,
    )


def build_controller_runtime_from_config(
    config: Mapping[str, Any],
    *,
    store: CoordinatorStore,
    external_handlers: Mapping[
        tuple[str, str],
        Callable[[dict, PipelineRunState, str], Mapping[str, Any]],
    ] | None = None,
    external_poll_handlers: Mapping[
        tuple[str, str],
        Callable[[dict, PipelineRunState, str, str], Mapping[str, Any]],
    ] | None = None,
    generic_transport_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
    custom_callback_factory: Callable[..., Any] | None = None,
    leaf_client: LeafGenerationClient,
    consumer_name: str = "controller",
    operation_catalogs: Mapping[Any, Any] | None = None,
    leaf_adapters: set[str] | frozenset[str] = frozenset(),
    external_adapters: set[str] | frozenset[str] = frozenset(),
    cancel_handler: Callable[[str, str], bool] | None = None,
) -> tuple[ControllerPipelineRuntime, CoordinatorCallbackMap]:
    """Build one explicit controller runtime from a reviewed callback map.

    This is the staging/bootstrap seam.  It reuses the supplied durable store
    and creates no background task; callers must explicitly attach the returned
    runtime to a controller application before a ``CoordinatorConsumer`` can
    run.  When ``generic_transport_factory`` is supplied, supported
    server-owned HTTP/flow/local-pipeline provider adapters are generated
    without a controller branch per provider; unsupported adapters still need
    explicit callbacks and therefore fail closed. ``custom_callback_factory``
    is the explicit deployment seam for reviewed non-generic adapters. It
    returns ``(execute_handlers, poll_handlers, unsupported)``; missing
    injections still fail the exact callback-map check below. The returned
    callback map is included for diagnostics and lifecycle ownership checks.
    """

    resolved_external_handlers: dict[
        tuple[str, str],
        Callable[[dict, PipelineRunState, str], Mapping[str, Any]],
    ] = dict(external_handlers or {})
    resolved_external_poll_handlers: dict[
        tuple[str, str],
        Callable[[dict, PipelineRunState, str, str], Mapping[str, Any]],
    ] = dict(external_poll_handlers or {})
    if generic_transport_factory is not None:
        # The generic bridge only fills bindings whose server-owned provider
        # adapter is explicitly allow-listed.  Custom QC/artifact/runtime
        # adapters still have to be supplied through the reviewed callback map;
        # CoordinatorCallbackMap.from_config below keeps that exactness check.
        from coordinator_provider_bridge import build_generic_external_callbacks

        generated, _unsupported = build_generic_external_callbacks(
            config,
            generic_transport_factory,
            operation_catalogs=operation_catalogs,
            leaf_adapters=leaf_adapters,
            external_adapters=external_adapters,
        )
        generated.update(resolved_external_handlers)
        resolved_external_handlers = generated
    if custom_callback_factory is not None:
        if not callable(custom_callback_factory):
            raise TypeError("custom_callback_factory must be callable")
        custom_result = custom_callback_factory(
            config,
            operation_catalogs=operation_catalogs,
        )
        if (
            not isinstance(custom_result, tuple)
            or len(custom_result) != 3
            or not isinstance(custom_result[0], Mapping)
            or not isinstance(custom_result[1], Mapping)
        ):
            raise AdapterBindingError(
                "custom_callback_factory must return "
                "(execute_handlers, poll_handlers, unsupported)"
            )
        custom_handlers, custom_poll_handlers, _unsupported = custom_result
        resolved_external_handlers.update(custom_handlers)
        resolved_external_poll_handlers.update(custom_poll_handlers)
    # Explicit maps are the final authority. A bootstrap can replace a
    # generated callback after reviewing the exact provider identity, but it
    # cannot add an unregistered one because CoordinatorCallbackMap checks it.
    resolved_external_handlers.update(dict(external_handlers or {}))
    resolved_external_poll_handlers.update(dict(external_poll_handlers or {}))

    callback_map = CoordinatorCallbackMap.from_config(
        config,
        external_handlers=resolved_external_handlers,
        external_poll_handlers=resolved_external_poll_handlers,
        leaf_client=leaf_client,
        operation_catalogs=operation_catalogs,
        leaf_adapters=leaf_adapters,
        external_adapters=external_adapters,
        cancel_handler=cancel_handler,
    )
    coordinator = PipelineCoordinator(
        store,
        callback_map.registry,
        callback_map.external,
        callback_map.leaf,
    )
    runtime = build_controller_runtime(
        coordinator=coordinator,
        store=store,
        consumer_name=consumer_name,
    )
    return runtime, callback_map


__all__ = [
    "AdapterBinding",
    "AdapterBindingError",
    "CoordinatorAdmissionError",
    "CoordinatorCallbackMap",
    "CoordinatorConsumer",
    "CoordinatorExternalExecutor",
    "CoordinatorJobMirror",
    "CoordinatorParentAdmission",
    "ReviewedAdapterRegistry",
    "ReviewedLeafGenerationClient",
    "build_controller_runtime",
    "build_controller_runtime_from_config",
]
