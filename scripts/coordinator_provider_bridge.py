"""Generic, reviewed provider adapters for the durable workflow coordinator.

The coordinator deliberately consumes synchronous callbacks because its durable
state machine runs in a worker thread.  ``PipelineProviderRuntime`` already
contains the bounded HTTP/flow/local-pipeline adapters used by the generic
WorkerPool path, but those adapters were not reusable from the coordinator.

This module is the narrow bridge between the two paths.  It only auto-binds
provider entries whose *server-owned* adapter is in the explicit allow-list:
``http_json``, ``http_openai_compatible``, ``http_flow`` or ``local_pipeline``.
Unsupported adapters remain visible to the caller and must receive a reviewed
custom callback.  No endpoint, command, credential or runtime choice is read
from a job request.

The transport factory is injected by the controller (normally a wrapper around
its existing aiohttp session), so this module is safe to exercise with a
recorder transport and cannot contact a provider by itself.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from collections.abc import Callable, Mapping
from typing import Any

from pipeline_coordinator import (
    PipelineRunState,
    QCVerdict,
    StageStatus,
    _deep_sanitize,
    _now_iso,
)
from pipeline_provider_runtime import PipelineProviderRuntime
from stage_dispatch import StageContext, StageResult
from workflow_coordinator_runtime import ReviewedAdapterRegistry


GENERIC_COORDINATOR_ADAPTERS = frozenset(
    {
        "http_json",
        "http_openai_compatible",
        "http_flow",
        "local_pipeline",
    }
)


class GenericCoordinatorBindingError(RuntimeError):
    """Raised when a generic coordinator binding cannot be constructed."""


def _provider_ids(config: Mapping[str, Any]) -> frozenset[str]:
    providers = config.get("pipeline_providers", {})
    if not isinstance(providers, Mapping):
        raise GenericCoordinatorBindingError("pipeline_providers must be an object")
    return frozenset(str(name) for name in providers if str(name).strip())


def _latest_output(state: PipelineRunState, stage_id: str) -> dict[str, Any]:
    attempts = state.stage_attempts.get(stage_id, [])
    if not attempts:
        return {}
    output = attempts[-1].output
    return copy.deepcopy(output) if isinstance(output, Mapping) else {}


def _stage_context(state: PipelineRunState) -> StageContext:
    """Rebuild the provider-runtime context from durable coordinator evidence."""

    request = state.request_params if isinstance(state.request_params, Mapping) else {}
    original_prompt = str(
        request.get("prompt") or request.get("brief") or request.get("description") or ""
    )
    context = StageContext(original_prompt=original_prompt)

    stage_defs = {
        str(stage.get("id")): stage
        for stage in state.compiled_pipeline.get("stages", [])
        if isinstance(stage, Mapping) and stage.get("id")
    }
    for stage_id, attempts in state.stage_attempts.items():
        if not attempts:
            continue
        latest = attempts[-1]
        status = getattr(latest.status, "value", latest.status)
        if status not in {StageStatus.COMPLETED.value, "completed"}:
            continue
        output = latest.output if isinstance(latest.output, Mapping) else {}
        context = context.with_artifact(stage_id, dict(output))
        stage_kind = str(stage_defs.get(stage_id, {}).get("kind") or "")
        if stage_kind in {"prompt_enhance", "preparation"}:
            enhanced = output.get("enhanced_prompt") or output.get("prompt")
            if enhanced is not None:
                context = context.with_prompt(stage_id, str(enhanced))
        verdict = latest.qc_verdict
        if verdict is not None:
            context = context.with_qc(stage_id, verdict.to_dict())

    # Coordinator owns the loop bound.  The generic one-stage dispatch should
    # not suppress a correction that the coordinator has explicitly scheduled.
    return context.with_delegate_depth(0)


def _seeded_results(
    state: PipelineRunState,
    current_stage_id: str,
) -> list[StageResult]:
    """Translate completed durable attempts into dependency evidence."""

    stage_defs = {
        str(stage.get("id")): stage
        for stage in state.compiled_pipeline.get("stages", [])
        if isinstance(stage, Mapping) and stage.get("id")
    }
    seeded: list[StageResult] = []
    for stage_id, attempts in state.stage_attempts.items():
        if stage_id == current_stage_id or not attempts:
            continue
        latest = attempts[-1]
        status = getattr(latest.status, "value", latest.status)
        if status not in {
            StageStatus.COMPLETED.value,
            StageStatus.FAILED.value,
            StageStatus.SKIPPED.value,
            "completed",
            "failed",
            "skipped",
        }:
            continue
        definition = stage_defs.get(stage_id, {})
        data = dict(latest.output) if isinstance(latest.output, Mapping) else {}
        if latest.qc_verdict is not None:
            data.setdefault("qc_pass", bool(latest.qc_verdict.passed))
            data.setdefault("report", latest.qc_verdict.report)
        result_status = "ok" if status == StageStatus.COMPLETED.value or status == "completed" else status
        seeded.append(
            StageResult(
                stage_id=stage_id,
                kind=str(definition.get("kind") or "delegate"),
                provider=str(definition.get("provider") or ""),
                status=result_status,
                data=data,
                error=str(latest.error or ""),
                retries=max(0, int(getattr(latest, "attempt", 1)) - 1),
            )
        )
    return seeded


def _artifact_from_data(data: Mapping[str, Any]) -> dict[str, str] | None:
    raw: Any = data.get("artifact") if isinstance(data.get("artifact"), Mapping) else data
    if not isinstance(raw, Mapping):
        return None
    path = raw.get("path_or_url") or raw.get("artifact_path") or raw.get("path")
    digest = raw.get("sha256") or raw.get("digest")
    if not isinstance(path, str) or not path.strip() or not isinstance(digest, str) or not digest.strip():
        return None
    return {
        "path_or_url": path[:2_000],
        "sha256": digest[:128],
    }


def _normalise_result(stage: Mapping[str, Any], result: StageResult) -> dict[str, Any]:
    """Map a provider-runtime result to the coordinator result contract."""

    data = _deep_sanitize(copy.deepcopy(result.data or {}))
    if not isinstance(data, Mapping):
        data = {}
    data = dict(data)
    raw_receipt = data.pop("raw", None)
    if isinstance(raw_receipt, Mapping):
        # Preserve provider provenance, not duplicate private reasoning traces
        # and the entire wire response in every downstream stage context.
        data["provider_receipt"] = {key: raw_receipt[key] for key in ("id", "model", "created") if key in raw_receipt}
    kind = str(stage.get("kind") or "")
    qc_verdict = None
    qc_pass = data.get("qc_pass")
    if kind == "qc" and isinstance(qc_pass, bool):
        qc_verdict = QCVerdict(
            stage_id=str(stage.get("id") or ""),
            passed=qc_pass,
            report=str(data.get("report") or data.get("qc_report") or "")[:10_000],
            raw_data=(str(data.get("raw_data"))[:10_000] if data.get("raw_data") is not None else None),
            scored_at=str(data.get("scored_at") or _now_iso()),
        )

    # The generic HTTP wrapper marks qc_pass=false as a failed StageResult so
    # dependency dispatch can stop ordinary successors.  At the coordinator
    # boundary that is a completed QC observation; the coordinator's explicit
    # loop handler must receive the verdict instead of retrying the QC stage.
    if kind == "qc" and qc_verdict is not None:
        status = "completed"
        error = None
    elif result.status == "ok":
        status = "completed"
        error = None
    elif result.status == "failed":
        status = "failed"
        error = str(result.error or "provider stage failed")[:10_000]
    else:
        status = "failed"
        error = f"provider stage returned unsupported nonterminal status {result.status!r}"

    output: dict[str, Any] = data
    artifact = _artifact_from_data(data)
    return {
        "status": status,
        "output": output,
        "error": error,
        "qc_verdict": qc_verdict,
        "artifact": artifact,
    }


class _GenericCoordinatorHandler:
    """Synchronous coordinator callback backed by one bounded async dispatch."""

    def __init__(
        self,
        config: Mapping[str, Any],
        provider_id: str,
        transport: Any,
        *,
        runtime_adapter: str,
        max_retries: int = 0,
    ) -> None:
        self._config = copy.deepcopy(dict(config))
        self._provider_id = provider_id
        self._transport = transport
        self._runtime_adapter = runtime_adapter
        self._max_retries = max(0, min(int(max_retries), 10))

    def __call__(self, stage: dict, state: PipelineRunState, _idempotency_key: str) -> Mapping[str, Any]:
        # Coordinator callbacks are invoked from CoordinatorConsumer's worker
        # thread.  Refuse an accidental event-loop call instead of nesting an
        # event loop and silently blocking the controller.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._execute(stage, state))
        raise GenericCoordinatorBindingError(
            "generic coordinator callbacks must run outside the controller event loop"
        )

    async def _execute(self, stage: dict, state: PipelineRunState) -> Mapping[str, Any]:
        dispatch_stage = copy.deepcopy(stage)
        if dispatch_stage.get("kind") == "preparation":
            dispatch_stage["kind"] = "prompt_enhance"
        params = dispatch_stage.setdefault("params", {})
        if not isinstance(params, dict):
            params = {}
            dispatch_stage["params"] = params
        resolved = dispatch_stage.get("resolved_inputs")
        if isinstance(resolved, Mapping):
            params["resolved_inputs"] = copy.deepcopy(dict(resolved))

        if self._runtime_adapter == "local_pipeline":
            provider_cfg = self._config.get("pipeline_providers", {}).get(self._provider_id, {})
            pipeline_id = provider_cfg.get("pipeline_id") or provider_cfg.get("local_pipeline_id")
            if pipeline_id and "pipeline_id" not in params:
                params["pipeline_id"] = str(pipeline_id)

        runtime = PipelineProviderRuntime(
            dict(self._config),
            _provider_ids(self._config),
        )
        transport_map = {}
        if self._runtime_adapter != "local_pipeline":
            if self._transport is None:
                raise GenericCoordinatorBindingError(
                    f"provider {self._provider_id!r} has no injected transport"
                )
            transport_map[self._provider_id] = self._transport

        contract = {
            "schema": state.compiled_pipeline.get("schema", ""),
            "stages": [dispatch_stage],
            # The durable coordinator owns correction scheduling. Forwarding
            # its loop to this single-stage executor would try to dispatch
            # Qwen here as well, outside the trusted queued leaf boundary.
            "loops": [],
            "max_loop_count": 5,
            "execution_mode": "generic",
        }
        executor = runtime.build_multi_provider_executor(
            contract,
            transport_map,
            max_retries=self._max_retries,
            max_corrections=5,
            max_delegate_depth=3,
        )
        stage_context = _stage_context(state)
        if stage.get("kind") == "qc" and state.artifact is not None:
            # Post-edit QC must judge the edited image, not rejudge the
            # original along with every historical correction candidate.
            stage_context = replace(stage_context, artifacts={"current": state.artifact.to_dict()})
        context, results = await executor.dispatch(
            [dispatch_stage],
            stage_context,
            seeded_results=_seeded_results(state, str(stage.get("id") or "")),
            stage_defs={str(dispatch_stage.get("id")): dispatch_stage},
        )
        del context  # context is represented by the bounded StageResult output
        if not results:
            raise GenericCoordinatorBindingError(
                f"provider {self._provider_id!r} returned no stage result"
            )
        return _normalise_result(stage, results[-1])


def build_generic_external_callbacks(
    config: Mapping[str, Any],
    transport_factory: Callable[[str, Mapping[str, Any]], Any],
    *,
    operation_catalogs: Mapping[Any, Any] | None = None,
    leaf_adapters: set[str] | frozenset[str] = frozenset(),
    external_adapters: set[str] | frozenset[str] = frozenset(),
    allowed_provider_adapters: set[str] | frozenset[str] = GENERIC_COORDINATOR_ADAPTERS,
    max_retries: int = 0,
) -> tuple[dict[tuple[str, str], Callable[..., Mapping[str, Any]]], tuple[tuple[str, str, str], ...]]:
    """Build generic callbacks for supported reviewed external providers.

    Returns ``(handlers, unsupported)``.  ``unsupported`` contains
    ``(provider, operation_adapter, provider_adapter)`` records that still
    require a custom callback.  The function never turns those records into
    HTTP calls, which keeps custom QC/artifact engines explicitly reviewed.
    """

    if not callable(transport_factory):
        raise TypeError("transport_factory must be callable")
    registry = ReviewedAdapterRegistry.from_config(
        config,
        operation_catalogs=operation_catalogs,
        leaf_adapters=leaf_adapters,
        external_adapters=external_adapters,
    )
    providers = config.get("pipeline_providers", {})
    if not isinstance(providers, Mapping):
        raise GenericCoordinatorBindingError("pipeline_providers must be an object")

    handlers: dict[tuple[str, str], Callable[..., Mapping[str, Any]]] = {}
    unsupported: list[tuple[str, str, str]] = []
    for binding in registry.summary():
        if binding["authority"] != "external":
            continue
        provider = binding["provider"]
        operation_adapter = binding["adapter"]
        provider_cfg = providers.get(provider)
        if not isinstance(provider_cfg, Mapping):
            raise GenericCoordinatorBindingError(
                f"provider {provider!r} has no server-owned configuration"
            )
        provider_adapter = str(provider_cfg.get("adapter") or "").strip()
        if provider_adapter not in allowed_provider_adapters:
            unsupported.append((provider, operation_adapter, provider_adapter))
            continue
        transport = (
            None
            if provider_adapter == "local_pipeline"
            else transport_factory(provider, provider_cfg)
        )
        handlers[(provider, operation_adapter)] = _GenericCoordinatorHandler(
            config,
            provider,
            transport,
            runtime_adapter=provider_adapter,
            max_retries=max_retries,
        )
    return handlers, tuple(sorted(unsupported))


__all__ = [
    "GENERIC_COORDINATOR_ADAPTERS",
    "GenericCoordinatorBindingError",
    "build_generic_external_callbacks",
]
