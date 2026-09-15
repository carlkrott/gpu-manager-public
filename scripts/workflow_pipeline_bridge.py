"""Compile ``workflow.v1`` graphs for the durable pipeline coordinator.

The repository has two deliberately different contracts:

* ``workflow.v1`` is the user/service-facing graph.  It owns stage IDs,
  operations, typed bindings and revision identity.
* ``pipeline_coordinator`` consumes a frozen list of stage records while it
  owns parent persistence, retries, child handles and recovery.

This module is the narrow, checked bridge between them.  It does not execute
providers and it does not create a second workflow schema.  The returned
mapping keeps ``schema``/``schema_version`` as ``workflow.v1`` and carries
the source graph's fingerprint and catalog identity alongside the coordinator
stage records.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pipeline_coordinator import StageAuthorityResolver
from workflow_capabilities import validate_workflow_capabilities
from workflow_contracts import WORKFLOW_SCHEMA_VERSION, validate_workflow_definition, workflow_fingerprint


BRIDGE_VERSION = "workflow-coordinator-bridge.v1"


class WorkflowCompilationError(ValueError):
    """Raised when a workflow cannot be safely frozen for coordination."""

    def __init__(self, errors: list[str] | tuple[str, ...] | str):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = tuple(str(error) for error in errors if str(error))
        super().__init__("workflow compilation failed: " + "; ".join(self.errors))


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_topological_order(stages: list[Mapping[str, Any]]) -> list[str]:
    """Return a deterministic topological order, retaining declaration ties."""

    positions = {str(stage["id"]): index for index, stage in enumerate(stages)}
    by_id = {str(stage["id"]): stage for stage in stages}
    dependencies = {
        stage_id: {
            str(dep)
            for dep in (stage.get("depends_on", []) or [])
        }
        for stage_id, stage in by_id.items()
    }
    result: list[str] = []
    remaining = set(by_id)
    while remaining:
        ready = sorted(
            (
                stage_id
                for stage_id in remaining
                if not (dependencies[stage_id] & remaining)
            ),
            key=positions.__getitem__,
        )
        if not ready:
            # The graph validator normally catches this.  Keep the bridge
            # defensive so it cannot emit a partial plan if called directly.
            raise WorkflowCompilationError("stages contain a dependency cycle")
        result.extend(ready)
        remaining.difference_update(ready)
    return result


def _retry_contract(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize workflow retry metadata to coordinator max-attempt semantics."""

    retry = stage.get("retry")
    if retry is None:
        return {"max_attempts": 1, "reconcile_before_retry": False}
    if not isinstance(retry, Mapping):
        raise WorkflowCompilationError(f"stage {stage.get('id')!r}.retry must be an object")

    raw_attempts = retry.get("max_attempts", 1)
    if isinstance(raw_attempts, bool) or not isinstance(raw_attempts, int):
        raise WorkflowCompilationError(
            f"stage {stage.get('id')!r}.retry.max_attempts must be an integer"
        )
    if raw_attempts < 1 or raw_attempts > 11:
        raise WorkflowCompilationError(
            f"stage {stage.get('id')!r}.retry.max_attempts must be between 1 and 11"
        )
    reconcile = retry.get("reconcile_before_retry", False)
    if not isinstance(reconcile, bool):
        raise WorkflowCompilationError(
            f"stage {stage.get('id')!r}.retry.reconcile_before_retry must be boolean"
        )
    return {
        "max_attempts": raw_attempts,
        "reconcile_before_retry": reconcile,
    }


def compile_workflow_pipeline(
    definition: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    service_name: str | None = None,
    service_revision: str = "",
    service_revision_fingerprint: str = "",
    service_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    provider_services: Mapping[str, str] | None = None,
    provider_resources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze a validated workflow/catalog pair for ``PipelineCoordinator``.

    ``provider_services`` and ``provider_resources`` are deployment-owned
    mappings.  They are never taken from request input; when absent, the
    provider ID remains the service identity so an explicit runtime resolver
    can reject or replace it before a leaf is submitted.
    """

    if not isinstance(definition, Mapping):
        raise WorkflowCompilationError("workflow must be an object")
    if not isinstance(catalog, Mapping):
        raise WorkflowCompilationError("operation catalog must be an object")

    errors = validate_workflow_capabilities(
        definition,
        catalog,
        service_definitions=service_definitions,
        runtime_profiles=runtime_profiles,
    )
    if errors:
        raise WorkflowCompilationError(errors)

    raw_stages = definition.get("stages")
    operations = catalog.get("operations", {})
    if not isinstance(raw_stages, list) or not isinstance(operations, Mapping):
        raise WorkflowCompilationError("workflow stages and catalog operations are required")

    provider_services = provider_services or {}
    provider_resources = provider_resources or {}
    stages_by_id = {
        str(stage["id"]): stage
        for stage in raw_stages
        if isinstance(stage, Mapping)
    }
    ordered_ids = _stable_topological_order(list(stages_by_id.values()))
    compiled_stages: list[dict[str, Any]] = []
    max_attempts = 1

    for stage_id in ordered_ids:
        stage = stages_by_id[stage_id]
        operation_id = str(stage["operation"])
        operation = operations[operation_id]
        if not isinstance(operation, Mapping):  # guarded by capability validation
            raise WorkflowCompilationError(
                f"stage {stage_id!r} operation {operation_id!r} is not an object"
            )

        provider_refs = operation.get("provider_refs", [])
        provider = stage.get("provider")
        if provider is None:
            if not isinstance(provider_refs, list) or len(provider_refs) != 1:
                raise WorkflowCompilationError(
                    f"stage {stage_id!r} must select exactly one provider"
                )
            provider = provider_refs[0]
        if not isinstance(provider, str) or not provider.strip():
            raise WorkflowCompilationError(f"stage {stage_id!r} has no provider")
        provider = provider.strip()

        adapter = stage.get("adapter") or operation.get("adapter")
        if not isinstance(adapter, str) or not adapter.strip():
            raise WorkflowCompilationError(f"stage {stage_id!r} has no adapter")
        adapter = adapter.strip()

        retry = _retry_contract(stage)
        max_attempts = max(max_attempts, int(retry["max_attempts"]))
        service = stage.get("service")
        if service is None:
            service = provider_services.get(provider)
        if service is not None and (not isinstance(service, str) or not service.strip()):
            raise WorkflowCompilationError(f"stage {stage_id!r}.service must be a non-empty string")

        resource_id = provider_resources.get(provider)
        params = {
            "operation_id": operation_id,
            "adapter": adapter,
            "bindings": copy.deepcopy(stage.get("bindings") or {}),
            "operation_inputs": copy.deepcopy(operation.get("inputs") or {}),
            "operation_outputs": copy.deepcopy(operation.get("outputs") or {}),
            "required_inputs": copy.deepcopy(operation.get("required_inputs") or []),
            "resource_class": operation.get("resource_class"),
        }
        compiled: dict[str, Any] = {
            "id": stage_id,
            "kind": str(stage["kind"]),
            "operation": operation_id,
            "adapter": adapter,
            "provider": provider,
            "depends_on": [str(dep) for dep in (stage.get("depends_on") or [])],
            "enabled_by_default": stage.get("enabled_by_default", True) is not False,
            "retry": retry,
            "params": params,
        }
        if service is not None:
            compiled["service"] = str(service)
            # This field is deployment-resolved and is consumed only by the
            # trusted leaf adapter; it is never accepted from the request.
            compiled["service_id"] = str(service)
        if resource_id is not None:
            compiled["resource_id"] = str(resource_id)
        if stage.get("runtime_profile") is not None:
            compiled["runtime_profile"] = str(stage["runtime_profile"])
        compiled_stages.append(compiled)

    raw_loop_count = definition.get("max_loop_count", 0)
    if isinstance(raw_loop_count, bool) or not isinstance(raw_loop_count, int):
        raise WorkflowCompilationError("max_loop_count must be an integer")
    if raw_loop_count < 0 or raw_loop_count > 5:
        raise WorkflowCompilationError("max_loop_count must be between 0 and 5")

    workflow_id = str(definition["id"])
    service_identity = service_name or str(definition.get("service_name") or workflow_id)
    result: dict[str, Any] = {
        # Keep the source schema visible.  This is a compiled coordinator
        # plan, not a second user-facing pipeline schema.
        "schema": WORKFLOW_SCHEMA_VERSION,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "compiler": BRIDGE_VERSION,
        "workflow_id": workflow_id,
        "workflow_revision": int(definition.get("revision", 1)),
        "workflow_fingerprint": workflow_fingerprint(definition),
        "catalog_revision": int(catalog.get("revision", 1)),
        "catalog_fingerprint": _fingerprint(catalog),
        "pipeline_id": workflow_id,
        "service_id": service_identity,
        "service_revision": service_revision,
        "service_revision_fingerprint": service_revision_fingerprint,
        "execution_mode": "coordinator",
        "max_retries": max(0, max_attempts - 1),
        "max_loop_count": raw_loop_count,
        "stages": compiled_stages,
        "inputs": copy.deepcopy(definition.get("inputs") or {}),
        "outputs": copy.deepcopy(definition.get("outputs") or []),
        "resource_policy": copy.deepcopy(definition.get("resource_policy") or {}),
        "activation": copy.deepcopy(definition.get("activation") or {}),
    }
    return result


class MappedStageAuthorityResolver(StageAuthorityResolver):
    """Resolve coordinator authority from reviewed deployment metadata.

    A service registration can select an existing adapter without adding a
    controller conditional.  Unknown providers/adapter identities remain a
    hard error rather than silently becoming external HTTP work.
    """

    def __init__(
        self,
        provider_authority: Mapping[str, str] | None = None,
        *,
        leaf_adapters: set[str] | frozenset[str] = frozenset(),
        external_adapters: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self._provider_authority = dict(provider_authority or {})
        self._leaf_adapters = frozenset(leaf_adapters)
        self._external_adapters = frozenset(external_adapters)

    def resolve(self, stage: dict) -> str:
        provider = str(stage.get("provider") or "")
        adapter = str(stage.get("adapter") or "")
        authority = self._provider_authority.get(provider)
        if authority is None and adapter in self._leaf_adapters:
            authority = "leaf"
        if authority is None and adapter in self._external_adapters:
            authority = "external"
        if authority not in {"leaf", "external"}:
            raise WorkflowCompilationError(
                f"stage {stage.get('id')!r} has no reviewed execution authority "
                f"for provider={provider!r}, adapter={adapter!r}"
            )
        return authority


__all__ = [
    "BRIDGE_VERSION",
    "WorkflowCompilationError",
    "compile_workflow_pipeline",
    "MappedStageAuthorityResolver",
]
