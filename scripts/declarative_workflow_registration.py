"""Compile a GUI service draft into a reviewed ``workflow.v1`` graph.

The GUI is allowed to describe *which* reviewed operations a service uses.  It
is not allowed to invent an adapter, endpoint, command, model path, or GPU
owner.  This module turns the small ``processing`` form into the existing
workflow/catalog contract and returns a bounded preview; it never writes a
registry or executes a provider.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from workflow_capabilities import validate_operation_catalog
from workflow_contracts import WORKFLOW_SCHEMA_VERSION, validate_workflow_definition
from workflow_pipeline_bridge import (
    WorkflowCompilationError,
    compile_workflow_pipeline,
)


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_KEYS = {
    "command",
    "exec",
    "shell",
    "privileged",
    "host_mounts",
    "docker_socket",
    "host_network",
    "credentials",
    "credential",
    "token",
    "secret",
}


class DraftWorkflowCompilationError(ValueError):
    """A GUI processing list cannot be mapped to reviewed operations."""

    def __init__(self, errors: list[str] | tuple[str, ...] | str):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = tuple(str(error) for error in errors if str(error))
        super().__init__("draft workflow compilation failed: " + "; ".join(self.errors))


def _type_name(value: Any) -> str | None:
    if isinstance(value, str) and _ID_RE.fullmatch(value.strip()):
        return value.strip()
    return None


def _scan_forbidden_keys(value: Any, *, path: str, errors: list[str]) -> None:
    """Keep GUI-owned nested metadata from becoming an execution escape hatch."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_KEYS:
                errors.append(f"{path}.{key_text} is not allowed in a declarative draft")
            _scan_forbidden_keys(child, path=f"{path}.{key_text}", errors=errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, path=f"{path}[{index}]", errors=errors)


def _workflow_inputs(service: Mapping[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    raw = service.get("input_contract")
    if raw is None:
        raw = service.get("input_schema", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        errors.append("input_contract must be an object to compile processing")
        return {}

    result: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        name_text = str(name)
        if not _ID_RE.fullmatch(name_text):
            errors.append(f"input_contract has invalid input name {name_text!r}")
            continue
        if isinstance(spec, Mapping):
            item = copy.deepcopy(dict(spec))
            type_name = _type_name(item.get("type"))
        else:
            item = {"type": spec}
            type_name = _type_name(spec)
        if type_name is None:
            errors.append(f"input_contract.{name_text}.type must be a type identifier")
            continue
        _scan_forbidden_keys(item, path=f"input_contract.{name_text}", errors=errors)
        item["type"] = type_name
        if "required" in item and not isinstance(item["required"], bool):
            errors.append(f"input_contract.{name_text}.required must be boolean")
        result[name_text] = item
    return result


def _provider_services(
    config: Mapping[str, Any],
    service_definitions: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Resolve only provider-to-service identities already owned by config."""

    providers = config.get("pipeline_providers", {})
    if not isinstance(providers, Mapping):
        return {}
    services = service_definitions or {}
    result: dict[str, str] = {}
    for provider_id, provider in providers.items():
        if not isinstance(provider, Mapping):
            continue
        target = provider.get("service_ref") or provider.get("service") or provider.get("endpoint_ref")
        if isinstance(target, str) and target in services:
            result[str(provider_id)] = target
    return result


def _input_binding(
    raw: Any,
    expected_type: str,
    *,
    path: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, str):
        raw = {"source": raw}
    if not isinstance(raw, Mapping):
        return None, f"{path} must be a binding object or source string"
    binding = copy.deepcopy(dict(raw))
    nested_errors: list[str] = []
    _scan_forbidden_keys(binding, path=path, errors=nested_errors)
    if nested_errors:
        return None, "; ".join(nested_errors)
    source = binding.get("source")
    if not isinstance(source, str) or not source.strip():
        return None, f"{path}.source must be a non-empty string"
    binding["source"] = source.strip()
    binding_type = binding.get("type", expected_type)
    if binding_type != expected_type:
        return None, f"{path}.type {binding_type!r} does not match {expected_type!r}"
    binding["type"] = expected_type
    return binding, None


def _source_stage(source: str) -> str | None:
    parts = source.split(".")
    if len(parts) == 3 and parts[0] == "stage":
        return parts[1]
    return None


def _stage_bindings(
    entry: Mapping[str, Any],
    operation: Mapping[str, Any],
    *,
    stage_id: str,
    workflow_inputs: Mapping[str, Mapping[str, Any]],
    prior_outputs: Mapping[str, list[tuple[str, str]]],
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    operation_inputs = operation.get("inputs", {})
    operation_outputs = operation.get("outputs", {})
    if not isinstance(operation_inputs, Mapping):
        operation_inputs = {}
    if not isinstance(operation_outputs, Mapping):
        operation_outputs = {}

    raw_bindings = entry.get("bindings")
    if raw_bindings is not None and not isinstance(raw_bindings, Mapping):
        errors.append(f"processing.{stage_id}.bindings must be an object")
        raw_bindings = {}
    raw_bindings = raw_bindings or {}
    raw_inputs = raw_bindings.get("inputs", entry.get("inputs", {}))
    raw_outputs = raw_bindings.get("outputs", entry.get("outputs", {}))
    if not isinstance(raw_inputs, Mapping):
        errors.append(f"processing.{stage_id}.inputs must be an object")
        raw_inputs = {}
    if not isinstance(raw_outputs, Mapping):
        errors.append(f"processing.{stage_id}.outputs must be an object")
        raw_outputs = {}

    inputs: dict[str, Any] = {}
    dependencies: set[str] = set()
    required = operation.get("required_inputs", [])
    required_set = set(required) if isinstance(required, list) else set(operation_inputs)
    for name, expected in operation_inputs.items():
        expected_type = _type_name(expected)
        if expected_type is None:
            errors.append(f"operation input {stage_id}.{name} has invalid type")
            continue
        raw = raw_inputs.get(name)
        if raw is None and name in workflow_inputs:
            raw = {"source": f"workflow.{name}"}
        if raw is None:
            candidates = prior_outputs.get(str(name), [])
            if len(candidates) == 1:
                raw = {"source": f"stage.{candidates[0][0]}.{name}"}
            elif len(candidates) > 1:
                errors.append(
                    f"processing.{stage_id}.inputs.{name} has ambiguous prior outputs"
                )
                continue
        if raw is None:
            if name in required_set:
                errors.append(f"processing.{stage_id}.inputs missing required input {name!r}")
            continue
        binding, error = _input_binding(
            raw,
            expected_type,
            path=f"processing.{stage_id}.inputs.{name}",
        )
        if error:
            errors.append(error)
            continue
        assert binding is not None
        source = binding["source"]
        dependency = _source_stage(source)
        if dependency:
            dependencies.add(dependency)
        inputs[str(name)] = binding

    outputs: dict[str, Any] = {}
    for name, expected in operation_outputs.items():
        expected_type = _type_name(expected)
        if expected_type is None:
            errors.append(f"operation output {stage_id}.{name} has invalid type")
            continue
        raw = raw_outputs.get(name, {})
        if isinstance(raw, str):
            raw = {"type": raw}
        if not isinstance(raw, Mapping):
            errors.append(f"processing.{stage_id}.outputs.{name} must be an object")
            continue
        output = copy.deepcopy(dict(raw))
        output_type = output.get("type", expected_type)
        if output_type != expected_type:
            errors.append(
                f"processing.{stage_id}.outputs.{name}.type {output_type!r} "
                f"does not match {expected_type!r}"
            )
            continue
        output["type"] = expected_type
        outputs[str(name)] = output

    explicit_dependencies = entry.get("depends_on", [])
    if not isinstance(explicit_dependencies, list) or any(
        not isinstance(item, str) or not item for item in explicit_dependencies
    ):
        errors.append(f"processing.{stage_id}.depends_on must be a list of stage IDs")
        explicit_dependencies = []
    dependencies.update(str(item) for item in explicit_dependencies)
    return inputs, outputs, sorted(dependencies)


def _workflow_outputs(
    service: Mapping[str, Any],
    stage_records: list[Mapping[str, Any]],
    operation_by_stage: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    raw_outputs = service.get("outputs", [])
    if isinstance(raw_outputs, Mapping):
        normalized_outputs: list[dict[str, Any]] = []
        for name, value in raw_outputs.items():
            if isinstance(value, Mapping):
                item = copy.deepcopy(dict(value))
                item.setdefault("name", str(name))
            else:
                item = {"name": str(name), "type": value}
            normalized_outputs.append(item)
        raw_outputs = normalized_outputs
    if not isinstance(raw_outputs, list):
        errors.append("outputs must be a list or object to compile processing")
        return []
    available: dict[str, list[tuple[str, str]]] = {}
    for stage in stage_records:
        operation = operation_by_stage.get(str(stage.get("id")), {})
        for name, type_name in (operation.get("outputs", {}) or {}).items():
            if isinstance(type_name, str):
                available.setdefault(str(name), []).append((str(stage["id"]), type_name))

    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_outputs):
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, Mapping):
            errors.append(f"outputs[{index}] must be a string or object")
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not _ID_RE.fullmatch(name):
            errors.append(f"outputs[{index}].name must be a valid output identifier")
            continue
        candidates = available.get(name, [])
        source_stage = raw.get("source_stage")
        if source_stage is None:
            if len(candidates) != 1:
                errors.append(
                    f"outputs[{index}] {name!r} must resolve to exactly one operation output"
                )
                continue
            source_stage, inferred_type = candidates[0]
        else:
            inferred_type = next(
                (type_name for stage_id, type_name in candidates if stage_id == source_stage),
                None,
            )
            if inferred_type is None:
                errors.append(f"outputs[{index}] references unknown source for {name!r}")
                continue
        output_type = raw.get("type", inferred_type)
        if output_type != inferred_type:
            errors.append(
                f"outputs[{index}].type {output_type!r} does not match {inferred_type!r}"
            )
            continue
        result.append({"name": name, "type": output_type, "source_stage": source_stage})
    return result


def compile_service_processing(
    service: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    service_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile a declarative service's processing list without side effects."""

    errors: list[str] = []
    service_name = str(service.get("name") or "").strip()
    if not _ID_RE.fullmatch(service_name):
        errors.append("service name is required for workflow compilation")
    errors.extend(validate_operation_catalog(catalog))
    _scan_forbidden_keys(
        service.get("processing"), path="processing", errors=errors
    )
    _scan_forbidden_keys(
        service.get("resource_requirements"),
        path="resource_requirements",
        errors=errors,
    )
    _scan_forbidden_keys(service.get("outputs"), path="outputs", errors=errors)
    stages_raw = service.get("processing")
    if not isinstance(stages_raw, list) or not stages_raw:
        errors.append("processing must be a non-empty list to compile a workflow")
        raise DraftWorkflowCompilationError(errors)

    workflow_inputs = _workflow_inputs(service, errors)
    resource_requirements = service.get("resource_requirements")
    if resource_requirements is not None and not isinstance(resource_requirements, Mapping):
        errors.append("resource_requirements must be an object to compile processing")
    operations = catalog.get("operations", {})
    if not isinstance(operations, Mapping):
        errors.append("operation catalog operations must be an object")
        raise DraftWorkflowCompilationError(errors)

    stage_records: list[dict[str, Any]] = []
    operation_by_stage: dict[str, Mapping[str, Any]] = {}
    prior_outputs: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()
    provider_services = _provider_services(
        config or {},
        service_definitions,
    )
    for index, raw in enumerate(stages_raw):
        path = f"processing[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{path} must be an object")
            continue
        stage_id = raw.get("stage", raw.get("id"))
        if not isinstance(stage_id, str) or not _ID_RE.fullmatch(stage_id):
            errors.append(f"{path}.stage must be a valid stage identifier")
            continue
        if stage_id in seen:
            errors.append(f"{path}.stage duplicates {stage_id!r}")
            continue
        seen.add(stage_id)
        operation_id = raw.get("operation")
        operation = operations.get(operation_id) if isinstance(operation_id, str) else None
        if not isinstance(operation, Mapping):
            errors.append(f"{path}.operation references unknown operation {operation_id!r}")
            continue
        operation_by_stage[stage_id] = operation
        kind = raw.get("kind", operation.get("kind"))
        if not isinstance(kind, str) or not kind.strip():
            errors.append(f"{path}.kind is required or must be declared by the catalog")
            kind = ""
        adapter = raw.get("adapter", operation.get("adapter"))
        if not isinstance(adapter, str) or not adapter.strip():
            errors.append(f"{path}.adapter is not declared by the catalog")
            adapter = ""
        if raw.get("adapter") is not None and raw.get("adapter") != operation.get("adapter"):
            errors.append(f"{path}.adapter does not match its catalog operation")
        providers = operation.get("provider_refs", [])
        provider = raw.get("provider")
        if provider is None and isinstance(providers, list) and len(providers) == 1:
            provider = providers[0]
        if not isinstance(provider, str) or not provider.strip():
            errors.append(f"{path}.provider must resolve to one catalog provider")
            provider = ""
        elif isinstance(providers, list) and provider not in providers:
            errors.append(f"{path}.provider is not allowed for operation {operation_id!r}")

        bindings, output_bindings, dependencies = _stage_bindings(
            raw,
            operation,
            stage_id=stage_id,
            workflow_inputs=workflow_inputs,
            prior_outputs=prior_outputs,
            errors=errors,
        )
        stage: dict[str, Any] = {
            "id": stage_id,
            "kind": kind.strip() if isinstance(kind, str) else kind,
            "operation": str(operation_id),
            "adapter": adapter.strip() if isinstance(adapter, str) else adapter,
            "provider": provider.strip() if isinstance(provider, str) else provider,
            "depends_on": dependencies,
            "bindings": {"inputs": bindings, "outputs": output_bindings},
        }
        for field in ("enabled_by_default", "runtime_profile", "retry"):
            if field in raw:
                stage[field] = copy.deepcopy(raw[field])
        service_target = raw.get("service")
        if service_target is None and isinstance(provider, str):
            service_target = provider_services.get(provider)
        if service_target is not None:
            stage["service"] = service_target
        stage_records.append(stage)
        for output_name, output_type in (operation.get("outputs", {}) or {}).items():
            if isinstance(output_type, str):
                prior_outputs.setdefault(str(output_name), []).append((stage_id, output_type))

    workflow_outputs = _workflow_outputs(
        service,
        stage_records,
        operation_by_stage,
        errors,
    )
    workflow: dict[str, Any] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "id": service_name,
        "revision": int(service.get("workflow_revision", 1) or 1),
        "status": "draft",
        "description": str(service.get("display") or service_name),
        "inputs": workflow_inputs,
        "stages": stage_records,
        "outputs": workflow_outputs,
        "resource_policy": copy.deepcopy(resource_requirements or {}),
        "activation": {
            "state": "draft",
            "required_gates": ["typed_graph_compiles", "reviewed_provider_bindings"],
        },
    }
    operation_catalog_ref = service.get("operation_catalog_ref")
    if isinstance(operation_catalog_ref, str) and operation_catalog_ref:
        workflow["operation_catalog_ref"] = operation_catalog_ref

    errors.extend(validate_workflow_definition(workflow))
    if errors:
        raise DraftWorkflowCompilationError(sorted(set(errors)))
    try:
        compiled = compile_workflow_pipeline(
            workflow,
            catalog,
            service_name=service_name,
            service_definitions=service_definitions,
            runtime_profiles=runtime_profiles,
            provider_services=provider_services,
        )
    except WorkflowCompilationError as exc:
        raise DraftWorkflowCompilationError(exc.errors) from exc
    return {
        "status": "compiled",
        "workflow": workflow,
        "compiled": compiled,
        "workflow_fingerprint": compiled["workflow_fingerprint"],
        "stage_count": len(stage_records),
        "stage_ids": [str(stage["id"]) for stage in stage_records],
    }


__all__ = ["DraftWorkflowCompilationError", "compile_service_processing"]
