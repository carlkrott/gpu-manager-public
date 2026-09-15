"""Capability and typed-binding checks for declarative workflows.

``workflow_contracts`` checks that a workflow is a well-formed DAG.  This
module adds the next, deliberately small, boundary: each stage operation must
be supplied by a reviewed operation catalog and its typed bindings must point
at declared workflow inputs or dependency outputs.  It performs no provider,
HTTP, process, filesystem, or model operations.

The catalog is server-owned metadata.  A user can choose an existing
operation/provider in a GUI, but cannot turn a service definition into an
arbitrary command or endpoint.  A new engine protocol still needs a reviewed
adapter and a catalog entry.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from workflow_contracts import validate_workflow_definition, workflow_fingerprint


OPERATION_CATALOG_SCHEMA = "operation-catalog.v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_FIELDS = {
    "command",
    "exec",
    "shell",
    "privileged",
    "host_mounts",
    "docker_socket",
    "host_network",
}


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_RE.fullmatch(value))


def _valid_type(value: Any) -> bool:
    return isinstance(value, str) and bool(_TYPE_RE.fullmatch(value))


def _scan_for_forbidden_fields(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_FIELDS:
                errors.append(f"{path}.{key_text} is not allowed")
            _scan_for_forbidden_fields(item, f"{path}.{key_text}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_for_forbidden_fields(item, f"{path}[{index}]", errors)


def _validate_ports(value: Any, path: str, errors: list[str]) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    result: dict[str, str] = {}
    for name, type_name in value.items():
        if not _valid_id(name):
            errors.append(f"{path} has invalid port name {name!r}")
        if not _valid_type(type_name):
            errors.append(f"{path}.{name} must be a type identifier")
        else:
            result[str(name)] = str(type_name)
    return result


def validate_operation_catalog(catalog: Mapping[str, Any]) -> list[str]:
    """Return structural errors for a reviewed operation catalog."""

    errors: list[str] = []
    if not isinstance(catalog, Mapping):
        return ["operation catalog must be an object"]
    if catalog.get("schema_version") != OPERATION_CATALOG_SCHEMA:
        errors.append(f"schema_version must be {OPERATION_CATALOG_SCHEMA!r}")
    revision = catalog.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append("revision must be a positive integer")
    operations = catalog.get("operations")
    if not isinstance(operations, Mapping) or not operations:
        return errors + ["operations must be a non-empty object"]
    for operation_id, operation in operations.items():
        path = f"operations.{operation_id}"
        if not _valid_id(operation_id):
            errors.append(f"{path}: operation id is invalid")
        if not isinstance(operation, Mapping):
            errors.append(f"{path} must be an object")
            continue
        adapter = operation.get("adapter")
        if not _nonempty_string(adapter):
            errors.append(f"{path}.adapter must be a non-empty string")
        _validate_ports(operation.get("inputs", {}), f"{path}.inputs", errors)
        _validate_ports(operation.get("outputs", {}), f"{path}.outputs", errors)
        required = operation.get("required_inputs", [])
        if not isinstance(required, list) or any(not _valid_id(item) for item in required):
            errors.append(f"{path}.required_inputs must be a list of valid port names")
        else:
            input_names = set(operation.get("inputs", {})) if isinstance(operation.get("inputs"), Mapping) else set()
            for name in required:
                if name not in input_names:
                    errors.append(f"{path}.required_inputs references unknown input {name!r}")
        providers = operation.get("provider_refs", [])
        if not isinstance(providers, list) or any(not _nonempty_string(item) for item in providers):
            errors.append(f"{path}.provider_refs must be a list of non-empty strings")
        _scan_for_forbidden_fields(operation, path, errors)
    return errors


def _ancestors(definition: Mapping[str, Any]) -> dict[str, set[str]]:
    stages = {
        str(stage.get("id")): stage
        for stage in definition.get("stages", [])
        if isinstance(stage, Mapping) and isinstance(stage.get("id"), str)
    }
    result: dict[str, set[str]] = {stage_id: set() for stage_id in stages}
    for stage_id in stages:
        declared = stages[stage_id].get("depends_on", [])
        pending = list(declared) if isinstance(declared, list) else []
        while pending:
            dependency = pending.pop()
            if dependency in result[stage_id]:
                continue
            if dependency not in stages:
                continue
            result[stage_id].add(dependency)
            declared = stages[dependency].get("depends_on", [])
            if isinstance(declared, list):
                pending.extend(declared)
    return result


def _binding_source(
    source: Any,
    *,
    stage_id: str,
    workflow_inputs: Mapping[str, Any],
    operation_outputs: Mapping[str, Mapping[str, str]],
    ancestors: Mapping[str, set[str]],
    path: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(source, str):
        return [f"{path}.source must be a string"]
    parts = source.split(".")
    if len(parts) == 2 and parts[0] == "workflow":
        if parts[1] not in workflow_inputs:
            errors.append(f"{path}.source references unknown workflow input {parts[1]!r}")
        return errors
    if len(parts) == 3 and parts[0] == "stage":
        source_stage, output_name = parts[1], parts[2]
        if source_stage not in operation_outputs:
            errors.append(f"{path}.source references unknown stage {source_stage!r}")
        elif source_stage not in ancestors.get(stage_id, set()):
            errors.append(
                f"{path}.source stage {source_stage!r} is not a dependency of {stage_id!r}"
            )
        elif output_name not in operation_outputs[source_stage]:
            errors.append(
                f"{path}.source references unknown output {output_name!r} on stage {source_stage!r}"
            )
        return errors
    return [
        f"{path}.source must use workflow.<input> or stage.<stage>.<output>"
    ]


def validate_workflow_capabilities(
    definition: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    service_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate operation, provider, service, profile and typed binding refs.

    ``service_definitions`` and ``runtime_profiles`` are optional so the same
    workflow graph can be checked before a deployment-specific registry is
    assembled.  When supplied, references are resolved strictly.
    """

    errors = list(validate_workflow_definition(definition))
    errors.extend(validate_operation_catalog(catalog))
    if errors:
        # Keep graph/catalog errors useful, but avoid dereferencing malformed
        # structures below.  Valid callers still receive all reference errors.
        if not isinstance(definition, Mapping) or not isinstance(catalog, Mapping):
            return sorted(set(errors))
    operations = catalog.get("operations", {}) if isinstance(catalog, Mapping) else {}
    if not isinstance(operations, Mapping):
        return sorted(set(errors))
    workflow_inputs = definition.get("inputs", {})
    if not isinstance(workflow_inputs, Mapping):
        workflow_inputs = {}
    ancestors = _ancestors(definition)
    operation_outputs: dict[str, Mapping[str, str]] = {}
    stage_entries = definition.get("stages", [])
    if not isinstance(stage_entries, list):
        return sorted(set(errors))

    # Resolve each operation and gather its declared output ports first so
    # downstream sources can be checked independent of declaration order.
    for index, stage in enumerate(stage_entries):
        path = f"stages[{index}]"
        if not isinstance(stage, Mapping):
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, str):
            continue
        operation_id = stage.get("operation")
        operation = operations.get(operation_id)
        if not isinstance(operation, Mapping):
            errors.append(f"{path}.operation references unknown operation {operation_id!r}")
            operation_outputs[stage_id] = {}
            continue
        output_ports = operation.get("outputs", {})
        operation_outputs[stage_id] = output_ports if isinstance(output_ports, Mapping) else {}

    for index, stage in enumerate(stage_entries):
        path = f"stages[{index}]"
        if not isinstance(stage, Mapping):
            continue
        stage_id = stage.get("id")
        operation_id = stage.get("operation")
        if not isinstance(stage_id, str) or not isinstance(operation_id, str):
            continue
        operation = operations.get(operation_id)
        if not isinstance(operation, Mapping):
            continue

        catalog_adapter = operation.get("adapter")
        if stage.get("adapter") is not None and stage.get("adapter") != catalog_adapter:
            errors.append(
                f"{path}.adapter {stage.get('adapter')!r} does not match operation adapter {catalog_adapter!r}"
            )
        providers = operation.get("provider_refs", [])
        provider = stage.get("provider")
        if provider is not None and isinstance(providers, list) and providers and provider not in providers:
            errors.append(f"{path}.provider {provider!r} is not allowed for operation {operation_id!r}")

        service = stage.get("service")
        if service is not None:
            if service_definitions is not None and service not in service_definitions:
                errors.append(f"{path}.service references unknown service {service!r}")
            service_definition = (
                service_definitions.get(service)
                if service_definitions is not None and isinstance(service_definitions.get(service), Mapping)
                else None
            )
            if service_definition is not None:
                profile = stage.get("runtime_profile")
                declared_profile = service_definition.get("runtime_profile")
                if profile and declared_profile and profile != declared_profile:
                    errors.append(
                        f"{path}.runtime_profile {profile!r} does not match service {service!r} profile {declared_profile!r}"
                    )

        profile = stage.get("runtime_profile")
        if profile and runtime_profiles is not None and profile not in runtime_profiles:
            errors.append(f"{path}.runtime_profile references unknown profile {profile!r}")

        operation_inputs = operation.get("inputs", {})
        if not isinstance(operation_inputs, Mapping):
            operation_inputs = {}
        required_inputs = operation.get("required_inputs", [])
        required_set = set(required_inputs) if isinstance(required_inputs, list) else set()
        bindings = stage.get("bindings", {})
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            errors.append(f"{path}.bindings must be an object")
            bindings = {}
        input_bindings = bindings.get("inputs", {})
        if input_bindings is None:
            input_bindings = {}
        if not isinstance(input_bindings, Mapping):
            errors.append(f"{path}.bindings.inputs must be an object")
            input_bindings = {}
        for required in required_set:
            if required not in input_bindings:
                errors.append(f"{path}.bindings.inputs missing required input {required!r}")
        for name, binding in input_bindings.items():
            binding_path = f"{path}.bindings.inputs.{name}"
            if name not in operation_inputs:
                errors.append(f"{binding_path} is not an input of operation {operation_id!r}")
                continue
            if not isinstance(binding, Mapping):
                errors.append(f"{binding_path} must be an object")
                continue
            if "optional" in binding and not isinstance(binding["optional"], bool):
                errors.append(f"{binding_path}.optional must be boolean")
            if name in required_set and binding.get("optional") is True:
                errors.append(f"{binding_path}.optional cannot be true for a required input")
            expected_type = operation_inputs[name]
            if binding.get("type") != expected_type:
                errors.append(
                    f"{binding_path}.type {binding.get('type')!r} does not match {expected_type!r}"
                )
            errors.extend(
                _binding_source(
                    binding.get("source"),
                    stage_id=stage_id,
                    workflow_inputs=workflow_inputs,
                    operation_outputs=operation_outputs,
                    ancestors=ancestors,
                    path=binding_path,
                )
            )

        output_bindings = bindings.get("outputs", {})
        if output_bindings is None:
            output_bindings = {}
        if not isinstance(output_bindings, Mapping):
            errors.append(f"{path}.bindings.outputs must be an object")
            output_bindings = {}
        for name, binding in output_bindings.items():
            binding_path = f"{path}.bindings.outputs.{name}"
            if name not in operation_outputs[stage_id]:
                errors.append(f"{binding_path} is not an output of operation {operation_id!r}")
                continue
            if not isinstance(binding, Mapping):
                errors.append(f"{binding_path} must be an object")
                continue
            if "optional" in binding and not isinstance(binding["optional"], bool):
                errors.append(f"{binding_path}.optional must be boolean")
            expected_type = operation_outputs[stage_id][name]
            if binding.get("type") != expected_type:
                errors.append(
                    f"{binding_path}.type {binding.get('type')!r} does not match {expected_type!r}"
                )

    for index, output in enumerate(definition.get("outputs", []) or []):
        path = f"outputs[{index}]"
        if not isinstance(output, Mapping):
            continue
        source_stage = output.get("source_stage")
        output_name = output.get("name")
        if source_stage is None or output_name is None:
            continue
        if source_stage not in operation_outputs:
            errors.append(f"{path}.source_stage references unknown stage {source_stage!r}")
            continue
        output_type = operation_outputs[source_stage].get(output_name)
        if output_type is None:
            errors.append(
                f"{path}.name references unknown output {output_name!r} on stage {source_stage!r}"
            )
        elif output.get("type") != output_type:
            errors.append(
                f"{path}.type {output.get('type')!r} does not match {output_type!r}"
            )
    return sorted(set(errors))


def workflow_capability_summary(
    definition: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Return safe compile metadata for preview/dry-run responses."""

    operations = catalog.get("operations", {}) if isinstance(catalog, Mapping) else {}
    stages = definition.get("stages", []) if isinstance(definition, Mapping) else []
    return {
        "workflow_id": definition.get("id") if isinstance(definition, Mapping) else None,
        "workflow_revision": definition.get("revision") if isinstance(definition, Mapping) else None,
        "workflow_fingerprint": workflow_fingerprint(definition),
        "catalog_schema": catalog.get("schema_version") if isinstance(catalog, Mapping) else None,
        "catalog_revision": catalog.get("revision") if isinstance(catalog, Mapping) else None,
        "stage_count": len(stages) if isinstance(stages, list) else 0,
        "operation_count": len(operations) if isinstance(operations, Mapping) else 0,
        "validation_errors": validate_workflow_capabilities(definition, catalog),
    }


__all__ = [
    "OPERATION_CATALOG_SCHEMA",
    "validate_operation_catalog",
    "validate_workflow_capabilities",
    "workflow_capability_summary",
]
