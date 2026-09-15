"""Pure validation for versioned, declarative GPU Manager workflows.

The controller should execute registered operations through reviewed adapters;
this module only validates the graph and its references.  It deliberately
does not load files, call providers, or interpret shell commands.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any


WORKFLOW_SCHEMA_VERSION = "workflow.v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STAGE_KINDS = {
    "preparation",
    "research",
    "validation",
    "generation",
    "transform",
    "qc",
    "delivery",
}


def validate_workflow_definition(definition: Mapping[str, Any]) -> list[str]:
    """Return structural errors for a workflow graph.

    Stage ``operation`` and adapter names are intentionally opaque strings;
    capability and authorization checks belong to the adapter registry.
    """

    if not isinstance(definition, Mapping):
        return ["workflow must be an object"]
    errors: list[str] = []
    if definition.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        errors.append(f"schema_version must be {WORKFLOW_SCHEMA_VERSION!r}")
    workflow_id = definition.get("id")
    if not isinstance(workflow_id, str) or not _ID_RE.fullmatch(workflow_id):
        errors.append("id must contain only letters, digits, '.', '_' or '-'")
    revision = definition.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append("revision must be a positive integer")

    stages = definition.get("stages")
    if not isinstance(stages, list) or not stages:
        return errors + ["stages must be a non-empty list"]

    stage_ids: set[str] = set()
    edges: dict[str, set[str]] = {}
    for index, stage in enumerate(stages):
        path = f"stages[{index}]"
        if not isinstance(stage, Mapping):
            errors.append(f"{path} must be an object")
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not _ID_RE.fullmatch(stage_id):
            errors.append(f"{path}.id must contain only letters, digits, '.', '_' or '-'")
            continue
        if stage_id in stage_ids:
            errors.append(f"{path}.id duplicates {stage_id!r}")
        stage_ids.add(stage_id)
        kind = stage.get("kind")
        if kind not in _STAGE_KINDS:
            errors.append(f"{path}.kind must be one of {sorted(_STAGE_KINDS)}")
        operation = stage.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            errors.append(f"{path}.operation must be a non-empty string")
        for field in ("adapter", "provider", "service", "runtime_profile"):
            if field in stage and stage[field] is not None and not isinstance(stage[field], str):
                errors.append(f"{path}.{field} must be a string")
        if "bindings" in stage and stage["bindings"] is not None and not isinstance(
            stage["bindings"], Mapping
        ):
            errors.append(f"{path}.bindings must be an object")
        if "enabled_by_default" in stage and not isinstance(
            stage["enabled_by_default"], bool
        ):
            errors.append(f"{path}.enabled_by_default must be boolean")
        depends_on = stage.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
            not isinstance(item, str) or not item for item in depends_on
        ):
            errors.append(f"{path}.depends_on must be a list of non-empty strings")
            depends_on = []
        edges[stage_id] = set(depends_on)
        retry = stage.get("retry")
        if retry is not None and not isinstance(retry, Mapping):
            errors.append(f"{path}.retry must be an object")

    unknown_dependencies = sorted(
        dependency
        for dependencies in edges.values()
        for dependency in dependencies
        if dependency not in stage_ids
    )
    for dependency in unknown_dependencies:
        errors.append(f"depends_on references unknown stage {dependency!r}")

    # Kahn's algorithm catches cycles without recursing on user-provided data.
    indegree = {stage_id: 0 for stage_id in stage_ids}
    children: dict[str, set[str]] = {stage_id: set() for stage_id in stage_ids}
    for stage_id, dependencies in edges.items():
        for dependency in dependencies & stage_ids:
            indegree[stage_id] += 1
            children[dependency].add(stage_id)
    ready = [stage_id for stage_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(stage_ids):
        errors.append("stages contain a dependency cycle")

    inputs = definition.get("inputs")
    if inputs is not None and not isinstance(inputs, Mapping):
        errors.append("inputs must be an object")
    outputs = definition.get("outputs")
    if outputs is not None and (
        not isinstance(outputs, list)
        or any(not isinstance(item, Mapping) for item in outputs)
    ):
        errors.append("outputs must be a list of objects")
    return errors


def workflow_fingerprint(definition: Mapping[str, Any]) -> str:
    """Return a stable identity for a workflow revision."""

    encoded = json.dumps(
        definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workflow_summary(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Return a non-secret summary suitable for a registry diagnostic."""

    stages = definition.get("stages", []) if isinstance(definition, Mapping) else []
    outputs = definition.get("outputs", []) if isinstance(definition, Mapping) else []
    return {
        "schema_version": definition.get("schema_version") if isinstance(definition, Mapping) else None,
        "id": definition.get("id") if isinstance(definition, Mapping) else None,
        "revision": definition.get("revision") if isinstance(definition, Mapping) else None,
        "stage_count": len(stages) if isinstance(stages, list) else 0,
        "output_count": len(outputs) if isinstance(outputs, list) else 0,
        "fingerprint": workflow_fingerprint(definition),
        "validation_errors": validate_workflow_definition(definition),
    }


__all__ = [
    "WORKFLOW_SCHEMA_VERSION",
    "validate_workflow_definition",
    "workflow_fingerprint",
    "workflow_summary",
]
