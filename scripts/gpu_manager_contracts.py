"""Small, shared contracts for GPU Manager control-plane boundaries.

This module deliberately contains no Redis, HTTP, systemd, or GPU calls.  It
gives the controller, broker, and offline tools the same answers for the few
values that must not drift between them: priority translation, registry
identity, active-job demand, and resource overlap.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
from collections.abc import Mapping, Iterable
from typing import Any
from urllib.parse import urlsplit

PUBLIC_ACTIVATION_STATUSES = frozenset({
    "enabled", "qualified", "ready", "active", "activated",
    "production-qualified",
})


def service_is_publicly_eligible(entry: Mapping[str, Any]) -> bool:
    """Shared admission/discovery gate for registry records."""
    if not isinstance(entry, Mapping) or entry.get("enabled", True) is False:
        return False
    metadata = entry.get("metadata")
    for field in ("activation_status", "qualification_status", "status"):
        for source in (entry, metadata if isinstance(metadata, Mapping) else None):
            if not isinstance(source, Mapping):
                continue
            value = source.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip().lower() in PUBLIC_ACTIVATION_STATUSES
    return True


def is_retired_minimax_comfyui_template(entry: Mapping[str, Any] | None) -> bool:
    """Identify historical MiniMax ComfyUI leaves even in old snapshots."""
    return bool(
        isinstance(entry, Mapping)
        and str(entry.get("backend") or "").strip() == "minimax-music3"
        and str(entry.get("workflow_backend") or "").strip().lower() == "comfyui"
    )

try:  # Support both ``PYTHONPATH=scripts`` and package-style imports.
    from workflow_contracts import validate_workflow_definition, workflow_summary
except ImportError:  # pragma: no cover - exercised only by package imports
    from .workflow_contracts import validate_workflow_definition, workflow_summary

try:
    from runtime_contracts import (
        model_set_fingerprint,
        model_set_summary,
        runtime_profile_summary,
        validate_model_set,
        validate_runtime_profile,
        validate_runtime_registry,
    )
except ImportError:  # pragma: no cover - exercised only by package imports
    from .runtime_contracts import (
        model_set_fingerprint,
        model_set_summary,
        runtime_profile_summary,
        validate_model_set,
        validate_runtime_profile,
        validate_runtime_registry,
    )


class PriorityConvention(StrEnum):
    """Numeric conventions already present at the public boundaries."""

    # Generation templates in gpu-manager use larger values for more urgency.
    GENERATION_HIGHER_IS_URGENT = "generation_higher_is_urgent"
    # Older dashboard/queue controls use smaller values for more urgency.
    LEGACY_LOWER_IS_URGENT = "legacy_lower_is_urgent"
    # Combined Gemma's broker uses 0/10/20, with zero most urgent.
    BROKER_LOWER_IS_URGENT = "broker_lower_is_urgent"


class PriorityClass(StrEnum):
    INTERACTIVE = "interactive"
    NORMAL = "normal"
    BACKGROUND = "background"


_PRIORITY_RANK = {
    PriorityClass.INTERACTIVE: 0,
    PriorityClass.NORMAL: 1,
    PriorityClass.BACKGROUND: 2,
}
_BROKER_VALUE = {
    PriorityClass.INTERACTIVE: 0,
    PriorityClass.NORMAL: 10,
    PriorityClass.BACKGROUND: 20,
}
_PRIORITY_NAMES = {
    "interactive": PriorityClass.INTERACTIVE,
    "high": PriorityClass.INTERACTIVE,
    "urgent": PriorityClass.INTERACTIVE,
    "normal": PriorityClass.NORMAL,
    "medium": PriorityClass.NORMAL,
    "med": PriorityClass.NORMAL,
    "background": PriorityClass.BACKGROUND,
    "low": PriorityClass.BACKGROUND,
}


@dataclass(frozen=True, slots=True)
class PriorityEnvelope:
    """A priority plus its source convention and broker translation."""

    raw: int
    priority_class: PriorityClass
    broker_priority: int
    convention: PriorityConvention

    @property
    def rank(self) -> int:
        return _PRIORITY_RANK[self.priority_class]

    def to_fields(self) -> dict[str, Any]:
        return {
            "priority": self.raw,
            "priority_class": self.priority_class.value,
            "broker_priority": self.broker_priority,
            "priority_convention": self.convention.value,
        }


def _coerce_int(value: Any, *, field: str = "priority") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != number):
        raise ValueError(f"{field} must be an integer")
    return number


def _class_for_numeric(value: int, convention: PriorityConvention) -> PriorityClass:
    if convention is PriorityConvention.GENERATION_HIGHER_IS_URGENT:
        # Established generation traffic uses 50 for normal Krea work and 60
        # for the higher-priority Qwen handoff. Keep that numeric ordering while
        # translating it into the broker's named lanes.
        if value >= 60:
            return PriorityClass.INTERACTIVE
        if value >= 10:
            return PriorityClass.NORMAL
        return PriorityClass.BACKGROUND
    if convention in {
        PriorityConvention.LEGACY_LOWER_IS_URGENT,
        PriorityConvention.BROKER_LOWER_IS_URGENT,
    }:
        if value <= 3:
            return PriorityClass.INTERACTIVE
        if value <= 10:
            return PriorityClass.NORMAL
        return PriorityClass.BACKGROUND
    raise ValueError(f"unsupported priority convention: {convention}")


def priority_envelope(
    value: Any = None,
    *,
    priority_class: str | PriorityClass | None = None,
    convention: PriorityConvention | str = PriorityConvention.GENERATION_HIGHER_IS_URGENT,
) -> PriorityEnvelope:
    """Normalize one priority without silently changing its convention.

    Named classes are preferred for new callers.  Numeric values remain
    lossless in ``raw`` while the broker receives its canonical 0/10/20 value.
    """

    convention = PriorityConvention(convention)
    if priority_class is not None:
        try:
            cls = PriorityClass(str(priority_class).lower())
        except ValueError:
            cls = _PRIORITY_NAMES.get(str(priority_class).lower())
        if cls is None:
            raise ValueError(f"unknown priority class: {priority_class!r}")
        raw = _BROKER_VALUE[cls] if value is None else _coerce_int(value)
    else:
        # Preserve the established defaults at each boundary: generation
        # templates use 50 as their normal admission value, while the older
        # lower-is-urgent queue/broker surfaces use 10 for normal work.
        default_raw = (
            50
            if convention is PriorityConvention.GENERATION_HIGHER_IS_URGENT
            else 10
        )
        raw = default_raw if value is None else _coerce_int(value)
        cls = _class_for_numeric(raw, convention)
    return PriorityEnvelope(raw, cls, _BROKER_VALUE[cls], convention)


def apply_priority_contract(
    payload: Mapping[str, Any],
    *,
    parent: Mapping[str, Any] | None = None,
    convention: PriorityConvention | str = PriorityConvention.GENERATION_HIGHER_IS_URGENT,
    allow_elevation: bool = False,
) -> dict[str, Any]:
    """Return a payload carrying explicit, propagated priority fields.

    A child inherits its parent's priority when it does not specify one.  A
    child may not silently elevate itself above its parent; callers that have
    an authorized escalation must opt in explicitly.
    """

    result = dict(payload)
    parent_env = None
    if parent:
        parent_env = priority_envelope(
            parent.get("priority"),
            priority_class=parent.get("priority_class"),
            convention=parent.get("priority_convention", convention),
        )

    has_child_priority = "priority" in result or "priority_class" in result
    if not has_child_priority and parent_env is not None:
        result.update(parent_env.to_fields())
        result["priority_source"] = "parent"
        return result

    env = priority_envelope(
        result.get("priority"),
        priority_class=result.get("priority_class"),
        convention=result.get("priority_convention", convention),
    )
    if parent_env is not None and env.rank < parent_env.rank and not allow_elevation:
        raise ValueError("priority elevation requires explicit authorization")
    result.update(env.to_fields())
    result.setdefault("priority_source", "request")
    return result


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROUTING_TYPES = {"llm", "generation", "orchestrator", "other"}
_SERVICE_MAPPING_FIELDS = {
    "input_contract",
    "input_schema",
    "output_contract",
    "resource_requirements",
    "admission",
    "workflow",
    "metadata",
}
_SERVICE_OBJECT_OR_LIST_FIELDS = {"processing", "outputs"}
_SERVICE_STRING_FIELDS = {
    "runtime_profile",
    "workflow_ref",
    "operation_catalog_ref",
    "source_lock_ref",
    "adapter",
}
_SERVICE_STRING_LIST_FIELDS = {
    "dependencies",
    "models",
    "model_targets",
    "capabilities",
    "aliases",
}
_DISALLOWED_DYNAMIC_FIELDS = {
    "command",
    "exec",
    "shell",
    "privileged",
    "host_mounts",
    "docker_socket",
    "host_network",
}


def _safe_reference(value: Any, *, allow_path: bool = False) -> bool:
    """Accept an inert registry reference, never an executable/path escape."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return False
    reference = value.strip().replace("\\", "/")
    if reference.startswith(("/", "~")) or "://" in reference:
        return False
    if any(part == ".." for part in reference.split("/")):
        return False
    if allow_path:
        return all(ord(char) >= 0x20 for char in reference)
    return bool(_NAME_RE.fullmatch(reference))


def validate_service_definition(name: str, service: Mapping[str, Any]) -> list[str]:
    """Validate the user-editable portion of a service definition.

    This is intentionally structural.  Engine-specific capability checks are
    performed by the runtime adapter, not guessed from a GUI form.
    """

    errors: list[str] = []
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        errors.append("name must contain only letters, digits, '.', '_' or '-'")
    if not isinstance(service, Mapping):
        return errors + ["service must be an object"]
    port = service.get("port", 0)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        errors.append("port must be an integer between 0 and 65535")
    routing_type = service.get("routing_group_type", "llm")
    if routing_type not in _ROUTING_TYPES:
        errors.append(f"routing_group_type must be one of {sorted(_ROUTING_TYPES)}")
    if "launch_args" in service and (
        not isinstance(service["launch_args"], list)
        or any(not isinstance(item, str) for item in service["launch_args"])
    ):
        errors.append("launch_args must be a list of strings")
    if "env_overrides" in service and (
        not isinstance(service["env_overrides"], Mapping)
        or any(not isinstance(k, str) or not isinstance(v, str)
               for k, v in service["env_overrides"].items())
    ):
        errors.append("env_overrides must map string names to string values")
    for field in ("systemd_unit", "routing_group", "bundle"):
        if field in service and service[field] is not None and not isinstance(service[field], str):
            errors.append(f"{field} must be a string")
    for field in _SERVICE_STRING_FIELDS:
        if field in service and service[field] is not None and not isinstance(service[field], str):
            errors.append(f"{field} must be a string")
        elif field in service and service[field] is not None:
            if field in {"workflow_ref", "operation_catalog_ref", "source_lock_ref"}:
                if not _safe_reference(service[field], allow_path=True):
                    errors.append(
                        f"{field} must be a relative registry reference"
                    )
            elif not _safe_reference(service[field]):
                errors.append(
                    f"{field} must be a simple registry identifier"
                )
    for field in _SERVICE_MAPPING_FIELDS:
        if field in service and service[field] is not None and not isinstance(service[field], Mapping):
            errors.append(f"{field} must be an object")
    if isinstance(service.get("workflow"), Mapping):
        errors.extend(
            f"workflow.{error}" for error in validate_workflow_definition(service["workflow"])
        )
    for field in _SERVICE_OBJECT_OR_LIST_FIELDS:
        value = service.get(field)
        if field not in service or value is None:
            continue
        if isinstance(value, Mapping):
            continue
        if not isinstance(value, list):
            errors.append(f"{field} must be an object or list")
            continue
        if field == "processing" and any(not isinstance(item, Mapping) for item in value):
            errors.append("processing list entries must be objects")
        if field == "outputs" and any(
            not isinstance(item, (str, Mapping)) for item in value
        ):
            errors.append("outputs list entries must be strings or objects")
    for field in _SERVICE_STRING_LIST_FIELDS:
        if field in service and (
            not isinstance(service[field], list)
            or any(not isinstance(item, str) or not item for item in service[field])
        ):
            errors.append(f"{field} must be a list of non-empty strings")
    for field in _DISALLOWED_DYNAMIC_FIELDS:
        if field in service:
            errors.append(f"{field} is not allowed in a service definition")
    return errors


def validate_registry(config: Mapping[str, Any]) -> list[str]:
    """Return structural errors without mutating the supplied registry."""

    if not isinstance(config, Mapping):
        return ["registry must be an object"]
    services = config.get("services", {})
    if not isinstance(services, Mapping):
        return ["services must be an object"]
    errors: list[str] = []
    for name, service in services.items():
        errors.extend(f"services.{name}: {error}" for error in validate_service_definition(name, service))
    errors.extend(validate_runtime_registry(config))
    errors.extend(checked_runtime_authority_errors(config))
    return errors


def service_registration_conflicts(
    config: Mapping[str, Any],
    service_name: str,
    candidate: Mapping[str, Any],
) -> list[str]:
    """Return conflicts introduced by enabling one service revision.

    Existing registries are intentionally allowed to load with legacy
    duplicate-port/unit findings so migration can be reviewed.  New or
    updated enabled services must not add another owner for an endpoint or
    systemd unit; otherwise a GUI edit can create a second process without a
    controller code change.
    """

    if not isinstance(candidate, Mapping) or candidate.get("enabled") is not True:
        return []
    services = config.get("services", {}) if isinstance(config, Mapping) else {}
    if not isinstance(services, Mapping):
        return []
    port = candidate.get("port")
    port_key = port if type(port) is int and port > 0 else None
    unit = candidate.get("systemd_unit")
    unit_key = unit.strip() if isinstance(unit, str) and unit.strip() else None
    if port_key is None and unit_key is None:
        return []

    conflicts: list[str] = []
    for other_name, other in services.items():
        if str(other_name) == str(service_name) or not isinstance(other, Mapping):
            continue
        if other.get("enabled") is not True:
            continue
        other_port = other.get("port")
        if port_key is not None and type(other_port) is int and other_port == port_key:
            conflicts.append(
                f"enabled service {service_name!r} shares port {port_key} "
                f"with {other_name!r}"
            )
        other_unit = other.get("systemd_unit")
        if (
            unit_key is not None
            and isinstance(other_unit, str)
            and other_unit.strip() == unit_key
        ):
            conflicts.append(
                f"enabled service {service_name!r} shares systemd unit "
                f"{unit_key!r} with {other_name!r}"
            )
    return sorted(set(conflicts))


def checked_runtime_authority_errors(config: Mapping[str, Any]) -> list[str]:
    """Reject active mixed lifecycle authorities for one local endpoint.

    A checked bundle and a legacy systemd bundle must not independently own
    services behind the same host port. Multiple checked profiles are safe only
    when they share one runtime instance/fence.
    """

    services = config.get("services", {}) if isinstance(config, Mapping) else {}
    bundles = config.get("bundles", {}) if isinstance(config, Mapping) else {}
    if not isinstance(services, Mapping) or not isinstance(bundles, Mapping):
        return []

    owners: dict[str, list[tuple[str, str | None]]] = {}
    for bundle_name, bundle in bundles.items():
        if not isinstance(bundle, Mapping):
            continue
        instance = bundle.get("runtime_instance")
        instance_id = instance if isinstance(instance, str) and instance else None
        members = bundle.get("services") or []
        if not isinstance(members, list):
            continue
        for service_name in members:
            service = services.get(service_name)
            if not isinstance(service, Mapping) or service.get("enabled") is not True:
                continue
            endpoint = service.get("endpoint")
            if isinstance(endpoint, str) and endpoint:
                parsed = urlsplit(endpoint)
                try:
                    endpoint_port = parsed.port
                except ValueError:
                    endpoint_port = None
                if parsed.hostname in {"localhost", "127.0.0.1", "::1"} and endpoint_port:
                    endpoint_key = f"loopback:{endpoint_port}"
                else:
                    endpoint_key = endpoint.rstrip("/")
            else:
                port = service.get("port")
                if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
                    continue
                endpoint_key = f"loopback:{port}"
            owners.setdefault(endpoint_key, []).append((str(bundle_name), instance_id))

    errors: list[str] = []
    for endpoint, endpoint_owners in sorted(owners.items()):
        checked_instances = {instance for _, instance in endpoint_owners if instance}
        if not checked_instances:
            continue
        legacy_bundles = sorted(name for name, instance in endpoint_owners if instance is None)
        if legacy_bundles:
            errors.append(
                f"runtime endpoint {endpoint!r} mixes checked and legacy lifecycle "
                f"authority; migrate bundles {legacy_bundles} to instance "
                f"{sorted(checked_instances)}"
            )
        if len(checked_instances) > 1:
            errors.append(
                f"runtime endpoint {endpoint!r} is owned by multiple checked runtime "
                f"instances: {sorted(checked_instances)}"
            )
    return errors


def registry_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 identity for routing-relevant config."""

    relevant = {
        key: config.get(key)
        for key in (
            "services",
            "generation_templates",
            "routing_groups",
            "bundles",
            "runtime_profiles",
            "model_sets",
            "service_revision_catalog",
            "scheduling",
            # Provider/workflow records are part of admission.  Omitting them
            # lets a GUI CAS update race through after a provider endpoint,
            # adapter, or workflow catalog has changed while the visible
            # service/template sections remain identical.
            "pipeline_providers",
            "service_pipelines",
            "pipelines",
            "workflows",
            "operation_catalogs",
            "source_locks",
        )
        if key in config
    }
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def registry_summary(config: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    """Build a safe, non-secret registry diagnostic for health endpoints."""

    services = config.get("services", {}) if isinstance(config, Mapping) else {}
    templates = config.get("generation_templates", {}) if isinstance(config, Mapping) else {}
    runtime_profiles = config.get("runtime_profiles", {}) if isinstance(config, Mapping) else {}
    model_sets = config.get("model_sets", {}) if isinstance(config, Mapping) else {}
    revision_catalog = config.get("service_revision_catalog", {}) if isinstance(config, Mapping) else {}
    revision_services = (
        revision_catalog.get("services", {})
        if isinstance(revision_catalog, Mapping)
        else {}
    )
    workflow_summaries = {
        str(name): workflow_summary(service["workflow"])
        for name, service in (services.items() if isinstance(services, Mapping) else [])
        if isinstance(service, Mapping) and isinstance(service.get("workflow"), Mapping)
    }
    return {
        "schema_version": 1,
        "owner": owner,
        "fingerprint": registry_fingerprint(config),
        "service_count": len(services) if isinstance(services, Mapping) else 0,
        "template_count": len(templates) if isinstance(templates, Mapping) else 0,
        "runtime_profile_count": len(runtime_profiles) if isinstance(runtime_profiles, Mapping) else 0,
        "model_set_count": len(model_sets) if isinstance(model_sets, Mapping) else 0,
        "revision_service_count": len(revision_services) if isinstance(revision_services, Mapping) else 0,
        "validation_errors": validate_registry(config),
        "legacy_findings": registry_legacy_findings(config),
        "workflow_summaries": workflow_summaries,
        "runtime_profile_summaries": {
            str(name): runtime_profile_summary(str(name), profile)
            for name, profile in (
                runtime_profiles.items() if isinstance(runtime_profiles, Mapping) else []
            )
            if isinstance(profile, Mapping)
        },
        "model_set_summaries": {
            str(name): model_set_summary(str(name), model_set)
            for name, model_set in (
                model_sets.items() if isinstance(model_sets, Mapping) else []
            )
            if isinstance(model_set, Mapping)
        },
    }


def registry_legacy_findings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Identify safe-to-report inactive, orphaned, or conflicting registry data.

    This is deliberately an inventory, not an automatic cleanup operation.  A
    disabled service may be an intentional maintenance choice, so callers must
    review these findings before deprecating or deleting anything.
    """

    findings: list[dict[str, Any]] = []
    services = config.get("services", {}) if isinstance(config, Mapping) else {}
    templates = config.get("generation_templates", {}) if isinstance(config, Mapping) else {}
    groups = config.get("routing_groups", {}) if isinstance(config, Mapping) else {}
    runtime_profiles = config.get("runtime_profiles", {}) if isinstance(config, Mapping) else {}
    model_sets = config.get("model_sets", {}) if isinstance(config, Mapping) else {}
    workflows = config.get("workflows", {}) if isinstance(config, Mapping) else {}
    pipeline_providers = config.get("pipeline_providers", {}) if isinstance(config, Mapping) else {}
    service_pipelines = config.get("service_pipelines", {}) if isinstance(config, Mapping) else {}
    legacy_pipelines = config.get("pipelines", {}) if isinstance(config, Mapping) else {}
    operation_catalogs = config.get("operation_catalogs", {}) if isinstance(config, Mapping) else {}
    source_locks = config.get("source_locks", {}) if isinstance(config, Mapping) else {}
    if not isinstance(services, Mapping):
        services = {}
    if not isinstance(templates, Mapping):
        templates = {}
    if not isinstance(groups, Mapping):
        groups = {}
    if not isinstance(runtime_profiles, Mapping):
        runtime_profiles = {}
    if not isinstance(model_sets, Mapping):
        model_sets = {}
    if not isinstance(workflows, Mapping):
        workflows = {}
    if not isinstance(pipeline_providers, Mapping):
        pipeline_providers = {}
    if not isinstance(service_pipelines, Mapping):
        service_pipelines = {}
    if not isinstance(legacy_pipelines, Mapping):
        legacy_pipelines = {}
    if not isinstance(operation_catalogs, Mapping):
        operation_catalogs = {}
    if not isinstance(source_locks, Mapping):
        source_locks = {}

    def add(code: str, severity: str, path: str, detail: str) -> None:
        findings.append({"code": code, "severity": severity, "path": path, "detail": detail})

    def declared_activation_status(entry: Mapping[str, Any]) -> tuple[str, str] | None:
        """Return the explicit activation/qualification/status value, if present."""

        # Activation is the authoritative public/private gate when both
        # fields are present; qualification remains useful for inventory.
        metadata = entry.get("metadata")
        for field in ("activation_status", "qualification_status", "status"):
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                return field, value.strip()
            if isinstance(metadata, Mapping):
                value = metadata.get(field)
                if isinstance(value, str) and value.strip():
                    return f"metadata.{field}", value.strip()
        return None

    public_statuses = {
        "enabled",
        "qualified",
        "ready",
        "active",
        "activated",
        "production-qualified",
    }
    retired_statuses = {"retired", "deprecated"}

    ports: dict[int, list[str]] = {}
    units: dict[str, list[str]] = {}
    referenced_groups: set[str] = set()
    referenced_services: set[str] = set()
    referenced_profiles: set[str] = set()
    referenced_model_sets: set[str] = set()
    referenced_workflows: set[str] = set()
    referenced_pipelines: set[str] = set()
    referenced_providers: set[str] = set()
    referenced_catalogs: set[str] = set()
    referenced_source_locks: set[str] = set()
    for name, service in services.items():
        if not isinstance(service, Mapping):
            continue
        path = f"services.{name}"
        if service.get("enabled") is False:
            add("inactive_service", "info", f"{path}.enabled", "service is disabled")
        if service.get("legacy") is True or service.get("deprecated") is True:
            add("legacy_marker", "warning", path, "service is explicitly marked legacy/deprecated")
        declared_status = declared_activation_status(service)
        if service.get("enabled") is True and declared_status:
            status_field, status = declared_status
            if status.lower() not in public_statuses:
                code = (
                    "retired_service_enabled"
                    if status.lower() in retired_statuses
                    else "active_unqualified_service"
                )
                add(
                    code,
                    "warning",
                    f"{path}.{status_field}",
                    f"enabled service declares non-public status {status!r}; keep it out of normal admission until qualified",
                )
        if (
            service.get("enabled", True) is True
            and service.get("systemd_unit")
            and service.get("type") in {"llm_backend", "generation_backend"}
            and not service.get("runtime_profile")
        ):
            add(
                "runtime_profile_missing",
                "warning",
                path,
                "active engine service has no immutable runtime_profile identity",
            )
        profile_name = service.get("runtime_profile")
        if isinstance(profile_name, str) and profile_name:
            referenced_profiles.add(profile_name)
        if profile_name and runtime_profiles and profile_name not in runtime_profiles:
            add(
                "runtime_profile_missing_manifest",
                "error",
                f"{path}.runtime_profile",
                f"profile {profile_name!r} is not defined in runtime_profiles",
            )
        workflow_ref = service.get("workflow_ref")
        if isinstance(workflow_ref, str) and workflow_ref:
            referenced_workflows.add(workflow_ref)
        service_pipeline_id = service.get("service_pipeline_id")
        if isinstance(service_pipeline_id, str) and service_pipeline_id:
            referenced_pipelines.add(service_pipeline_id)
        provider_id = service.get("provider")
        if isinstance(provider_id, str) and provider_id:
            referenced_providers.add(provider_id)
        for field, target in (
            ("operation_catalog_ref", referenced_catalogs),
            ("source_lock_ref", referenced_source_locks),
        ):
            reference = service.get(field)
            if isinstance(reference, str) and reference:
                target.add(reference)
        inline_workflow = service.get("workflow")
        if isinstance(inline_workflow, Mapping):
            workflow_id = inline_workflow.get("id")
            if isinstance(workflow_id, str) and workflow_id:
                referenced_workflows.add(workflow_id)
            for stage in inline_workflow.get("stages", []):
                if isinstance(stage, Mapping):
                    stage_profile = stage.get("runtime_profile")
                    if isinstance(stage_profile, str) and stage_profile:
                        referenced_profiles.add(stage_profile)
        port = service.get("port")
        if isinstance(port, int) and not isinstance(port, bool) and port > 0:
            ports.setdefault(port, []).append(str(name))
        unit = service.get("systemd_unit")
        if isinstance(unit, str) and unit:
            units.setdefault(unit, []).append(str(name))
        group = service.get("routing_group")
        if isinstance(group, str) and group:
            referenced_groups.add(group)

    for port, names in sorted(ports.items()):
        if len(names) > 1:
            add("duplicate_port", "warning", "services", f"port {port} is shared by {sorted(names)}")
    for unit, names in sorted(units.items()):
        if len(names) > 1:
            add("duplicate_systemd_unit", "warning", "services", f"unit {unit} is shared by {sorted(names)}")

    for name, template in templates.items():
        if not isinstance(template, Mapping):
            continue
        path = f"generation_templates.{name}"
        backend = template.get("backend")
        if isinstance(backend, str) and backend:
            referenced_services.add(backend)
            if services and backend not in services:
                add("missing_template_backend", "error", f"{path}.backend", f"service {backend!r} is not registered")
        if (
            str(backend or "").strip() == "minimax-music3"
            and str(template.get("workflow_backend") or "").strip().lower() == "comfyui"
        ):
            add(
                "legacy_minimax_comfyui_template",
                "warning",
                path,
                "historical MiniMax ComfyUI leaf is inventory-only; use the native audio.cpp MiniMax Music 3 workflow",
            )
        group = template.get("routing_group")
        if isinstance(group, str) and group:
            referenced_groups.add(group)
            if groups and group not in groups:
                add("missing_routing_group", "warning", f"{path}.routing_group", f"routing group {group!r} is not defined")
        workflow_ref = template.get("workflow_ref")
        if isinstance(workflow_ref, str) and workflow_ref:
            referenced_workflows.add(workflow_ref)
        service_pipeline_id = template.get("service_pipeline_id")
        if isinstance(service_pipeline_id, str) and service_pipeline_id:
            referenced_pipelines.add(service_pipeline_id)
        provider_id = template.get("provider")
        if isinstance(provider_id, str) and provider_id:
            referenced_providers.add(provider_id)
        template_status = declared_activation_status(template)
        if template_status:
            status_field, status = template_status
            if status.lower() in {"qualification_only", "deprecated", "retired"}:
                add("inactive_template", "info", f"{path}.{status_field}", f"{status_field}={status}")
            elif status.lower() not in public_statuses:
                add(
                    "unqualified_template",
                    "warning",
                    f"{path}.{status_field}",
                    f"template declares non-public status {status!r}; keep it out of normal admission",
                )

    # Keep referenced-but-not-yet-public control-plane records visible as
    # inventory too.  They are intentionally not removed or made inactive by
    # this read-only audit; activation blockers remain the source of truth.
    for section_name, records, inactive_code, unqualified_code in (
        ("runtime_profiles", runtime_profiles, "inactive_runtime_profile", "unqualified_runtime_profile"),
        ("model_sets", model_sets, "inactive_model_set", "unqualified_model_set"),
        ("workflows", workflows, "inactive_workflow", "unqualified_workflow"),
        ("operation_catalogs", operation_catalogs, "inactive_operation_catalog", "unqualified_operation_catalog"),
        ("source_locks", source_locks, "inactive_source_lock", "unqualified_source_lock"),
    ):
        for record_name, record in records.items():
            if not isinstance(record, Mapping):
                continue
            declared_status = declared_activation_status(record)
            if not declared_status:
                continue
            status_field, status = declared_status
            normalized = status.lower()
            if normalized in public_statuses:
                continue
            retired = normalized in retired_statuses
            add(
                inactive_code if retired else unqualified_code,
                "info" if retired else "warning",
                f"{section_name}.{record_name}.{status_field}",
                f"record declares non-public status {status!r}; keep it out of normal admission",
            )

    for workflow in workflows.values():
        if not isinstance(workflow, Mapping):
            continue
        for field, target in (
            ("operation_catalog_ref", referenced_catalogs),
            ("source_lock_ref", referenced_source_locks),
        ):
            reference = workflow.get(field)
            if isinstance(reference, str) and reference:
                target.add(reference)
        for stage in workflow.get("stages", []):
            if not isinstance(stage, Mapping):
                continue
            provider_id = stage.get("provider")
            if isinstance(provider_id, str) and provider_id:
                referenced_providers.add(provider_id)
            stage_profile = stage.get("runtime_profile")
            if isinstance(stage_profile, str) and stage_profile:
                referenced_profiles.add(stage_profile)

    # Canonical service_pipelines are the effective runtime graph when present;
    # retain the legacy ``pipelines`` section only as an explicit inventory so
    # a stale duplicate cannot disappear from the audit.
    overlap = sorted(set(service_pipelines) & set(legacy_pipelines))
    if overlap:
        add(
            "duplicate_pipeline_registry",
            "warning",
            "pipelines",
            f"pipeline IDs are present in both service_pipelines and pipelines: {overlap}",
        )
    for registry_name, pipeline_registry in (
        ("service_pipelines", service_pipelines),
        ("pipelines", legacy_pipelines),
    ):
        for pipeline_id, pipeline in pipeline_registry.items():
            if not isinstance(pipeline, Mapping):
                continue
            for field, target in (
                ("operation_catalog_ref", referenced_catalogs),
                ("source_lock_ref", referenced_source_locks),
            ):
                reference = pipeline.get(field)
                if isinstance(reference, str) and reference:
                    target.add(reference)
            for stage in pipeline.get("stages", []):
                if isinstance(stage, Mapping):
                    provider_id = stage.get("provider")
                    if isinstance(provider_id, str) and provider_id:
                        referenced_providers.add(provider_id)

    for provider_id, provider in pipeline_providers.items():
        if not isinstance(provider, Mapping):
            continue
        if provider.get("enabled") is False:
            add(
                "inactive_provider",
                "info",
                f"pipeline_providers.{provider_id}.enabled",
                "provider is disabled",
            )
    for provider_id in sorted(set(pipeline_providers) - referenced_providers):
        add(
            "orphaned_pipeline_provider",
            "info",
            f"pipeline_providers.{provider_id}",
            "provider is not referenced by a workflow, service pipeline, template, or service",
        )

    for pipeline_id in sorted(set(service_pipelines) - referenced_pipelines):
        add(
            "orphaned_service_pipeline",
            "info",
            f"service_pipelines.{pipeline_id}",
            "pipeline is not referenced by a registered service or template",
        )
    for pipeline_id in sorted(set(legacy_pipelines) - set(service_pipelines)):
        add(
            "legacy_pipeline_registry",
            "warning",
            f"pipelines.{pipeline_id}",
            "pipeline exists only in the legacy registry section",
        )
    for catalog_id in sorted(set(operation_catalogs) - referenced_catalogs):
        add(
            "orphaned_operation_catalog",
            "info",
            f"operation_catalogs.{catalog_id}",
            "operation catalog is not referenced by a service or workflow",
        )
    for lock_id in sorted(set(source_locks) - referenced_source_locks):
        add(
            "orphaned_source_lock",
            "info",
            f"source_locks.{lock_id}",
            "source lock is not referenced by a service or workflow",
        )

    for group in sorted(set(groups) - referenced_groups):
        add("orphaned_routing_group", "info", f"routing_groups.{group}", "group is not referenced by a service or generation template")

    for profile in sorted(set(runtime_profiles) - referenced_profiles):
        add(
            "orphaned_runtime_profile",
            "info",
            f"runtime_profiles.{profile}",
            "profile is not referenced by a registered service or inline workflow",
        )
    for profile_name, profile in runtime_profiles.items():
        if not isinstance(profile, Mapping):
            continue
        model_set_name = profile.get("model_set")
        if isinstance(model_set_name, str) and model_set_name:
            referenced_model_sets.add(model_set_name)
            if model_sets and model_set_name not in model_sets:
                add(
                    "model_set_missing_manifest",
                    "error",
                    f"runtime_profiles.{profile_name}.model_set",
                    f"model set {model_set_name!r} is not defined in model_sets",
                )
    for model_set_name in sorted(set(model_sets) - referenced_model_sets):
        add(
            "orphaned_model_set",
            "info",
            f"model_sets.{model_set_name}",
            "model set is not referenced by a runtime profile",
        )
    for workflow in sorted(set(workflows) - referenced_workflows):
        add(
            "orphaned_workflow",
            "info",
            f"workflows.{workflow}",
            "workflow is not referenced by a registered service",
        )

    return sorted(findings, key=lambda item: (item["severity"], item["code"], item["path"]))


def registry_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Return changed JSON paths; values are intentionally not exposed."""

    changes: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b), key=str):
                child = f"{path}.{key}" if path else str(key)
                if key not in a or key not in b:
                    changes.append(child)
                else:
                    walk(a[key], b[key], child)
            return
        if a != b:
            changes.append(path or "$")

    walk(left, right, "")
    return changes


_DEMAND_STATES = ("queued", "claimed", "loading", "running", "accepted", "in_flight")


def demand_counts(records: Iterable[Any]) -> dict[str, int]:
    """Count durable job states, including claimed/loading work."""

    counts = Counter()
    for record in records:
        status = record.get("status") if isinstance(record, Mapping) else getattr(record, "status", None)
        status = getattr(status, "value", status)
        if isinstance(status, bytes):
            status = status.decode("utf-8", errors="replace")
        if status:
            counts[str(status)] += 1
    result = {state: int(counts.get(state, 0)) for state in _DEMAND_STATES}
    result["active_demand"] = sum(result[state] for state in _DEMAND_STATES)
    result["terminal"] = sum(counts.get(state, 0) for state in ("completed", "failed", "cancelled", "outcome_unknown"))
    return result


def resources_conflict(left: Iterable[str], right: Iterable[str]) -> bool:
    """Whether two leases share any canonical resource identity."""

    return bool(set(str(item) for item in left) & set(str(item) for item in right))


def service_qualification_blockers(
    registry: Mapping[str, Any], service: Mapping[str, Any]
) -> list[str]:
    """Return the shared, fail-closed activation gate for one service.

    Offline package tooling and the live controller must use the same answer;
    otherwise a package can be advertised as activation-ready and then be
    rejected by the controller (or, worse, omit a qualification layer).
    """

    blockers: list[str] = []
    profiles = registry.get("runtime_profiles") if isinstance(registry, Mapping) else None
    model_sets = registry.get("model_sets") if isinstance(registry, Mapping) else None

    def check_model_set(profile_name: str, profile: Mapping[str, Any], context: str) -> None:
        # Older registries remain readable, but a service using a checked
        # profile cannot be activated without its component identity.
        if not isinstance(model_sets, Mapping):
            blockers.append(f"{context}_model_sets_registry_missing:{profile_name}")
            return
        model_set_name = profile.get("model_set")
        if not isinstance(model_set_name, str) or not model_set_name:
            blockers.append(f"{context}_model_set_missing:{profile_name}")
            return
        model_set = model_sets.get(model_set_name)
        if not isinstance(model_set, Mapping):
            blockers.append(
                f"{context}_model_set_not_resolved:{profile_name}:{model_set_name}"
            )
            return
        if validate_model_set(model_set_name, model_set):
            blockers.append(
                f"{context}_model_set_invalid:{profile_name}:{model_set_name}"
            )
        status = model_set.get("status")
        if status != "qualified":
            blockers.append(
                f"{context}_model_set_not_qualified:{profile_name}:{model_set_name}:{status or 'missing'}"
            )
        if profile.get("model_set_revision") != model_set.get("revision") or profile.get(
            "model_set_fingerprint"
        ) != model_set_fingerprint(model_set):
            blockers.append(
                f"{context}_model_set_binding_stale:{profile_name}:{model_set_name}"
            )
    references = (
        ("workflow_ref", "workflow", "workflows", "workflow_ref_not_resolved"),
        (
            "operation_catalog_ref",
            "operation_catalog",
            "operation_catalogs",
            "operation_catalog_ref_not_resolved",
        ),
        ("source_lock_ref", "source_lock", "source_locks", "source_lock_ref_not_resolved"),
    )
    resolved: dict[str, Mapping[str, Any]] = {}
    for field, inline_field, section_name, missing_code in references:
        reference = service.get(field)
        inline = service.get(inline_field)
        if isinstance(inline, Mapping):
            resolved[field] = inline
            continue
        if not isinstance(reference, str) or not reference:
            continue
        section = registry.get(section_name) if isinstance(registry, Mapping) else None
        document = section.get(reference) if isinstance(section, Mapping) else None
        if not isinstance(document, Mapping):
            blockers.append(missing_code)
        else:
            resolved[field] = document

    metadata = service.get("metadata")
    service_status = (
        metadata.get("activation_status") if isinstance(metadata, Mapping) else None
    )
    if service_status not in {"qualified", "ready", "active"}:
        blockers.append(f"service_not_qualified:{service_status or 'missing'}")

    workflow = resolved.get("workflow_ref")
    if workflow is not None:
        status = workflow.get("status")
        if status not in {"qualified", "active", "production-qualified"}:
            blockers.append(
                f"workflow_not_qualified:{service.get('workflow_ref') or 'inline'}:{status or 'missing'}"
            )
        registry_services = registry.get("services")
        for stage in workflow.get("stages", []):
            if not isinstance(stage, Mapping) or stage.get("enabled_by_default") is False:
                continue
            stage_id = str(stage.get("id") or "unknown")
            stage_service_name = stage.get("service")
            if isinstance(stage_service_name, str) and stage_service_name:
                stage_service = (
                    registry_services.get(stage_service_name)
                    if isinstance(registry_services, Mapping)
                    else None
                )
                if not isinstance(stage_service, Mapping):
                    blockers.append(
                        f"workflow_stage_service_missing:{stage_id}:{stage_service_name}"
                    )
                elif stage_service.get("enabled") is not True:
                    blockers.append(
                        f"workflow_stage_service_not_enabled:{stage_id}:{stage_service_name}"
                    )
            stage_profile_name = stage.get("runtime_profile")
            if isinstance(stage_profile_name, str) and stage_profile_name:
                stage_profile = (
                    profiles.get(stage_profile_name)
                    if isinstance(profiles, Mapping)
                    else None
                )
                if not isinstance(stage_profile, Mapping):
                    blockers.append(
                        f"workflow_stage_profile_missing:{stage_id}:{stage_profile_name}"
                    )
                    continue
                stage_profile_metadata = stage_profile.get("metadata")
                stage_profile_status = (
                    stage_profile_metadata.get("activation_status")
                    if isinstance(stage_profile_metadata, Mapping)
                    else None
                )
                if stage_profile_status not in {"qualified", "ready", "active"}:
                    blockers.append(
                        "workflow_stage_profile_not_qualified:"
                        f"{stage_id}:{stage_profile_name}:{stage_profile_status or 'missing'}"
                    )
                check_model_set(
                    stage_profile_name, stage_profile, f"workflow_stage:{stage_id}"
                )

    catalog = resolved.get("operation_catalog_ref")
    if catalog is not None:
        catalog_metadata = catalog.get("metadata")
        status = catalog.get("status")
        if status is None and isinstance(catalog_metadata, Mapping):
            status = catalog_metadata.get("activation_status")
        if status not in {"qualified", "active", "production-qualified"}:
            blockers.append(
                "operation_catalog_not_qualified:"
                f"{service.get('operation_catalog_ref') or 'inline'}:{status or 'missing'}"
            )

    # A graph/catalog pair is not activation-ready merely because its
    # references resolve.  Run the same typed-binding check used by the
    # coordinator bridge so a GUI-created service cannot publish a graph that
    # would later lose an input, provider or output at runtime.  This remains
    # an offline check: no provider, process or filesystem operation occurs.
    if workflow is not None and catalog is not None:
        try:
            from workflow_capabilities import validate_workflow_capabilities

            graph_errors = validate_workflow_capabilities(
                workflow,
                catalog,
                service_definitions=(
                    registry.get("services")
                    if isinstance(registry.get("services"), Mapping)
                    else None
                ),
                runtime_profiles=profiles if isinstance(profiles, Mapping) else None,
            )
        except Exception as exc:
            graph_errors = [f"validator_error:{type(exc).__name__}"]
        if graph_errors:
            blockers.append(
                "workflow_graph_invalid:"
                + ",".join(str(error) for error in graph_errors[:8])
            )

    profile_name = service.get("runtime_profile")
    if isinstance(profile_name, str) and profile_name:
        profile = profiles.get(profile_name) if isinstance(profiles, Mapping) else None
        if not isinstance(profile, Mapping):
            blockers.append("runtime_profile_not_resolved")
        else:
            profile_metadata = profile.get("metadata")
            status = (
                profile_metadata.get("activation_status")
                if isinstance(profile_metadata, Mapping)
                else None
            )
            if status not in {"qualified", "ready", "active"}:
                blockers.append(
                    f"runtime_profile_not_qualified:{profile_name}:{status or 'missing'}"
                )
            check_model_set(profile_name, profile, "runtime_profile")

    source_lock = resolved.get("source_lock_ref")
    if source_lock is not None:
        status = source_lock.get("status")
        if status not in {"qualified", "activated", "active", "production-qualified"}:
            blockers.append(
                "source_lock_not_qualified:"
                f"{service.get('source_lock_ref') or 'inline'}:{status or 'missing'}"
            )
    return sorted(set(blockers))


__all__ = [
    "PriorityClass", "PriorityConvention", "PriorityEnvelope", "priority_envelope",
    "apply_priority_contract", "validate_service_definition", "validate_registry",
    "service_registration_conflicts",
    "checked_runtime_authority_errors",
    "registry_fingerprint", "registry_summary", "registry_diff", "demand_counts",
    "registry_legacy_findings", "resources_conflict", "validate_runtime_profile",
    "validate_runtime_registry", "runtime_profile_summary",
    "service_qualification_blockers",
    "PUBLIC_ACTIVATION_STATUSES", "service_is_publicly_eligible",
    "is_retired_minimax_comfyui_template",
]
