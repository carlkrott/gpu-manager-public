"""Pure contracts for reviewed runtime profiles and observed instances.

Runtime profiles describe an immutable engine/build/model contract.  They do
not contain launch commands and they do not claim that a process is healthy.
The controller or a reviewed adapter supplies the observed instance evidence
at transition time.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any


RUNTIME_PROFILE_SCHEMA_VERSION = "runtime-profile.v1"
MODEL_SET_SCHEMA_VERSION = "model-set.v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATES = {"stopped", "loading", "ready", "draining", "faulted"}
_MODEL_SET_STATUSES = {"draft", "qualified", "retired"}
_MODEL_SET_KINDS = {"model-runtime", "control-plane"}
_DISALLOWED_FIELDS = {
    "command",
    "exec",
    "shell",
    "privileged",
    "host_mounts",
    "docker_socket",
    "host_network",
}
_MODEL_SET_DISALLOWED_FIELDS = _DISALLOWED_FIELDS | {
    "path",
    "url",
    "endpoint",
    "environment",
    "env",
    "secret",
    "credentials",
    "mounts",
}


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def _dangerous_model_set_fields(value: Any, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in _MODEL_SET_DISALLOWED_FIELDS:
                errors.append(f"{child_path} is not allowed in a model set")
            errors.extend(_dangerous_model_set_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_dangerous_model_set_fields(child, f"{prefix}[{index}]"))
    return errors


def validate_model_set(name: str, model_set: Mapping[str, Any]) -> list[str]:
    """Validate a portable model/component identity without host launch data."""

    errors: list[str] = []
    if not isinstance(name, str) or not _ID_RE.fullmatch(name):
        errors.append("name must contain only letters, digits, '.', '_' or '-'")
    if not isinstance(model_set, Mapping):
        return errors + ["model set must be an object"]
    if model_set.get("schema_version") != MODEL_SET_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MODEL_SET_SCHEMA_VERSION!r}")
    if model_set.get("id") != name:
        errors.append("id must match the registry model-set name")
    revision = model_set.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append("revision must be a positive integer")
    status = model_set.get("status")
    if status not in _MODEL_SET_STATUSES:
        errors.append(f"status must be one of {sorted(_MODEL_SET_STATUSES)}")
    kind = model_set.get("kind")
    if kind not in _MODEL_SET_KINDS:
        errors.append(f"kind must be one of {sorted(_MODEL_SET_KINDS)}")

    components = model_set.get("components")
    if not isinstance(components, list):
        errors.append("components must be a list")
        components = []
    elif kind == "model-runtime" and not components:
        errors.append("model-runtime sets must declare at least one component")
    seen: set[str] = set()
    component_roles: set[str] = set()
    component_devices: set[str] = set()
    for index, component in enumerate(components):
        prefix = f"components[{index}]"
        if not isinstance(component, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        component_id = component.get("id")
        if not isinstance(component_id, str) or not _ID_RE.fullmatch(component_id):
            errors.append(f"{prefix}.id must be a safe identifier")
        elif component_id in seen:
            errors.append(f"{prefix}.id is duplicated")
        else:
            seen.add(component_id)
        for field in ("role", "artifact", "precision", "device"):
            value = component.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
            elif status == "qualified" and value.strip().lower().startswith(
                ("unresolved", "unknown")
            ):
                errors.append(
                    f"{prefix}.{field} may not be unresolved in a qualified model set"
                )
        if isinstance(component.get("role"), str) and component["role"].strip():
            component_roles.add(component["role"])
        if isinstance(component.get("device"), str) and component["device"].strip():
            component_devices.add(component["device"])
        if "compatible_with" in component and not _string_list(component["compatible_with"]):
            errors.append(f"{prefix}.compatible_with must be a list of component IDs")
        digest = component.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
        ):
            errors.append(f"{prefix}.sha256 must be null or lowercase SHA-256")
        if status == "qualified" and digest is None:
            errors.append(f"{prefix}.sha256 is required for a qualified model set")

    budgets = model_set.get("memory_budget_mib")
    if not isinstance(budgets, Mapping):
        errors.append("memory_budget_mib must be an object")
    else:
        if kind == "model-runtime" and not budgets:
            errors.append("model-runtime sets must declare memory budgets")
        for device, amount in budgets.items():
            if not isinstance(device, str) or not device.strip():
                errors.append("memory_budget_mib keys must be non-empty strings")
            if amount is None and status != "qualified":
                continue
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                errors.append(
                    f"memory_budget_mib.{device} must be a non-negative integer"
                    + (" for a qualified model set" if status == "qualified" else " or null")
                )
            elif status == "qualified" and kind == "model-runtime" and amount == 0:
                errors.append(
                    f"memory_budget_mib.{device} must be positive for a qualified model runtime"
                )
        for device in sorted(component_devices - set(str(key) for key in budgets)):
            errors.append(
                f"memory_budget_mib must declare component device {device!r}"
            )

    compatibility = model_set.get("compatibility")
    if not isinstance(compatibility, Mapping):
        errors.append("compatibility must be an object")
    else:
        for field in ("engines", "requires_roles", "accepts", "produces"):
            if field in compatibility and not _string_list(compatibility[field]):
                errors.append(f"compatibility.{field} must be a list of non-empty strings")
        required_roles = compatibility.get("requires_roles")
        if _string_list(required_roles):
            for role in sorted(set(required_roles) - component_roles):
                errors.append(
                    f"compatibility.requires_roles references undeclared role {role!r}"
                )
        if status == "qualified" and kind == "model-runtime":
            for field in ("engines", "requires_roles", "accepts", "produces"):
                if not compatibility.get(field):
                    errors.append(
                        f"compatibility.{field} is required for a qualified model runtime"
                    )
    errors.extend(_dangerous_model_set_fields(model_set))
    return errors


def model_set_fingerprint(model_set: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        model_set, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_set_summary(name: str, model_set: Mapping[str, Any]) -> dict[str, Any]:
    components = model_set.get("components", []) if isinstance(model_set, Mapping) else []
    component_maps = [item for item in components if isinstance(item, Mapping)]
    return {
        "id": name,
        "schema_version": model_set.get("schema_version") if isinstance(model_set, Mapping) else None,
        "revision": model_set.get("revision") if isinstance(model_set, Mapping) else None,
        "status": model_set.get("status") if isinstance(model_set, Mapping) else None,
        "kind": model_set.get("kind") if isinstance(model_set, Mapping) else None,
        "component_count": len(component_maps),
        "components_with_hashes": sum(
            1 for component in component_maps if component.get("sha256") is not None
        ),
        "memory_budget_mib": dict(model_set.get("memory_budget_mib", {}))
        if isinstance(model_set, Mapping) and isinstance(model_set.get("memory_budget_mib"), Mapping)
        else {},
        "fingerprint": model_set_fingerprint(model_set),
        "validation_errors": validate_model_set(name, model_set),
    }


def validate_runtime_profile(name: str, profile: Mapping[str, Any]) -> list[str]:
    """Return structural errors for one declarative runtime profile."""

    errors: list[str] = []
    if not isinstance(name, str) or not _ID_RE.fullmatch(name):
        errors.append("name must contain only letters, digits, '.', '_' or '-'")
    if not isinstance(profile, Mapping):
        return errors + ["profile must be an object"]
    if profile.get("schema_version") != RUNTIME_PROFILE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RUNTIME_PROFILE_SCHEMA_VERSION!r}")
    revision = profile.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append("revision must be a positive integer")
    for field in ("engine", "adapter"):
        value = profile.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    process = profile.get("process", {})
    if process is not None and not isinstance(process, Mapping):
        errors.append("process must be an object")
    elif isinstance(process, Mapping):
        if "unit" in process and process["unit"] is not None and not isinstance(process["unit"], str):
            errors.append("process.unit must be a string")
        port = process.get("port", 0)
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            errors.append("process.port must be an integer between 0 and 65535")
        if "health_path" in process and not isinstance(process["health_path"], str):
            errors.append("process.health_path must be a string")

    model_set = profile.get("model_set")
    if model_set is not None and not isinstance(model_set, (str, Mapping)):
        errors.append("model_set must be a string or object")
    if "model_set_revision" in profile:
        revision = profile["model_set_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            errors.append("model_set_revision must be a positive integer")
    if "model_set_fingerprint" in profile and not _SHA256_RE.fullmatch(
        str(profile["model_set_fingerprint"])
    ):
        errors.append("model_set_fingerprint must be a lowercase SHA-256")
    if "capabilities" in profile and not _string_list(profile["capabilities"]):
        errors.append("capabilities must be a list of non-empty strings")
    if "resource_requirements" in profile and not isinstance(
        profile["resource_requirements"], Mapping
    ):
        errors.append("resource_requirements must be an object")
    if "identity_fields" in profile and not _string_list(profile["identity_fields"]):
        errors.append("identity_fields must be a list of non-empty strings")
    for field in _DISALLOWED_FIELDS:
        if field in profile:
            errors.append(f"{field} is not allowed in a runtime profile")
    return errors


def validate_runtime_registry(registry: Mapping[str, Any]) -> list[str]:
    """Validate optional runtime profiles and their first-class model sets."""

    if not isinstance(registry, Mapping):
        return ["registry must be an object"]
    if "runtime_profiles" not in registry and "model_sets" not in registry:
        # Existing registries may carry the string field before profile
        # manifests are introduced. Keep that compatibility path explicit;
        # once the section exists, references are checked strictly.
        return []
    profiles = registry.get("runtime_profiles", {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, Mapping):
        return ["runtime_profiles must be an object"]
    errors: list[str] = []
    for name, profile in profiles.items():
        errors.extend(
            f"runtime_profiles.{name}: {error}"
            for error in validate_runtime_profile(str(name), profile)
        )
    has_model_sets = "model_sets" in registry
    model_sets = registry.get("model_sets", {})
    if model_sets is None:
        model_sets = {}
    if not isinstance(model_sets, Mapping):
        return errors + ["model_sets must be an object"]
    for name, model_set in model_sets.items():
        errors.extend(
            f"model_sets.{name}: {error}"
            for error in validate_model_set(str(name), model_set)
        )
    if has_model_sets:
        for profile_name, profile in profiles.items():
            if not isinstance(profile, Mapping):
                continue
            model_set_name = profile.get("model_set")
            if isinstance(model_set_name, str) and model_set_name not in model_sets:
                errors.append(
                    f"runtime_profiles.{profile_name}.model_set references unknown model set {model_set_name!r}"
                )
                continue
            if isinstance(model_set_name, str):
                model_set = model_sets.get(model_set_name)
                if not isinstance(model_set, Mapping):
                    continue
                if profile.get("model_set_revision") != model_set.get("revision"):
                    errors.append(
                        f"runtime_profiles.{profile_name}.model_set_revision does not match {model_set_name!r}"
                    )
                if profile.get("model_set_fingerprint") != model_set_fingerprint(model_set):
                    errors.append(
                        f"runtime_profiles.{profile_name}.model_set_fingerprint does not match {model_set_name!r}"
                    )
    services = registry.get("services", {})
    if isinstance(services, Mapping):
        for service_name, service in services.items():
            if not isinstance(service, Mapping):
                continue
            profile_name = service.get("runtime_profile")
            if profile_name and profile_name not in profiles:
                errors.append(
                    f"services.{service_name}.runtime_profile references unknown profile {profile_name!r}"
                )
    return errors


def runtime_profile_fingerprint(profile: Mapping[str, Any]) -> str:
    """Return a stable identity for a profile revision."""

    encoded = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_profile_summary(name: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return safe metadata for health and registry diagnostics."""

    process = profile.get("process", {}) if isinstance(profile, Mapping) else {}
    model_set = profile.get("model_set") if isinstance(profile, Mapping) else None
    return {
        "name": name,
        "schema_version": profile.get("schema_version") if isinstance(profile, Mapping) else None,
        "revision": profile.get("revision") if isinstance(profile, Mapping) else None,
        "engine": profile.get("engine") if isinstance(profile, Mapping) else None,
        "adapter": profile.get("adapter") if isinstance(profile, Mapping) else None,
        # A string model-set ID is safe to expose; manifests may contain
        # private paths or hashes and are intentionally summarized, not copied.
        "model_set": model_set if isinstance(model_set, str) else None,
        "model_set_revision": profile.get("model_set_revision")
        if isinstance(profile, Mapping)
        else None,
        "model_set_fingerprint": profile.get("model_set_fingerprint")
        if isinstance(profile, Mapping)
        else None,
        "model_set_declared": model_set is not None,
        "process_port": process.get("port") if isinstance(process, Mapping) else None,
        "capabilities": list(profile.get("capabilities", []))
        if isinstance(profile, Mapping) and isinstance(profile.get("capabilities"), list)
        else [],
        "fingerprint": runtime_profile_fingerprint(profile),
        "validation_errors": validate_runtime_profile(name, profile),
    }


def validate_runtime_observation(
    profile_name: str,
    profile: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> list[str]:
    """Check adapter evidence against the selected profile revision.

    This is intentionally fail-closed for identity fields, but it does not
    infer readiness from an HTTP status or from VRAM alone.
    """

    errors = validate_runtime_profile(profile_name, profile)
    if not isinstance(observation, Mapping):
        return errors + ["observation must be an object"]
    if observation.get("profile_name") != profile_name:
        errors.append("observation.profile_name does not match selected profile")
    if observation.get("profile_fingerprint") != runtime_profile_fingerprint(profile):
        errors.append("observation.profile_fingerprint does not match selected profile")
    if observation.get("state") not in _STATES:
        errors.append(f"observation.state must be one of {sorted(_STATES)}")
    process_generation = observation.get("process_generation")
    if isinstance(process_generation, bool) or not isinstance(process_generation, int) or process_generation < 1:
        errors.append("observation.process_generation must be a positive integer")
    for field in ("process_ready", "model_ready", "accepting"):
        if not isinstance(observation.get(field), bool):
            errors.append(f"observation.{field} must be boolean")
    if observation.get("state") == "ready" and not (
        observation.get("process_ready") and observation.get("model_ready")
    ):
        errors.append("ready observation requires process_ready and model_ready")
    expected_model_set = profile.get("model_set")
    if expected_model_set is not None and observation.get("model_set") != expected_model_set:
        errors.append("observation.model_set does not match selected profile")
    return errors


__all__ = [
    "RUNTIME_PROFILE_SCHEMA_VERSION",
    "MODEL_SET_SCHEMA_VERSION",
    "validate_model_set",
    "model_set_fingerprint",
    "model_set_summary",
    "validate_runtime_profile",
    "validate_runtime_registry",
    "runtime_profile_fingerprint",
    "runtime_profile_summary",
    "validate_runtime_observation",
]
