"""Load and compile a declarative GPU Manager service package.

This is a source-only staging/GUI helper.  Given an explicit repository root
and service-definition path it resolves only safe relative manifest references,
loads the matching workflow/catalog/profile files, and returns a bounded
validation summary.  It never follows network URLs, executes fields, starts a
service, or writes the active registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from gpu_manager_contracts import service_qualification_blockers, validate_service_definition
from runtime_contracts import (
    model_set_summary,
    validate_model_set,
    validate_runtime_profile,
    validate_runtime_registry,
)
from workflow_capabilities import validate_workflow_capabilities, workflow_capability_summary
from workflow_contracts import validate_workflow_definition, workflow_summary
from workflow_pipeline_bridge import WorkflowCompilationError, compile_workflow_pipeline


PACKAGE_SCHEMA = "service-package-preview.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PackageLoadError(ValueError):
    """A package path or manifest could not be resolved safely."""


def _safe_manifest_path(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise PackageLoadError("manifest reference must be a non-empty string")
    normalized = reference.strip().replace("\\", "/")
    if normalized.startswith(("/", "~")) or "\x00" in normalized or "://" in normalized:
        raise PackageLoadError("manifest reference must be a relative local path")
    if any(part == ".." for part in normalized.split("/")):
        raise PackageLoadError("manifest reference may not escape the package root")
    package_root = root.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(package_root)
    except ValueError as exc:
        raise PackageLoadError("manifest reference escapes the package root") from exc
    if not candidate.is_file():
        raise PackageLoadError(f"manifest does not exist: {normalized}")
    return candidate


def _read_json(root: Path, reference: str) -> tuple[Path, Any]:
    path = _safe_manifest_path(root, reference)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PackageLoadError(f"unable to read manifest: {reference}") from exc
    except json.JSONDecodeError as exc:
        raise PackageLoadError(f"manifest is not valid JSON: {reference}") from exc


def _load_service_files(root: Path) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    services: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    directory = root / "config" / "service-definitions"
    if not directory.is_dir():
        return services, ["service-definitions directory is missing"]
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: unable to load service manifest ({exc})")
            continue
        if not isinstance(raw, Mapping):
            errors.append(f"{path.name}: service manifest must be an object")
            continue
        candidates = (
            raw.get("services")
            if isinstance(raw.get("services"), Mapping)
            else {raw.get("name"): raw}
        )
        for name, service in candidates.items():
            if not isinstance(name, str) or not isinstance(service, Mapping):
                errors.append(f"{path.name}: service manifest must contain named objects")
                continue
            definition_errors = validate_service_definition(name, service)
            errors.extend(f"{path.name}: {error}" for error in definition_errors)
            if not definition_errors:
                services[name] = service
    return services, errors


def _load_profiles(root: Path) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    profiles: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    directory = root / "config" / "runtime-profiles"
    if not directory.is_dir():
        return profiles, ["runtime-profiles directory is missing"]
    for path in sorted(directory.glob("*.runtime-profile.json")):
        name = path.name.removesuffix(".runtime-profile.json")
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: unable to load runtime profile ({exc})")
            continue
        profile_errors = validate_runtime_profile(name, profile)
        errors.extend(f"{path.name}: {error}" for error in profile_errors)
        if not profile_errors:
            profiles[name] = profile
    return profiles, errors


def _load_model_sets(root: Path) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    model_sets: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    directory = root / "config" / "model-sets"
    if not directory.is_dir():
        return model_sets, ["model-sets directory is missing"]
    for path in sorted(directory.glob("*.model-set.json")):
        name = path.name.removesuffix(".model-set.json")
        try:
            model_set = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: unable to load model set ({exc})")
            continue
        model_set_errors = validate_model_set(name, model_set)
        errors.extend(f"{path.name}: {error}" for error in model_set_errors)
        if not model_set_errors:
            model_sets[name] = model_set
    return model_sets, errors


def _source_lock_summary(lock: Mapping[str, Any], service_name: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if lock.get("schema_version") != "workflow-source-lock.v1":
        errors.append("source lock schema_version must be 'workflow-source-lock.v1'")
    if lock.get("service") != service_name:
        errors.append("source lock service does not match service definition")
    sources = lock.get("sources", {})
    if not isinstance(sources, Mapping):
        errors.append("source lock sources must be an object")
        sources = {}
    for name, source in sources.items():
        if not isinstance(source, Mapping) or not _SHA256_RE.fullmatch(str(source.get("sha256", ""))):
            errors.append(f"source lock source {name!r} must carry a lowercase SHA-256")
    return {
        "schema_version": lock.get("schema_version"),
        "service": lock.get("service"),
        "revision": lock.get("revision"),
        "status": lock.get("status"),
        "source_count": len(sources),
        "fingerprint": hashlib.sha256(
            json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }, errors


def load_service_package(root: str | Path, service_reference: str) -> dict[str, Any]:
    """Compile a service definition and its local declarative dependencies."""

    package_root = Path(root)
    service_path, raw_service = _read_json(package_root, service_reference)
    errors: list[str] = []
    if not isinstance(raw_service, Mapping):
        raise PackageLoadError("service definition must be an object")
    service_name = raw_service.get("name")
    if not isinstance(service_name, str) or not service_name:
        raise PackageLoadError("service definition must carry a name")
    errors.extend(validate_service_definition(service_name, raw_service))

    all_services, service_errors = _load_service_files(package_root)
    errors.extend(service_errors)
    profiles, profile_errors = _load_profiles(package_root)
    errors.extend(profile_errors)
    model_sets, model_set_errors = _load_model_sets(package_root)
    errors.extend(model_set_errors)
    errors.extend(
        validate_runtime_registry(
            {"runtime_profiles": profiles, "model_sets": model_sets}
        )
    )

    workflow: Mapping[str, Any] | None = None
    catalog: Mapping[str, Any] | None = None
    source_lock: Mapping[str, Any] | None = None
    for field, target in (
        ("workflow_ref", "workflow"),
        ("operation_catalog_ref", "catalog"),
        ("source_lock_ref", "source_lock"),
    ):
        reference = raw_service.get(field)
        if not reference:
            errors.append(f"service is missing {field}")
            continue
        try:
            _, loaded = _read_json(package_root, reference)
        except PackageLoadError as exc:
            errors.append(f"{field}: {exc}")
            continue
        if not isinstance(loaded, Mapping):
            errors.append(f"{field} must resolve to an object")
            continue
        if target == "workflow":
            workflow = loaded
        elif target == "catalog":
            catalog = loaded
        else:
            source_lock = loaded

    workflow_view = None
    capability_view = None
    coordinator_view = None
    source_view = None
    if workflow is not None:
        errors.extend(validate_workflow_definition(workflow))
        workflow_view = workflow_summary(workflow)
    if workflow is not None and catalog is not None:
        errors.extend(
            validate_workflow_capabilities(
                workflow,
                catalog,
                service_definitions=all_services,
                runtime_profiles=profiles,
            )
        )
        capability_view = workflow_capability_summary(workflow, catalog)
        try:
            compiled = compile_workflow_pipeline(
                workflow,
                catalog,
                service_name=service_name,
                service_definitions=all_services,
                runtime_profiles=profiles,
                # Service manifests may not know deployment endpoint names;
                # the preview still proves graph/order/typed contract and
                # leaves provider-to-runtime binding to the host overlay.
            )
            coordinator_view = {
                "schema": compiled.get("schema"),
                "compiler": compiled.get("compiler"),
                "workflow_fingerprint": compiled.get("workflow_fingerprint"),
                "catalog_fingerprint": compiled.get("catalog_fingerprint"),
                "stage_count": len(compiled.get("stages", [])),
                "stage_ids": [stage.get("id") for stage in compiled.get("stages", [])],
                "execution_mode": compiled.get("execution_mode"),
            }
        except WorkflowCompilationError as exc:
            errors.extend(f"coordinator_compile: {error}" for error in exc.errors)
    elif workflow is not None or catalog is not None:
        errors.append("workflow and operation catalog must both resolve")
    if source_lock is not None:
        source_view, source_errors = _source_lock_summary(source_lock, service_name)
        errors.extend(source_errors)

    qualification_registry = {
        "services": all_services,
        "runtime_profiles": profiles,
        "model_sets": model_sets,
        "workflows": (
            {raw_service["workflow_ref"]: workflow}
            if workflow is not None and isinstance(raw_service.get("workflow_ref"), str)
            else {}
        ),
        "operation_catalogs": (
            {raw_service["operation_catalog_ref"]: catalog}
            if catalog is not None
            and isinstance(raw_service.get("operation_catalog_ref"), str)
            else {}
        ),
        "source_locks": (
            {raw_service["source_lock_ref"]: source_lock}
            if source_lock is not None
            and isinstance(raw_service.get("source_lock_ref"), str)
            else {}
        ),
    }
    activation_blockers = service_qualification_blockers(
        qualification_registry, raw_service
    )

    referenced_profiles: set[str] = set()
    referenced_model_sets: set[str] = set()
    service_profile = raw_service.get("runtime_profile")
    if isinstance(service_profile, str) and service_profile:
        referenced_profiles.add(service_profile)
    if isinstance(workflow, Mapping):
        for stage in workflow.get("stages", []):
            if isinstance(stage, Mapping):
                if stage.get("enabled_by_default") is False:
                    continue
                stage_profile = stage.get("runtime_profile")
                if isinstance(stage_profile, str) and stage_profile:
                    referenced_profiles.add(stage_profile)
    for profile_name in sorted(referenced_profiles):
        profile = profiles.get(profile_name)
        if profile is None:
            activation_blockers.append(f"runtime_profile_missing:{profile_name}")
            continue
        metadata = profile.get("metadata", {})
        activation = metadata.get("activation_status") if isinstance(metadata, Mapping) else None
        if activation not in {"qualified", "ready"}:
            activation_blockers.append(
                f"runtime_profile_not_qualified:{profile_name}:{activation or 'unspecified'}"
            )
        model_set_name = profile.get("model_set")
        if isinstance(model_set_name, str) and model_set_name:
            referenced_model_sets.add(model_set_name)
            if model_set_name not in model_sets:
                activation_blockers.append(f"model_set_missing:{model_set_name}")

    definition_ready = not errors

    return {
        "schema_version": PACKAGE_SCHEMA,
        "service": {
            "name": service_name,
            "path": str(service_path.relative_to(package_root.resolve())),
            "enabled": bool(raw_service.get("enabled", False)),
            "runtime_profile": raw_service.get("runtime_profile"),
            "fingerprint": hashlib.sha256(
                json.dumps(raw_service, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        },
        "workflow": workflow_view,
        "capabilities": capability_view,
        "coordinator": coordinator_view,
        "source_lock": source_view,
        "profile_count": len(profiles),
        "model_set_count": len(model_sets),
        "referenced_model_sets": {
            name: model_set_summary(name, model_sets[name])
            for name in sorted(referenced_model_sets)
            if name in model_sets
        },
        "service_manifest_count": len(all_services),
        "validation_errors": sorted(set(errors)),
        "definition_ready": definition_ready,
        "activation_blockers": sorted(set(activation_blockers)),
        "activation_ready": (
            definition_ready
            and not bool(raw_service.get("enabled", False))
            and not activation_blockers
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="package/repository root")
    parser.add_argument("service", help="relative service definition path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = load_service_package(args.root, args.service)
    except PackageLoadError as exc:
        print(json.dumps({"schema_version": PACKAGE_SCHEMA, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["validation_errors"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PACKAGE_SCHEMA", "PackageLoadError", "load_service_package"]
