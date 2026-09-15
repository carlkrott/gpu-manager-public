#!/usr/bin/env python3
"""Stage a verified service-package export into a new registry candidate.

Both inputs and the output are explicit. The command verifies every exported
file hash, refuses undeclared extra files and merge conflicts, recompiles the
service package, and writes a new disabled registry candidate. It never edits
the live registry or activates a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from declarative_package import load_service_package
from gpu_manager_contracts import (
    registry_fingerprint,
    service_qualification_blockers,
    validate_registry,
)


EXPORT_SCHEMA = "gpu-manager-service-package-export.v2"
IMPORT_SCHEMA = "gpu-manager-service-package-import-preview.v2"
CONTROL_PLANE_CONTRACT = 2
_RUNTIME_PROFILE_ROLES = frozenset(
    {
        "runtime-profile",
        "qualification-only-runtime-profile",
        "candidate-runtime-profile",
    }
)
_MODEL_SET_ROLES = frozenset(
    {
        "model-set",
        "qualification-only-model-set",
        "candidate-model-set",
    }
)


class PackageImportError(RuntimeError):
    """An exported package cannot be safely staged."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise PackageImportError("exported file path must be non-empty")
    normalized = reference.strip().replace("\\", "/")
    if normalized.startswith(("/", "~")) or "://" in normalized:
        raise PackageImportError(f"exported file path must be relative: {reference!r}")
    unresolved = root / normalized
    if unresolved.is_symlink():
        raise PackageImportError(f"exported file may not be a symlink: {reference}")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PackageImportError(f"exported file escapes package root: {reference!r}") from exc
    if not path.is_file():
        raise PackageImportError(f"exported file is missing: {reference}")
    return path


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageImportError(f"cannot load JSON document {path.name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PackageImportError(f"JSON document must be an object: {path.name}")
    return value


def _verify_export(
    root: Path,
    expected_manifest_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, tuple[Path, str]]]:
    manifest_path = root / "package-export.json"
    normalized_digest = expected_manifest_sha256.strip().lower()
    if (
        len(normalized_digest) != 64
        or any(character not in "0123456789abcdef" for character in normalized_digest)
    ):
        raise PackageImportError("expected manifest SHA-256 must be 64 lowercase hex characters")
    if _sha256(manifest_path) != normalized_digest:
        raise PackageImportError("package export manifest hash mismatch")
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != EXPORT_SCHEMA:
        raise PackageImportError("unsupported package export schema")
    minimum_contract = manifest.get("minimum_control_plane_contract")
    if not isinstance(minimum_contract, int) or isinstance(minimum_contract, bool):
        raise PackageImportError("package has no valid minimum control-plane contract")
    if minimum_contract > CONTROL_PLANE_CONTRACT:
        raise PackageImportError(
            "package requires a newer control-plane contract: "
            f"{minimum_contract} > {CONTROL_PLANE_CONTRACT}"
        )
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise PackageImportError("package export has no file records")
    declared: dict[str, tuple[Path, str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PackageImportError(f"files[{index}] must be an object")
        reference = record.get("path")
        role = record.get("role")
        expected = record.get("export_sha256")
        if not isinstance(reference, str) or reference in declared:
            raise PackageImportError(f"files[{index}] has a missing or duplicate path")
        if not isinstance(role, str) or not role:
            raise PackageImportError(f"files[{index}] has no role")
        path = _safe_file(root, reference)
        if _sha256(path) != expected:
            raise PackageImportError(f"exported file hash mismatch: {reference}")
        declared[reference] = (path, role)

    actual = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.name != "package-export.json"
    }
    if actual != set(declared):
        extra = sorted(actual - set(declared))
        missing = sorted(set(declared) - actual)
        raise PackageImportError(
            f"package file inventory mismatch; extra={extra}, missing={missing}"
        )
    return manifest, declared


def _merge_named(target: dict[str, Any], additions: Mapping[str, Any], section: str) -> None:
    for name, value in additions.items():
        if name in target and target[name] != value:
            raise PackageImportError(f"registry conflict at {section}.{name}")
        target[name] = value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise PackageImportError("output registry already exists; refusing overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PackageImportError("output registry already exists; refusing overwrite") from exc


def stage_service_package(
    package: str | Path,
    base_registry: str | Path,
    output_registry: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Build a new disabled registry candidate from one verified export."""

    package_root = Path(package).resolve()
    export_manifest, files = _verify_export(
        package_root, expected_manifest_sha256
    )
    service_reference = export_manifest.get("service", {}).get("path")
    if not isinstance(service_reference, str):
        raise PackageImportError("package export has no service path")
    compiled = load_service_package(package_root, service_reference)
    if compiled.get("validation_errors"):
        raise PackageImportError(
            "exported service no longer compiles: "
            + "; ".join(str(item) for item in compiled["validation_errors"])
        )

    base_path = Path(base_registry)
    base = dict(_json(base_path))
    base_errors = validate_registry(base)
    if base_errors:
        raise PackageImportError("base registry is invalid: " + "; ".join(base_errors))
    before_fingerprint = registry_fingerprint(base)

    candidate = json.loads(json.dumps(base))
    for section in (
        "services",
        "runtime_profiles",
        "model_sets",
        "workflows",
        "operation_catalogs",
        "source_locks",
        "routing_groups",
        "bundles",
        "generation_templates",
    ):
        if section not in candidate:
            candidate[section] = {}
        elif not isinstance(candidate[section], dict):
            raise PackageImportError(f"base registry {section} must be an object")

    service_documents: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    model_sets: dict[str, Any] = {}
    workflow_documents: list[Mapping[str, Any]] = []
    catalog_documents: list[Mapping[str, Any]] = []
    source_documents: list[Mapping[str, Any]] = []
    fragments: list[Mapping[str, Any]] = []
    for reference, (path, role) in files.items():
        if role in {"service-definition", "generation-leaf-definition"}:
            document = _json(path)
            name = document.get("name")
            if not isinstance(name, str) or not name:
                raise PackageImportError(f"service file has no name: {reference}")
            if document.get("enabled") is not False:
                raise PackageImportError(f"imported service must be disabled: {name}")
            if name in service_documents:
                raise PackageImportError(f"duplicate imported service definition: {name}")
            service_documents[name] = dict(document)
        elif role in _RUNTIME_PROFILE_ROLES:
            name = path.name.removesuffix(".runtime-profile.json")
            if name in profiles:
                raise PackageImportError(f"duplicate imported runtime profile: {name}")
            profiles[name] = dict(_json(path))
        elif role in _MODEL_SET_ROLES:
            document = _json(path)
            name = path.name.removesuffix(".model-set.json")
            if document.get("id") != name:
                raise PackageImportError(
                    f"model-set id does not match filename: {reference}"
                )
            if name in model_sets:
                raise PackageImportError(f"duplicate imported model set: {name}")
            model_sets[name] = dict(document)
        elif role == "workflow-definition":
            workflow_documents.append(_json(path))
        elif role == "operation-catalog":
            catalog_documents.append(_json(path))
        elif role == "external-source-lock":
            source_documents.append(_json(path))
        elif role == "portable-registry-fragment":
            fragments.append(_json(path))

    main_service = service_documents.get(compiled["service"]["name"])
    if main_service is None:
        raise PackageImportError("main service definition is absent from export")
    if len(workflow_documents) != 1 or len(catalog_documents) != 1 or len(source_documents) != 1:
        raise PackageImportError(
            "package must contain exactly one workflow, operation catalog and source lock"
        )
    for document in workflow_documents:
        reference = main_service.get("workflow_ref")
        if not isinstance(reference, str):
            raise PackageImportError("main service has no workflow_ref")
        _merge_named(candidate["workflows"], {reference: document}, "workflows")
    for document in catalog_documents:
        reference = main_service.get("operation_catalog_ref")
        if not isinstance(reference, str):
            raise PackageImportError("main service has no operation_catalog_ref")
        _merge_named(candidate["operation_catalogs"], {reference: document}, "operation_catalogs")
    for document in source_documents:
        reference = main_service.get("source_lock_ref")
        if not isinstance(reference, str):
            raise PackageImportError("main service has no source_lock_ref")
        _merge_named(candidate["source_locks"], {reference: document}, "source_locks")
    _merge_named(candidate["runtime_profiles"], profiles, "runtime_profiles")
    _merge_named(candidate["model_sets"], model_sets, "model_sets")
    _merge_named(candidate["services"], service_documents, "services")
    for fragment in fragments:
        for section in ("routing_groups", "bundles", "generation_templates"):
            additions = fragment.get(section, {})
            if not isinstance(additions, Mapping):
                raise PackageImportError(f"registry fragment {section} must be an object")
            _merge_named(candidate[section], additions, section)

    candidate_errors = validate_registry(candidate)
    if candidate_errors:
        raise PackageImportError("staged registry is invalid: " + "; ".join(candidate_errors))

    blockers = list(compiled.get("activation_blockers", []))
    blockers.extend(service_qualification_blockers(candidate, main_service))
    requirements = export_manifest.get("requirements", {})
    registry_contracts = (
        requirements.get("required_registry_contracts", {})
        if isinstance(requirements, Mapping)
        else {}
    )
    if isinstance(registry_contracts, Mapping):
        backend = registry_contracts.get("backend")
        if isinstance(backend, str):
            backend_service = candidate["services"].get(backend)
            if not isinstance(backend_service, Mapping):
                blockers.append(f"runtime_backend_missing:{backend}")
            elif backend_service.get("enabled") is not True:
                blockers.append(f"runtime_backend_not_enabled:{backend}")
        backends = registry_contracts.get("backends")
        if isinstance(backends, list):
            for backend_name in backends:
                if not isinstance(backend_name, str):
                    continue
                backend_service = candidate["services"].get(backend_name)
                if not isinstance(backend_service, Mapping):
                    blockers.append(f"runtime_backend_missing:{backend_name}")
                elif backend_service.get("enabled") is not True:
                    blockers.append(f"runtime_backend_not_enabled:{backend_name}")
        bundle = registry_contracts.get("bundle")
        if isinstance(bundle, str) and bundle not in (candidate.get("bundles") or {}):
            blockers.append(f"runtime_bundle_missing:{bundle}")
        bundles = registry_contracts.get("bundles")
        if isinstance(bundles, list):
            for bundle_name in bundles:
                if isinstance(bundle_name, str) and bundle_name not in (candidate.get("bundles") or {}):
                    blockers.append(f"runtime_bundle_missing:{bundle_name}")

    _write_new_json(Path(output_registry), candidate)
    return {
        "schema_version": IMPORT_SCHEMA,
        "package": export_manifest.get("name"),
        "output_registry": str(Path(output_registry).resolve()),
        "base_registry_fingerprint": before_fingerprint,
        "candidate_registry_fingerprint": registry_fingerprint(candidate),
        "services_staged": sorted(service_documents),
        "model_sets_staged": sorted(model_sets),
        "generation_templates_staged": sorted(
            name for fragment in fragments for name in (fragment.get("generation_templates") or {})
        ),
        "definition_ready": bool(compiled.get("definition_ready")),
        "activation_ready": not blockers,
        "activation_blockers": sorted(set(str(item) for item in blockers)),
        "writes_live_registry": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="exported package directory")
    parser.add_argument("base_registry", type=Path, help="explicit base registry JSON")
    parser.add_argument("output_registry", type=Path, help="new staged registry JSON")
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="trusted SHA-256 printed by the export command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = stage_service_package(
            args.package,
            args.base_registry,
            args.output_registry,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    except PackageImportError as exc:
        print(json.dumps({"schema_version": IMPORT_SCHEMA, "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_PLANE_CONTRACT",
    "IMPORT_SCHEMA",
    "PackageImportError",
    "stage_service_package",
]
