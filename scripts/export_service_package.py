#!/usr/bin/env python3
"""Export one reviewed GPU Manager service package to a clean directory.

The source manifest is an explicit file allowlist with SHA-256 identities. The
exporter compiles the declarative service before copying anything, sanitizes
JSON documents, refuses symlinks/path traversal, and never writes the live
registry. The destination must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from declarative_package import load_service_package
from sanitize_registry import sanitize_registry


PACKAGE_MANIFEST_SCHEMA = "gpu-manager-service-package.v2"
PACKAGE_EXPORT_SCHEMA = "gpu-manager-service-package-export.v2"
CONTROL_PLANE_CONTRACT = 2
_PRIVATE_TEXT = re.compile(
    rb"/(?:home|mnt|root|opt)/|https?://(?:localhost|127\.0\.0\.1|10\.|192\.168\.|100\.64\.)"
    + rb"|redis" + rb"s?://[^\s/@]*:[^\s/@]+@"
    + rb"|redis" + rb"-cli\b[^\n]*(?:\s-a(?:\s|=)|--pass(?:word)?(?:\s|=))",
    re.IGNORECASE,
)


class PackageExportError(RuntimeError):
    """A package could not be exported without weakening its contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise PackageExportError("package file path must be a non-empty string")
    normalized = reference.strip().replace("\\", "/")
    if normalized.startswith(("/", "~")) or "://" in normalized:
        raise PackageExportError(f"package file path must be relative: {reference!r}")
    unresolved = root / normalized
    if unresolved.is_symlink():
        raise PackageExportError(f"package file may not be a symlink: {reference}")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PackageExportError(f"package file escapes source root: {reference!r}") from exc
    if not candidate.is_file():
        raise PackageExportError(f"package file is missing or not a regular file: {reference}")
    return candidate


def _load_manifest(root: Path, reference: str) -> tuple[Path, Mapping[str, Any]]:
    path = _safe_source(root, reference)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageExportError(f"cannot load package manifest: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PackageExportError("package manifest must be an object")
    if value.get("schema_version") != PACKAGE_MANIFEST_SCHEMA:
        raise PackageExportError(f"unsupported package manifest schema: {value.get('schema_version')!r}")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise PackageExportError("package manifest name is required")
    if not isinstance(value.get("service_ref"), str):
        raise PackageExportError("package manifest service_ref is required")
    minimum_contract = value.get("minimum_control_plane_contract")
    if not isinstance(minimum_contract, int) or isinstance(minimum_contract, bool):
        raise PackageExportError("minimum_control_plane_contract must be an integer")
    if minimum_contract > CONTROL_PLANE_CONTRACT:
        raise PackageExportError(
            "package requires a newer control-plane contract: "
            f"{minimum_contract} > {CONTROL_PLANE_CONTRACT}"
        )
    if not isinstance(value.get("files"), list) or not value["files"]:
        raise PackageExportError("package manifest files must be a non-empty list")
    return path, value


def _render_file(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix.lower() != ".json":
        if _PRIVATE_TEXT.search(raw):
            raise PackageExportError(
                f"non-JSON package file contains a private path, endpoint, or credential transport: {path.name}"
            )
        return raw
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageExportError(f"invalid JSON package file: {path.name}") from exc
    return (json.dumps(sanitize_registry(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def export_service_package(
    root: str | Path,
    manifest_reference: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Validate and export one service package into a new directory."""

    package_root = Path(root).resolve()
    manifest_path, manifest = _load_manifest(package_root, manifest_reference)
    destination_path = Path(destination)
    if destination_path.exists():
        raise PackageExportError("destination already exists; refusing to merge or overwrite")

    compile_result = load_service_package(package_root, manifest["service_ref"])
    if compile_result.get("validation_errors"):
        raise PackageExportError(
            "service package does not compile: "
            + "; ".join(str(item) for item in compile_result["validation_errors"])
        )
    if compile_result.get("service", {}).get("enabled") is not False:
        raise PackageExportError("portable exports must contain a disabled service definition")

    selected: list[tuple[str, Path, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(manifest["files"]):
        if not isinstance(item, Mapping):
            raise PackageExportError(f"files[{index}] must be an object")
        reference = item.get("path")
        expected = item.get("sha256")
        role = item.get("role")
        if not isinstance(reference, str) or reference in seen:
            raise PackageExportError(f"files[{index}] has a missing or duplicate path")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise PackageExportError(f"files[{index}] must carry a SHA-256")
        if not isinstance(role, str) or not role:
            raise PackageExportError(f"files[{index}] must carry a role")
        source = _safe_source(package_root, reference)
        if _sha256_file(source) != expected:
            raise PackageExportError(f"package source hash mismatch: {reference}")
        selected.append((reference, source, role))
        seen.add(reference)

    required = {
        manifest["service_ref"],
        compile_result["service"]["path"],
    }
    if isinstance(manifest.get("provenance_ref"), str) and manifest["provenance_ref"]:
        required.add(manifest["provenance_ref"])
    missing = sorted(required - seen)
    if missing:
        raise PackageExportError("package manifest omits required service file: " + ", ".join(missing))

    destination_path.mkdir(parents=True, exist_ok=False)
    exported_files: list[dict[str, str]] = []
    try:
        for reference, source, role in selected:
            rendered = _render_file(source)
            target = destination_path / reference
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rendered)
            exported_files.append(
                {
                    "path": reference,
                    "role": role,
                    "source_sha256": _sha256_file(source),
                    "export_sha256": _sha256_bytes(rendered),
                }
            )

        exported_compile = load_service_package(
            destination_path, manifest["service_ref"]
        )
        if exported_compile.get("validation_errors"):
            raise PackageExportError(
                "sanitized export does not compile: "
                + "; ".join(
                    str(item) for item in exported_compile["validation_errors"]
                )
            )
        export_manifest = {
            "schema_version": PACKAGE_EXPORT_SCHEMA,
            "minimum_control_plane_contract": manifest[
                "minimum_control_plane_contract"
            ],
            "name": manifest["name"],
            "service": exported_compile["service"],
            "provenance_ref": manifest.get("provenance_ref"),
            "source_manifest": str(manifest_path.relative_to(package_root)),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "requirements": sanitize_registry(
                {
                    key: manifest[key]
                    for key in (
                        "required_adapter_ids",
                        "required_private_overlays",
                        "required_registry_contracts",
                    )
                    if key in manifest
                }
            ),
            "files": exported_files,
            "portable": True,
            "live_registry_included": False,
            "runtime_state_included": False,
            "credential_material_policy": (
                "excluded; sanitizer found no declared credential fields"
            ),
            "activation_required": True,
        }
        export_manifest_path = destination_path / "package-export.json"
        export_manifest_path.write_text(
            json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(destination_path, ignore_errors=True)
        raise

    return {
        "schema_version": PACKAGE_EXPORT_SCHEMA,
        "name": manifest["name"],
        "destination": str(destination_path.resolve()),
        "file_count": len(exported_files),
        "manifest_sha256": _sha256_file(export_manifest_path),
        "service_fingerprint": exported_compile["service"]["fingerprint"],
        "activation_required": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="repository/package root")
    parser.add_argument("manifest", help="relative service-package manifest path")
    parser.add_argument("output", type=Path, help="new export directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = export_service_package(args.root, args.manifest, args.output)
    except PackageExportError as exc:
        print(json.dumps({"schema_version": PACKAGE_EXPORT_SCHEMA, "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PACKAGE_EXPORT_SCHEMA",
    "PACKAGE_MANIFEST_SCHEMA",
    "CONTROL_PLANE_CONTRACT",
    "PackageExportError",
    "export_service_package",
]
