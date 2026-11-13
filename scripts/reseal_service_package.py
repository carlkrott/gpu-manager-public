#!/usr/bin/env python3
"""Generic service-package manifest check and reseal helper.

Reads a service package manifest containing a `files` array, validates all paths
relative to an explicitly supplied source root, verifies against traversal and
symlink escapes, computes SHA-256 digests, and updates only file hashes in a new
manifest while requiring an explicit revision bump if source bytes have changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
from typing import Any, Mapping


class ResealError(RuntimeError):
    """Raised when package manifest validation or reseal fails."""


PackageResealError = ResealError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source_path(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ResealError("package file path must be a non-empty string")
    normalized = reference.strip().replace("\\", "/")
    if (
        normalized.startswith(("/", "~"))
        or "://" in normalized
        or "\x00" in normalized
        or Path(normalized).is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or bool(PureWindowsPath(normalized).drive)
    ):
        raise ResealError(f"package file path must be relative: {reference!r}")
    parts = Path(normalized).parts
    if any(part == ".." for part in parts):
        raise ResealError(f"package file path contains '..' traversal: {reference!r}")

    current = root.resolve()
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ResealError(f"package path component is a symlink: {reference!r}")

    resolved = current.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ResealError(f"package path escapes source root: {reference!r}") from exc

    if not resolved.is_file():
        raise ResealError(f"package file is missing or not a regular file: {reference!r}")
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResealError(f"manifest file does not exist: {str(path)!r}")
    try:
        raw_text = path.read_text(encoding="utf-8")
        manifest = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResealError(f"unable to load manifest JSON from {str(path)!r}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ResealError("package manifest must be a JSON object")
    if not isinstance(manifest.get("schema_version"), str) or not manifest["schema_version"]:
        raise ResealError("package manifest schema_version is required")
    if not isinstance(manifest.get("name"), str) or not manifest["name"]:
        raise ResealError("package manifest name is required")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ResealError("package manifest files must be a list")
    if not files:
        raise ResealError("package manifest files list must be non-empty")

    seen_paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ResealError(f"files[{index}] must be a JSON object")
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ResealError(f"files[{index}] missing valid path")
        if rel_path in seen_paths:
            raise ResealError(f"duplicate file path in manifest: {rel_path!r}")
        seen_paths.add(rel_path)
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != hashlib.sha256().digest_size * 2
            or any(character not in "0123456789abcdefABCDEF" for character in sha256)
        ):
            raise ResealError(f"files[{index}] missing valid sha256")

    return manifest


def _write_json_atomically(target_path: Path, data: Mapping[str, Any]) -> None:
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_prefix = f".{target_path.name}."
    descriptor, temp_file = tempfile.mkstemp(prefix=temp_prefix, dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Linking the completed temporary file is create-only: unlike
        # os.replace(), it cannot silently overwrite an output made by a
        # concurrent caller.
        try:
            os.link(temp_file, target_path)
        except FileExistsError as exc:
            raise ResealError(
                f"output manifest already exists; refusing overwrite: {str(target_path)!r}"
            ) from exc
        os.unlink(temp_file)
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        try:
            os.unlink(temp_file)
        except OSError:
            pass
        raise


def check_manifest(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Inspect and verify service package files against source root."""
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise ResealError(f"source root directory does not exist: {str(root)!r}")
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_manifest(manifest_file)

    changed: list[str] = []
    for file_entry in manifest["files"]:
        rel_path = file_entry["path"]
        expected_sha = file_entry["sha256"]
        resolved_file = _safe_source_path(source_root, rel_path)
        actual_sha = _sha256_file(resolved_file)

        if actual_sha != expected_sha.lower():
            changed.append(rel_path)

    return {
        "ok": len(changed) == 0,
        "changed": changed,
        "file_count": len(manifest["files"]),
        "revision": manifest.get("revision"),
    }


check_service_package = check_manifest


def reseal_manifest(
    root: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    revision: int | None = None,
) -> dict[str, Any]:
    """Reseal service package manifest with updated file hashes and revision."""
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise ResealError(f"source root directory does not exist: {str(root)!r}")
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_manifest(manifest_file)
    target_path = Path(output_path).resolve()

    if target_path.exists():
        raise ResealError(f"output path already exists: {str(target_path)!r}")

    actual_hashes: dict[str, str] = {}
    changed: list[str] = []

    for file_entry in manifest["files"]:
        rel_path = file_entry["path"]
        expected_sha = file_entry["sha256"]
        resolved_file = _safe_source_path(source_root, rel_path)
        actual_sha = _sha256_file(resolved_file)
        actual_hashes[rel_path] = actual_sha
        if actual_sha != expected_sha.lower():
            changed.append(rel_path)

    current_revision = manifest.get("revision")

    if changed:
        if revision is None:
            raise ResealError("source bytes changed; explicit revision bump is required")
        if current_revision is not None and revision == current_revision:
            raise ResealError(
                f"new revision {revision!r} must differ from current revision {current_revision!r}"
            )
        if isinstance(current_revision, int) and isinstance(revision, int):
            if revision <= current_revision:
                raise ResealError(
                    f"new revision {revision} must be greater than current revision {current_revision}"
                )
        target_revision: int | str | None = revision
    else:
        target_revision = revision if revision is not None else current_revision

    new_manifest = copy.deepcopy(manifest)
    if target_revision is not None:
        new_manifest["revision"] = target_revision

    for file_entry in new_manifest["files"]:
        rel_path = file_entry["path"]
        file_entry["sha256"] = actual_hashes[rel_path]

    _write_json_atomically(target_path, new_manifest)

    return {
        "ok": True,
        "changed": changed,
        "file_count": len(new_manifest["files"]),
        "revision": target_revision,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check and reseal GPU Manager service package manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify package manifest")
    check_parser.add_argument("root", type=Path, help="source root directory")
    check_parser.add_argument("manifest", type=Path, help="service package manifest path")

    reseal_parser = subparsers.add_parser("reseal", help="Reseal package manifest")
    reseal_parser.add_argument("root", type=Path, help="source root directory")
    reseal_parser.add_argument("manifest", type=Path, help="service package manifest path")
    reseal_parser.add_argument("output", type=Path, help="output manifest path")
    reseal_parser.add_argument(
        "--revision",
        type=int,
        default=None,
        help="explicit new revision number",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            result = check_manifest(args.root, args.manifest)
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
            return 0 if result["ok"] else 1
        elif args.command == "reseal":
            result = reseal_manifest(
                args.root,
                args.manifest,
                args.output,
                revision=args.revision,
            )
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
            return 0
    except ResealError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    return 0


__all__ = [
    "ResealError",
    "PackageResealError",
    "check_manifest",
    "check_service_package",
    "reseal_manifest",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
