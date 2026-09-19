#!/usr/bin/env python3
"""Export an exact, source-only GPUManager publication allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
from typing import Any

MAX_FILE_BYTES = 5 * 1024 * 1024
SCHEMA = "gpumanager.public-files.v1"
EXPORT_SCHEMA = "gpumanager.exported-files.v1"
_ALLOWED_DISPOSITIONS = frozenset(
    {"core", "portable_optional_integration", "installation_derived_rewrite", "utility"}
)
_BINARY_SUFFIXES = frozenset(
    {
        ".gguf", ".ggml", ".bin", ".pt", ".ckpt", ".safetensors", ".onnx",
        ".h5", ".pkl", ".joblib", ".db", ".sqlite", ".sqlite3", ".aof",
        ".so", ".pyd", ".dll", ".exe", ".whl", ".zip", ".tar", ".tgz",
        ".gz", ".bz2", ".xz", ".7z", ".rar", ".deb", ".rpm", ".dmg", ".iso",
    }
)
_ARCHIVE_MAGICS = (b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"Rar!", b"!<arch>\n")


class ExportError(ValueError):
    """Raised when an export would violate the publication contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ExportError("path must be a non-empty relative string")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExportError(f"path must be relative without traversal: {value!r}")
    if "\\" in value:
        raise ExportError(f"path must use POSIX separators: {value!r}")
    return path.as_posix()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read manifest: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA:
        raise ExportError(f"manifest schema must be {SCHEMA!r}")
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise ExportError("manifest files must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ExportError("manifest entries must be objects")
        relative = _validate_relative_path(entry.get("path"))
        if relative in seen:
            raise ExportError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        disposition = entry.get("disposition")
        if disposition not in _ALLOWED_DISPOSITIONS:
            raise ExportError(f"unsupported disposition for {relative}: {disposition!r}")
        allow_large = entry.get("allow_large", False)
        if type(allow_large) is not bool:
            raise ExportError(f"allow_large must be boolean for {relative}")
        if allow_large and not isinstance(entry.get("rationale"), str):
            raise ExportError(f"allow_large requires rationale for {relative}")
        result.append({"path": relative, "disposition": disposition, "allow_large": allow_large})
    declared_count = document.get("file_count")
    if declared_count is not None and declared_count != len(result):
        raise ExportError("manifest file_count does not match files")
    return sorted(result, key=lambda item: item["path"])


def _assert_regular_source(root: Path, relative: str) -> Path:
    candidate = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ExportError(f"source path contains symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ExportError(f"source escapes root or is missing: {relative}") from exc
    if not candidate.is_file() or not stat.S_ISREG(candidate.stat().st_mode):
        raise ExportError(f"source is missing or not a regular file: {relative}")
    return candidate


def _assert_public_content(path: Path, entry: dict[str, Any], *, allow_large: bool) -> None:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES and not (allow_large and entry["allow_large"]):
        raise ExportError(f"file exceeds {MAX_FILE_BYTES} byte limit: {entry['path']}")
    if path.suffix.lower() in _BINARY_SUFFIXES:
        raise ExportError(f"binary/model/archive file is not publishable: {entry['path']}")
    with path.open("rb") as handle:
        sample = handle.read(4096)
    if b"\0" in sample or any(sample.startswith(magic) for magic in _ARCHIVE_MAGICS):
        raise ExportError(f"binary/archive content is not publishable: {entry['path']}")



def export_public_source(
    source_root: Path,
    manifest_path: Path,
    destination: Path,
    *,
    allow_large: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve(strict=True)
    manifest_path = Path(manifest_path)
    if destination.exists():
        raise ExportError(f"destination already exists: {destination}")
    if manifest_path.is_symlink():
        raise ExportError("manifest may not be a symlink")
    entries = _load_manifest(manifest_path)
    prepared: list[tuple[dict[str, Any], Path]] = []
    for entry in entries:
        source = _assert_regular_source(source_root, entry["path"])
        _assert_public_content(source, entry, allow_large=allow_large)
        prepared.append((entry, source))

    try:
        destination.mkdir(parents=True, mode=0o750)
        records: list[dict[str, Any]] = []
        for entry, source in prepared:
            target = destination / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
            target.chmod(0o644)
            records.append({
                "path": entry["path"],
                "sha256": _sha256(target),
                "size": target.stat().st_size,
                "disposition": entry["disposition"],
            })
        export_manifest = {
            "schema_version": EXPORT_SCHEMA,
            "file_count": len(records),
            "source_manifest_sha256": _sha256(manifest_path),
            "files": records,
        }
        export_path = destination / "release" / "export-manifest.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(export_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        export_path.chmod(0o644)
        return {
            "schema_version": EXPORT_SCHEMA,
            "destination": str(destination),
            "file_count": len(records),
            "manifest_sha256": _sha256(export_path),
        }
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)
    source_root = args.source_root or Path(__file__).resolve().parents[1]
    manifest = args.manifest or source_root / "release/public-files.json"
    try:
        result = export_public_source(source_root, manifest, args.destination, allow_large=args.allow_large)
    except (OSError, ExportError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "exported", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
