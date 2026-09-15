#!/usr/bin/env python3
"""Fail-closed privacy checks for a source-only public payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from sanitize_registry import sanitize_registry


MAX_FILE_BYTES = 5 * 1024 * 1024
_BINARY_SUFFIXES = frozenset(
    {".gguf", ".bin", ".pt", ".ckpt", ".safetensors", ".onnx", ".pkl", ".db", ".sqlite", ".so", ".dll", ".exe", ".whl", ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
)
_ARCHIVE_MAGICS = (b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"Rar!", b"!<arch>\n")
_PRIVATE_REPORT_NAME = re.compile(
    r"(?:^|[-_.])(?:audit|gitleaks-summary|credential-candidate|tracked-text-snapshot|existing-export)(?:[-_.]|$)",
    re.IGNORECASE,
)
_ABSOLUTE_PROVENANCE = re.compile(
    r"(?<![A-Za-z0-9_])(?:/home/[^\s\"']+|/mnt/[^\s\"']+|/opt/[^\s\"']+|/Users/[^\s\"']+|[A-Za-z]:[\\/][^\s\"']+)",
)
_CREDENTIAL_URL = re.compile(
    r"https?://[^\s\"'<>:/]+:[^\s\"'<>@]+@|"
    r"https?://[^\s\"'<>]+[?#](?:[^\s\"'<>]*(?:pass(?:word)?|token|secret|api[_-]?key|credential|auth)[^\s\"'<>]*)",
    re.IGNORECASE,
)
_PEM = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?(?:PRIVATE KEY|CERTIFICATE)-----")


def _record(violations: list[dict[str, str]], path: str, rule: str, detail: str) -> None:
    violations.append({"path": path, "rule": rule, "detail": detail})


def _manifest_paths(root: Path, manifest: Path | None) -> list[Path]:
    if manifest is None:
        return sorted(
            path for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    entries = document.get("files") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError("manifest files must be a list")
    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("manifest entries must contain path strings")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manifest path is not relative: {entry['path']!r}")
        paths.append(root / relative)
    return paths


def _check_structured(path: Path, relative: str, violations: list[dict[str, str]]) -> None:
    if path.suffix.lower() != ".json" or relative == "release/public-files.json":
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _record(violations, relative, "invalid_json", str(exc))
        return
    if sanitize_registry(value) != value:
        _record(
            violations,
            relative,
            "sanitizer_would_change_payload",
            "structured payload contains a private path, endpoint, or secret-looking value",
        )


def check_public_payload(root: Path, *, manifest: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    violations: list[dict[str, str]] = []
    try:
        paths = _manifest_paths(root, manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "file_count": 0, "violations": [{"path": str(manifest or root), "rule": "invalid_manifest", "detail": str(exc)}]}

    checked = 0
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            _record(violations, str(path), "path_outside_root", "manifest path escapes payload root")
            continue
        checked += 1
        if not path.is_file():
            _record(violations, relative, "missing_file", "manifested payload file is missing")
            continue
        if _PRIVATE_REPORT_NAME.search(path.name):
            _record(violations, relative, "private_report_filename", "private audit/report filename is not publishable")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            _record(violations, relative, "size_limit", f"file exceeds {MAX_FILE_BYTES} bytes")
            continue
        sample = path.read_bytes()[:4096]
        if path.suffix.lower() in _BINARY_SUFFIXES or b"\0" in sample or any(sample.startswith(magic) for magic in _ARCHIVE_MAGICS):
            _record(violations, relative, "binary_or_archive", "binary/model/archive payload is not publishable")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _record(violations, relative, "non_utf8_payload", "public source payload must be UTF-8 text")
            continue
        if relative.startswith("tests/"):
            continue
        if relative != "scripts/check_public_payload.py":
            for match in _ABSOLUTE_PROVENANCE.finditer(text):
                _record(violations, relative, "absolute_provenance_forbidden", match.group(0)[:160])
            for match in _CREDENTIAL_URL.finditer(text):
                _record(violations, relative, "credential_url_forbidden", match.group(0)[:160])
            for match in _PEM.finditer(text):
                _record(violations, relative, "key_or_certificate_forbidden", match.group(0))
        _check_structured(path, relative, violations)

    return {"ok": not violations, "file_count": checked, "violations": violations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    report = check_public_payload(args.root, manifest=args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MAX_FILE_BYTES", "check_public_payload", "main"]
