from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check_public_payload import check_public_payload  # noqa: E402
from export_public_source import ExportError, export_public_source  # noqa: E402


def _fixture(
    tmp_path: Path,
    *,
    path: str = "scripts/example.py",
    data: bytes = b"print('ok')\n",
    include_public_manifest: bool = False,
):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    target = source / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    files = [{
        "path": path,
        "disposition": "core",
    }]
    if include_public_manifest:
        files.append({
            "path": "release/public-files.json",
            "disposition": "utility",
        })
    manifest = {
        "schema_version": "gpumanager.public-files.v1",
        "file_count": len(files),
        "files": files,
    }
    manifest_path = source / "release" / "public-files.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path


def test_allowlist_without_receipt_fields_generates_and_verifies_receipt(tmp_path):
    source, manifest = _fixture(tmp_path, include_public_manifest=True)
    (source / "scripts/example.py").write_bytes(b"print('changed')\n")
    destination = tmp_path / "export"

    result = export_public_source(source, manifest, destination)

    receipt = json.loads((destination / "release/export-manifest.json").read_text())
    entry = next(item for item in receipt["files"] if item["path"] == "scripts/example.py")
    assert result["file_count"] == 2
    assert set(entry) == {"path", "sha256", "size", "disposition"}
    assert entry["path"] == "scripts/example.py"
    assert entry["size"] == len(b"print('changed')\n")
    assert check_public_payload(destination, manifest=destination / "release/export-manifest.json")["ok"]

    (destination / "scripts/example.py").write_text("tampered\n", encoding="utf-8")
    report = check_public_payload(destination, manifest=destination / "release/export-manifest.json")
    assert not report["ok"]
    assert any(item["rule"] == "receipt_hash_mismatch" for item in report["violations"])


def test_exports_exact_allowlist_and_omits_unlisted_files(tmp_path):
    source, manifest = _fixture(tmp_path)
    (source / "private.txt").write_text("not selected", encoding="utf-8")
    destination = tmp_path / "export"

    result = export_public_source(source, manifest, destination)

    assert result["file_count"] == 1
    assert (destination / "scripts/example.py").read_bytes() == b"print('ok')\n"
    assert not (destination / "private.txt").exists()
    assert (destination / "release/export-manifest.json").is_file()
    assert (destination / "scripts/example.py").stat().st_mode & 0o777 == 0o644


def test_receipt_rejects_extra_regular_files(tmp_path):
    source, manifest = _fixture(tmp_path, include_public_manifest=True)
    destination = tmp_path / "export"
    export_public_source(source, manifest, destination)

    (destination / "unlisted.py").write_text("not in the allowlist\n", encoding="utf-8")
    report = check_public_payload(destination, manifest=destination / "release/export-manifest.json")

    assert not report["ok"]
    assert any(
        item["rule"] == "receipt_unlisted_file" and item["path"] == "unlisted.py"
        for item in report["violations"]
    )


def test_receipt_rejects_public_manifest_with_swapped_disposition(tmp_path):
    source, manifest = _fixture(tmp_path, include_public_manifest=True)
    destination = tmp_path / "export"
    export_public_source(source, manifest, destination)

    public_manifest = destination / "release/public-files.json"
    swapped = json.loads(public_manifest.read_text())
    swapped["files"][0]["disposition"] = "utility"
    public_manifest.write_text(json.dumps(swapped), encoding="utf-8")
    report = check_public_payload(destination, manifest=destination / "release/export-manifest.json")

    assert not report["ok"]
    assert any(item["rule"] == "receipt_manifest_mismatch" for item in report["violations"])
    assert any(item["rule"] == "receipt_manifest_parity_mismatch" for item in report["violations"])


def test_rejects_invalid_paths_before_writing(tmp_path):
    source, manifest = _fixture(tmp_path, path="../outside.py")
    with pytest.raises(ExportError, match="relative|traversal"):
        export_public_source(source, manifest, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_rejects_absolute_paths(tmp_path):
    source, manifest = _fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["files"][0]["path"] = "/etc/passwd"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExportError, match="relative"):
        export_public_source(source, manifest, tmp_path / "export")


def test_rejects_symlink_source(tmp_path):
    source, manifest = _fixture(tmp_path)
    real = source / "scripts/example.py"
    real.unlink()
    (source / "actual.py").write_text("not copied", encoding="utf-8")
    real.symlink_to(source / "actual.py")
    with pytest.raises(ExportError, match="symlink"):
        export_public_source(source, manifest, tmp_path / "export")


def test_rejects_missing_source_and_records_changed_source_bytes(tmp_path):
    source, manifest = _fixture(tmp_path)
    (source / "scripts/example.py").unlink()
    with pytest.raises(ExportError, match="missing"):
        export_public_source(source, manifest, tmp_path / "missing")

    source, manifest = _fixture(tmp_path / "hash")
    (source / "scripts/example.py").write_bytes(b"print('xx')\n")
    destination = tmp_path / "hash-export"
    export_public_source(source, manifest, destination)
    receipt = json.loads((destination / "release/export-manifest.json").read_text())
    assert receipt["files"][0]["size"] == len(b"print('xx')\n")


def test_rejects_binary_and_oversized_files(tmp_path):
    source, manifest = _fixture(tmp_path, path="models/example.gguf", data=b"GGUF\0binary")
    with pytest.raises(ExportError, match="binary"):
        export_public_source(source, manifest, tmp_path / "binary-export")

    source, manifest = _fixture(tmp_path / "large", data=b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(ExportError, match="size|exceeds"):
        export_public_source(source, manifest, tmp_path / "large-export")


def test_rejects_existing_destination_and_duplicate_manifest_paths(tmp_path):
    source, manifest = _fixture(tmp_path)
    destination = tmp_path / "export"
    destination.mkdir()
    with pytest.raises(ExportError, match="exists"):
        export_public_source(source, manifest, destination)

    source, manifest = _fixture(tmp_path / "duplicates")
    data = json.loads(manifest.read_text())
    data["files"].append(dict(data["files"][0]))
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExportError, match="duplicate"):
        export_public_source(source, manifest, tmp_path / "duplicate-export")
