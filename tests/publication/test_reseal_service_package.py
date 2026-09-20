from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "reseal_service_package.py"
sys.path.insert(0, str(ROOT / "scripts"))
from reseal_service_package import ResealError, check_manifest, reseal_manifest  # noqa: E402


def _manifest(root: Path, *, revision: int = 1) -> Path:
    payload = b"print('one')\n"
    source = root / "scripts" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    manifest = root / "package.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "example.package.v1",
                "revision": revision,
                "name": "example",
                "files": [
                    {
                        "path": "scripts/example.py",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "role": "source",
                    }
                ],
                "metadata": {"keep": True},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_check_accepts_manifest_and_reports_clean_files(tmp_path: Path):
    manifest = _manifest(tmp_path)

    result = check_manifest(tmp_path, manifest)

    assert result["ok"] is True
    assert result["changed"] == []
    assert result["file_count"] == 1


def test_reseal_updates_only_hashes_in_new_manifest(tmp_path: Path):
    manifest = _manifest(tmp_path)
    original = json.loads(manifest.read_text(encoding="utf-8"))
    (tmp_path / "scripts/example.py").write_bytes(b"print('two')\n")
    output = tmp_path / "sealed" / "package.json"

    result = reseal_manifest(tmp_path, manifest, output, revision=2)

    sealed = json.loads(output.read_text(encoding="utf-8"))
    assert result["changed"] == ["scripts/example.py"]
    assert sealed["revision"] == 2
    assert sealed["metadata"] == original["metadata"]
    assert sealed["files"][0]["role"] == original["files"][0]["role"]
    assert sealed["files"][0]["sha256"] == hashlib.sha256(b"print('two')\n").hexdigest()
    assert json.loads(manifest.read_text(encoding="utf-8")) == original


def test_reseal_requires_revision_bump_for_changed_bytes(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "scripts/example.py").write_bytes(b"changed\n")

    with pytest.raises(ResealError, match="revision"):
        reseal_manifest(tmp_path, manifest, tmp_path / "out.json")

    with pytest.raises(ResealError, match="revision"):
        reseal_manifest(tmp_path, manifest, tmp_path / "out.json", revision=1)
    assert not (tmp_path / "out.json").exists()


def test_rejects_absolute_parent_and_symlink_escape_paths(tmp_path: Path):
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for path in ("/etc/passwd", "../outside.py", "scripts/../outside.py"):
        data["files"][0]["path"] = path
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ResealError, match="relative|traversal"):
            check_manifest(tmp_path, manifest)

    data["files"][0]["path"] = "escape.py"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "outside.py").write_text("private", encoding="utf-8")
    (tmp_path / "escape.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ResealError, match="symlink"):
        check_manifest(tmp_path, manifest)


def test_rejects_missing_duplicate_and_existing_output(tmp_path: Path):
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"][0]["path"] = "missing.py"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ResealError, match="missing"):
        check_manifest(tmp_path, manifest)

    manifest = _manifest(tmp_path / "duplicate")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"].append(dict(data["files"][0]))
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ResealError, match="duplicate"):
        check_manifest(manifest.parent, manifest)

    manifest = _manifest(tmp_path / "existing")
    output = manifest.parent / "out.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ResealError, match="exists"):
        reseal_manifest(manifest.parent, manifest, output)
    assert output.read_text(encoding="utf-8") == "keep"


def test_cli_check_and_reseal_are_json_and_fail_closed(tmp_path: Path):
    manifest = _manifest(tmp_path)
    check = subprocess.run(
        [sys.executable, str(SCRIPT), "check", str(tmp_path), str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0
    assert json.loads(check.stdout)["ok"] is True

    (tmp_path / "scripts/example.py").write_bytes(b"cli change\n")
    output = tmp_path / "out" / "package.json"
    reseal = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "reseal",
            str(tmp_path),
            str(manifest),
            str(output),
            "--revision",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reseal.returncode == 0
    assert json.loads(reseal.stdout)["changed"] == ["scripts/example.py"]
    assert output.is_file()


def test_cli_requires_explicit_source_root():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
