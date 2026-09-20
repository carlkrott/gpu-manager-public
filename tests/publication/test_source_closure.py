from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "release" / "public-files.json"
EXPORT_MANIFEST = ROOT / "release" / "export-manifest.json"
OPTIONAL_ADAPTERS = (
    "minimax_music3_adapter.py",
    "minimax_music3_generation_runner.py",
    "minimax_music3_preparation_contract.py",
    "minimax_custom_callbacks.py",
    "minimax_recovered_callbacks.py",
)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _receipt() -> dict:
    return json.loads(EXPORT_MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_module_path(name: str) -> Path | None:
    module_file = ROOT / "scripts" / f"{name}.py"
    package_init = ROOT / "scripts" / name / "__init__.py"
    if module_file.is_file():
        return module_file
    if package_init.is_file():
        return package_init
    return None


def test_manifest_is_complete_and_disposition_bound() -> None:
    entries = _manifest()["files"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert all(set(entry) == {"path", "disposition"} for entry in entries)

    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink(), entry["path"]

    assert _manifest()["file_count"] == len(entries)

    shipped_python = {
        path.relative_to(ROOT).as_posix()
        for base in (ROOT / "scripts", ROOT / "tests")
        for path in base.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert shipped_python <= set(paths)


def test_generated_receipt_is_exact_byte_bound() -> None:
    entries = _receipt()["files"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert _receipt()["file_count"] == len(entries)

    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink(), entry["path"]
        assert entry["size"] == path.stat().st_size, entry["path"]
        assert entry["sha256"] == _sha256(path), entry["path"]


def test_direct_local_imports_are_manifested() -> None:
    manifest_paths = {entry["path"] for entry in _manifest()["files"]}
    for path in (ROOT / "scripts").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name.split(".")[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                local = _local_module_path(name)
                if local is not None:
                    assert local.relative_to(ROOT).as_posix() in manifest_paths


def test_core_module_origins_are_candidate_only() -> None:
    modules = (
        "api_auth",
        "declarative_workflow_registration",
        "execution_boundary",
        "pipeline_provider_runtime",
        "sanitize_registry",
    )
    code = (
        "import importlib\n"
        f"mods = {modules!r}\n"
        "for name in mods:\n"
        "    module = importlib.import_module(name)\n"
        "    print(name + '=' + (module.__file__ or ''))\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(ROOT / "scripts"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        name, origin = line.split("=", 1)
        assert name in modules
        assert Path(origin).resolve().is_relative_to((ROOT / "scripts").resolve())


def test_optional_adapters_have_no_installation_root_defaults() -> None:
    for name in OPTIONAL_ADAPTERS:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert not re.search(r"/(?:home|mnt|opt|Users)/", text)
        assert "Path.home()" not in text


def test_optional_adapters_import_without_runtime_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OPTIONAL_ADAPTERS:
        module_name = name[:-3]
        monkeypatch.delenv("GPU_MANAGER_MINIMAX_SOURCE_ROOT", raising=False)
        monkeypatch.delenv("GPU_MANAGER_MINIMAX_AUDIO_CPP_ROOT", raising=False)
        monkeypatch.delenv("GPU_MANAGER_MINIMAX_MUSIC3_SOURCE_ROOT", raising=False)
        monkeypatch.delenv("GPU_MANAGER_MINIMAX_MUSIC3_EVIDENCE_ROOT", raising=False)
        monkeypatch.delenv("GPU_MANAGER_MINIMAX_AUDIO_CPP_RUNNER", raising=False)
        module = importlib.import_module(module_name)
        assert module.__file__ is not None


def test_optional_adapters_fail_closed_when_requested_without_overlay() -> None:
    adapter = importlib.import_module("minimax_music3_adapter")
    with pytest.raises(adapter.MiniMaxAdapterError):
        adapter.MiniMaxAdapterConfig.from_environment()

    recovered = importlib.import_module("minimax_recovered_callbacks")
    assert recovered.SOURCE_ROOT is None
    assert recovered.SEAM_ROOT is None
    with pytest.raises(RuntimeError, match="MINIMAX_SOURCE_ROOT_REQUIRED"):
        recovered._sources()


def test_optional_pure_contracts_accept_synthetic_inputs(tmp_path: Path) -> None:
    preparation = importlib.import_module("minimax_music3_preparation_contract")
    plan = {
        "queries": [
            {"component": "key", "query": "identify the harmonic key", "desired_attributes": ["tonality"], "audio_use": "descriptive"},
            {"component": "instrument", "query": "describe guitar register performance", "desired_attributes": ["register"], "audio_use": "descriptive"},
            {"component": "mixing", "query": "describe stereo depth and balance", "desired_attributes": ["panning"], "audio_use": "descriptive"},
            {"component": "mastering", "query": "describe loudness and true peak", "desired_attributes": ["headroom"], "audio_use": "descriptive"},
        ]
    }
    assert preparation.validate_query_plan(plan)

    callbacks = importlib.import_module("minimax_custom_callbacks")
    config = {
        "pipeline_providers": {
            "synthetic-qc": {"managed": "external", "adapter": "local_audio_qc"}
        }
    }
    catalogs = {
        "synthetic": {
            "operations": {
                "qc": {"adapter": "local_audio_qc", "provider_refs": ["synthetic-qc"]}
            }
        }
    }
    _, _, unsupported = callbacks.build_minimax_custom_callbacks(config, operation_catalogs=catalogs)
    assert unsupported
    policy = callbacks.MiniMaxCustomCallbackPolicy(audio_root=tmp_path)
    handlers, _, unsupported = callbacks.build_minimax_custom_callbacks(
        config,
        operation_catalogs=catalogs,
        policy=policy,
        qc_runner=lambda **_: {"status": "pass"},
    )
    assert ("synthetic-qc", "local_audio_qc") in handlers
    assert not unsupported
