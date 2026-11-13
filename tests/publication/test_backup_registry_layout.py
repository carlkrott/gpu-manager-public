"""Service-registry resolution for the scoped backup capture.

Since r139 the registry lives outside the release, beside the runtime-state
document, so a release that no longer ships ``config/services.json`` must fall
back to that sibling copy instead of crashing the scheduled backup.

The capture path is exercised against fixture directories only. Redis is
stubbed at the client constructor: no Redis connection is opened and no
command is issued by these tests.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import backup_gpu_manager_state as backup  # noqa: E402


RELEASE_REGISTRY = json.dumps({"services": {"legacy": {"name": "release-copy"}}}).encode()
SIBLING_REGISTRY = json.dumps({"services": {"r139": {"name": "sibling-copy"}}}).encode()


class _StubRedis:
    """Stand-in for ``redis.Redis`` that never opens a connection."""

    instances: list["_StubRedis"] = []

    def __init__(self, **options):
        self.options = options
        self.evaluated = False
        _StubRedis.instances.append(self)

    def eval(self, _script, _numkeys, *_args):
        self.evaluated = True
        return []


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Redis client so the capture never opens a connection."""
    _StubRedis.instances = []
    monkeypatch.setattr(backup.redis, "Redis", _StubRedis)


def _layout(
    tmp_path: Path,
    *,
    release_registry: bytes | None = None,
    sibling_registry: bytes | None = None,
) -> argparse.Namespace:
    """Build a fixture deployment and the matching capture arguments."""
    release = tmp_path / "release"
    (release / "scripts").mkdir(parents=True)
    runtime_state = tmp_path / "home" / ".config" / "gpu-manager" / "runtime-state.json"
    runtime_state.parent.mkdir(parents=True)
    runtime_state.write_text(json.dumps({"schema": "gpu-manager-runtime-state.v1"}))
    host_fences = tmp_path / "fences.json"
    host_fences.write_text(json.dumps({"schema": "gpu-manager-host-fences.v1"}))
    native_tasks = tmp_path / "native-tasks"
    native_tasks.mkdir()

    if release_registry is not None:
        (release / "config").mkdir()
        (release / "config" / "services.json").write_bytes(release_registry)
    if sibling_registry is not None:
        (runtime_state.parent / "services.json").write_bytes(sibling_registry)

    return argparse.Namespace(
        destination=tmp_path / "backups",
        release=release,
        runtime_state=runtime_state,
        host_fences=host_fences,
        native_tasks=native_tasks,
    )


def _captured(snapshot: Path) -> dict[str, bytes]:
    document = json.loads(gzip.decompress(snapshot.read_bytes()))
    return {label: base64.b64decode(item["data"]) for label, item in document["files"].items()}


def _registry_source(snapshot: Path) -> str:
    document = json.loads(gzip.decompress(snapshot.read_bytes()))
    return document["registry_source"]


def test_registry_falls_back_to_sibling_when_release_has_none(tmp_path: Path) -> None:
    """r139 layout: the release ships no config/services.json at all."""
    args = _layout(tmp_path, sibling_registry=SIBLING_REGISTRY)

    snapshot = backup.capture(args)

    captured = _captured(snapshot)
    assert captured["registry.json"] == SIBLING_REGISTRY
    assert _StubRedis.instances and _StubRedis.instances[0].evaluated
    assert _registry_source(snapshot) == str(
        args.runtime_state.resolve().parent / "services.json"
    )


def test_release_registry_still_wins_when_both_copies_exist(tmp_path: Path) -> None:
    """The old in-release layout is unchanged even once the sibling exists."""
    args = _layout(tmp_path, release_registry=RELEASE_REGISTRY, sibling_registry=SIBLING_REGISTRY)

    snapshot = backup.capture(args)

    assert _captured(snapshot)["registry.json"] == RELEASE_REGISTRY


def test_release_config_without_registry_still_falls_back(tmp_path: Path) -> None:
    """A release that keeps config/ for other files still has no registry."""
    args = _layout(tmp_path, sibling_registry=SIBLING_REGISTRY)
    args.release.joinpath("config").mkdir()
    args.release.joinpath("config", "gpu-manager.json").write_text("{}")

    snapshot = backup.capture(args)

    assert _captured(snapshot)["registry.json"] == SIBLING_REGISTRY


def test_missing_registry_names_both_candidate_paths(tmp_path: Path) -> None:
    """No registry anywhere must fail loudly, naming both paths."""
    args = _layout(tmp_path)
    release_path = args.release.resolve() / "config/services.json"
    sibling_path = args.runtime_state.resolve().parent / "services.json"

    with pytest.raises(FileNotFoundError) as failure:
        backup.capture(args)

    message = str(failure.value)
    assert str(release_path) in message
    assert str(sibling_path) in message
    assert not list(args.destination.glob("*")) if args.destination.exists() else True