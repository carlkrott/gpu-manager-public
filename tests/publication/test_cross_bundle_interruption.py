"""Regression coverage for linked-bundle ownership and ComfyUI interruption."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden in this regression")


@pytest.fixture
def facade(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    real_socket = socket.socket
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    spec = importlib.util.spec_from_file_location("cross_bundle_facade", SCRIPTS / "gpu-manager.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(socket, "socket", real_socket)
    return module


def test_rollback_preserves_another_active_child_bundle(facade, monkeypatch):
    bundles = {
        "parent": {"gpu_id": "gpu-parent", "linked_to": ["child"]},
        "child": {"gpu_id": "gpu-child", "linked_from": ["parent"]},
    }
    states = {
        "gpu-parent": type("FakeState", (), {"worker_in_flight": 0})(),
        "gpu-child": type("FakeState", (), {"worker_in_flight": 1})(),
    }
    stopped = []

    monkeypatch.setattr(facade, "_get_gpu_state", lambda gpu_id: states.get(gpu_id))

    async def fake_unload(name):
        stopped.append(name)
        return True

    monkeypatch.setattr(facade, "_unload_bundle", fake_unload)
    report = asyncio.run(facade._rollback_bundle_graph("parent", bundles=bundles))

    assert report["ok"] is True
    assert report["attempted"] == ["parent"]
    assert report["preserved"] == ["child"]
    assert stopped == ["parent"]


def test_execution_interrupted_is_terminal_comfyui_status(facade):
    # ComfyUI reports interrupted prompts as status messages, not completed=True.
    message = {
        "prompt_id": "prompt-123",
        "status": {
            "completed": False,
            "messages": [["execution_interrupted", {"node_id": "17"}]],
        },
    }

    assert facade._comfyui_execution_interrupted(message) is True
    assert facade._comfyui_execution_interrupted(
        {"status": {"completed": False, "messages": []}}
    ) is False
