"""Capacity summary excludes integrated graphics while device identity stays visible."""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


def _deny(*args, **kwargs):
    raise AssertionError("process creation forbidden in fake-inventory tests")


@pytest.fixture
def facade(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    (tmp_path / "home").mkdir()
    (tmp_path / "xdg").mkdir()
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    spec = importlib.util.spec_from_file_location("gpu_capacity_facade", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_GPU_INVENTORY = [
    {"card": "card0", "gpu_id": "rx6950", "gpu_type": "dGPU", "status": "observed",
     "vram_total_mib": 16368, "vram_used_mib": 8000},
    {"card": "card1", "gpu_id": "mi50", "gpu_type": "dGPU", "status": "observed",
     "vram_total_mib": 16368, "vram_used_mib": 4000},
    {"card": "card2", "gpu_id": "igpu", "gpu_type": "iGPU", "status": "observed",
     "vram_total_mib": 0, "vram_used_mib": None},
]


def test_capacity_excludes_igpu_but_identity_inventory_keeps_it(facade):
    capacity = facade._gpu_capacity_snapshot({}, FAKE_GPU_INVENTORY, 123.0)
    assert capacity["gpu_count"] == 2
    assert [gpu["gpu_id"] for gpu in capacity["gpus"]] == ["rx6950", "mi50"]
    assert [gpu["capacity_state"] for gpu in capacity["gpus"]] == ["measured", "measured"]
    configured_capacity = facade._gpu_capacity_snapshot(
        {"gpu_devices": {
            "rx6950": {"card": "card0"},
            "mi50": {"card": "card1"},
            "igpu": {"card": "card2"},
        }},
        FAKE_GPU_INVENTORY,
        123.0,
    )
    assert configured_capacity["gpu_count"] == 2
    assert [gpu["gpu_id"] for gpu in configured_capacity["gpus"]] == ["rx6950", "mi50"]
    # The identity/list endpoint continues to receive all observed devices.
    assert [gpu["gpu_id"] for gpu in FAKE_GPU_INVENTORY] == ["rx6950", "mi50", "igpu"]
    assert FAKE_GPU_INVENTORY[-1]["gpu_type"] == "iGPU"
