"""Dashboard accuracy: header VRAM without the iGPU, calm banner for a full-by-design card.

Live symptoms (2026-10-10):
- Header said "VRAM 84.5% (30.4 / 36.0 GiB)": an integrated GPU's 4 GiB BIOS
  carve-out (system RAM) was added to two 16 GiB discrete cards.
- A red "CRITICAL: VRAM nearly full" banner showed for a discrete card whose
  configured idle bundle (one llama-server) fills it by design.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


def _deny(*args, **kwargs):
    raise AssertionError("process creation forbidden in parser tests")


@pytest.fixture
def facade(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    spec = importlib.util.spec_from_file_location("vram_glance_facade", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device(card, pci, name, gpu_type, total, used, fdinfo=None):
    return {
        "Info": {
            "PCI": pci,
            "GPU Type": gpu_type,
            "DevicePath": {"card": f"/dev/dri/{card}", "DeviceName": name},
        },
        "VRAM": {
            "Total VRAM": {"unit": "MiB", "value": total},
            "Total VRAM Usage": {"unit": "MiB", "value": used},
            "Total GTT": {"unit": "MiB", "value": 0},
            "Total GTT Usage": {"unit": "MiB", "value": 0},
        },
        "fdinfo": fdinfo or {},
    }


AMDGPU_TOP = {
    "devices": [
        _device("card1", "0000:0b:00.0", "Discrete A", "dGPU", 16368, 14376),
        _device("card2", "0000:0c:00.0", "Integrated", "APU", 4096, 859),
        _device("card0", "0000:0a:00.0", "Discrete B", "dGPU", 16368, 15797),
    ]
}


def test_apu_excluded_from_discrete_totals(facade):
    out = facade._parse_amdgpu_top(json.dumps(AMDGPU_TOP))
    assert out["dgpu_total_vram_mib"] == 16368 * 2
    assert out["dgpu_used_vram_mib"] == 14376 + 15797
    assert out["dgpu_usage_pct"] == round((14376 + 15797) / (16368 * 2) * 100, 1)
    assert out["apu_vram_mib"] == 4096


def test_legacy_all_device_totals_unchanged(facade):
    out = facade._parse_amdgpu_top(json.dumps(AMDGPU_TOP))
    assert out["total_vram_mib"] == 16368 * 2 + 4096
    assert out["used_vram_mib"] == 14376 + 859 + 15797


def test_apu_still_listed_and_flagged(facade):
    out = facade._parse_amdgpu_top(json.dumps(AMDGPU_TOP))
    flags = {d["card"]: d["integrated"] for d in out["devices"]}
    assert flags == {"card1": False, "card2": True, "card0": False}


def test_missing_gpu_type_counts_as_discrete(facade):
    raw = {"devices": [_device("card0", "0000:0a:00.0", "X", None, 8192, 4096)]}
    out = facade._parse_amdgpu_top(json.dumps(raw))
    assert out["dgpu_total_vram_mib"] == 8192
    assert out["apu_vram_mib"] == 0
    assert out["devices"][0]["integrated"] is False


def test_parse_error_shape_has_no_discrete_keys(facade):
    out = facade._parse_amdgpu_top("not json")
    # The page falls back to the legacy totals when the split is absent.
    assert "dgpu_total_vram_mib" not in out


# ── Dashboard JS: _vramBundleHolder decides calm vs alarm ──

def _dashboard_js():
    src = FACADE.read_text()
    html = re.search(r'DASHBOARD_HTML = r"""(.*?)"""', src, re.S).group(1)
    js = max(re.findall(r"<script>(.*?)</script>", html, re.S), key=len)
    parts = []
    for name in ("_findBundleForGpu", "_findBundleByName", "_vramBundleHolder"):
        m = re.search(rf"^function {name}\(.*?^}}\n", js, re.S | re.M)
        assert m, name
        parts.append(m.group(0))
    return "\n".join(parts)


def _holder(state, gpu):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    prog = (
        f"const state = {json.dumps(state)};\n{_dashboard_js()}\n"
        f"process.stdout.write(JSON.stringify(_vramBundleHolder({json.dumps(gpu)})));"
    )
    res = subprocess.run([node, "-e", prog], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


STATE = {
    "gpu_states": {"gpu-b": {"loaded_bundle": "aux-idle"}},
    "bundles": {"aux-idle": {"services": ["aux"], "idle": True}},
    "services": {"services": {"aux": {"port": 8146}}},
}


def test_bundle_accounts_for_full_card():
    gpu = {
        "card": "card0", "gpu_id": "gpu-b", "vram_used_mib": 15797,
        "processes": [
            {"name": "llama-server", "service_port": 8146, "vram_mib": 15783},
            {"name": "Xorg", "service_port": None, "vram_mib": 3},
        ],
    }
    h = _holder(STATE, gpu)
    assert h == {"bundle": "aux-idle", "idle": True, "bundleMib": 15783, "otherMib": 14}


def test_foreign_vram_is_not_explained_away():
    gpu = {
        "card": "card0", "gpu_id": "gpu-b", "vram_used_mib": 15797,
        "processes": [
            {"name": "llama-server", "service_port": 8146, "vram_mib": 9000},
            {"name": "python", "service_port": None, "vram_mib": 6700},
        ],
    }
    h = _holder(STATE, gpu)
    assert h["otherMib"] == 15797 - 9000  # >= 512 MiB: page keeps the alarm


def test_no_bundle_or_no_matching_process_returns_null():
    gpu = {"card": "card1", "gpu_id": "other", "vram_used_mib": 100, "processes": []}
    assert _holder(STATE, gpu) is None
    gpu = {
        "card": "card0", "gpu_id": "gpu-b", "vram_used_mib": 15797,
        "processes": [{"name": "llama-server", "service_port": 9999, "vram_mib": 15783}],
    }
    assert _holder(STATE, gpu) is None
