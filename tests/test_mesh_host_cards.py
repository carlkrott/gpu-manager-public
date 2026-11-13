"""Dashboard accuracy: VRAM-tab host chips and mesh host cards.

Live symptoms (2026-10-10):
- Host chips read "host-a (local) | host-a | host-a | host-b | host-c":
  two mesh members whose address is on this machine got chips of their own.
- A down member (no answer on its port) showed a red card with a configured
  RAM size as if measured; healthy members without a host agent did the same.
- Cards said "unknown" for the model while the peer reported one.
- System RAM was shown three times in the VRAM tab header.
"""
from __future__ import annotations

import asyncio
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
    spec = importlib.util.spec_from_file_location("mesh_cards_facade", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dashboard_html():
    src = FACADE.read_text()
    return re.search(r'DASHBOARD_HTML = r"""(.*?)"""', src, re.S).group(1)


def _dashboard_js():
    js = max(re.findall(r"<script>(.*?)</script>", _dashboard_html(), re.S), key=len)
    parts = []
    for name in ("_meshEsc", "_meshHostChips", "_meshCardView"):
        m = re.search(rf"^function {name}\(.*?^}}\n", js, re.S | re.M)
        assert m, name
        parts.append(m.group(0))
    return "\n".join(parts)


def _js(expr):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    prog = f"{_dashboard_js()}\nprocess.stdout.write(JSON.stringify({expr}));"
    res = subprocess.run([node, "-e", prog], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


MESH = {
    "local_machine": "host-a",
    "nodes": {
        "svc-big": {"healthy": False, "host": "192.0.2.1", "port": 8101,
                     "machine": "host-a", "gpu": "dGPU 16GB", "model": "unknown",
                     "ram_used_gb": 19.4, "is_local": False, "on_this_host": True,
                     "stats_live": False},
        "svc-cpu": {"healthy": True, "host": "192.0.2.1", "port": 8090,
                         "machine": "host-a", "gpu": "CPU-only", "model": "unknown",
                         "ram_used_gb": 7.3, "is_local": False, "on_this_host": True,
                         "stats_live": False, "model_live": "model-a.gguf"},
        "svc-embed": {"healthy": True, "host": "192.0.2.2", "port": 8080,
                            "machine": "host-b", "gpu": "unknown", "model": "unknown",
                            "vram_live_mib": 466, "vram_live_total_mib": 2048,
                            "ram_used_gb": 2.3, "is_local": False, "on_this_host": False,
                            "stats_live": True},
        "svc-laptop": {"healthy": False, "host": "192.0.2.3", "port": 8086,
                            "machine": "host-c (laptop)", "model": "model-c.gguf",
                            "ram_used_gb": 11, "is_local": False, "on_this_host": False,
                            "stats_live": False},
        "cloud:x": {"healthy": True, "machine": "Cloud", "is_cloud": True},
    },
}


def test_host_chips_one_per_machine_and_this_host_members_fold_into_local():
    chips = _js(f"_meshHostChips({json.dumps(MESH)})")
    assert [c["label"] for c in chips] == ["host-a (local)", "host-b", "host-c (laptop)"]
    assert chips[0]["key"] == "local"


def test_host_chips_dedupe_by_machine_label():
    data = {"local_machine": "a", "nodes": {
        "m1": {"machine": "host-b"}, "m2": {"machine": "host-b"}, "m3": {}}}
    chips = _js(f"_meshHostChips({json.dumps(data)})")
    assert [c["label"] for c in chips] == ["a (local)", "host-b", "m3"]


def test_down_member_says_down_and_shows_no_configured_size():
    view = _js(f"_meshCardView('svc-big', {json.dumps(MESH['nodes']['svc-big'])})")
    assert view["tone"] == "down"
    assert view["value"] == "down"
    assert view["sub"] == "no answer on 192.0.2.1:8101"
    assert "19.4" not in json.dumps(view)
    assert "this host" in view["detail"]


def test_up_member_without_agent_is_labelled_as_config_estimate():
    view = _js(f"_meshCardView('svc-cpu', {json.dumps(MESH['nodes']['svc-cpu'])})")
    assert view["tone"] == "config"
    assert view["value"] == "7.3 GB RAM"
    assert "size from config, not measured" in view["detail"]
    assert view["sub"] == "model-a.gguf"  # live model id wins over "unknown"
    assert view["label"] == "host-a · svc-cpu"


def test_member_with_agent_shows_measured_vram():
    view = _js(f"_meshCardView('svc-embed', {json.dumps(MESH['nodes']['svc-embed'])})")
    assert view["tone"] == "live"
    assert view["value"] == "0.5 / 2.0 GiB"
    assert "not measured" not in view["detail"]
    assert view["sub"] == "model not reported"


def test_old_payload_without_stats_live_falls_back_to_vram_presence():
    node = dict(MESH["nodes"]["svc-embed"])
    node.pop("stats_live")
    assert _js(f"_meshCardView('n', {json.dumps(node)})")["tone"] == "live"
    node = dict(MESH["nodes"]["svc-cpu"])
    node.pop("stats_live")
    assert _js(f"_meshCardView('n', {json.dumps(node)})")["tone"] == "config"


def test_mesh_escape():
    assert _js("_meshEsc('<img src=x onerror=\"a\">&\\'')") == (
        "&lt;img src=x onerror=&quot;a&quot;&gt;&amp;&#39;")
    assert _js("_meshEsc(null)") == ""


def test_update_vram_tab_escapes_card_text():
    js = max(re.findall(r"<script>(.*?)</script>", _dashboard_html(), re.S), key=len)
    body = re.search(r"^async function updateVramTab\(\) \{.*?^}\n", js, re.S | re.M).group(0)
    assert "_meshHostChips(data)" in body
    assert "_meshCardView(" in body
    assert "_meshEsc(view.label)" in body and "_meshEsc(view.sub)" in body


def test_system_ram_shown_once_in_vram_header():
    html = _dashboard_html()
    for gone in ('id="card-ram-total"', 'id="card-ram-used"', 'id="card-ram-free"',
                 'id="ram-info-block"', 'id="ram-total"', 'id="ram-pct"'):
        assert gone not in html, gone
    assert 'id="card-proc"' in html  # still written unguarded by updateVramTab
    assert "System RAM Allocation" in html


# ── Server side: additive /mesh/data fields ──

@pytest.mark.parametrize("raw,want", [
    ("model-a-Q4_K_M.gguf", "model-a-Q4_K_M.gguf"),
    ("/models/x/model-b.gguf", "model-b.gguf"),
    ("  model-c-8B  ", "model-c-8B"),
    ("<script>alert(1)</script>", None),
    ("a" * 97, None),
    ("", None),
    (None, None),
    (42, None),
])
def test_peer_model_label(facade, raw, want):
    assert facade._peer_model_label(raw) == want


def test_peer_status_additive_fields(facade, monkeypatch):
    cfg = {"providers": {"local": {"instances": {
        "here-down": {"host": "192.0.2.1", "port": 8101, "model": "unknown", "ram_used_gb": 19.4},
        "remote-up": {"host": "192.0.2.2", "port": 8080, "model": "unknown"},
        "remote-bad": {"host": "192.0.2.2", "port": 1, "model": "unknown"},
    }}}}
    mon = facade.PeerMonitor(cfg, {})

    async def health(host, port):
        return host == "192.0.2.2"

    async def slots(host, port):
        return [], "/m/<b>bad</b>" if port == 1 else "model-b.gguf"

    monkeypatch.setattr(mon, "_check_remote_health", health)
    monkeypatch.setattr(mon, "_get_remote_slots", slots)
    monkeypatch.setattr(mon, "_is_this_host", lambda host: host == "192.0.2.1")
    asyncio.run(mon.check_all())

    down, up = mon.peer_status["here-down"], mon.peer_status["remote-up"]
    assert down["on_this_host"] is True and down["stats_live"] is False
    assert down["model_live"] is None
    assert down["model"] == "unknown" and down["ram_used_gb"] == 19.4  # unchanged
    assert up["on_this_host"] is False and up["stats_live"] is False  # no agent
    assert up["model_live"] == "model-b.gguf"
    assert up["is_local"] is False
    assert mon.peer_status["remote-bad"]["model_live"] is None  # "<b>bad</b>" dropped


def test_is_this_host_loopback_and_foreign(facade):
    mon = facade.PeerMonitor({}, {})
    assert mon._is_this_host("127.0.0.1") is True
    assert mon._is_this_host("localhost") is True
    assert mon._is_this_host("192.0.2.1") is False  # TEST-NET-1, never local
    assert mon._is_this_host("example.invalid") is False
