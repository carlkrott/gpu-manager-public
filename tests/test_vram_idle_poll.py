"""The pusher loop must not run `sudo amdgpu_top` every second while nothing watches.

Live evidence 2026-10-07 21:15: 162 sudo journal lines/min from the 1 Hz loop with no
dashboard attached. On-demand consumers keep refreshing through _poll_vram() (TTL).
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


def _deny(*args, **kwargs):
    raise AssertionError("process creation forbidden during loop proof")


@pytest.fixture
def facade(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir(); xdg.mkdir()
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    spec = importlib.util.spec_from_file_location("vram_idle_poll_facade", FACADE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_should_poll_truth_table(facade):
    f = facade._vram_loop_should_poll
    assert f(True, True, 0, every=5)          # dashboard attached: every tick
    assert f(False, False, 0, every=5)        # cold cache: poll at once
    assert not f(False, True, 1, every=5)     # idle and warm: skip
    assert not f(False, True, 4, every=5)
    assert f(False, True, 5, every=5)         # Nth idle tick
    assert f(False, True, 1, every=1)         # every=1 restores 1 Hz


def test_default_every_is_slower_than_one_hertz(facade):
    assert facade.VRAM_IDLE_POLL_EVERY >= 2


def _run_loop(facade, monkeypatch, ticks, clients):
    polls = {"n": 0}
    seen = {"ticks": 0}

    async def fake_poll():
        polls["n"] += 1
        return {"processes": [], "gpus": {}}

    class Sched:
        def get_queue_status(self):
            seen["ticks"] += 1
            return {}
        def verify_state_consistency(self): pass
        def get_lifecycle_status(self): return {"state": "idle"}

    async def noop(*a, **k): return {}

    monkeypatch.setattr(facade, "_poll_vram", fake_poll)
    monkeypatch.setattr(facade, "scheduler", Sched())
    monkeypatch.setattr(facade, "GPU_PROBE_STARTUP_GRACE_S", 0)
    monkeypatch.setattr(facade, "VRAM_POLL_INTERVAL", 0)
    monkeypatch.setattr(facade, "VRAM_IDLE_POLL_EVERY", 5)
    monkeypatch.setattr(facade, "_vram_cache", {})
    monkeypatch.setattr(facade, "_detect_silent_oom_via_dmesg", lambda since_ts=None: [])
    monkeypatch.setattr(facade, "_broadcast_ws", noop)
    monkeypatch.setattr(facade, "_check_all_services_health", noop)
    monkeypatch.setattr(facade, "_check_nginx_status", noop)
    monkeypatch.setattr(facade, "_resolve_bundles", lambda: {})
    monkeypatch.setattr(facade, "_ws_clients", set(clients))

    async def main():
        task = asyncio.create_task(facade._vram_pusher_loop())
        while seen["ticks"] < ticks:
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(asyncio.wait_for(main(), timeout=10))
    return polls["n"], seen["ticks"]


def test_idle_loop_polls_every_fifth_tick(facade, monkeypatch):
    polls, ticks = _run_loop(facade, monkeypatch, 12, clients=[])
    assert ticks >= 12
    # tick 1 (cold cache), then every 5th tick afterwards
    assert polls == 1 + (ticks - 1) // 5


def test_loop_polls_every_tick_with_dashboard_attached(facade, monkeypatch):
    polls, ticks = _run_loop(facade, monkeypatch, 12, clients=[object()])
    assert polls >= 12 and polls >= ticks - 1
