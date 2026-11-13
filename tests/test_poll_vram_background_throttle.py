"""Hermetic coverage for idle-policy sharing in background VRAM readers."""
from __future__ import annotations

import ast
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
    raise AssertionError("process creation forbidden in background throttle proof")


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
    spec = importlib.util.spec_from_file_location("background_vram_facade", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _tick_reaper(facade, monkeypatch):
    class Logger:
        def debug(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass

    reaper = facade.GPUScheduler.__new__(facade.GPUScheduler)
    reaper.logger = Logger()
    reaper._expected_pids = set()
    reaper._terminated_pids = {}
    reaper._last_orphan_reaper_at = 0
    reaper._get_service_main_pid = lambda unit: asyncio.sleep(0, result=None)
    monkeypatch.setattr(facade, "_load_services_config", lambda: {"services": {}})
    await reaper._orphan_reaper_tick()


async def _tick_peer_monitor(facade, monkeypatch):
    monitor = facade.PeerMonitor(
        {"providers": {"local": {"instances": {
            "local": {"host": "localhost", "port": 1234, "local": True}
        }}}},
        {},
    )
    async def up(*args, **kwargs): return True
    async def slots(*args, **kwargs): return ([], 0)
    monkeypatch.setattr(monitor, "_check_local_service", up)
    monkeypatch.setattr(monitor, "_get_llama_slots", slots)
    await monitor.check_all()


def _caller_cache_window(caller, facade):
    """Reuse window each background caller is entitled to.

    The idle pusher's own policy is ``VRAM_IDLE_POLL_EVERY * VRAM_POLL_INTERVAL``
    (5 s). The orphan reaper runs every ``ORPHAN_REAPER_INTERVAL_S`` (30 s), so it
    passes that cadence as the window and may legitimately reuse a cache older
    than 5 s. PeerMonitor still takes the bare 5 s default.
    """
    if caller is _tick_reaper:
        return facade.ORPHAN_REAPER_INTERVAL_S
    return facade.VRAM_IDLE_POLL_EVERY * facade.VRAM_POLL_INTERVAL


@pytest.mark.parametrize("caller", [_tick_reaper, _tick_peer_monitor], ids=["orphan-reaper", "peer-monitor"])
@pytest.mark.parametrize("cache_age", [3.5, 6.0, None], ids=["fresh", "stale", "empty"])
def test_background_callers_share_fresh_cache_or_refresh_once(facade, monkeypatch, caller, cache_age):
    _assert_probe_count(facade, monkeypatch, caller, cache_age)


@pytest.mark.parametrize("caller", [_tick_reaper, _tick_peer_monitor], ids=["orphan-reaper", "peer-monitor"])
@pytest.mark.parametrize("cache_age", [59.0, 60.0], ids=["inside-reaper-window", "past-reaper-window"])
def test_reaper_window_does_not_admit_data_older_than_its_cadence(facade, monkeypatch, caller, cache_age):
    """A widened window is bounded: data at/over the cadence must be re-probed."""
    _assert_probe_count(facade, monkeypatch, caller, cache_age)


def _probe_count(facade, monkeypatch, caller, cache_age):
    """Drive one caller tick against a cache of the given age; return probe count."""
    polls = []
    result = {"per_device": [], "gpus": []}
    async def fake_poll_once():
        polls.append("poll")
        return result

    monkeypatch.setattr(facade, "_poll_vram_once", fake_poll_once)
    monkeypatch.setattr(facade, "_vram_poll_lock", None)
    monkeypatch.setattr(facade, "_vram_cache", result if cache_age is not None else {})
    monkeypatch.setattr(
        facade,
        "_vram_cache_ts",
        facade.time.monotonic() - cache_age if cache_age is not None else 0,
    )
    asyncio.run(caller(facade, monkeypatch))
    return len(polls)


def _assert_probe_count(facade, monkeypatch, caller, cache_age):
    # An empty cache must always force exactly one probe for every caller, so a
    # widened window can never leave a background reader blind.
    window = _caller_cache_window(caller, facade)
    assert _probe_count(facade, monkeypatch, caller, cache_age) == (
        0 if cache_age is not None and cache_age < window else 1
    )


def test_reaper_window_is_pinned_to_the_reaper_cadence(facade, monkeypatch):
    """The widened window must BE the reaper cadence, not just some wider number.

    Ages straddle the real constant by 5 s on each side, so a window that is
    narrower than the cadence (data silently re-probed every tick) or wider than
    it (data silently reused past a full scan cycle) both fail. PeerMonitor is
    asserted un-widened in the same pass, since its 5 s window is far below both
    ages and must re-probe either way.
    """
    cadence = facade.ORPHAN_REAPER_INTERVAL_S
    for caller, cache_age, expected_polls, why in (
        (_tick_reaper, cadence - 5.0, 0, "just inside the cadence: reuse allowed"),
        (_tick_reaper, cadence + 5.0, 1, "just past the cadence: must re-probe"),
        (_tick_peer_monitor, cadence - 5.0, 1, "PeerMonitor is not reaper-scoped"),
        (_tick_peer_monitor, cadence + 5.0, 1, "PeerMonitor is not reaper-scoped"),
    ):
        actual = _probe_count(facade, monkeypatch, caller, cache_age)
        assert actual == expected_polls, (
            f"{caller.__name__} at cache_age={cache_age}s ({why}): "
            f"expected {expected_polls} probe(s), got {actual} — the reaper's "
            f"window is not pinned to ORPHAN_REAPER_INTERVAL_S={cadence}s"
        )


def _function_node(tree, name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _awaits_poll_vram(node):
    return any(
        isinstance(item, ast.Await)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == "_poll_vram"
        for item in ast.walk(node)
    )


def _awaits_background_helper(node):
    return any(
        isinstance(item, ast.Await)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == "_poll_vram_background"
        for item in ast.walk(node)
    )


def test_on_demand_callers_keep_direct_fresh_poll_path():
    tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    expected = {
        "_observability_snapshot", "handle_vram_list", "_check_gpu_vram_ceiling",
        "_websocket_handler", "_mcp_call_tool", "_verify_bundle_runtime_for_commit",
        "ensure_services_running", "_restore_after_gpu_work",
    }
    for name in expected:
        assert _awaits_poll_vram(_function_node(tree, name)), name
    peer_monitor = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "PeerMonitor")
    peer_check = next(node for node in peer_monitor.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "check_all")
    assert _awaits_background_helper(peer_check)
    scheduler = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "GPUScheduler")
    reaper = next(node for node in scheduler.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_orphan_reaper_tick")
    assert _awaits_background_helper(reaper)
