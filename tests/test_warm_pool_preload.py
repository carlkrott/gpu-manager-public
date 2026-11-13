"""Catalog 5e: warm-pool pre-load (queue look-ahead), flag GPU_MANAGER_WARM_POOL_PRELOAD.

Drives the real facade with fake GPU state, bundles, head jobs and load call:
nothing loads, nothing sleeps for real and no GPU is touched.
"""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FACADE = REPO / "scripts" / "gpu-manager.py"
sys.path.insert(0, str(REPO / "scripts"))
from gpu_manager import warm_pool  # noqa: E402

BUNDLES = {
    "bA": {"gpu_id": "gpu_a", "idle": False, "priority": 5, "services": ["svcA"]},
    "bB": {"gpu_id": "gpu_b", "idle": False, "priority": 5, "services": ["svcB"]},
    "bOld": {"gpu_id": "gpu_b", "idle": False, "priority": 5, "services": ["svcOld"]},
    "idleB": {"gpu_id": "gpu_b", "idle": True, "priority": 1, "services": ["Qwen-Dual"]},
    "strata": {"gpu_id": "gpu_b", "idle": False, "priority": 3, "evict_only_when_idle": True},
}
GROUP_BUNDLE = {"A": "bA", "B": "bB"}


def gpu(loaded=None, **kw):
    # Real state class, so a renamed field breaks these tests instead of silently passing.
    state = _load_facade().GpuBundleState("gpu_x")
    state.loaded_bundle, state.loaded_bundles = loaded, [loaded] if loaded else []
    for key, value in kw.items():
        assert hasattr(state, key), key
        setattr(state, key, value)
    return state


@functools.cache
def _load_facade():
    spec = importlib.util.spec_from_file_location("gpu_manager_warm_pool_test", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def facade():
    return _load_facade()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(facade, monkeypatch):
    clock = Clock()
    states = {"gpu_a": gpu("bA", worker_in_flight=1), "gpu_b": gpu("bOld")}
    loads, heads, fit = [], {"B": ["job-1"]}, {"errors": []}

    async def fake_load(bundle, reason="", priority=None):
        loads.append((bundle, reason))
        return True

    def fake_eval(config, snapshot, owners=None, **kw):
        return [{"rule": r, "component": "c", "severity": "error"} for r in fit["errors"]]

    m = facade
    monkeypatch.setattr(m, "_warm_pool_memory", warm_pool.WarmPoolMemory(clock=clock))
    monkeypatch.setattr(m, "_last_next_decision", {"next_decision": "demand:A"})
    monkeypatch.setattr(m, "_last_pending_groups", [{"group": "A"}, {"group": "B"}])
    monkeypatch.setattr(m, "_resolve_bundles", lambda: BUNDLES)
    monkeypatch.setattr(m, "_resolve_scheduler_bundle_for_group",
                        lambda g: (GROUP_BUNDLE[g], 5, None) if g in GROUP_BUNDLE else (None, None, "none"))
    monkeypatch.setattr(m, "_bundle_physical_gpu_set", lambda b, bundles=None: {BUNDLES[b]["gpu_id"]})
    monkeypatch.setattr(m, "_resolve_group_target_gpus", lambda g: [])
    monkeypatch.setattr(m, "_bundle_resident_on_all_gpus",
                        lambda b, bundles: states[BUNDLES[b]["gpu_id"]].loaded_bundle == b)
    monkeypatch.setattr(m, "_scheduler_precheck_head_jobs", lambda g: {"head_job_ids": list(heads.get(g, []))})
    monkeypatch.setattr(m, "_get_gpu_state", lambda gid: states.get(gid))
    monkeypatch.setattr(m, "_bundle_transition_is_active", lambda: False)
    monkeypatch.setattr(m, "_active_group_still_draining", lambda g: False)
    monkeypatch.setattr(m, "scheduler", None)
    monkeypatch.setattr(m, "_bundle_worker_in_flight", lambda b, bundles=None: 0)
    monkeypatch.setattr(m, "_auto_evict_bundle_blockers", lambda b, bundles=None: [])
    monkeypatch.setattr(m, "_bundle_tracker_in_flight_jobs", lambda b, bundles=None: [])
    monkeypatch.setattr(m, "_get_linked_bundles", lambda b, bundles=None: [])
    monkeypatch.setattr(m, "_load_services_config", lambda: {})
    monkeypatch.setattr(m, "_vram_cache", {"gpus": [{"gpu_id": "gpu_b"}]})
    monkeypatch.setattr(m._fit_validator, "evaluate", fake_eval)
    monkeypatch.setattr(m, "_evict_then_load_bundle", fake_load)
    return types.SimpleNamespace(m=m, clock=clock, states=states, loads=loads, heads=heads, fit=fit)


def tick(e):
    async def run():
        started = await e.m._warm_pool_preload_tick()
        task = e.m._warm_pool_memory.task
        if task is not None:
            await task
        return started
    return asyncio.run(run())


def test_preloads_next_group_bundle_over_finished_bundle(env):
    assert tick(env) == {"group": "B", "bundle": "bB", "job_id": "job-1"}
    assert env.loads == [("bB", "warm_pool:B")]


@pytest.mark.parametrize("state,expected", [
    (gpu("idleB"), "idle_member_resident"),
    (gpu(None), "idle_member_slot"),
    (gpu("llm:Qwen-Dual"), "llm_resident"),
    (gpu("strata"), "protected_occupant"),
    (gpu("bOld", worker_in_flight=1), "gpu_running_job"),
    (gpu("bOld", pending_requests=2), "gpu_running_job"),
    (gpu("bOld", loading=True), "gpu_in_transition"),
    (gpu("bA"), "occupant_has_demand"),
])
def test_never_displaces_idle_protected_or_busy(env, caplog, state, expected):
    env.states["gpu_b"] = state
    with caplog.at_level(logging.INFO):
        assert tick(env) is None
    assert env.loads == []
    assert expected in caplog.text


def test_empty_gpu_without_idle_member_is_allowed():
    assert warm_pool.gpu_blocker(gpu(None), bundles=BUNDLES, gpu_has_idle_member=False,
                                 pending_bundles=set()) is None


def test_skips_gpu_needed_by_head_group(env, caplog):
    env.m._last_pending_groups[:] = []
    GROUP_BUNDLE["C"] = "bOld"  # same GPU as bB, ranked above it
    try:
        env.m._last_pending_groups.extend([{"group": "A"}, {"group": "C"}, {"group": "B"}])
        env.states["gpu_b"] = gpu(None)
        env.heads["C"] = ["job-c"]
        with caplog.at_level(logging.INFO):
            assert tick(env) is None
        assert "gpu_needed_by_higher_pending_group" in caplog.text
    finally:
        del GROUP_BUNDLE["C"]


def test_no_preload_without_live_demand_or_queued_head_job(env):
    env.m._last_next_decision["next_decision"] = "hold_dwell_time"
    assert tick(env) is None
    env.m._last_next_decision["next_decision"] = "demand:A"
    env.heads["B"] = []
    assert tick(env) is None
    assert env.loads == []


def test_fit_validator_gates(env, caplog):
    env.fit["errors"] = ["over_budget"]
    with caplog.at_level(logging.INFO):
        assert tick(env) is None
    assert "fit_refused:over_budget:c" in caplog.text
    env.fit["errors"] = []
    env.m._vram_cache = {}
    assert tick(env) is None
    assert env.loads == []


def test_hysteresis_prevents_thrash_after_auto_evict(env):
    assert tick(env) is not None
    # Auto-evict unloads bB; the idle restore would now follow. Same job, many ticks:
    env.states["gpu_b"] = gpu(None)
    for _ in range(5):
        assert tick(env) is None
    # A finished bundle comes back with a NEW job inside the cooldown: still no preload.
    env.states["gpu_b"] = gpu("bOld")
    env.heads["B"] = ["job-2"]
    assert tick(env) is None
    # The first job is never preloaded for twice, even after the cooldown.
    env.clock.now += warm_pool.BUNDLE_COOLDOWN_S + 1
    env.heads["B"] = ["job-1"]
    assert tick(env) is None
    env.heads["B"] = ["job-2"]
    assert tick(env) is not None
    assert [b for b, _ in env.loads] == ["bB", "bB"]


def test_skip_is_logged_once_per_job(env, caplog):
    env.states["gpu_b"] = gpu("idleB")
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            tick(env)
    assert caplog.text.count("warm_pool: skip") == 1


class _Stop(BaseException):
    pass


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_loop_calls_warm_pool_only_when_flag_on(facade, monkeypatch, flag):
    m = facade
    calls = {"tick": 0, "dispatch": 0, "sleep": 0}

    async def fake_sleep(_s):
        calls["sleep"] += 1
        if calls["sleep"] > 2:
            raise _Stop

    async def fake_dispatch(decision):
        calls["dispatch"] += 1
        return dict(decision, dispatch_performed=False)

    async def fake_tick():
        calls["tick"] += 1

    if flag is None:
        monkeypatch.delenv(warm_pool.FLAG, raising=False)
    else:
        monkeypatch.setenv(warm_pool.FLAG, flag)
    monkeypatch.setattr(m, "asyncio", types.SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(m, "_phase3_scheduling_config", lambda: {
        "scheduler_owns_load": True, "scheduler_dry_run_mode": False, "proactive_scheduling_enabled": True})
    monkeypatch.setattr(m, "_refresh_phase3_scheduler_state", lambda: {})
    monkeypatch.setattr(m, "_scheduler_evaluate_load", lambda now: {"action": "noop"})
    monkeypatch.setattr(m, "_dispatch_scheduler_decision", fake_dispatch)
    monkeypatch.setattr(m, "_record_load_history", lambda d: None)
    monkeypatch.setattr(m, "_warm_pool_preload_tick", fake_tick)
    with pytest.raises(_Stop):
        asyncio.run(m._proactive_dispatch_loop())
    assert calls["dispatch"] >= 1
    assert calls["tick"] == (calls["dispatch"] if flag == "1" else 0)
