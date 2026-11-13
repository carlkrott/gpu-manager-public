"""Binding proof for the extracted scheduler helpers (scripts/gpu_manager/scheduler_helpers.py).

Only synthetic facade state is used: no socket, subprocess or event loop is ever
created, and the guards below fail the test if a facade load or a helper call
tries to.  ``_TargetStateUnreadable`` stays a facade class; the helpers resolve
it, ``_queue_depth_observation`` and the other scheduler-state names through the
facade dictionary, not through a module pointer.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"

NAMES = (
    "_record_load_history",
    "_record_head_job_precheck",
    "_group_circuit_state",
    "_active_group_still_draining",
    "_target_gpus_hold_warm_bundle",
    "_resolve_target_bundle_components",
    "_resolve_group_target_gpus",
    "_scheduler_precheck_head_jobs",
    "_scheduler_run_precheck",
    "_refresh_phase3_scheduler_state",
    "_scheduler_evaluate_load",
    "_resolve_scheduler_bundle_for_group",
    "_is_repeat_idle_decision",
    "_dispatch_scheduler_decision",
    "_proactive_dispatch_loop",
    "_should_suppress_proactive_spacing",
)
ASYNC_NAMES = {"_dispatch_scheduler_decision", "_proactive_dispatch_loop"}


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def scheduler_helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from gpu_manager import scheduler_helpers as module

    return module


@pytest.fixture
def facades(tmp_path, monkeypatch):
    """Load two fresh, unregistered facades under hermetic guards."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    # aiohttp subclasses socket.socket at import time; load it before the guard.
    import aiohttp.web  # noqa: F401

    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(subprocess, "run", _deny)
    monkeypatch.setattr(os, "system", _deny)

    loaded = []
    for name in ("scheduler_helpers_facade_one", "scheduler_helpers_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def test_symbols_are_bound_to_their_own_facade(facades, scheduler_helpers):
    one, two = facades
    assert scheduler_helpers.NAMES == NAMES
    assert len(NAMES) == 16
    for facade in facades:
        for name in NAMES:
            function = getattr(facade, name)
            template = getattr(scheduler_helpers, name)
            assert function is not template
            assert function.__globals__ is vars(facade)
            assert function.__module__ == facade.__name__
            assert function.__qualname__ == template.__qualname__
            assert function.__doc__ == template.__doc__
            assert str(inspect.signature(function)) == str(inspect.signature(template))
            assert inspect.iscoroutinefunction(function) == (name in ASYNC_NAMES)
            assert function.__code__ is template.__code__
        assert facade.__name__ not in sys.modules  # no sys.modules dependency
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)
    assert len(scheduler_helpers.bind({})) == len(NAMES)


def test_facade_state_stays_facade_owned_and_isolated(facades, scheduler_helpers):
    one, two = facades
    for name in ("_load_history", "_head_job_pre_checks", "_idle_decision_signature"):
        assert not hasattr(scheduler_helpers, name)
    assert vars(one)["_load_history"] is not vars(two)["_load_history"]
    one._record_load_history({"action": "noop"})
    assert vars(one)["_load_history"] == [{"action": "noop"}]
    assert vars(two)["_load_history"] == []  # the other facade's helper does not see it
    for index in range(60):
        two._record_load_history({"n": index})
    assert len(vars(two)["_load_history"]) == 50 and vars(two)["_load_history"][0] == {"n": 59}
    assert vars(one)["_load_history"] == [{"action": "noop"}]

    for index in range(25):
        one._record_head_job_precheck({"n": index})
    assert len(vars(one)["_head_job_pre_checks"]) == 20
    assert vars(one)["_head_job_pre_checks"][0] == {"n": 24}
    assert vars(two)["_head_job_pre_checks"] == []


def test_helpers_write_and_read_known_facade_globals(facades, scheduler_helpers):
    one, two = facades
    assert one._group_circuit_state("g", 0.0, {}) == {"state": "closed"}
    vars(one)["_group_failure_state"]["g"] = {"state": "open"}
    assert one._group_circuit_state("g", 0.0, {}) == {"state": "open"}
    assert two._group_circuit_state("g", 0.0, {}) == {"state": "closed"}

    # _is_repeat_idle_decision writes its `global` into the facade only.
    idle = {"action": "noop", "reason": "r", "target_gpus": ["0"]}
    assert vars(one)["_idle_decision_signature"] is None
    assert one._is_repeat_idle_decision(dict(idle)) is False
    assert vars(one)["_idle_decision_signature"] == ("noop", "r", ("0",))
    assert one._is_repeat_idle_decision(dict(idle)) is True
    assert vars(two)["_idle_decision_signature"] is None
    assert one._is_repeat_idle_decision({"action": "dispatch"}) is False
    assert vars(one)["_idle_decision_signature"] is None

    # _active_group_still_draining reads the facade's _drain_cutoffs and clock.
    ticks = iter([1000.0, 1000.0])
    vars(one)["_drain_cutoffs"]["other"] = 990.0
    vars(one)["time"] = types.SimpleNamespace(time=lambda: next(ticks))
    try:
        assert one._active_group_still_draining("target") is True
        assert one._active_group_still_draining("other") is False
    finally:
        del vars(one)["time"]
        import time as real_time

        vars(one)["time"] = real_time
    assert two._active_group_still_draining("target") is False


def test_pure_helper_matches_its_original_contract(facades):
    one = facades[0]
    suppress = one._should_suppress_proactive_spacing
    window = vars(one)["_PROACTIVE_SPACING_WINDOW_SEC"]
    base = dict(now=100.0, action="a", last_action_recorded_at=100.0, last_decision_action="a", dry_run=True)
    assert suppress(**base) is True
    assert suppress(**{**base, "dry_run": False}) is False
    assert suppress(**{**base, "last_decision_action": None}) is False
    assert suppress(**{**base, "last_decision_action": "b"}) is False
    assert suppress(**{**base, "now": 100.0 + window}) is False


def test_cross_module_names_resolve_through_the_facade(facades, scheduler_helpers, monkeypatch):
    """Scheduler-state names, a facade class and sibling helpers are looked up in the facade."""
    one, two = facades
    # The template module imports none of them: they exist only in the facade dictionaries.
    for name in ("_queue_depth_observation", "_TargetStateUnreadable", "_get_all_gpu_states"):
        assert not hasattr(scheduler_helpers, name)

    # _scheduler_run_precheck -> sibling helpers replaced in facade one only.
    recorded = []
    monkeypatch.setattr(one, "_scheduler_precheck_head_jobs", lambda group: {"group": group})
    monkeypatch.setattr(one, "_record_head_job_precheck", recorded.append)
    record = one._scheduler_run_precheck("grp")
    assert recorded == [record] and record["group"] == "grp" and "recorded_at" in record
    assert vars(two)["_head_job_pre_checks"] == [] and vars(one)["_head_job_pre_checks"] == []

    # _refresh_phase3_scheduler_state -> scheduler-state `_queue_depth_observation` via the facade.
    seen = []

    def observe(queue, service, group):
        seen.append((queue, service, group))
        return {"depth": 7, "known": True, "reason": "fake"}

    def phase3(groups, gpu_states, templates):
        return {"groups": groups}

    monkeypatch.setattr(one, "_get_all_gpu_states", lambda: {})
    monkeypatch.setattr(one, "_services_config", {"services": {"svc": {"routing_group": "rg", "type": "llm_backend"}}})
    monkeypatch.setattr(one, "queue_engine", object())
    monkeypatch.setattr(one, "_queue_depth_observation", observe)
    monkeypatch.setattr(one, "_get_phase3_scheduler_state", phase3)
    state = one._refresh_phase3_scheduler_state()
    assert seen == [(one.queue_engine, "rg", "rg")]
    assert state["groups"]["rg"]["queue_lag"] == 7 and state["groups"]["rg"]["type"] == "llm"
    assert state["groups"]["rg"]["queue_lag_reason"] == "fake"
    # Facade two still has its own, unreplaced scheduler-state function.
    assert two._queue_depth_observation is not observe

    # The facade class is resolved at call time as well: replace it and the helper raises the new one.
    class Replacement(Exception):
        pass

    def unreadable(gpu_id):
        raise OSError("state unreadable")

    for facade in (one, two):
        monkeypatch.setattr(facade, "_resolve_group_target_gpus", lambda group: ["0"])
        monkeypatch.setattr(facade, "_resolve_target_bundle_components", lambda group: ("bundle", []))
        monkeypatch.setattr(facade, "_get_gpu_state", unreadable)
    original_two = two._TargetStateUnreadable
    monkeypatch.setattr(one, "_TargetStateUnreadable", Replacement)
    with pytest.raises(Replacement):
        one._target_gpus_hold_warm_bundle("grp")
    with pytest.raises(original_two) as caught:
        two._target_gpus_hold_warm_bundle("grp")
    assert (caught.value.group, caught.value.gpu_id) == ("grp", "0")


def test_unbound_templates_cannot_resolve_facade_names(scheduler_helpers):
    with pytest.raises(NameError):
        scheduler_helpers._record_load_history({"action": "noop"})
    with pytest.raises(NameError):
        scheduler_helpers._scheduler_run_precheck("grp")


def test_bind_rejects_non_dict(scheduler_helpers):
    with pytest.raises(TypeError):
        scheduler_helpers.bind(object())
    before = set(sys.modules)
    bound = scheduler_helpers.bind({"__name__": "synthetic_facade"})
    assert set(sys.modules) == before
    assert all(function.__module__ == "synthetic_facade" for function in bound)
