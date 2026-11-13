"""Binding proof for the extracted scheduler state (scripts/gpu_manager/scheduler_state.py).

Only synthetic queues, trackers, supervisors and managers are used: no socket,
subprocess or event loop is ever created, and the guards below fail the test if
a facade load or a scheduler-state call tries to.  The actual extracted
``_RestoreAdmissionGuard`` is held to the same modeled proof as the class
binding proof in ``tests/test_memory_policy_binding.py``.
"""
from __future__ import annotations

import asyncio
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
    "_queue_depth_observation",
    "_logical_group_quiescent",
    "_wait_for_logical_group_quiescence",
    "_log_can_consume_pause",
    "_get_dual_swap_lock",
    "_RestoreAdmissionGuard",
)
FUNCTION_NAMES = NAMES[:-1]


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def scheduler_state(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from gpu_manager import scheduler_state as module

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
    for name in ("scheduler_state_facade_one", "scheduler_state_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


class _FakeManager:
    def __init__(self):
        self._evict_lock = asyncio.Lock()


def _run(coro):
    """Drive a coroutine that never suspends (uncontended locks) without an event loop.

    The module guard denies socket creation and an event loop needs a socketpair.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("coroutine suspended; the proof only uses uncontended locks")


def test_symbols_are_bound_to_their_own_facade(facades, scheduler_state):
    one, two = facades
    assert scheduler_state.NAMES == NAMES
    for facade in facades:
        for name in FUNCTION_NAMES:
            function = getattr(facade, name)
            template = getattr(scheduler_state, name)
            assert function is not template
            assert function.__globals__ is vars(facade)
            assert function.__module__ == facade.__name__
            assert function.__qualname__ == template.__qualname__
            assert function.__doc__ == template.__doc__
            assert str(inspect.signature(function)) == str(inspect.signature(template))
            assert inspect.iscoroutinefunction(function) == inspect.iscoroutinefunction(template)
    for name in FUNCTION_NAMES:
        assert getattr(one, name) is not getattr(two, name)
    assert inspect.iscoroutinefunction(one._wait_for_logical_group_quiescence)
    assert len(scheduler_state.bind({})) == len(NAMES)


def test_actual_guard_class_is_a_per_facade_bound_class(facades, scheduler_state):
    one, two = facades
    template = scheduler_state._RestoreAdmissionGuard
    for facade in facades:
        bound = facade._RestoreAdmissionGuard
        assert bound is not template and isinstance(bound, type)
        assert bound.__bases__ == (object,)
        assert bound.__name__ == "_RestoreAdmissionGuard"
        assert bound.__qualname__ == template.__qualname__
        assert bound.__module__ == facade.__name__
        assert bound.__doc__ == template.__doc__
        for name in ("__init__", "__aenter__", "__aexit__"):
            method = vars(bound)[name]
            assert method.__globals__ is vars(facade)
            assert inspect.iscoroutinefunction(method) == inspect.iscoroutinefunction(vars(template)[name])
            assert str(inspect.signature(method)) == str(inspect.signature(vars(template)[name]))
            assert method.__qualname__ == f"_RestoreAdmissionGuard.{name}"
        assert facade.__name__ not in sys.modules  # no sys.modules dependency
    assert one._RestoreAdmissionGuard is not two._RestoreAdmissionGuard
    # The unbound template cannot resolve the facade lock: this is the defect the binder removes.
    with pytest.raises(NameError):
        template(_FakeManager())


def test_class_identity_is_stable_within_one_binding_and_distinct_per_bind(facades, scheduler_state):
    one = facades[0]
    bound = one._RestoreAdmissionGuard
    manager = _FakeManager()
    first, second = one._RestoreAdmissionGuard(manager), one._RestoreAdmissionGuard(manager)
    assert type(first) is type(second) is bound
    assert isinstance(first, one._RestoreAdmissionGuard)
    assert not isinstance(first, scheduler_state._RestoreAdmissionGuard)
    # Identity is per bind() call: the facade binds once and keeps the result.
    again = scheduler_state.bind(vars(one))
    assert again[-1] is not bound
    assert one._RestoreAdmissionGuard is bound


def test_guard_sees_replaced_dual_swap_lock_after_import_and_releases_on_failure(facades, monkeypatch):
    one = facades[0]
    manager = _FakeManager()

    async def scenario():
        first = one._RestoreAdmissionGuard(manager)
        real_lock = first._admission_lock
        assert real_lock is one._get_dual_swap_lock()  # facade-global lookup, not a copy
        replacement = asyncio.Lock()
        monkeypatch.setattr(one, "_get_dual_swap_lock", lambda: replacement)
        second = one._RestoreAdmissionGuard(manager)  # constructed AFTER import and AFTER the replacement
        assert second._admission_lock is replacement
        assert first._admission_lock is real_lock  # earlier instance is unchanged
        async with second as entered:
            assert entered is second
            assert replacement.locked() and manager._evict_lock.locked()
            assert not real_lock.locked()
        assert not replacement.locked() and not manager._evict_lock.locked()

        class Boom:
            class _Lock:
                async def acquire(self):
                    raise RuntimeError("evict lock failed")

            _evict_lock = _Lock()

        third = one._RestoreAdmissionGuard(Boom())
        with pytest.raises(RuntimeError, match="evict lock failed"):
            await third.__aenter__()
        assert not replacement.locked()  # admission lock released, error re-raised

        # __aexit__ returns False so body exceptions propagate and both locks release.
        with pytest.raises(ValueError, match="body failed"):
            async with one._RestoreAdmissionGuard(manager):
                raise ValueError("body failed")
        assert not replacement.locked() and not manager._evict_lock.locked()

    _run(scenario())


def test_facade_lock_state_stays_facade_owned_and_isolated(facades, scheduler_state, monkeypatch):
    one, two = facades
    assert not hasattr(scheduler_state, "_dual_swap_lock")
    assert not hasattr(scheduler_state, "_can_consume_last_log")
    assert vars(one)["_dual_swap_lock"] is None and vars(two)["_dual_swap_lock"] is None
    assert isinstance(vars(one)["_can_consume_last_log"], dict)
    assert vars(one)["_can_consume_last_log"] is not vars(two)["_can_consume_last_log"]

    sentinel = object()
    monkeypatch.setattr(one, "_get_dual_swap_lock", lambda: sentinel)
    manager = _FakeManager()
    assert one._RestoreAdmissionGuard(manager)._admission_lock is sentinel
    lock_two = two._RestoreAdmissionGuard(manager)._admission_lock
    assert isinstance(lock_two, asyncio.Lock) and lock_two is not sentinel
    assert lock_two is two._get_dual_swap_lock()
    # The lazily created lock lives in facade two's globals only.
    assert vars(two)["_dual_swap_lock"] is lock_two
    assert vars(one)["_dual_swap_lock"] is None
    # Repeated calls reuse the facade's lock.
    assert two._get_dual_swap_lock() is lock_two


class _ObservingQueue:
    def get_queue_depth_observation(self, service, routing_group=None):
        return {"depth": "3", "known": True, "reason": "ok", "pending": 2, "stream": "s", "error": "x" * 400}


class _LegacyQueue:
    def __init__(self):
        self.calls = []

    def get_queue_depth(self, service, routing_group=None, **kwargs):
        self.calls.append(kwargs)
        if "strict" in kwargs:
            raise TypeError("unexpected keyword argument 'strict'")
        return 4


def test_queue_depth_observation_behaviour_matches_the_original_contract(facades):
    one = facades[0]
    observe = one._queue_depth_observation
    assert observe(None, "svc") == {
        "depth": 0, "known": False, "reason": "queue_engine_unavailable", "pending": 0,
    }
    seen = observe(_ObservingQueue(), "svc", "grp")
    assert seen["depth"] == 3 and seen["known"] is True and seen["pending"] == 2
    assert len(seen["error"]) == 240
    legacy = _LegacyQueue()
    assert observe(legacy, "svc") == {
        "depth": 4, "known": True, "reason": "legacy_integer_api", "pending": 0,
    }
    assert legacy.calls == [{"strict": False}, {}]

    class Broken:
        def get_queue_depth(self, *args, **kwargs):
            raise RuntimeError("down")

    assert observe(Broken(), "svc")["reason"] == "depth_error:RuntimeError"
    with pytest.raises(RuntimeError):
        observe(Broken(), "svc", strict=True)


class _Tracker:
    def __init__(self, status="available", active=0):
        self.status, self.active = status, active

    def demand_snapshot(self, routing_group=None):
        return {"status": self.status, "active_demand": self.active}


class _Queue:
    def __init__(self, depth=0, pending=()):
        self.depth, self.pending = depth, list(pending)

    def get_queue_depth_observation(self, service, routing_group=None):
        return {"depth": self.depth, "known": True, "reason": "ok"}

    def get_pending(self, service, routing_group=None, strict=False):
        return self.pending


def test_logical_group_quiescence_reads_facade_globals_at_call_time(facades, monkeypatch):
    one = facades[0]
    monkeypatch.setattr(one, "queue_engine", _Queue())
    monkeypatch.setattr(one, "worker_supervisor", None)
    monkeypatch.setattr(one, "job_tracker", None)
    empty, snapshot = one._logical_group_quiescent("grp")
    assert empty is False and snapshot["error"] == "job demand tracker unavailable"

    monkeypatch.setattr(one, "job_tracker", _Tracker())
    empty, snapshot = one._logical_group_quiescent("grp")
    assert empty is True and snapshot["queue_depth_known"] is True

    pool = types.SimpleNamespace(routing_groups=("grp",), active_workers=2)
    monkeypatch.setattr(one, "worker_supervisor", types.SimpleNamespace(_pools={"p": pool}))
    empty, snapshot = one._logical_group_quiescent("grp")
    assert empty is False and snapshot["active_workers"] == 2

    monkeypatch.setattr(one, "worker_supervisor", None)
    monkeypatch.setattr(one, "queue_engine", _Queue(depth=1))
    assert one._logical_group_quiescent("grp")[0] is False
    monkeypatch.setattr(one, "queue_engine", _Queue(pending=["x"]))
    assert one._logical_group_quiescent("grp")[0] is False
    monkeypatch.setattr(one, "queue_engine", _Queue())
    monkeypatch.setattr(one, "job_tracker", _Tracker(status="unavailable"))
    assert one._logical_group_quiescent("grp")[1]["error"] == "job demand unavailable"
    monkeypatch.setattr(one, "job_tracker", _Tracker(active=1))
    assert one._logical_group_quiescent("grp")[0] is False


def test_wait_for_quiescence_uses_replaced_facade_helpers_and_clock(facades, monkeypatch):
    one = facades[0]
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(one, "asyncio", types.SimpleNamespace(sleep=fake_sleep, Lock=asyncio.Lock))
    answers = iter([(True, {"n": 1}), (True, {"n": 2})])
    monkeypatch.setattr(one, "_logical_group_quiescent", lambda group: next(answers))
    assert _run(one._wait_for_logical_group_quiescence("grp", timeout=5.0, settle_seconds=0.25)) is True
    assert sleeps == [0.25]

    sleeps.clear()
    clock = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(one, "time", types.SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(one, "_logical_group_quiescent", lambda group: (False, {"busy": True}))
    assert _run(one._wait_for_logical_group_quiescence("grp", timeout=5.0)) is False
    assert sleeps == [0.5]


def test_log_can_consume_pause_writes_only_the_facade_state(facades, monkeypatch):
    one, two = facades
    ticks = iter([100.0, 105.0, 111.0])
    monkeypatch.setattr(one, "time", types.SimpleNamespace(time=lambda: next(ticks)))
    one._log_can_consume_pause("svc", "grp", "reason", "w1")
    key = ("svc", "grp", "reason")
    assert one._can_consume_last_log == {key: 100.0}
    one._log_can_consume_pause("svc", "grp", "reason", "w1")  # inside the 10s window: unchanged
    assert one._can_consume_last_log[key] == 100.0
    one._log_can_consume_pause("svc", "grp", "reason", "w1")
    assert one._can_consume_last_log[key] == 111.0
    assert two._can_consume_last_log == {}


def test_bind_rejects_non_dict(scheduler_state):
    with pytest.raises(TypeError):
        scheduler_state.bind(object())
    before = set(sys.modules)
    scheduler_state.bind({"__name__": "synthetic_facade"})
    assert set(sys.modules) == before
