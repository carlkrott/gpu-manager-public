"""Binding and behavior proof for the facade-bound bundle model classes.

Baseline = the same class source executed directly in a facade namespace (what
``scripts/gpu-manager.py`` did before the extraction).  Candidate = the class a real
facade gets from ``bundle_model.bind(globals())``.
"""
from __future__ import annotations

import ast
import asyncio
import enum
import importlib.util
import logging
import os
import pickle
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"
MODULE = SCRIPTS / "gpu_manager" / "bundle_model.py"
NAMES = ("GpuState", "GpuBundleState", "BundleCoordinator")


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401 -- aiohttp subclasses socket.socket at import time

    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    result = []
    for name in ("bundle_model_facade_one", "bundle_model_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def _bundle_model_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        from gpu_manager import bundle_model
    finally:
        sys.path.remove(str(SCRIPTS))
    return bundle_model


def _baseline_facade(name, **extra):
    """A namespace that defines the three classes directly, as the old facade did."""
    tree = ast.parse(MODULE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in NAMES]
    assert [n.name for n in nodes] == list(NAMES)
    namespace = {
        "__name__": name,
        "Enum": enum.Enum,
        "asyncio": asyncio,
        "logger": logging.getLogger("bundle-model-baseline"),
        "DRAIN_TIMEOUT": 60,
        "deque": __import__("collections").deque,
        "BUNDLE_STATE_SCHEMA_VERSION": 5,
        "_sync_legacy_gpu_state": lambda *args: None,
        "time": types.SimpleNamespace(time=lambda: 1000.0),
        **extra,
    }
    exec(compile(ast.Module(nodes, []), "<baseline-facade>", "exec"), namespace)
    return namespace


def _shape(cls):
    return {
        "name": cls.__name__,
        "qualname": cls.__qualname__,
        "doc": cls.__doc__,
        "bases": cls.__bases__,
        "metaclass": type(cls),
        "keys": sorted(vars(cls)),
    }


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("coroutine unexpectedly suspended in hermetic proof")


def _outcome(call):
    try:
        return ("ok", call())
    except Exception as exc:  # noqa: BLE001 -- the failure itself is the observation
        return ("raised", type(exc).__name__, str(exc))


def test_real_source_shape_and_one_time_facade_binding(facades):
    source = FACADE.read_text()
    tree = ast.parse(source)
    top_level_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    assert not set(NAMES) & top_level_classes
    assert "GpuState, GpuBundleState, BundleCoordinator = _bind_bundle_model(globals())" in source
    assert source.count("_bind_bundle_model(globals())") == 1
    assert "_gpu_states: dict[str, GpuBundleState] = {}" in source
    bundle_model = _bundle_model_module()
    assert bundle_model.NAMES == NAMES
    templates = [n.name for n in ast.parse(MODULE.read_text()).body if isinstance(n, ast.ClassDef)]
    assert [name for name in templates if name in NAMES] == list(NAMES)
    for facade in facades:
        for cls in (facade.GpuState, facade.GpuBundleState, facade.BundleCoordinator):
            assert cls.__module__ == facade.__name__
        assert facade.GpuState is not bundle_model.GpuState
        assert isinstance(facade.bundle_coordinator, facade.BundleCoordinator)
        assert facade._gpu_states == {}


def test_gpu_state_enum_matches_baseline_identity_members_and_repr(facades):
    for facade in facades:
        base = _baseline_facade(facade.__name__)["GpuState"]
        cand = facade.GpuState
        assert _shape(cand) == _shape(base)
        assert list(cand.__members__) == list(base.__members__)
        assert [(m.name, m.value) for m in cand] == [(m.name, m.value) for m in base]
        assert [repr(m) for m in cand] == [repr(m) for m in base]
        assert [str(m) for m in cand] == [str(m) for m in base]
        assert [format(m) for m in cand] == [format(m) for m in base]
        assert repr(cand) == repr(base)
        assert cand._member_type_ is base._member_type_ and cand._member_names_ == base._member_names_
        for member in cand:
            assert cand[member.name] is member
            assert cand(member.value) is member
            assert getattr(cand, member.name) is member
            assert isinstance(member, cand) and not isinstance(member, base)
            cand_reduce, base_reduce = member.__reduce_ex__(2), base[member.name].__reduce_ex__(2)
            assert (cand_reduce[0] is cand, cand_reduce[1:]) == (base_reduce[0] is base, base_reduce[1:])
        assert _outcome(lambda: cand("nope")) == _outcome(lambda: base("nope"))
        assert _outcome(lambda: cand["NOPE"]) == _outcome(lambda: base["NOPE"])
        assert _outcome(lambda: setattr(cand, "LLM_LOADED", 1)) == _outcome(
            lambda: setattr(base, "LLM_LOADED", 1)
        )
        assert _outcome(lambda: setattr(cand.LLM_LOADED, "value", 1)) == _outcome(
            lambda: setattr(base.LLM_LOADED, "value", 1)
        )


def test_gpu_state_pickle_matches_baseline_for_unregistered_and_registered_facade(facades, monkeypatch):
    one = facades[0]
    base = _baseline_facade(one.__name__)["GpuState"]
    # Unregistered facade: the baseline cannot pickle, so the candidate must fail identically.
    assert one.__name__ not in sys.modules
    for name in base.__members__:
        expected = _outcome(lambda: pickle.dumps(base[name]))
        assert expected[0] == "raised" and expected[1] == "PicklingError"
        assert _outcome(lambda: pickle.dumps(one.GpuState[name])) == expected
    # Registered facade (the production situation): lookup by qualified name returns the
    # bound class because the facade assigns it to the same attribute name.
    monkeypatch.setitem(sys.modules, one.__name__, one)
    for member in one.GpuState:
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(member, protocol)) is member
    assert pickle.loads(pickle.dumps(one.GpuState)) is one.GpuState


def test_two_facades_get_isolated_classes_and_members(facades):
    one, two = facades
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)
        assert getattr(one, name) is getattr(one, name)
    assert one.GpuState.GPU_WORK is not two.GpuState.GPU_WORK
    assert not isinstance(one.GpuState.GPU_WORK, two.GpuState)
    assert one.GpuState.GPU_WORK.value == two.GpuState.GPU_WORK.value
    assert one.GpuBundleState._VALID_OOM_STATES == two.GpuBundleState._VALID_OOM_STATES
    assert one.bundle_coordinator is not two.bundle_coordinator
    one._gpu_states["gpu0"] = one.GpuBundleState("gpu0")
    assert two._gpu_states == {}


def _methods(cls):
    for value in vars(cls).values():
        if isinstance(value, types.FunctionType):
            yield value
        elif isinstance(value, (staticmethod, classmethod)):
            yield value.__func__
        elif isinstance(value, property):
            yield from (f for f in (value.fget, value.fset, value.fdel) if f)


def test_methods_resolve_facade_globals_at_call_time_and_rebind_per_facade(facades, monkeypatch):
    one, two = facades
    for facade in facades:
        for cls in (facade.GpuBundleState, facade.BundleCoordinator):
            functions = list(_methods(cls))
            assert functions
            for function in functions:
                assert function.__globals__ is vars(facade)
                assert function.__module__ == facade.__name__
    assert isinstance(vars(one.GpuBundleState)["from_dict"], classmethod)
    # A facade global replaced after bind is seen by the next call; the other facade is not.
    monkeypatch.setattr(one, "time", types.SimpleNamespace(time=lambda: 111.0))
    monkeypatch.setattr(two, "time", types.SimpleNamespace(time=lambda: 222.0))
    gs_one, gs_two = one.GpuBundleState("gpu0"), two.GpuBundleState("gpu0")
    gs_one.touch_activity()
    gs_two.touch_activity()
    assert (gs_one.last_activity_at, gs_two.last_activity_at) == (111.0, 222.0)
    synced = []
    monkeypatch.setattr(one, "_sync_legacy_gpu_state", lambda gpu_id, state: synced.append((gpu_id, state)))
    gs_one.set_loading(True, source="test")
    assert synced == [("gpu0", gs_one)] and gs_one.events[-1]["ts"] == 111.0
    gs_two.set_loading(True, source="test")
    assert synced == [("gpu0", gs_one)]
    # A facade-global lookup inside a classmethod and inside a coroutine method.
    monkeypatch.setattr(one, "BUNDLE_STATE_SCHEMA_VERSION", 99)
    restored = one.GpuBundleState.from_dict(gs_one.to_dict())
    assert type(restored) is one.GpuBundleState and restored.gpu_id == "gpu0"
    sentinel = object()
    monkeypatch.setattr(
        one,
        "asyncio",
        types.SimpleNamespace(
            Lock=asyncio.Lock,
            get_event_loop=lambda: types.SimpleNamespace(create_future=lambda: sentinel),
        ),
    )
    coordinator = one.BundleCoordinator()
    future = _run(coordinator.register_job("job", "bundle", ["g0", "g1"], 5, {}))
    assert future is sentinel
    assert coordinator._active_jobs["job"]["created_at"] == 111.0
    assert coordinator._active_jobs["job"]["gpu_ids"] == {"g0", "g1"}


def test_gpu_bundle_state_behavior_matches_baseline(facades):
    for facade in facades:
        base_ns = _baseline_facade(facade.__name__)
        base_ns["time"] = facade.time = types.SimpleNamespace(time=lambda: 1000.0)
        base_ns["_sync_legacy_gpu_state"] = lambda *a: None
        facade._sync_legacy_gpu_state = lambda *a: None
        results = []
        for cls in (base_ns["GpuBundleState"], facade.GpuBundleState):
            gs = cls("gpu0")
            gs.set_loading(True, source="t")
            gs.set_evicting(True, source="t")
            gs.set_draining(True, source="t")
            gs.set_oom_state("bogus", source="t", reason="r")
            gs.enqueue("b1", 1, {"x": 1})
            gs.enqueue("b2", 9, {"x": 2})
            gs.start_drain()
            gs.request_started()
            gs.request_finished()
            drained = _run(gs.wait_drain())
            results.append((gs.to_dict(), gs.events, drained, gs.dequeue_next()["bundle_name"]))
        assert results[0] == results[1]


def test_wait_drain_default_snapshots_bind_time_value(facades, monkeypatch):
    import inspect

    bundle_model = _bundle_model_module()
    for facade in facades:
        assert inspect.signature(facade.GpuBundleState.wait_drain).parameters["timeout"].default == 60
    monkeypatch.setattr(facades[0], "DRAIN_TIMEOUT", 7)
    assert inspect.signature(facades[0].GpuBundleState.wait_drain).parameters["timeout"].default == 60
    rebound = bundle_model.bind({**vars(facades[0]), "DRAIN_TIMEOUT": 7})
    assert inspect.signature(rebound[1].wait_drain).parameters["timeout"].default == 7
    assert inspect.signature(facades[1].GpuBundleState.wait_drain).parameters["timeout"].default == 60


def test_bind_rejects_non_dict_and_does_not_touch_sys_modules(facades):
    bundle_model = _bundle_model_module()
    with pytest.raises(TypeError):
        bundle_model.bind(object())
    before = set(sys.modules)
    result = bundle_model.bind(vars(facades[0]))
    assert len(result) == len(NAMES) == 3
    assert set(sys.modules) == before
    assert result[0] is not facades[0].GpuState
