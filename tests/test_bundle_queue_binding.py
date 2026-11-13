"""Binding proof for the extracted bundle queues and PriorityTier Enum."""
from __future__ import annotations

import ast
import asyncio
import contextlib
import enum
import importlib.util
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
MODULE = SCRIPTS / "gpu_manager" / "bundle_queue.py"
NAMES = (
    "BundleRequestQueue", "_get_bundle_request_queue", "PriorityTier", "PortProxy",
    "_start_all_port_proxies", "_stop_all_port_proxies", "_BundleRequestQueue",
    "_get_bundle_queue", "_check_bundle_services_healthy", "priority_to_tier",
)

# Frozen from the pre-extraction facade at tranche-3 base a89f648.
PRIORITY_TIER_SOURCE = '''
class PriorityTier(Enum):
    HIGH = 1       # Interactive requests - guaranteed 1 slot
    MEDIUM = 2     # Normal batch work
    LOW = 3        # Background/preemptible work
'''


def _deny(*args, **kwargs):
    raise AssertionError("socket/process creation forbidden in binding proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    result = []
    for name in ("bundle_queue_facade_one", "bundle_queue_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def _baseline_enum(name):
    namespace = {"Enum": enum.Enum, "__name__": name}
    exec(compile(PRIORITY_TIER_SOURCE, "<frozen-priority-tier>", "exec"), namespace)
    return namespace["PriorityTier"]


def _result(call):
    try:
        return ("ok", call())
    except Exception as exc:  # compare the baseline failure contract
        return ("raised", type(exc).__name__, str(exc))


def _run(coro):
    try:
        return coro.send(None)
    except StopIteration as done:
        return done.value


def test_facade_binds_all_names_and_preserves_enum_identity(facades):
    tree = ast.parse(FACADE.read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert not {"PriorityTier", "BundleRequestQueue", "_BundleRequestQueue", "PortProxy"} & classes
    assert not {"_get_bundle_request_queue", "_get_bundle_queue", "priority_to_tier"} & functions
    module_tree = ast.parse(MODULE.read_text())
    assert tuple(ast.literal_eval(next(n.value for n in module_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAMES" for t in n.targets)))) == NAMES
    for facade in facades:
        base = _baseline_enum(facade.__name__)
        cand = facade.PriorityTier
        assert cand.__name__ == base.__name__ == "PriorityTier"
        assert cand.__qualname__ == base.__qualname__ == "PriorityTier"
        assert cand.__module__ == facade.__name__
        assert [(m.name, m.value) for m in cand] == [("HIGH", 1), ("MEDIUM", 2), ("LOW", 3)]
        assert [(m.name, m.value) for m in cand] == [(m.name, m.value) for m in base]
        assert repr(cand) == repr(base)
        assert [repr(m) for m in cand] == [repr(m) for m in base]
        assert [str(m) for m in cand] == [str(m) for m in base]
        for member in cand:
            assert cand[member.name] is member
            assert cand(member.value) is member
            assert facade.PRIORITY_TIER_MAP.keys() >= {member}
            assert (member.__reduce_ex__(2)[0] is cand, member.__reduce_ex__(2)[1:]) == (
                base[member.name].__reduce_ex__(2)[0] is base,
                base[member.name].__reduce_ex__(2)[1:],
            )
        assert _result(lambda: cand("bad")) == _result(lambda: base("bad"))


def test_distinct_queue_classes_registry_isolation_and_timeout_snapshot(facades):
    one, two = facades
    assert one.BundleRequestQueue is not one._BundleRequestQueue
    assert two.BundleRequestQueue is not two._BundleRequestQueue
    assert one.BundleRequestQueue.__module__ == one._BundleRequestQueue.__module__ == one.__name__
    assert one._get_bundle_request_queue("x") is one._get_bundle_request_queue("x")
    assert one._get_bundle_queue("x") is one._get_bundle_queue("x")
    assert isinstance(one._get_bundle_request_queue("x"), one.BundleRequestQueue)
    assert isinstance(one._get_bundle_queue("x"), one._BundleRequestQueue)
    assert one._get_bundle_request_queue("x") is not two._get_bundle_request_queue("x")
    assert one._get_bundle_queue("x") is not two._get_bundle_queue("x")
    request_a, request_b = one.BundleRequestQueue("a"), one.BundleRequestQueue("b")
    assert request_a._pending is not request_b._pending
    assert request_a._lock is not request_b._lock
    a, b = one._BundleRequestQueue("a"), one._BundleRequestQueue("b")
    a.mark_served()
    a.signal_ready()
    assert a.to_dict()["total_served"] == 1 and a.is_ready
    assert b.to_dict()["total_served"] == 0 and not b.is_ready
    assert a.to_dict() is not b.to_dict()
    assert one._BundleRequestQueue.wait_for_ready.__defaults__ == (300,)
    one.BUNDLE_PROXY_WAIT_TIMEOUT = 9
    assert one._BundleRequestQueue.wait_for_ready.__defaults__ == (300,)
    from gpu_manager import bundle_queue
    custom_globals = {"__name__": "custom_queue_facade", "BUNDLE_PROXY_WAIT_TIMEOUT": 9}
    bound = dict(zip(NAMES, bundle_queue.bind(custom_globals)))
    assert bound["_BundleRequestQueue"].wait_for_ready.__defaults__ == (9,)


def test_bound_functions_resolve_patched_facade_dependencies(facades, monkeypatch):
    facade = facades[0]
    calls = []
    class Event:
        def __init__(self): calls.append("event")
        def is_set(self): return False
        def clear(self): pass
        def set(self): pass
        async def wait(self): calls.append("wait")
    monkeypatch.setattr(facade.asyncio, "Event", Event)
    q = facade._BundleRequestQueue("patched")
    assert calls == ["event"]
    facade._services_config = {"services": {"patched-service": {"port": 4567}}}
    health_calls = []
    monkeypatch.setattr(socket, "create_connection", lambda address, timeout: (health_calls.append((address, timeout)), contextlib.nullcontext())[1])
    assert facade._check_bundle_services_healthy({"services": ["patched-service"]}) is True
    assert health_calls == [(('localhost', 4567), 2)]
    class PriorityClass:
        value = "interactive"
    monkeypatch.setattr(facade, "priority_envelope", lambda *args, **kwargs: types.SimpleNamespace(priority_class=PriorityClass()))
    monkeypatch.setattr(facade, "PriorityConvention", types.SimpleNamespace(GENERATION_HIGHER_IS_URGENT="patched"))
    assert facade.priority_to_tier(17) is facade.PriorityTier.HIGH
    assert facade.priority_to_tier.__globals__ is facade.__dict__


def test_port_proxy_uses_fake_server_and_connection(facades, monkeypatch):
    facade = facades[0]
    calls = []
    class Server:
        def close(self): calls.append("close")
        async def wait_closed(self): calls.append("wait_closed")
        def is_serving(self): return True
    class Reader:
        async def read(self, size): return b""
    class Writer:
        def get_extra_info(self, name): return ("peer", 1)
        def write(self, data): pass
        async def drain(self): pass
        def close(self): pass
    handlers = []
    async def start_server(handler, host, port):
        handlers.append(handler)
        calls.append(("start", host, port, callable(handler)))
        return Server()
    async def open_connection(host, port):
        calls.append(("connect", host, port))
        return Reader(), Writer()
    monkeypatch.setattr(facade.asyncio, "start_server", start_server)
    monkeypatch.setattr(facade.asyncio, "open_connection", open_connection)
    async def fake_wait_for(coroutine, timeout):
        return await coroutine
    monkeypatch.setattr(facade.asyncio, "wait_for", fake_wait_for)
    async def fake_gather(*coroutines):
        for coroutine in coroutines:
            _run(coroutine)
    monkeypatch.setattr(facade.asyncio, "gather", fake_gather)
    proxy = facade.PortProxy("test", 1234, "backend", 4321, facade.logger)
    _run(proxy.start())
    assert proxy.is_running
    _run(handlers[0](Reader(), Writer()))
    _run(proxy.stop())
    assert calls == [("start", "0.0.0.0", 1234, True), ("connect", "backend", 4321), "close", "wait_closed"]
    assert facade._check_bundle_services_healthy.__globals__ is facade.__dict__


def test_priority_tier_pickle_failure_matches_unregistered_baseline(facades):
    facade = facades[0]
    base = _baseline_enum(facade.__name__)
    assert facade.__name__ not in sys.modules
    for name in base.__members__:
        expected = _result(lambda: pickle.dumps(base[name]))
        assert expected[0:2] == ("raised", "PicklingError")
        assert _result(lambda: pickle.dumps(facade.PriorityTier[name])) == expected
