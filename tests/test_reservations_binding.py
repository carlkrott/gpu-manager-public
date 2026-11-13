"""Hermetic binding proof for facade-owned GPU reservation helpers."""
from __future__ import annotations

import importlib.util
import inspect
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"
NAMES = (
    "_phase5_reservation_config", "_derive_group_reservations",
    "_validate_reservation_config", "_record_reservation_change",
    "reserve_gpus", "release_gpus", "expire_reservations",
    "_reservation_persistence_path", "_persist_reservations",
    "_record_ingress_blocked_by_reservation",
)


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def bound_facades(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from gpu_manager import reservations

    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    import aiohttp.web  # noqa: F401 -- load socket subclasses before the guard
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(subprocess, "run", _deny)
    monkeypatch.setattr(os, "system", _deny)

    facades = []
    for name in ("reservations_facade_one", "reservations_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        facade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(facade)
        facades.append(facade)
    return reservations, facades


def test_functions_bind_to_independent_facade_globals_and_preserve_metadata(bound_facades):
    module, (one, two) = bound_facades
    assert module.NAMES == NAMES
    assert len(module.bind({})) == len(NAMES)
    for facade in (one, two):
        for name in NAMES:
            function, template = getattr(facade, name), getattr(module, name)
            assert function is not template
            assert function.__globals__ is vars(facade)
            assert function.__module__ == facade.__name__
            assert function.__qualname__ == template.__qualname__
            assert function.__doc__ == template.__doc__
            assert function.__annotations__ == template.__annotations__
            assert inspect.signature(function) == inspect.signature(template)
            assert function.__code__ is template.__code__
        assert facade.__name__ not in sys.modules
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)


def test_dependency_reads_are_late_bound_in_each_facade(bound_facades, monkeypatch):
    _, (one, two) = bound_facades
    monkeypatch.setattr(one, "_service_is_publicly_qualified", lambda template: True)
    monkeypatch.setattr(two, "_service_is_publicly_qualified", lambda template: False)
    monkeypatch.setattr(one, "_service_group_priority", lambda *args: 17)
    monkeypatch.setattr(two, "_service_group_priority", lambda *args: 29)
    template = {"routing_group": "g", "bundle": None}
    first = one._derive_group_reservations({}, {"svc": template})
    second = two._derive_group_reservations({}, {"svc": template})
    assert first["g"]["priority"] == 17
    assert "g" not in second
    one._services_config = {"generation_templates": {"t": {"evict_first": True}}}
    two._services_config = {"generation_templates": {"t": {"evict_first": True}}}
    monkeypatch.setattr(one, "_phase3_scheduling_config", lambda: {"scheduler_owns_load": True})
    monkeypatch.setattr(two, "_phase3_scheduling_config", lambda: {"scheduler_owns_load": False})
    # _validate_reservation_config resolves this dependency at call time.
    one_warnings = one._validate_reservation_config({})
    two_warnings = two._validate_reservation_config({})
    assert any("scheduler_owns_load=true" in warning for warning in one_warnings)
    assert not any("scheduler_owns_load=true" in warning for warning in two_warnings)


def test_reservation_mutations_stay_in_facade_state(bound_facades, monkeypatch):
    _, (one, two) = bound_facades
    monkeypatch.setattr(one, "_persist_reservations", lambda: {"ok": True})
    monkeypatch.setattr(two, "_persist_reservations", lambda: {"ok": True})
    ok, details = one.reserve_gpus("g", ["0"], token="tok", hold_s=60)
    assert ok and details["token"] == "tok"
    assert vars(one)["_gpu_reservations"]["0"]["group"] == "g"
    assert vars(two)["_gpu_reservations"] == {}
    one._record_ingress_blocked_by_reservation("g")
    assert vars(one)["_ingress_blocked_by_reservation"] == 1
    assert vars(one)["_ingress_blocked_by_reservation_by_group"] == {"g": 1}
    assert vars(two)["_ingress_blocked_by_reservation"] == 0
    released, result = one.release_gpus("g", "tok")
    assert released and result["released"] == ["0"]
    assert vars(one)["_gpu_reservations"] == {}
    assert vars(two)["_gpu_reservations"] == {}


def test_default_and_path_resolution_are_facade_local(bound_facades):
    _, (one, two) = bound_facades
    assert inspect.signature(one.reserve_gpus).parameters["hold_s"].default == 120
    one._services_config = {"gpu_reservations_path": "/tmp/one-reservations.json"}
    two._services_config = {"gpu_reservations_path": "/tmp/two-reservations.json"}
    assert one._reservation_persistence_path() == "/tmp/one-reservations.json"
    assert two._reservation_persistence_path() == "/tmp/two-reservations.json"


def test_binding_rejects_non_dict_and_templates_do_not_resolve_facade_names(bound_facades):
    module, _ = bound_facades
    with pytest.raises(TypeError):
        module.bind(object())
    with pytest.raises(NameError):
        module._record_ingress_blocked_by_reservation("g")
