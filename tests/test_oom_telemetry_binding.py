"""Binding contract for extracted facade-bound OOM telemetry helpers."""
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
    "_junction_crit_aborted", "_parse_psi_avg10", "_psi_full_stall_pct",
    "_parse_cgroup_memory_events", "_cgroup_events_oom", "_journal_timestamp",
    "_parse_prior_boot_thrash", "_prior_boot_oom_rate_per_minute",
    "_prior_boot_is_thrash", "_oom_counters_snapshot",
    "_watchdog_should_escalate", "_should_admit_generation",
    "_watchdog_ladder_snapshot",
)


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def oom_telemetry(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from gpu_manager import oom_telemetry

    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    import aiohttp.web  # noqa: F401 -- import before forbidding socket creation
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(subprocess, "run", _deny)
    monkeypatch.setattr(os, "system", _deny)

    facades = []
    for name in ("oom_telemetry_facade_one", "oom_telemetry_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        facade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(facade)
        facades.append(facade)
    return oom_telemetry, facades


def test_helpers_are_bound_to_independent_facade_globals(oom_telemetry):
    module, facades = oom_telemetry
    one, two = facades
    assert module.NAMES == NAMES
    assert len(module.bind({"_TASK8A_PSI_FULL_ABORT_PCT": 5.0})) == len(NAMES)
    for facade in facades:
        for name in NAMES:
            function = getattr(facade, name)
            template = getattr(module, name)
            assert function is not template
            assert function.__globals__ is vars(facade)
            assert function.__module__ == facade.__name__
            assert function.__qualname__ == template.__qualname__
            assert function.__doc__ == template.__doc__
            assert function.__annotations__ == template.__annotations__
            assert inspect.signature(function) == inspect.signature(template)
            assert inspect.iscoroutinefunction(function) is inspect.iscoroutinefunction(template)
            assert function.__code__ is template.__code__
        assert facade.__name__ not in sys.modules
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)


def test_global_reads_are_late_bound_and_facade_local(oom_telemetry):
    _, (one, two) = oom_telemetry
    one.is_junction_critical = lambda row: True
    two.is_junction_critical = lambda row: False
    per_gpu = {"0": {"temp_junction_c": 80, "temp_junction_crit_c": 100}}
    assert one._junction_crit_aborted(per_gpu) is True
    assert two._junction_crit_aborted(per_gpu) is False
    one.is_junction_critical = lambda row: False
    assert one._junction_crit_aborted(per_gpu) is False
    assert two._junction_crit_aborted(per_gpu) is False


def test_psi_threshold_default_is_facade_snapshot(oom_telemetry):
    _, (one, two) = oom_telemetry
    assert inspect.signature(one._psi_full_stall_pct).parameters["abort_threshold_pct"].default == 5.0
    assert inspect.signature(two._psi_full_stall_pct).parameters["abort_threshold_pct"].default == 5.0
    one._TASK8A_PSI_FULL_ABORT_PCT = 12.5
    assert inspect.signature(one._psi_full_stall_pct).parameters["abort_threshold_pct"].default == 5.0
    assert inspect.signature(two._psi_full_stall_pct).parameters["abort_threshold_pct"].default == 5.0
    rebound = dict(vars(one), _TASK8A_PSI_FULL_ABORT_PCT=12.5)
    module, _ = oom_telemetry
    bound = dict(zip(module.NAMES, one._bind_oom_telemetry(rebound)))
    assert inspect.signature(bound["_psi_full_stall_pct"]).parameters["abort_threshold_pct"].default == 12.5


def test_binding_does_not_register_facades_or_change_coroutine_status(oom_telemetry):
    module, facades = oom_telemetry
    assert all(not inspect.iscoroutinefunction(getattr(facades[0], n)) for n in NAMES)
    assert facades[0].__name__ not in sys.modules
    before = set(sys.modules)
    module.bind({"__name__": "unregistered_facade", "_TASK8A_PSI_FULL_ABORT_PCT": 5.0})
    assert "unregistered_facade" not in sys.modules
    assert facades[0].__name__ not in sys.modules
    assert before <= set(sys.modules)
