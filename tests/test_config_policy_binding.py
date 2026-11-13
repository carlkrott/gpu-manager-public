"""Binding proof for the extracted config policy (scripts/gpu_manager/config_policy.py).

Only synthetic config dicts and tmp-file restart-policy evidence are used:
no systemd, systemctl, socket or subprocess is ever touched, and the guards
below fail the test if a facade load tries to create one.
"""
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
    "_validate_linked_bundle_graph",
    "_validate_config_policy",
    "_validate_config_for_observability",
    "_expected_restart_policy",
    "_systemd_restart_policy_drift",
    "_oom_damping_shape",
    "_count_pinned_compute_idles",
    "_validate_idle_policy",
)


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during import proof")


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

    loaded = []
    with monkeypatch.context() as guard:
        guard.setattr(socket, "socket", _deny)
        guard.setattr(subprocess, "Popen", _deny)
        guard.setattr(os, "system", _deny)
        for name in ("config_policy_facade_one", "config_policy_facade_two"):
            spec = importlib.util.spec_from_file_location(name, FACADE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
    return loaded


@pytest.fixture
def no_subprocess(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(subprocess, "run", _deny)
    monkeypatch.setattr(os, "system", _deny)


def _cat_evidence(path: Path, unit: str, body: str) -> Path:
    path.write_text(f"=== {unit}.service (systemctl cat) ===\n{body}\n=== other.service (systemctl cat) ===\nRestartSec=999\n")
    return path


def test_functions_are_bound_to_their_own_facade(facades):
    one, two = facades
    for facade in facades:
        for name in NAMES:
            assert getattr(facade, name).__globals__ is vars(facade)
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)
        assert getattr(one, name).__module__ == one.__name__
        assert getattr(one, name).__qualname__ == name


def test_signatures_and_docstrings_are_preserved(facades):
    one = facades[0]
    assert list(inspect.signature(one._validate_idle_policy).parameters) == [
        "idle_map", "compute_gpu_ids", "allow_count", "thermal_provenance",
    ]
    assert inspect.signature(one._validate_idle_policy).parameters["allow_count"].default == 1
    assert inspect.signature(one._expected_restart_policy).parameters["unit"].default == "gemma-cpu"
    assert inspect.signature(one._oom_damping_shape).parameters["unit"].default == "gemma-cpu"
    assert list(inspect.signature(one._systemd_restart_policy_drift).parameters) == [
        "unit", "dropin_dir", "systemctl_cat_path",
    ]
    assert one._validate_linked_bundle_graph.__doc__.startswith("Fail-closed validation")
    assert one._validate_config_for_observability.__doc__.startswith("Combine graph")


def test_validators_on_synthetic_configs(facades):
    one = facades[0]
    assert one._validate_linked_bundle_graph("junk") == ([], [])
    assert one._validate_linked_bundle_graph({"bundles": {}}) == ([], [])
    assert one._validate_config_policy([]) == ([], [])

    errors, warnings = one._validate_config_policy({
        "gpu_devices": {"g0": {"vram_target_mib": 10, "vram_ceiling_mib": 5}},
        "scheduling": {
            "scheduler_owns_load": True,
            "pinned_service": "p",
            "idle_service": "i",
        },
        "generation_templates": {"t": {"evict_first": ["x"]}},
    })
    assert any("evict_first is forbidden" in e for e in errors)
    assert any("vram_target_mib=10 exceeds vram_ceiling_mib=5" in e for e in errors)
    assert any("pinned_service > " in w for w in warnings)

    errors, _ = one._validate_config_policy({
        "gpu_devices": {"g1": {"vram_target_mib": "abc"}},
        "scheduling": {"idle_job": {"enabled": True}, "idle_service": "x"},
    })
    assert any("must be an integer" in e for e in errors)
    assert any("idle_job.enabled=true conflicts" in e for e in errors)

    errors, _ = one._validate_linked_bundle_graph({
        "bundles": {"a": {"linked_to": ["a"]}},
    })
    assert errors  # self-link is graph corruption


def test_observability_combines_and_reads_facade_authority(facades, monkeypatch):
    one, two = facades
    config = {"scheduling": {"idle_job": {"enabled": True}, "idle_service": "x"}}
    base_errors, _ = one._validate_config_for_observability(config)
    assert any("idle_job.enabled=true conflicts" in e for e in base_errors)

    # Rebinding a facade global AFTER import must be seen by the bound function.
    monkeypatch.setattr(one, "checked_runtime_authority_errors", lambda cfg: ["authority:one"])
    rebound, _ = one._validate_config_for_observability(config)
    assert rebound == [*base_errors, "authority:one"]
    # The second, unregistered facade keeps its own binding.
    other, _ = two._validate_config_for_observability(config)
    assert other == base_errors
    assert "authority:one" not in other


def test_restart_policy_tables_and_damping_follow_facade_globals(facades, monkeypatch):
    one, two = facades
    assert one._expected_restart_policy() == {
        "RestartSec": 120, "StartLimitIntervalSec": 900, "StartLimitBurst": 5,
    }
    assert one._expected_restart_policy("llm-primary")["RestartSec"] == 10
    assert one._expected_restart_policy("no-such-unit") == {}
    # Mutating a returned policy never leaks into the table.
    one._expected_restart_policy("gpu-manager")["RestartSec"] = 1
    assert one._expected_restart_policy("gpu-manager")["RestartSec"] == 10

    shape = one._oom_damping_shape("gemma-cpu")
    assert shape["watchdog_observability"] == "covers_amplification"
    assert shape["expected_respawn_window_s"] == 120
    assert shape["expected_damping_factor"] == 1
    assert one._oom_damping_shape("no-such-unit")["watchdog_observability"] == "unknown_policy"

    monkeypatch.setattr(one, "OOM_WATCHDOG_POLL_INTERVAL_S", 1000)
    assert one._oom_damping_shape("gemma-cpu")["watchdog_observability"] == "unknown_policy"
    assert one._oom_damping_shape("gemma-cpu")["watchdog_interval_s"] == 1000
    assert two._oom_damping_shape("gemma-cpu")["watchdog_observability"] == "covers_amplification"

    monkeypatch.setattr(one, "_expected_restart_policy", lambda unit="x": {"RestartSec": 5000, "StartLimitBurst": 0})
    patched = one._oom_damping_shape()
    assert patched["expected_respawn_window_s"] == 5000
    assert patched["expected_damping_factor"] == 0
    assert two._oom_damping_shape()["expected_respawn_window_s"] == 120


def test_systemd_drift_reads_only_supplied_evidence(facades, tmp_path, monkeypatch, no_subprocess):
    one, two = facades
    dropins = tmp_path / "dropins"
    dropins.mkdir()
    ok_body = "RestartSec=120\nStartLimitIntervalSec=900s\nStartLimitBurst=5"
    cat = _cat_evidence(tmp_path / "cat.txt", "gemma-cpu", ok_body)
    assert one._systemd_restart_policy_drift("gemma-cpu", dropins, cat) == {}
    assert one._systemd_restart_policy_drift("no-such-unit", dropins, cat) == {}

    (dropins / "10-override.conf").write_text("[Service]\nRestartSec=30\nStartLimitBurst=oops\n# c\n")
    drift = one._systemd_restart_policy_drift("gemma-cpu", dropins, cat)
    assert drift == {"RestartSec": ("30", 120), "StartLimitBurst": ("oops", 5)}

    missing = _cat_evidence(tmp_path / "missing.txt", "gemma-cpu", "RestartSec=120")
    assert set(one._systemd_restart_policy_drift("gemma-cpu", tmp_path / "none", missing)) == {
        "StartLimitIntervalSec", "StartLimitBurst",
    }
    with pytest.raises(ValueError, match="no section for llm-primary"):
        one._systemd_restart_policy_drift("llm-primary", dropins, cat)

    # The drift check resolves _expected_restart_policy from the facade it was bound to.
    monkeypatch.setattr(one, "_expected_restart_policy", lambda unit="x": {"RestartSec": 7})
    assert one._systemd_restart_policy_drift("gemma-cpu", tmp_path / "none", cat) == {
        "RestartSec": ("120", 7),
    }
    assert two._systemd_restart_policy_drift("gemma-cpu", tmp_path / "none", cat) == {}


def test_idle_policy_and_count_use_facade_measurements(facades, monkeypatch):
    one, two = facades
    idle = {"0": "svc-a", 1: "svc-b", "2": "", "9": "other"}
    assert one._count_pinned_compute_idles(idle, ["0", 1, "2"]) == 2
    assert one._count_pinned_compute_idles("junk", ["0"]) == 0

    assert one._validate_idle_policy("junk", ["0"]) == (False, "idle_policy_input_invalid")
    assert one._validate_idle_policy({}, "0") == (False, "idle_policy_input_invalid")
    assert one._validate_idle_policy({}, ["0"], allow_count=True) == (False, "allow_count_invalid")
    assert one._validate_idle_policy({"0": "a"}, ["0"]) == (True, "")
    assert one._validate_idle_policy(idle, ["0", "1"], allow_count=0) == (False, "compute_idles_not_allowed")
    assert one._validate_idle_policy(idle, ["0", "1"]) == (False, "two_compute_idles_without_thermal_provenance")
    good = {"junction_p99_c": 80, "junction_cooled_c": 60}
    assert one._validate_idle_policy(idle, ["0", "1"], thermal_provenance=good) == (
        True, "two_compute_idles_with_thermal_provenance",
    )
    hot = {"junction_p99_c": 111, "junction_cooled_c": 60}
    assert one._validate_idle_policy(idle, ["0", "1"], thermal_provenance=hot) == (
        False, "junction_at_or_above_crit",
    )
    assert one._validate_idle_policy(idle, ["0", "1"], thermal_provenance={"junction_p99_c": 80}) == (
        False, "thermal_provenance_incomplete",
    )

    # Memory-policy ownership stays in the facade: patching its measurement helper is seen here.
    monkeypatch.setattr(one, "_positive_measurement", lambda value: None)
    assert one._validate_idle_policy(idle, ["0", "1"], thermal_provenance=good) == (
        False, "thermal_provenance_incomplete",
    )
    assert two._validate_idle_policy(idle, ["0", "1"], thermal_provenance=good)[0] is True

    monkeypatch.setattr(one, "_count_pinned_compute_idles", lambda m, c: 0)
    assert one._validate_idle_policy(idle, ["0", "1"], allow_count=0) == (True, "")
    assert two._validate_idle_policy(idle, ["0", "1"], allow_count=0)[0] is False


def test_bind_requires_a_globals_dict():
    sys.path.insert(0, str(SCRIPTS))
    try:
        from gpu_manager import config_policy
    finally:
        sys.path.remove(str(SCRIPTS))
    with pytest.raises(TypeError):
        config_policy.bind(object())
    assert config_policy.NAMES == NAMES
    bound = config_policy.bind({})
    assert len(bound) == len(NAMES)
    assert all(fn.__globals__ == {} for fn in bound)
