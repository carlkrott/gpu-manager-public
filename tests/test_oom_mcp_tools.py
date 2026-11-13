"""Host tests for the four Phase 5 OOM MCP tools.

Covers ``oom_history``, ``oom_recovery_status``, ``oom_recovery_force_reset``
and ``requeue_oom_jobs`` through ``_mcp_call_tool`` in ``scripts/gpu-manager.py``.

No Redis, no HTTP and no GPU is touched: the Redis-backed helpers, the HTTP
proxy and the per-GPU state objects are replaced by in-memory fakes on a
facade loaded under the same hermetic guards as ``test_oom_retry_binding.py``.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during import proof")


@pytest.fixture
def facade(tmp_path, monkeypatch):
    """Load one fresh facade under hermetic guards and stub the lazy singletons."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    # aiohttp subclasses socket.socket at import time; load it before the guard.
    import aiohttp.web  # noqa: F401

    with monkeypatch.context() as guard:
        guard.setattr(socket, "socket", _deny)
        guard.setattr(subprocess, "Popen", _deny)
        guard.setattr(os, "system", _deny)
        spec = importlib.util.spec_from_file_location("oom_mcp_tools_facade", FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    module._oom_redis_pool = None
    # _mcp_call_tool lazily builds scheduler/vram and loads services config;
    # pre-seed them so dispatch never reaches those constructors.
    monkeypatch.setattr(module, "scheduler", object())
    monkeypatch.setattr(module, "vram", object())
    monkeypatch.setattr(module, "_services_config", {"services": {"stub": {}}})
    monkeypatch.setattr(module, "_gpu_states", {})
    monkeypatch.setattr(module, "_oom_recovery_in_progress", {})
    monkeypatch.setattr(module, "_oom_force_reset_last_call_ts", {})
    return module


def _call(facade, name, arguments=None):
    return asyncio.run(facade._mcp_call_tool(name, arguments or {}))


# ---------------------------------------------------------------- oom_history


def _ring(facade, monkeypatch, events):
    """Replace the ring reader with one that mirrors the real contract.

    The real ``_recent_oom_events(limit)`` returns the newest ``limit`` entries
    of a bounded ring (oom_detection.py: ``list(ring)[-limit:]``).
    """
    ring = list(events)
    calls = []

    def fake_recent(limit=100):
        calls.append(limit)
        return list(ring)[-limit:]

    monkeypatch.setattr(facade, "_recent_oom_events", fake_recent)
    return ring, calls


def _events(n):
    return [
        {"ts": 1000.0 + i, "gpu_id": "GPU0" if i % 2 == 0 else "GPU1", "seq": i}
        for i in range(n)
    ]


def test_oom_history_success_returns_schema_and_newest_first_window(facade, monkeypatch):
    ring, _ = _ring(facade, monkeypatch, _events(5))
    result = _call(facade, "oom_history", {"limit": 3})
    assert result["schema_version"] == 1
    assert [e["seq"] for e in result["events"]] == [2, 3, 4]
    # count is the number of matching events before the limit is applied.
    assert result["count"] == 5
    assert result["filter"] == {"gpu_id": None, "since": None, "limit": 3}


def test_oom_history_defaults_to_fifty(facade, monkeypatch):
    _ring(facade, monkeypatch, _events(7))
    result = _call(facade, "oom_history")
    assert result["filter"]["limit"] == facade.OOM_HISTORY_DEFAULT_LIMIT == 50
    assert result["count"] == 7


def test_oom_history_clamps_limit_to_max(facade, monkeypatch):
    _ring(facade, monkeypatch, _events(3))
    result = _call(facade, "oom_history", {"limit": 5000})
    assert result["filter"]["limit"] == facade.OOM_HISTORY_MAX_LIMIT == 1000


def test_oom_history_limit_zero_returns_no_events(facade, monkeypatch):
    _ring(facade, monkeypatch, _events(4))
    result = _call(facade, "oom_history", {"limit": 0})
    assert result["events"] == []
    assert result["count"] == 4


def test_oom_history_since_filter(facade, monkeypatch):
    _ring(facade, monkeypatch, _events(6))
    result = _call(facade, "oom_history", {"since": 1003.0, "limit": 50})
    assert [e["seq"] for e in result["events"]] == [3, 4, 5]
    assert result["count"] == 3
    assert result["filter"]["since"] == 1003.0


def test_oom_history_gpu_filter_returns_newest_matching_events(facade, monkeypatch):
    # GPU0 events are seq 0,2,4,6,8; GPU1 events are seq 1,3,5,7,9.
    # The three newest GPU0 events are 4, 6 and 8. Filtering must happen
    # before the limit window is applied, otherwise GPU1 traffic crowds
    # GPU0 history out of the answer.
    _ring(facade, monkeypatch, _events(10))
    result = _call(facade, "oom_history", {"gpu_id": "GPU0", "limit": 3})
    assert all(e["gpu_id"] == "GPU0" for e in result["events"])
    assert [e["seq"] for e in result["events"]] == [4, 6, 8]
    assert result["count"] == 5


def test_oom_history_is_read_only(facade, monkeypatch):
    ring, _ = _ring(facade, monkeypatch, _events(4))
    before = [dict(e) for e in ring]
    _call(facade, "oom_history", {"gpu_id": "GPU1", "since": 1000.0, "limit": 2})
    assert ring == before
    assert facade._oom_force_reset_last_call_ts == {}
    assert facade._oom_recovery_in_progress == {}


# ------------------------------------------------------- oom_recovery_status


def _http_recorder(facade, monkeypatch, reply=None):
    calls = []

    async def fake_http(method, path, body=None, **kwargs):
        calls.append((method, path, body))
        return reply if reply is not None else {"ok": True, "gpus": {}}

    monkeypatch.setattr(facade, "_mcp_http", fake_http)
    return calls


def test_oom_recovery_status_all_gpus_proxies_plain_get(facade, monkeypatch):
    calls = _http_recorder(facade, monkeypatch, reply={"ok": True, "gpus": {"GPU0": {}}})
    result = _call(facade, "oom_recovery_status")
    assert result == {"ok": True, "gpus": {"GPU0": {}}}
    assert calls == [("GET", "/oom/recovery/status", None)]


def test_oom_recovery_status_single_gpu_adds_query(facade, monkeypatch):
    calls = _http_recorder(facade, monkeypatch)
    _call(facade, "oom_recovery_status", {"gpu_id": "GPU1"})
    assert calls == [("GET", "/oom/recovery/status?gpu_id=GPU1", None)]


def test_oom_recovery_status_is_read_only(facade, monkeypatch):
    _http_recorder(facade, monkeypatch)
    sentinel = object()
    monkeypatch.setitem(facade._gpu_states, "GPU0", sentinel)
    monkeypatch.setitem(facade._oom_force_reset_last_call_ts, "GPU0", 1.0)
    _call(facade, "oom_recovery_status", {"gpu_id": "GPU0"})
    assert facade._gpu_states == {"GPU0": sentinel}
    assert facade._oom_force_reset_last_call_ts == {"GPU0": 1.0}


# ----------------------------------------------------- oom_recovery_force_reset


class FakeBundleState:
    def __init__(self, oom_state):
        self.oom_state = oom_state
        self.set_calls = []

    def set_oom_state(self, state, source=""):
        self.set_calls.append((state, source))
        self.oom_state = state


def test_force_reset_requires_gpu_id(facade):
    fake = FakeBundleState("oom_recovering")
    facade._gpu_states["GPU0"] = fake
    result = _call(facade, "oom_recovery_force_reset", {"reason": "no id"})
    assert result == {"schema_version": 1, "ok": False, "error": "gpu_id required"}
    assert fake.set_calls == []


def test_force_reset_success_clears_state_and_records_timestamp(facade, monkeypatch):
    fake = FakeBundleState("oom_recovering")
    facade._gpu_states["GPU0"] = fake
    facade._oom_recovery_in_progress["GPU0"] = 123.0
    before = time.time()
    result = _call(
        facade,
        "oom_recovery_force_reset",
        {"gpu_id": "GPU0", "reason": "operator escape hatch"},
    )
    assert result == {
        "schema_version": 1,
        "ok": True,
        "gpu_id": "GPU0",
        "old_state": "oom_recovering",
        "new_state": "idle",
        "reason": "operator escape hatch",
    }
    assert fake.set_calls == [("idle", "oom_recovery_force_reset (reason=operator escape hatch)")]
    assert "GPU0" not in facade._oom_recovery_in_progress
    assert facade._oom_force_reset_last_call_ts["GPU0"] >= before


def test_force_reset_second_call_inside_window_is_throttled(facade):
    fake = FakeBundleState("oom_recovering")
    facade._gpu_states["GPU0"] = fake
    first = _call(facade, "oom_recovery_force_reset", {"gpu_id": "GPU0"})
    assert first["ok"] is True
    fake.oom_state = "oom_recovering"
    second = _call(facade, "oom_recovery_force_reset", {"gpu_id": "GPU0"})
    assert second["ok"] is False
    assert second["throttled"] is True
    assert 0 < second["retry_after_s"] <= facade.OOM_FORCE_RESET_MIN_INTERVAL_S
    # Throttled call must not touch the state object again.
    assert len(fake.set_calls) == 1
    assert fake.oom_state == "oom_recovering"


def test_force_reset_unknown_gpu_is_refused_without_side_effects(facade):
    result = _call(facade, "oom_recovery_force_reset", {"gpu_id": "GPU9"})
    assert result == {"schema_version": 1, "ok": False, "error": "unknown gpu_id: GPU9"}
    assert "GPU9" not in facade._oom_force_reset_last_call_ts


def test_force_reset_window_expires(facade, monkeypatch):
    fake = FakeBundleState("oom_recovering")
    facade._gpu_states["GPU0"] = fake
    facade._oom_force_reset_last_call_ts["GPU0"] = time.time() - (
        facade.OOM_FORCE_RESET_MIN_INTERVAL_S + 1
    )
    result = _call(facade, "oom_recovery_force_reset", {"gpu_id": "GPU0"})
    assert result["ok"] is True


# ------------------------------------------------------------ requeue_oom_jobs


def _dead_letters():
    return [
        {"job_id": "job-a", "_cls": "eligible", "tracker_payload_hash": "h-a",
         "tracker_service_name": "svc", "tracker_routing_group": "rg",
         "tracker_artifact_refs": "[\"ref-a\"]"},
        {"job_id": "job-b", "_cls": "not_eligible", "tracker_payload_hash": "h-b",
         "tracker_service_name": "svc", "tracker_routing_group": "rg",
         "tracker_artifact_refs": ""},
        {"job_id": "job-c", "_cls": "eligible", "tracker_payload_hash": "h-c",
         "tracker_service_name": "svc", "tracker_routing_group": "rg",
         "tracker_artifact_refs": ""},
    ]


@pytest.fixture
def requeue_fakes(facade, monkeypatch):
    dead = _dead_letters()
    atomic_calls = []

    async def fake_get_dead_letters(limit=100):
        return [dict(e) for e in dead]

    async def fake_check(entries):
        return [
            {
                "job_id": e["job_id"],
                "classification": e["_cls"],
                "tracker_artifact_refs": e.get("tracker_artifact_refs", ""),
                "tracker_payload_hash": e.get("tracker_payload_hash"),
                "tracker_service_name": e.get("tracker_service_name"),
                "tracker_routing_group": e.get("tracker_routing_group"),
            }
            for e in entries
        ]

    async def fake_atomic(item, **kwargs):
        atomic_calls.append((item["job_id"], kwargs))
        return {"outcome": "retried", "orig_job_id": item["job_id"]}

    monkeypatch.setattr(facade, "_get_dead_letters", fake_get_dead_letters)
    monkeypatch.setattr(facade, "_oom_retry_check_redis_state", fake_check)
    monkeypatch.setattr(facade, "_atomic_retry_oom_job", fake_atomic)
    return atomic_calls


def test_requeue_without_job_ids_is_rejected_and_writes_nothing(facade, requeue_fakes):
    result = _call(facade, "requeue_oom_jobs", {})
    assert result["outcome"] == "default_all_rejected"
    assert result["dry_run"] is True
    assert result["archive_immutable"] is True
    assert [c["job_id"] for c in result["classifications"]] == ["job-a", "job-b", "job-c"]
    assert requeue_fakes == []


def test_requeue_default_is_dry_run_and_reports_missing_ids(facade, requeue_fakes):
    result = _call(facade, "requeue_oom_jobs", {"job_ids": ["job-a", "nope"]})
    assert result["outcome"] == "classified"
    assert result["dry_run"] is True
    assert [c["job_id"] for c in result["classifications"]] == ["job-a"]
    assert result["missing_job_ids"] == ["nope"]
    assert result["archive_immutable"] is True
    assert requeue_fakes == []


def test_requeue_empty_job_ids_classifies_nothing(facade, requeue_fakes):
    result = _call(facade, "requeue_oom_jobs", {"job_ids": []})
    assert result["outcome"] == "classified"
    assert result["classifications"] == []
    assert result["missing_job_ids"] == []
    assert requeue_fakes == []


def test_requeue_live_run_without_idempotency_key_is_refused(facade, requeue_fakes):
    result = _call(facade, "requeue_oom_jobs", {"job_ids": ["job-a"], "dry_run": False})
    assert result["outcome"] == "refused"
    assert result["reason"] == "missing_idempotency_key"
    assert requeue_fakes == []


def test_requeue_live_run_retries_only_eligible_and_refuses_rest(facade, requeue_fakes):
    result = _call(
        facade,
        "requeue_oom_jobs",
        {
            "job_ids": ["job-a", "job-b", "missing-x"],
            "dry_run": False,
            "idempotency_key": "batch-1",
            "actor": "owner",
            "reason": "test",
        },
    )
    assert result["outcome"] == "processed"
    assert result["dry_run"] is False
    assert result["missing_job_ids"] == ["missing-x"]
    assert result["archive_immutable"] is True
    refused = [r for r in result["results"] if r["outcome"] == "refused"]
    assert refused == [{"outcome": "refused", "orig_job_id": "job-b", "reason": "not_eligible"}]
    assert [c[0] for c in requeue_fakes] == ["job-a"]
    _, kwargs = requeue_fakes[0]
    assert kwargs["idempotency_key"] == "batch-1:job-a"
    assert kwargs["actor"] == "owner"
    assert kwargs["reason"] == "test"
    assert kwargs["payload_hash"] == "h-a"
    assert kwargs["artifact_refs"] == ["ref-a"]
    assert kwargs["service_name"] == "svc"
    assert kwargs["routing_group"] == "rg"


def test_requeue_dry_run_false_string_is_not_treated_as_live(facade, requeue_fakes):
    # Only a literal boolean False disables the dry-run default.
    result = _call(facade, "requeue_oom_jobs", {"job_ids": ["job-a"], "dry_run": "false"})
    assert result["outcome"] == "classified"
    assert result["dry_run"] is True
    assert requeue_fakes == []
