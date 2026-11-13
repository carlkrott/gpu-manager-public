"""Binding proof for the extracted OOM retry (scripts/gpu_manager/oom_retry.py).

No Redis is ever contacted: ``_oom_redis_pool`` is replaced by in-memory fakes
on each synthetic facade, and socket/process creation is denied while the
facades load.
"""
from __future__ import annotations

import asyncio
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
    "_oom_retry_lua_script",
    "_normalize_dead_letter_entry",
    "_oom_retry_check_redis_state",
    "_generate_oom_retry_id",
    "_generate_oom_retry_job_id",
    "_atomic_retry_oom_job",
    "_oom_retry_correlation_snapshot",
)


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during import proof")


class FakePool:
    """In-memory stand-in for the Redis client; records every call."""

    def __init__(self, eval_result=None, audit=None):
        self.eval_result = eval_result
        self.audit = audit or {}
        self.calls = []

    def eval(self, script, numkeys, *args):
        self.calls.append(("eval", script, numkeys, args))
        if isinstance(self.eval_result, Exception):
            raise self.eval_result
        return self.eval_result

    def scan(self, cursor=0, match=None, count=None):
        self.calls.append(("scan", cursor, match, count))
        return 0, list(self.audit)

    def hgetall(self, key):
        self.calls.append(("hgetall", key))
        return self.audit.get(key, {})


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
        for name in ("oom_retry_facade_one", "oom_retry_facade_two"):
            spec = importlib.util.spec_from_file_location(name, FACADE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
    for facade in loaded:
        # Never leave a real pool reachable from a synthetic facade.
        facade._oom_redis_pool = None
    return loaded


def test_functions_are_bound_to_their_own_facade(facades):
    one, two = facades
    for facade in facades:
        for name in NAMES:
            assert getattr(facade, name).__globals__ is vars(facade)
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)
        assert getattr(one, name).__module__ == one.__name__
    assert inspect.iscoroutinefunction(one._atomic_retry_oom_job)
    assert inspect.iscoroutinefunction(one._oom_retry_check_redis_state)
    assert inspect.iscoroutinefunction(one._oom_retry_correlation_snapshot)
    assert not inspect.iscoroutinefunction(one._oom_retry_lua_script)


def test_signatures_and_defaults_are_preserved(facades):
    one = facades[0]
    sig = inspect.signature(one._atomic_retry_oom_job)
    assert list(sig.parameters) == [
        "dead_letter",
        "actor",
        "reason",
        "idempotency_key",
        "payload_hash",
        "artifact_refs",
        "service_name",
        "routing_group",
    ]
    kinds = {n: p.kind for n, p in sig.parameters.items()}
    assert kinds["dead_letter"] is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        kinds[n] is inspect.Parameter.KEYWORD_ONLY for n in list(kinds)[1:]
    )
    assert all(
        sig.parameters[n].default is None
        for n in ("payload_hash", "artifact_refs", "service_name", "routing_group")
    )
    assert inspect.signature(one._oom_retry_correlation_snapshot).parameters["limit"].default == 100
    assert list(inspect.signature(one._oom_retry_check_redis_state).parameters) == ["entries"]
    assert list(inspect.signature(one._generate_oom_retry_id).parameters) == ["orig_job_id"]
    assert list(inspect.signature(one._generate_oom_retry_job_id).parameters) == ["retry_id"]
    assert one._normalize_dead_letter_entry.__doc__.startswith("Return a sanitized copy")
    assert one._atomic_retry_oom_job.__qualname__ == "_atomic_retry_oom_job"


def test_pure_helpers_and_lua_come_from_facade_global(facades, monkeypatch):
    one, two = facades
    assert one._oom_retry_lua_script() == one._OOM_RETRY_LUA
    monkeypatch.setattr(one, "_OOM_RETRY_LUA", "return 1")
    assert one._oom_retry_lua_script() == "return 1"
    assert two._oom_retry_lua_script() == two._OOM_RETRY_LUA != "return 1"
    retry_id = one._generate_oom_retry_id("job/with spaces")
    assert retry_id.startswith("retry-job_with_spaces-")
    assert one._generate_oom_retry_job_id(retry_id) == f"job-retry-{retry_id}"
    assert one._normalize_dead_letter_entry("junk") == {"job_id": "", "gpu_id": ""}
    assert one._normalize_dead_letter_entry({"job_id": " a ", "gpu_id": None})["job_id"] == "a"


def test_rebinding_facade_globals_after_import_is_seen(facades, monkeypatch):
    one = facades[0]
    entry = {"job_id": "orig-1", "gpu_id": "0", "payload_hash": "h"}
    refused = asyncio.run(
        one._atomic_retry_oom_job(entry, actor="a", reason="r", idempotency_key="k")
    )
    assert refused["outcome"] == "refused" and refused["reason"] == "redis_unavailable"

    pool = FakePool(eval_result=["submitted", "rid", "1-0"])
    monkeypatch.setattr(one, "_oom_redis_pool", pool)
    monkeypatch.setattr(one, "_generate_oom_retry_id", lambda orig: "retry-fixed")
    result = asyncio.run(
        one._atomic_retry_oom_job(
            entry, actor="a", reason="r", idempotency_key="k", routing_group="g1"
        )
    )
    assert result["outcome"] == "submitted"
    assert result["retry_id"] == "retry-fixed"
    assert result["new_job_id"] == "job-retry-retry-fixed"
    assert result["stream_entry_id"] == "1-0"
    (call,) = pool.calls
    assert call[0] == "eval" and call[1] == one._OOM_RETRY_LUA and call[2] == 5
    assert "queue:g1" in call[3]


def test_two_facades_keep_isolated_state(facades, monkeypatch):
    one, two = facades
    pool_one = FakePool(eval_result=["budget_exceeded", "3", "3"])
    monkeypatch.setattr(one, "_oom_redis_pool", pool_one)
    entry = {"job_id": "orig-2", "gpu_id": "1"}
    over = asyncio.run(
        one._atomic_retry_oom_job(entry, actor="a", reason="r", idempotency_key="k")
    )
    assert over["outcome"] == "budget_exceeded"
    assert over["attempt"] == "3" and over["attempt_limit"] == "3"
    # The second facade never saw the pool and stays fail-closed.
    unavailable = asyncio.run(
        two._atomic_retry_oom_job(entry, actor="a", reason="r", idempotency_key="k")
    )
    assert unavailable["reason"] == "redis_unavailable"
    assert asyncio.run(two._oom_retry_correlation_snapshot())["redis_available"] is False
    assert asyncio.run(two._oom_retry_check_redis_state([entry])) != []
    assert len(pool_one.calls) == 1


def test_atomic_retry_failure_and_refusal_paths(facades, monkeypatch):
    one = facades[0]
    entry = {"job_id": "orig-3", "gpu_id": "0"}
    run = lambda e, **kw: asyncio.run(  # noqa: E731
        one._atomic_retry_oom_job(e, actor="a", reason="r", **kw)
    )
    assert run({"job_id": ""}, idempotency_key="k")["reason"] == "blank_job_id"
    assert run(entry, idempotency_key="")["reason"] == "missing_idempotency_key"
    monkeypatch.setattr(one, "_oom_redis_pool", FakePool(eval_result=RuntimeError("boom")))
    failed = run(entry, idempotency_key="k")
    assert failed["outcome"] == "exec_failed" and failed["reason"] == "boom"
    monkeypatch.setattr(one, "_oom_redis_pool", FakePool(eval_result="not-a-list"))
    assert run(entry, idempotency_key="k")["reason"] == "malformed_lua_response"
    monkeypatch.setattr(one, "_oom_redis_pool", FakePool(eval_result=["replayed", "retry-old"]))
    replay = run(entry, idempotency_key="k")
    assert replay["outcome"] == "replayed"
    assert replay["new_job_id"] == "job-retry-retry-old"
    blocked = run(dict(entry, classification="missing_tracker"), idempotency_key="k")
    assert blocked["reason"] == "ineligible:missing_tracker"


def test_correlation_snapshot_reads_rebound_pool(facades, monkeypatch):
    one = facades[0]
    prefix = one._OOM_RETRY_AUDIT_PREFIX
    pool = FakePool(audit={
        f"{prefix}r1": {"orig_job_id": "orig-9"},
        f"{prefix}r2": {"orig_job_id": ""},
    })
    monkeypatch.setattr(one, "_oom_redis_pool", pool)
    snapshot = asyncio.run(one._oom_retry_correlation_snapshot())
    assert snapshot["redis_available"] is True
    assert snapshot["total_retries"] == 1
    assert list(snapshot["by_orig"]) == ["orig-9"]
    assert {c[0] for c in pool.calls} == {"scan", "hgetall"}


def test_bind_requires_a_globals_dict():
    sys.path.insert(0, str(SCRIPTS))
    try:
        from gpu_manager import oom_retry
    finally:
        sys.path.remove(str(SCRIPTS))
    with pytest.raises(TypeError):
        oom_retry.bind(object())
    assert oom_retry.NAMES == NAMES
    bound = oom_retry.bind({})
    assert len(bound) == len(NAMES)
    assert all(fn.__globals__ == {} for fn in bound)
