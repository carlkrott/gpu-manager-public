"""Hermetic proof for facade-bound OOM detection helpers."""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"
NAMES = (
    "_detect_oom_in_history", "_verify_oom_via_vram", "_detect_silent_oom_via_dmesg",
    "_get_force_recover_lock", "_get_bundle_load_lock", "_get_bundle_admission_lock",
    "_PhysicalResourceSetLock", "_get_physical_resource_lock",
    "_reset_physical_resource_locks_for_tests", "_bundle_physical_gpu_set",
    "_recent_oom_events", "_init_oom_redis", "_close_oom_redis",
    "_dead_letter_identity", "_matching_dead_letter_members", "_add_to_dead_letter",
    "_get_dead_letters", "_refresh_dead_letter_cache",
)


def _deny(*_args, **_kwargs):
    raise AssertionError("subprocess/network side effect forbidden in binding proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir(); xdg.mkdir()
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    loaded = []
    for name in ("oom_detection_facade_one", "oom_detection_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        assert spec is not None and spec.loader is not None
        facade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(facade)
        loaded.append(facade)
    return loaded


def test_extracted_definitions_match_independently_frozen_base_and_bind_globals(facades, monkeypatch):
    try:
        base = subprocess.run(
            ["git", "show", "a89f648:scripts/gpu-manager.py"], cwd=REPO,
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("base revision a89f648 not available (release tree without git history)")
    old = {n.name: n for n in ast.walk(ast.parse(base))
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in NAMES}
    from gpu_manager import oom_detection
    assert tuple(oom_detection.NAMES) == NAMES
    for name in NAMES:
        template = next(n for n in ast.walk(ast.parse(Path(oom_detection.__file__).read_text()))
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name)
        assert ast.dump(template, include_attributes=False) == ast.dump(old[name], include_attributes=False)
    for facade in facades:
        for name in NAMES:
            value = getattr(facade, name)
            if isinstance(value, type):
                assert all(method.__globals__ is vars(facade) for method in vars(value).values()
                           if callable(method) and getattr(method, "__globals__", None) is not None)
            else:
                assert value.__globals__ is vars(facade)
    assert facades[0]._PhysicalResourceSetLock is not facades[1]._PhysicalResourceSetLock
    assert facades[0]._get_force_recover_lock("svc") is facades[0]._get_force_recover_lock("svc")
    assert facades[0]._get_force_recover_lock("svc") is not facades[1]._get_force_recover_lock("svc")


def test_facade_dependencies_and_state_are_late_bound(facades, monkeypatch):
    one, two = facades
    monkeypatch.setattr(one, "_recent_oom_events_log", [{"id": 1}])
    monkeypatch.setattr(two, "_recent_oom_events_log", [{"id": 2}])
    assert one._recent_oom_events() == [{"id": 1}]
    assert two._recent_oom_events() == [{"id": 2}]
    one._reset_physical_resource_locks_for_tests()
    lock = one._get_physical_resource_lock(["gpu-b", "gpu-a"])
    assert lock is one._get_physical_resource_lock(["gpu-a", "gpu-b"])
    assert lock.resource_ids == ("gpu-a", "gpu-b")
    assert two._get_physical_resource_lock(["gpu-a", "gpu-b"]) is not lock
    replacement = lambda *_a, **_kw: ["child"]
    monkeypatch.setattr(one, "_get_linked_bundles", replacement)
    monkeypatch.setattr(one, "_resolve_bundles", lambda: {
        "root": {"gpu_id": "1"}, "child": {"gpu_id": "2"},
    })
    assert one._bundle_physical_gpu_set("root") == ("1", "2")


def test_fake_redis_lifecycle_dead_letters_and_cache_are_facade_owned(facades, monkeypatch):
    one, two = facades
    class FakeRedis:
        def __init__(self): self.rows = {}; self.closed = False
        def ping(self): return True
        def close(self): self.closed = True
        def zrange(self, _key, _start, _stop): return list(self.rows)
        def zadd(self, _key, mapping): self.rows.update(mapping)
        def zrem(self, _key, *members):
            for member in members: self.rows.pop(member, None)
        def zremrangebyrank(self, *_args): pass
        def expire(self, *_args): return True
        def zrevrange(self, _key, _start, _stop):
            return [row for row, _score in sorted(self.rows.items(), key=lambda item: item[1], reverse=True)]
    client = FakeRedis()
    class RedisModule:
        @staticmethod
        def Redis(**_kwargs): return client
    monkeypatch.setattr(one, "_queue_redis", None)
    monkeypatch.setattr(one, "_redis_mod", RedisModule)
    monkeypatch.setattr(one, "_redis_connection_kwargs", lambda: {})
    asyncio.run(one._init_oom_redis())
    assert one._oom_redis_pool is client
    assert asyncio.run(one._add_to_dead_letter("job-1", "gpu-0", "oom", "failed"))
    assert asyncio.run(one._get_dead_letters()) == [{
        "job_id": "job-1", "gpu_id": "gpu-0", "reason": "oom", "last_state": "failed",
        "ts": next(iter(client.rows.values())),
    }]
    before = one._dead_letter_cache_refresh_ts
    assert asyncio.run(one._refresh_dead_letter_cache()) == 1
    assert one._dead_letter_cache[0]["job_id"] == "job-1"
    assert one._dead_letter_cache_refresh_ts >= before
    assert two._oom_redis_pool is None and two._dead_letter_cache == []
    asyncio.run(one._close_oom_redis())
    assert client.closed and one._oom_redis_pool is None


def test_resource_lock_acquisition_order_and_cancellation_unwind(facades):
    facade = facades[0]
    facade._reset_physical_resource_locks_for_tests()
    lock = facade._get_physical_resource_lock(["b", "a"])
    assert lock.resource_ids == ("a", "b")
    async def run():
        async with lock:
            assert lock.locked()
        assert not lock.locked()
        # Cancel a composite waiting on its second atom; the first is released.
        a, b = lock._atoms
        await b.acquire()
        waiter = asyncio.create_task(lock.__aenter__())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not a.locked()
        b.release()
    asyncio.run(run())
