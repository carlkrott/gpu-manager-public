"""Fake-only lifecycle/follower/status contract for gpu-guard.

The wiring API is deliberately dependency-injected before app startup:

    register_gpu_guard(app, *, config, proc_root, sys_root, receipt_root,
                       collector, monotonic_clock, process_factory,
                       auth_decision)

``config`` contains ``{"enabled": bool, "interval_seconds": float}``.
The app stores the service at ``app["gpu_guard"]``; its cleanup context owns
all follower/task lifetime.  The follower launches journalctl with an argv
list (never a shell), parses bounded MESSAGE fields, and retries with capped
backoff.  The authenticated status route is ``GET /v1/gpu-guard/status``.
All process, path, time, sampling, and auth behavior below is fake/injected.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gpu_manager import gpu_guard


class FakeProcess:
    def __init__(self, *, stdout=b"", returncode=0):
        self.stdout = _FakeStream(stdout)
        self.returncode = returncode
        self.terminated = 0
        self.waited = 0

    async def wait(self):
        self.waited += 1
        return self.returncode

    def terminate(self):
        self.terminated += 1


class _FakeStream:
    def __init__(self, payload):
        self.payload = payload
        self.reads = []

    async def readline(self):
        self.reads.append(True)
        if not self.payload:
            return b""
        line, separator, rest = self.payload.partition(b"\n")
        self.payload = rest if separator else b""
        return line + (b"\n" if separator else b"")


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now
        self.sleeps = []

    def monotonic(self):
        return self.now

    async def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def _register(app, tmp_path, *, enabled=True, interval=0.01, collector=None,
              factory=None, clock=None, auth=None):
    selected_clock = clock or FakeClock()
    return gpu_guard.register_gpu_guard(
        app,
        config={"enabled": enabled, "interval_seconds": interval},
        proc_root=tmp_path / "proc",
        sys_root=tmp_path / "sys",
        receipt_root=tmp_path / "receipts",
        collector=collector or (lambda *_args, **_kwargs: {"sampled_at": 1.0, "clients": []}),
        monotonic_clock=selected_clock.monotonic,
        sleep=selected_clock.sleep,
        process_factory=factory or (lambda argv: FakeProcess()),
        auth_decision=auth or (lambda _request: True),
        managed_units={"gpu-manager.service"},
        desktop_allowlist={"kwin_x11", "plasmashell"},
    )


def test_disabled_config_has_no_task_process_or_receipt_files(tmp_path):
    app = web.Application()
    made = []
    _register(app, tmp_path, enabled=False, factory=lambda argv: made.append(argv))
    async def exercise():
        async with TestClient(TestServer(app)) as client:
            assert client.app["gpu_guard"].enabled is False
            assert made == []
            assert not (tmp_path / "receipts").exists()
            await client.get("/v1/gpu-guard/status")
            assert made == []
            assert not (tmp_path / "receipts").exists()
    asyncio.run(exercise())


def test_default_receipt_root_uses_home_state_dir_and_creates_private_directory(tmp_path, monkeypatch):
    home = tmp_path / "fake-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    app = web.Application()
    service = gpu_guard.register_gpu_guard(
        app,
        config={"enabled": False},
        proc_root=tmp_path / "proc",
        sys_root=tmp_path / "sys",
    )
    expected = home / ".local" / "state" / "gpu-manager" / "gpu-guard"
    assert service.receipt_root == expected
    assert not expected.exists()

    service.recorder.record("GPU reset begin", {}, state=service.state,
                            snapshot={"sampled_at": 1.0, "clients": []})
    assert expected.is_dir()
    assert expected.stat().st_mode & 0o777 == 0o700


def test_cleanup_is_idempotent_and_cancels_owned_task_once(tmp_path):
    app = web.Application()
    service = _register(app, tmp_path)
    async def exercise():
        await service.start()
        await service.close()
        await service.close()
        assert service.cleanup_count == 1
        assert service.task.done()
    asyncio.run(exercise())


def test_collector_exception_is_contained_and_interval_has_floor_and_backoff(tmp_path):
    clock = FakeClock()
    app = web.Application()
    service = _register(app, tmp_path, collector=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fake")), clock=clock)
    async def exercise():
        await service.start()
        await service.run_one_pass()
        assert service.last_error == "collector_error"
        assert service.interval_seconds >= 0.5
        assert service.interval_seconds > 0.5  # repeated slow/error pass backs off
        await service.close()
    asyncio.run(exercise())


def test_journal_follower_uses_exact_argv_and_bounds_records(tmp_path):
    process = FakeProcess(stdout=(
        b'{"MESSAGE":"GPU reset begin"}\n'
        b'{bad json}\n'
        + b'{"MESSAGE":"' + b'x' * 70000 + b'"}\n'
    ))
    calls = []
    follower = gpu_guard.JournalFollower(
        process_factory=lambda argv: calls.append(argv) or process,
        receipt_recorder=lambda *_a, **_k: None,
        sleep=lambda _delay: asyncio.sleep(0),
    )
    async def exercise():
        await follower.start()
        await follower.read_available()
        assert calls == [["journalctl", "-k", "-f", "--output=json", "--no-pager", "-n", "0"]]
        assert "-k" in calls[0]
        assert calls[0][calls[0].index("-n") + 1] == "0"
        assert "--unit" not in calls[0]
        assert follower.malformed_records == 1
        assert follower.oversized_records == 1
        assert follower.accepted_records == 1
        await follower.close()
        assert process.terminated == 1 and process.waited == 1
    asyncio.run(exercise())


def test_follower_denial_and_exit_backoff_is_bounded_and_reaped(tmp_path):
    clock = FakeClock()
    processes = []
    def factory(_argv):
        proc = FakeProcess(returncode=1)
        processes.append(proc)
        return proc
    follower = gpu_guard.JournalFollower(
        process_factory=factory, receipt_recorder=lambda *_a, **_k: None,
        sleep=clock.sleep, monotonic_clock=clock.monotonic,
    )
    async def exercise():
        for _ in range(7):
            await follower.restart_after_exit()
        assert follower.retry_delays == [1, 2, 4, 8, 16, 30, 30]
        assert all(proc.waited == 1 for proc in processes)
        await follower.close()
    asyncio.run(exercise())


def test_status_route_respects_auth_and_never_returns_token_material(tmp_path):
    decisions = []
    app = web.Application()
    _register(app, tmp_path, enabled=False, auth=lambda request: decisions.append(request) or False)
    async def exercise():
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/v1/gpu-guard/status")
            assert response.status in (401, 403)
            assert len(decisions) == 1
            assert "authorization" not in json.dumps(await response.text()).lower()
    asyncio.run(exercise())


def test_status_success_contains_only_observational_state(tmp_path):
    app = web.Application()
    _register(app, tmp_path, enabled=False, auth=lambda _request: True)
    async def exercise():
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/v1/gpu-guard/status")
            assert response.status == 200
            body = await response.json()
            assert body["enabled"] is False
            assert body["status"] in {"off", "unknown"}
            assert "token" not in json.dumps(body).lower()
    asyncio.run(exercise())


def test_synthetic_fault_to_authenticated_status_and_off_switch(tmp_path):
    """Exercise sampling, fault receipt, status, sampler failure and disabled mode."""
    samples = iter((
        {"sampled_at": 1.0, "clients": [
            {"device": "0000:03:00.0", "drm_client_id": "managed-1",
             "engine_gfx_ns": 100, "engine_compute_ns": 0,
             "holders": [{"pid": 11, "starttime": 111, "comm": "worker",
                          "cmdline": "worker", "unit": "gpu-manager.service"}]},
            {"device": "0000:03:00.0", "drm_client_id": "desktop-1",
             "engine_gfx_ns": 50, "engine_compute_ns": 0,
             "holders": [{"pid": 22, "starttime": 222, "comm": "kwin_x11",
                          "cmdline": "kwin_x11", "unit": ""}]},
        ], "kfd_holders": [{"pid": 33, "starttime": 333, "comm": "compute",
                            "cmdline": "compute", "unit": ""}],
            "unreadable": [{"pid": 44, "comm": "denied", "reason": "permission_denied"}],
            "unreadable_total": 1},
        {"sampled_at": 2.0, "clients": [
            {"device": "0000:03:00.0", "drm_client_id": "managed-1",
             "engine_gfx_ns": 175, "engine_compute_ns": 0,
             "holders": [{"pid": 11, "starttime": 111, "comm": "worker",
                          "cmdline": "worker", "unit": "gpu-manager.service"}]},
            {"device": "0000:03:00.0", "drm_client_id": "desktop-1",
             "engine_gfx_ns": 75, "engine_compute_ns": 0,
             "holders": [{"pid": 22, "starttime": 222, "comm": "kwin_x11",
                          "cmdline": "kwin_x11", "unit": ""}]},
        ], "kfd_holders": [{"pid": 33, "starttime": 333, "comm": "compute",
                            "cmdline": "compute", "unit": ""}],
            "unreadable": [{"pid": 44, "comm": "denied", "reason": "permission_denied"}],
            "unreadable_total": 1},
    ))

    def collector(*_args, **_kwargs):
        try:
            return next(samples)
        except StopIteration:
            raise OSError("synthetic sampler failure")

    fault_process = FakeProcess(stdout=(
        b'{"MESSAGE":"amdgpu: GPU reset begin"}\n'
        b'{"MESSAGE":"amdgpu: GPU Hang detected"}\n'
    ))
    app = web.Application()
    service = _register(app, tmp_path, collector=collector,
                        factory=lambda _argv: fault_process,
                        auth=lambda request: request.headers.get("X-Test-Auth") == "ok")

    async def exercise():
        await service.run_one_pass()
        await service.run_one_pass()
        assert service.state.active_clients()
        assert {row["classification"] for row in service.state.active_clients()} == {"managed", "desktop"}
        await service.run_one_pass()
        assert service.last_error == "collector_error"
        await service.follower.start()
        await service.follower.read_available()
        assert service.follower.accepted_records == 2
        assert service.recorder._receipt is not None
        receipt = json.loads(service.recorder._receipt.read_text(encoding="utf-8"))
        assert len(receipt["matched_lines"]) == 2
        context = receipt["context"]
        assert {item["classification"] for item in context["active_clients"]} == {"managed", "desktop"}
        assert context["kfd_holders"][0]["pid"] == 33
        assert context["unreadable"][0]["pid"] == 44
        await service.close()
        async with TestClient(TestServer(app)) as client:
            denied = await client.get("/v1/gpu-guard/status")
            assert denied.status in (401, 403)
            response = await client.get("/v1/gpu-guard/status", headers={"X-Test-Auth": "ok"})
            assert response.status == 200
            body = await response.json()
            assert body["enabled"] is True
            assert body["last_error"] == "collector_error"
            assert body["latest_receipt"]
            assert body["kfd_holders"][0]["pid"] == 33
            assert body["unreadable"][0]["pid"] == 44

        off_app = web.Application()
        off_started = []
        off_service = _register(off_app, tmp_path / "off", enabled=False,
                                factory=lambda argv: off_started.append(argv))
        async with TestClient(TestServer(off_app)) as client:
            response = await client.get("/v1/gpu-guard/status")
            assert response.status == 200
            body = await response.json()
            assert body["enabled"] is False and body["status"] == "off"
            assert off_started == []
            assert not (tmp_path / "off" / "receipts").exists()
        assert off_service.cleanup_count == 1

    asyncio.run(exercise())
