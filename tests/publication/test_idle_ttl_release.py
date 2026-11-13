"""Fenced generation idle-TTL release and both-GPU baseline restoration.

The bundle unload and idle restore talk to real aiohttp TestServers (a fake
ComfyUI stop endpoint and fake idle-LLM health endpoints); no GPU, Redis or
live service is used.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


def _load():
    spec = importlib.util.spec_from_file_location("candidate_gpm_idle_ttl", CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Env:
    """Wires a scheduler + fake backends around the loaded controller module."""

    def __init__(self, module, monkeypatch, comfy_url: str, ttl: float = 20.0):
        self.m = module
        self.unloads: list[str] = []
        self.restores: list[dict] = []
        self.session = None
        self.comfy_url = comfy_url
        self.in_flight = 0
        self.restore_result = {
            "gpu0": {"restored": True, "unit": "rx"},
            "gpu1": {"restored": True, "unit": "mi"},
        }
        now = time.time()
        gs = module.GpuBundleState("gpu0")
        gs.loaded_bundle = "comfyui"
        gs.bundle_loaded_at = now - 600
        gs.last_activity_at = now - 300
        self.gpu = gs
        sched = object.__new__(module.GPUScheduler)
        sched.logger = logging.getLogger("idle-ttl-test")
        sched._comfyui_queue = []
        sched._acestep_queue = []
        sched._current_backend = None
        sched._THRESHOLD_CACHE_TTL = 10.0
        sched._cached_threshold = 30.0
        sched._cached_threshold_at = time.time()
        sched._placement_orchestrator = None
        self.sched = sched
        cfg = {
            "scheduling": {"generation_idle_ttl_seconds": ttl},
            "generation_templates": {},
            "services": {},
        }
        monkeypatch.setattr(module, "vram", SimpleNamespace(
            gpu_state="llm_loaded",
            _llm_evicted=False,
            mark_llm_evicted=lambda v, source="": setattr(self, "evicted", v),
        ), raising=False)
        monkeypatch.setattr(module, "_load_services_config", lambda: cfg)
        monkeypatch.setattr(module, "_maintenance_mode_enabled", lambda s: False)
        monkeypatch.setattr(module, "_resolve_bundles", lambda: {
            "comfyui": {"services": ["comfyui"]},
            "Gemma-idle": {"idle": True, "services": ["gemma"]},
        })
        monkeypatch.setattr(module, "_get_all_gpu_states", lambda: {"gpu0": gs})
        monkeypatch.setattr(module, "_auto_evict_bundle_blockers", lambda *a, **k: [])
        monkeypatch.setattr(
            module, "_bundle_worker_in_flight", lambda *a, **k: self.in_flight
        )
        monkeypatch.setattr(module, "capacity_tracker", None, raising=False)
        monkeypatch.setattr(module, "queue_engine", None, raising=False)
        monkeypatch.setattr(module, "_get_linked_bundles", lambda *a, **k: [])
        monkeypatch.setattr(
            module, "_expand_bundle_transition_members", lambda m, **k: m
        )

        async def passthrough(members, op, reason=""):
            return await op()

        monkeypatch.setattr(module, "_run_bundle_group_transition", passthrough)

        async def unload(name):
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.comfy_url}/stop/{name}") as r:
                    ok = r.status == 200
            if ok:
                self.unloads.append(name)
                gs.loaded_bundle = None
            return ok

        monkeypatch.setattr(module, "_unload_bundle", unload)

        async def restore(snapshot=None):
            self.restores.append(dict(self.restore_result))
            return self.restore_result

        monkeypatch.setattr(module, "_restore_both_idle_llms", restore)


@pytest.fixture
def module():
    return _load()


async def _comfy_server():
    stops: list[str] = []

    async def stop(request):
        stops.append(request.match_info["name"])
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/stop/{name}", stop)
    server = TestServer(app)
    await server.start_server()
    return server, stops


def _run(module, monkeypatch, scenario, ttl=20.0):
    async def go():
        server, stops = await _comfy_server()
        try:
            env = _Env(module, monkeypatch, str(server.make_url("")).rstrip("/"), ttl)
            await scenario(env)
            return env, stops
        finally:
            await server.close()

    return asyncio.run(go())


def test_completed_generation_released_after_ttl_and_both_baselines_restored(
    module, monkeypatch
):
    async def scenario(env):
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario)
    assert env.unloads == ["comfyui"] and stops == ["comfyui"]
    assert len(env.restores) == 1
    report = env.sched._last_idle_restore_report
    assert sorted(report["results"]) == ["gpu0", "gpu1"] and not report["partial"]
    assert env.evicted is True


def test_before_ttl_does_not_release(module, monkeypatch):
    async def scenario(env):
        env.gpu.last_activity_at = time.time() - 5
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario, ttl=60.0)
    assert env.unloads == [] and stops == [] and env.restores == []


def test_per_bundle_ttl_is_bounded(module):
    f = module._bundle_idle_ttl_seconds
    assert f({"idle_ttl_seconds": 1}, {}, 30) == module._IDLE_TTL_MIN_S
    assert f({"idle_ttl_seconds": 10**9}, {}, 30) == module._IDLE_TTL_MAX_S
    assert f({}, {"generation_idle_ttl_seconds": 90}, 30) == 90
    assert f({"idle_ttl_seconds": "bad"}, {}, 30) == 30
    assert f({"idle_ttl_seconds": 45}, {"generation_idle_ttl_seconds": 90}, 30) == 45


def test_explicit_idle_hold_wins_until_released(module, monkeypatch):
    async def scenario(env):
        env.sched.acquire_idle_hold("h1", "gpu0", ttl_seconds=600)
        await env.sched._auto_evict_idle_bundles()
        assert env.unloads == []
        env.sched.release_idle_hold("h1")
        await env.sched._auto_evict_idle_bundles()

    env, _ = _run(module, monkeypatch, scenario)
    assert env.unloads == ["comfyui"]


def test_unknown_hold_state_fails_closed(module, monkeypatch):
    async def scenario(env):
        def boom(gpu_id):
            raise RuntimeError("orchestrator down")

        env.sched.is_idle_held = boom
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario)
    assert env.unloads == [] and stops == []


def test_work_arriving_under_the_fence_cancels_release(module, monkeypatch):
    async def scenario(env):
        async def work_arrives_then_run(members, op, reason=""):
            env.in_flight = 1  # a worker claims the bundle after the pre-checks
            return await op()

        monkeypatch.setattr(
            env.m, "_run_bundle_group_transition", work_arrives_then_run
        )
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario)
    assert env.unloads == [] and stops == [] and env.restores == []
    # the cancelled release is not treated as a failed unload attempt
    assert env.gpu.last_evict_attempt_at == 0.0


def test_new_activity_changes_generation_and_cancels(module, monkeypatch):
    async def scenario(env):
        def blockers(*a, **k):
            env.gpu.last_activity_at = time.time()  # job touched the bundle
            return []

        monkeypatch.setattr(env.m, "_auto_evict_bundle_blockers", lambda *a, **k: [])
        original = env.sched._auto_evict_revalidate

        def revalidate(*a, **k):
            env.gpu.last_activity_at = time.time()
            return original(*a, **k)

        env.sched._auto_evict_revalidate = revalidate
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario)
    assert env.unloads == [] and stops == []


def test_queued_legacy_work_blocks_release(module, monkeypatch):
    async def scenario(env):
        env.sched._comfyui_queue = [SimpleNamespace(status="queued")]
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario)
    assert env.unloads == [] and stops == []


def test_partial_restore_failure_is_reported_not_swallowed(module, monkeypatch, caplog):
    async def scenario(env):
        env.restore_result = {
            "gpu0": {"restored": False, "reason": "bundle_admission_failed"},
            "gpu1": {"restored": True},
        }
        with caplog.at_level(logging.WARNING, logger="idle-ttl-test"):
            await env.sched._auto_evict_idle_bundles()

    env, _ = _run(module, monkeypatch, scenario)
    report = env.sched._last_idle_restore_report
    assert env.unloads == ["comfyui"]
    assert report["partial"] is True and report["failed_gpus"] == ["gpu0"]
    assert any("partial" in r.message for r in caplog.records)


def test_restore_exception_is_typed_partial(module, monkeypatch):
    async def scenario(env):
        async def boom(snapshot=None):
            raise RuntimeError("systemd gone")

        monkeypatch.setattr(env.m, "_restore_both_idle_llms", boom)
        await env.sched._auto_evict_idle_bundles()

    env, _ = _run(module, monkeypatch, scenario)
    report = env.sched._last_idle_restore_report
    assert report["partial"] is True and "RuntimeError" in report["error"]
    assert env.unloads == ["comfyui"]


def test_idle_bundle_never_released(module, monkeypatch):
    async def scenario(env):
        env.gpu.loaded_bundle = "Gemma-idle"
        await env.sched._auto_evict_idle_bundles()

    env, stops = _run(module, monkeypatch, scenario)
    assert env.unloads == [] and env.restores == []
