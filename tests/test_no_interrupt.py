"""No-interrupt rule: running work and its queue are never interrupted, and a
GPU switches service only when idle.

Drives the real gpu-manager facade (hyphenated file, loaded via
``spec_from_file_location``) through the real scheduler loop, LLM relief and
legacy LLM start path.  The boundaries that can stop or evict a service (LLM
drain, LLM evict, the generation batch runner, subprocess creation, the clock
and sleeps) are replaced with fakes, so nothing sleeps for real, nothing
spawns a process and no GPU is touched.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FACADE = REPO / "scripts" / "gpu-manager.py"


class _StopLoop(BaseException):
    """Escapes the scheduler's ``except Exception`` so a test can end the loop."""


@pytest.fixture(scope="module")
def facade():
    scripts = str(REPO / "scripts")
    added = scripts not in sys.path
    if added:
        sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("gpu_manager_no_interrupt", FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if added and scripts in sys.path:
            sys.path.remove(scripts)
    return module


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    monotonic = time

    def advance(self, seconds):
        self.now += float(seconds)


def _fake_asyncio(real, clock, on_sleep):
    """Real asyncio namespace with a clock-advancing, non-sleeping ``sleep`` and no subprocesses."""
    ns = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__")})

    async def sleep(delay=0, result=None):
        clock.advance(delay or 0)
        on_sleep(delay or 0)
        await real.sleep(0)
        return result

    async def forbidden_subprocess(*args, **kwargs):
        raise RuntimeError(f"subprocess forbidden in no-interrupt tests: {args[:3]}")

    ns.sleep = sleep
    ns.create_subprocess_exec = forbidden_subprocess
    return ns


class FakeLLM:
    def __init__(self, active=0, queued=0):
        self.active_requests = active
        self.queued_requests = queued
        self.health = []

    def update_health(self, healthy):
        self.health.append(bool(healthy))


class FakeVram:
    """Scripted drain: each call pops the next result; ``idle_after_drain`` clears active requests."""

    def __init__(self, m, drain_script, llm, idle_after_drain=False):
        self.gpu_state = m.GpuState.LLM_LOADED
        self._llm_evicted = False
        self._drain_script = list(drain_script)
        self._llm = llm
        self._idle_after_drain = idle_after_drain
        self.drain_calls = []
        self.evict_calls = []
        self.restore_calls = []

    async def drain_llm(self, timeout=None):
        self.drain_calls.append(timeout)
        result = self._drain_script.pop(0) if self._drain_script else False
        if result and self._idle_after_drain:
            self._llm.active_requests = 0
        return result

    async def evict_llm(self, reason="gpu_work_start", **_kwargs):
        self.evict_calls.append(reason)
        return True

    async def restore_llm(self, reason="unknown"):
        self.restore_calls.append(reason)
        return True

    def mark_llm_evicted(self, value, source="unknown"):
        self._llm_evicted = bool(value)


def _job(job_id, backend="comfyui"):
    return types.SimpleNamespace(
        id=job_id,
        status="queued",
        priority_tier=None,  # filled by caller
        backend=backend,
        bundle_name=None,
    )


def _make_scheduler(m):
    sched = object.__new__(m.GPUScheduler)
    sched.logger = logging.getLogger("test-no-interrupt")
    sched._comfyui_queue = []
    sched._acestep_queue = []
    sched._pinned_service = None
    sched._pinned_service_since = None
    sched._running = False
    sched._current_backend = None
    sched._last_orphan_reaper_at = 0.0
    sched._idle_check_event = asyncio.Event()
    sched.total_llm_reliefs = 0
    sched._restoring = False
    sched.run_calls = []
    sched.restore_calls = 0

    async def fake_run_comfyui_batch():
        sched.run_calls.append("comfyui")
        for job in sched._comfyui_queue:
            if job.status == "queued":
                job.status = "done"
                break

    async def fake_restore_after_gpu_work():
        sched.restore_calls += 1

    async def fake_ensure_idle_service():
        return None

    async def fake_auto_evict_idle_bundles():
        return None

    sched._run_comfyui_batch = fake_run_comfyui_batch
    sched._run_acestep_job = fake_run_comfyui_batch
    sched._restore_after_gpu_work = fake_restore_after_gpu_work
    sched._ensure_idle_service = fake_ensure_idle_service
    sched._auto_evict_idle_bundles = fake_auto_evict_idle_bundles
    return sched


def _install_loop_fakes(monkeypatch, m, clock, vram, llm, stop_after_sleeps):
    sleeps = {"n": 0}

    def on_sleep(_delay):
        sleeps["n"] += 1
        if sleeps["n"] > stop_after_sleeps:
            raise _StopLoop()

    monkeypatch.setattr(m, "asyncio", _fake_asyncio(asyncio, clock, on_sleep))
    monkeypatch.setattr(m, "time", clock)
    monkeypatch.setattr(m, "vram", vram)
    monkeypatch.setattr(m, "llm", llm)
    monkeypatch.setattr(m, "_get_all_gpu_states", lambda: {})
    monkeypatch.setattr(m, "scheduler", None)
    return sleeps


def _queued_comfyui(m, count=1):
    jobs = []
    for i in range(count):
        job = _job(f"c{i}")
        job.priority_tier = m.PriorityTier.MEDIUM
        jobs.append(job)
    return jobs


async def _run_loop(sched):
    try:
        await sched._scheduler_loop()
    except _StopLoop:
        pass


# ── 1. LLM drain timeout must not force an eviction ───────────────────

def test_scheduler_drain_timeout_does_not_force_evict(facade, monkeypatch):
    """A busy LLM (drain never completes) is never evicted; GPU work waits."""
    m = facade
    clock = FakeClock()
    llm = FakeLLM(active=1, queued=0)
    vram = FakeVram(m, drain_script=[False] * 20, llm=llm)
    _install_loop_fakes(monkeypatch, m, clock, vram, llm, stop_after_sleeps=6)
    sched = _make_scheduler(m)
    sched._comfyui_queue = _queued_comfyui(m)

    asyncio.run(_run_loop(sched))

    assert vram.evict_calls == [], "LLM evicted while requests were still active"
    assert sched.run_calls == [], "GPU job started while LLM still resident"
    assert llm.active_requests == 1


# ── 2. Deferred switch happens only at idle, then queued work proceeds ─

def test_deferred_switch_happens_at_idle_and_queue_resumes(facade, monkeypatch):
    """Drain fails once (LLM busy), then succeeds (idle): evict and GPU job follow only then."""
    m = facade
    clock = FakeClock()
    llm = FakeLLM(active=1, queued=3)
    vram = FakeVram(m, drain_script=[False, True], llm=llm, idle_after_drain=True)
    _install_loop_fakes(monkeypatch, m, clock, vram, llm, stop_after_sleeps=10)
    sched = _make_scheduler(m)
    sched._comfyui_queue = _queued_comfyui(m)

    asyncio.run(_run_loop(sched))

    assert len(vram.drain_calls) >= 2, "switch was not retried after the busy drain"
    assert vram.evict_calls == ["gpu_work_start"], "evict must happen exactly once, at idle"
    assert sched.run_calls == ["comfyui"], "GPU job must run once the LLM is idle"
    assert sched.restore_calls >= 1, "queued LLM work never resumed after GPU went idle"


# ── 3. LLM relief must not evict with requests still queued/active ─────

def _relief_fixture(m, monkeypatch, clock, llm, drain_script):
    vram = FakeVram(m, drain_script=drain_script, llm=llm)
    monkeypatch.setattr(m, "asyncio", _fake_asyncio(asyncio, clock, lambda _d: None))
    monkeypatch.setattr(m, "time", clock)
    monkeypatch.setattr(m, "vram", vram)
    monkeypatch.setattr(m, "llm", llm)
    monkeypatch.setattr(m, "_load_services_config", lambda: {"scheduling": {}})
    monkeypatch.setattr(m, "_maintenance_mode_enabled", lambda _s: False)
    monkeypatch.setattr(m, "_get_idle_service_name", lambda *_a, **_k: "llm-primary")
    sched = _make_scheduler(m)
    for name in ("_lifecycle_begin", "_lifecycle_step_begin", "_lifecycle_step_done",
                 "_lifecycle_error", "_lifecycle_complete"):
        monkeypatch.setattr(sched, name, lambda *a, **k: None)
    return sched, vram


def test_relief_timeout_with_queued_requests_does_not_evict(facade, monkeypatch):
    """Queue never drains within LLM_RELIEF_DRAIN_TIMEOUT: relief must not re-evict and strand it."""
    m = facade
    clock = FakeClock()
    llm = FakeLLM(active=0, queued=12)
    sched, vram = _relief_fixture(m, monkeypatch, clock, llm, drain_script=[False] * 5)

    asyncio.run(sched._do_llm_relief())

    assert vram.evict_calls == [], "relief re-evicted the LLM with requests still queued"
    assert llm.queued_requests == 12


def test_relief_drain_with_active_requests_does_not_evict(facade, monkeypatch):
    """Queue empties, but an active request is still running: relief must not evict over it."""
    m = facade
    clock = FakeClock()
    llm = FakeLLM(active=1, queued=0)
    sched, vram = _relief_fixture(m, monkeypatch, clock, llm, drain_script=[False] * 5)

    asyncio.run(sched._do_llm_relief())

    assert vram.evict_calls == [], "relief evicted the LLM with an active request in flight"
    assert llm.active_requests == 1


# ── 4. LLM burst must not preempt a generation job mid-run ─────────────

def test_llm_burst_does_not_interrupt_running_generation_job(facade, monkeypatch):
    """High-priority LLM burst (queue depth over INTERRUPT_QUEUE_DEPTH) waits for the
    in-flight generation job instead of force-evicting it."""
    m = facade
    forced = []

    async def fake_force_evict(self, gpu_id):
        forced.append(gpu_id)
        return True

    async def fake_port_unhealthy(_port):
        return False

    monkeypatch.setattr(m, "asyncio", _fake_asyncio(asyncio, FakeClock(), lambda _d: None))
    monkeypatch.setattr(m, "_check_port_healthy", fake_port_unhealthy)
    monkeypatch.setattr(m, "_get_bundle_for_service", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_bundle_worker_in_flight", lambda *_a, **_k: 1)
    monkeypatch.setattr(m, "_services_config", {
        "services": {"llm-burst": {"port": 8099, "systemd_unit": "llm-burst.service"}}
    })
    monkeypatch.setattr(m, "scheduler", None)
    monkeypatch.setattr(m, "llm", FakeLLM())
    gpu = m.GpuBundleState("gpu_primary")
    monkeypatch.setattr(m, "_get_gpu_state", lambda _gid: gpu)
    sched = _make_scheduler(m)
    monkeypatch.setattr(type(sched), "_force_evict_gpu", fake_force_evict)

    gpu.loaded_bundle = "generation-bundle"
    gpu.bundle_priority = 1
    for _ in range(m.GPUScheduler.INTERRUPT_QUEUE_DEPTH + 1):
        gpu.enqueue("llm:llm-burst", priority=5, request_data={"service_name": "llm-burst"})

    entry = {"bundle_name": "llm:llm-burst", "priority": 5,
             "request_data": {"service_name": "llm-burst", "future": None}}
    result = asyncio.run(sched._process_gpu_queue_llm("gpu_primary", entry))

    assert forced == [], "LLM burst force-evicted a generation job that was mid-run"
    assert result is False, "burst must be re-queued, not started over a running job"
