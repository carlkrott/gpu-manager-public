"""Bundle eviction gaps (audit 2026-11-12 B1-B3).

B1: broker-owned members are drained before their bundle is unloaded.
B2: a failed unload aborts the eviction; the GPU is not marked evicted.
B3: a reservation-only bundle (no services) is freed by evicting its owner.
B7: an active resident unit (Breccia) on a target GPU is never stopped; queued callers
    (worker reserve, per-GPU bundle queue, ACE-Step) WAIT, direct callers refuse.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

BUNDLES = {
    "Dual-idle": {"gpu_id": "rx", "services": ["Dual"], "linked_to": ["Dual-mi50"]},
    "Dual-mi50": {"gpu_id": "mi50", "services": [], "linked_from": ["Dual-idle"]},
    "Other-mi50": {"gpu_id": "mi50", "services": ["Other"]},
}


def _run(monkeypatch, *, unload_ok=True, drain_ok=True):
    spec = importlib.util.spec_from_file_location("candidate_gpm_evict_b", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls: list[tuple[str, str]] = []

    async def drain(svc):
        calls.append(("drain", svc))
        return drain_ok

    async def unload(name):
        calls.append(("unload", name))
        return unload_ok

    async def vram_below(**_kw):
        return True

    monkeypatch.setattr(m, "bundle_lifecycle_controller", object())
    monkeypatch.setattr(m, "scheduler", SimpleNamespace(_gpu_state={}))
    monkeypatch.setattr(m, "_load_services_config", lambda: {"services": {}, "scheduling": {}})
    monkeypatch.setattr(m, "_resolve_bundles", lambda: BUNDLES)
    monkeypatch.setattr(m, "_drain_combined_gemma_member_before_stop", drain)
    monkeypatch.setattr(m, "_unload_bundle", unload)
    monkeypatch.setattr(m, "llm", SimpleNamespace(update_health=lambda _v: None))
    monkeypatch.setattr(m, "_get_gpu_state", lambda g: m.GpuBundleState(g))
    vram = m.VRAMLifecycleManager(None, logging.getLogger("evict-b"))
    vram._wait_for_vram_below = vram_below
    ok = asyncio.run(vram.evict_llm("test", target_gpu="mi50", admission_authorized=True))
    return ok, calls, vram


def test_b1_b3_reservation_share_evicts_owner_after_drain(monkeypatch):
    ok, calls, vram = _run(monkeypatch)
    assert ok is True
    assert calls == [("drain", "Dual"), ("unload", "Dual-idle"), ("drain", "Other"), ("unload", "Other-mi50")]
    assert vram._gpu_evicted["mi50"] is True


def test_b2_failed_unload_aborts_without_marking_evicted(monkeypatch):
    ok, calls, vram = _run(monkeypatch, unload_ok=False)
    assert ok is False
    assert calls == [("drain", "Dual"), ("unload", "Dual-idle")]
    assert not vram._gpu_evicted.get("mi50")


def test_b1_failed_drain_does_not_stop_member(monkeypatch):
    ok, calls, _ = _run(monkeypatch, drain_ok=False)
    assert ok is False
    assert calls == [("drain", "Dual")]


def test_b7_active_resident_unit_refuses_load_without_touching_anything(monkeypatch):
    spec = importlib.util.spec_from_file_location("candidate_gpm_evict_b7", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = {"services": {}, "gpu_devices": {"mi50": {"resident_units": ["breccia-mi50.service"]}}}
    monkeypatch.setattr(m, "_load_services_config", lambda: cfg)
    monkeypatch.setattr(m, "_resolve_bundles", lambda: BUNDLES)
    monkeypatch.setattr(m, "_get_gpu_state", lambda g: m.GpuBundleState(g))
    monkeypatch.setattr(m, "scheduler", SimpleNamespace(_gpu_state={}))
    seen: list[tuple] = []

    class _Proc:
        def __init__(self, rc):
            self.rc = rc

        async def wait(self):
            return self.rc

    async def fake_exec(*argv, **_kw):
        seen.append(argv)
        return _Proc(0 if argv[-1] == "breccia-mi50.service" else 3)

    async def must_not_run(*_a, **_kw):
        raise AssertionError("nothing may be stopped or loaded")

    monkeypatch.setattr(m.asyncio, "create_subprocess_exec", fake_exec)
    for name in ("_unload_bundle", "_load_bundle", "_drain_combined_gemma_member_before_stop"):
        monkeypatch.setattr(m, name, must_not_run)
    assert asyncio.run(m._active_resident_unit("rx")) is None
    assert asyncio.run(m._evict_then_load_bundle_unlocked("Other-mi50")) is False
    assert all(a[:2] == ("systemctl", "is-active") for a in seen)
    try:
        asyncio.run(m._evict_then_load_bundle_unlocked("Other-mi50", reason="worker_reserve:x"))
        raise AssertionError("worker reservations must defer (WAIT), not fail")
    except m.BundleAdmissionDeferred:
        pass
    for reason in ("gpu-queue:job1", "acestep_job"):
        assert m._admission_waits(reason)
    assert not m._admission_waits("nginx_image") and not m._admission_waits("mcp_submit_generation")
    cfg["gpu_devices"]["mi50"]["resident_units"] = ["not-running.service"]
    assert asyncio.run(m._active_resident_unit("mi50")) is None


def test_b7_gpu_queue_deferral_requeues_instead_of_failing(monkeypatch):
    spec = importlib.util.spec_from_file_location("candidate_gpm_evict_b7q", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    gpu = m.GpuBundleState("mi50")
    done: list[bool] = []

    async def deferred(*_a, **_kw):
        raise m.BundleAdmissionDeferred("resident tenant breccia-mi50.service holds mi50")

    async def ready(*_a):
        return True

    async def complete(_job, success=True):
        done.append(success)

    monkeypatch.setattr(m, "_evict_then_load_bundle", deferred)
    monkeypatch.setattr(m, "_get_gpu_state", lambda _g: gpu)
    monkeypatch.setattr(m, "bundle_coordinator", SimpleNamespace(mark_gpu_ready=ready, complete_job=complete))
    sched = SimpleNamespace(logger=logging.getLogger("b7q"))
    entry = {"bundle_name": "Other-mi50", "priority": 5,
             "request_data": {"bundle_job_id": "j1", "retry_count": 0}}
    assert asyncio.run(m.GPUScheduler._process_gpu_queue_bundle(sched, "mi50", entry)) is False
    assert done == [] and len(gpu.priority_queue) == 1
    entry["request_data"]["retry_count"] = 30
    asyncio.run(m.GPUScheduler._process_gpu_queue_bundle(sched, "mi50", entry))
    assert done == [False]
