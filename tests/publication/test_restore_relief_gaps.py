"""Restore/relief gaps (audit 2026-11-12 B4 + llm_relief).

B4: an unhealthy restore keeps _llm_evicted (so it is retried) and backs off;
    an explicitly empty scheduling.idle_services restores nothing (no legacy
    `systemctl start` of the cloud members' router unit).
llm_relief: relief no longer forces _llm_evicted False, which made
    restore_llm() a no-op.
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


class _NoGuard:
    def __init__(self, _vram):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _module(monkeypatch, *, scheduling, idle_entries=(), load_ok=True):
    spec = importlib.util.spec_from_file_location("candidate_gpm_restore_b4", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls: list[str] = []

    async def no_sleep(_s):
        return None

    async def transition(name, fn, **_kw):
        calls.append(f"load:{name}")
        return load_ok

    async def subprocess_exec(*args, **_kw):
        calls.append("exec:" + " ".join(args))
        return SimpleNamespace(wait=lambda: no_sleep(0))

    async def rejoin(*_a, **_kw):
        return True

    monkeypatch.setattr(m, "_RestoreAdmissionGuard", _NoGuard)
    monkeypatch.setattr(m, "_generation_release_blockers", lambda **_kw: [])
    monkeypatch.setattr(m, "_load_services_config", lambda: {"services": {
        "Cloud": {"enabled": True, "priority": 1, "systemd_unit": "fleet-llm-router.service"}},
        "scheduling": scheduling})
    monkeypatch.setattr(m, "_get_all_idle_services", lambda: list(idle_entries))
    monkeypatch.setattr(m, "_get_idle_bundle", lambda g: {"bundle_name": f"{g}-idle"})
    monkeypatch.setattr(m, "_run_bundle_transition", transition)
    monkeypatch.setattr(m, "_rejoin_combined_gemma_member_after_start", rejoin)
    monkeypatch.setattr(m, "_get_gpu_state", lambda g: m.GpuBundleState(g))
    monkeypatch.setattr(m.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(m.asyncio, "create_subprocess_exec", subprocess_exec)
    vram = m.VRAMLifecycleManager(None, logging.getLogger("restore-b4"))
    vram.mark_llm_evicted(True, source="test")
    return m, vram, calls


def test_b4_empty_idle_map_restores_nothing(monkeypatch):
    _m, vram, calls = _module(monkeypatch, scheduling={"idle_services": {}, "idle_service": ""})
    assert asyncio.run(vram.restore_llm("t")) is True
    assert calls == [] and vram._llm_evicted is False


def test_b4_unhealthy_restore_stays_evicted_and_backs_off(monkeypatch):
    entry = {"gpu_id": "rx", "service_name": "Local"}
    m, vram, calls = _module(monkeypatch, scheduling={"idle_services": {"rx": "Local"}},
                             idle_entries=[entry], load_ok=False)
    assert asyncio.run(vram.restore_llm("t")) is False
    assert vram._llm_evicted is True and vram._restore_failures == 1
    assert vram._restore_retry_at > m.time.monotonic()
    assert asyncio.run(vram.restore_llm("t")) is False  # inside backoff: no new load
    assert calls == ["load:rx-idle"]
    vram._restore_retry_at = 0.0
    async def healthy_load(*_a, **_kw):
        return True

    monkeypatch.setattr(m, "_run_bundle_transition", healthy_load)
    assert asyncio.run(vram.restore_llm("t")) is True
    assert (vram._llm_evicted, vram._restore_failures) == (False, 0)


def _relief(monkeypatch, evicted):
    spec = importlib.util.spec_from_file_location("candidate_gpm_relief", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    seen: list[bool] = []

    async def restore_llm(reason):
        seen.append(fake._llm_evicted)
        return False  # stop relief after the restore step

    fake = SimpleNamespace(_llm_evicted=evicted, gpu_state=None, restore_llm=restore_llm)
    fake.mark_llm_evicted = lambda v, source="": setattr(fake, "_llm_evicted", v)
    monkeypatch.setattr(m, "vram", fake)
    monkeypatch.setattr(m, "_load_services_config", lambda: {"scheduling": {}})
    sched = SimpleNamespace(_restoring=False, total_llm_reliefs=0, logger=logging.getLogger("relief"),
                            _lifecycle_begin=lambda *a, **k: None, _lifecycle_step_begin=lambda *a: None,
                            _lifecycle_step_done=lambda *a: None, _lifecycle_error=lambda *a: None,
                            _lifecycle_complete=lambda: None)
    asyncio.run(m.GPUScheduler._do_llm_relief(sched))
    return seen, fake


def test_relief_restores_with_evicted_flag_intact(monkeypatch):
    seen, fake = _relief(monkeypatch, evicted=True)
    assert seen == [True] and fake._llm_evicted is True


def test_relief_on_resident_llm_leaves_flag(monkeypatch):
    seen, fake = _relief(monkeypatch, evicted=False)
    assert seen == [False] and fake._llm_evicted is False


def _oom_module(monkeypatch, active: bool):
    spec = importlib.util.spec_from_file_location("candidate_gpm_oom_b6", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls: list[str] = []

    class _Proc:
        def __init__(self, rc):
            self.rc = rc

        async def communicate(self):
            return self.rc, b""

        async def wait(self):
            return self.rc

    async def subprocess_exec(*args, **_kw):
        calls.append(" ".join(args))
        return _Proc(0 if (args[1] != "is-active" or active) else 3)

    async def unhealthy(*_a, **_kw):
        return False

    async def rejoin(name, **_kw):
        calls.append(f"rejoin:{name}")
        return True

    svc = {"enabled": True, "type": "llm_backend", "systemd_unit": "m.service", "port": 1,
           "routing_group": m._COMBINED_GEMMA_ROUTING_GROUP}
    monkeypatch.setattr(m, "_services_config", {"services": {"M": svc}})
    monkeypatch.setattr(m, "_combined_gemma_broker_enabled", lambda: True)
    monkeypatch.setattr(m, "_get_bundle_for_service", lambda _n: None)
    monkeypatch.setattr(m, "_probe_health_inline", unhealthy)
    monkeypatch.setattr(m, "_rejoin_combined_gemma_member_after_start", rejoin)
    monkeypatch.setattr(m, "vram", object())
    monkeypatch.setattr(m, "scheduler", object())
    monkeypatch.setattr(m.asyncio, "create_subprocess_exec", subprocess_exec)
    return m, calls


def test_b6_oom_restore_never_restarts_a_running_broker_member(monkeypatch):
    m, calls = _oom_module(monkeypatch, active=True)
    res = asyncio.run(m._restore_both_idle_llms({"rx": "M"}))
    assert res["rx"]["reason"] == "broker_member_unhealthy_left_to_broker"
    assert not any("try-restart" in c for c in calls)


def test_b6_oom_restore_rejoins_a_started_broker_member(monkeypatch):
    m, calls = _oom_module(monkeypatch, active=False)
    res = asyncio.run(m._restore_both_idle_llms({"rx": "M"}))
    assert calls[-2:] == ["systemctl start m.service", "rejoin:M"]
    assert res["rx"]["restored"] is True and res["rx"]["rejoined"] is True
