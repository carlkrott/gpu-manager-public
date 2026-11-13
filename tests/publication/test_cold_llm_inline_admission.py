"""Durable owner: a cold bundle-backed LLM is admitted from the proxy.

The per-GPU LLM queue is consumed only by the legacy scheduler loop, which the
durable owner never starts (smallllm.chat 503 after 300 s, 2026-11-13).
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

BUNDLES = {
    "Small-idle": {"bundle_name": "Small-idle", "priority": 2},
    "Idle": {"bundle_name": "Idle", "priority": 1, "idle": True},
    "Gen": {"bundle_name": "Gen", "priority": 5},
}


def _run(monkeypatch, loaded, admitted=True, bundled=True, resident=False):
    spec = importlib.util.spec_from_file_location("candidate_gpm_cold_admit", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls: list[tuple] = []

    async def admit(name, priority):
        calls.append((name, priority))
        return admitted, None

    monkeypatch.setattr(m, "_resolve_bundles", lambda: BUNDLES)
    monkeypatch.setattr(m, "_get_bundle_for_service",
                        lambda _n, bundles=None: BUNDLES["Small-idle"] if bundled else None)
    monkeypatch.setattr(m, "_gpu_states", {"rx": SimpleNamespace(loaded_bundle=None, loaded_bundles=loaded)})
    monkeypatch.setattr(m, "_admit_bundle_backed_llm_service", admit)
    monkeypatch.setattr(m, "_bundle_resident_on_all_gpus", lambda _n, _b: resident)
    monkeypatch.setattr(m, "_bundle_physical_gpu_set", lambda _n, bundles=None: {"rx"})
    monkeypatch.setattr(m, "llm", SimpleNamespace(active_requests=0))
    return asyncio.run(m._admit_cold_llm_service_inline("Small", "rx")), calls, m


def test_idle_occupant_is_replaced(monkeypatch):
    refusal, calls, m = _run(monkeypatch, ["Idle"])
    assert refusal is None and calls == [("Small", 2)] and m.llm.active_requests == 0


def test_higher_priority_tenant_is_never_displaced(monkeypatch):
    refusal, calls, _ = _run(monkeypatch, ["Idle", "Gen"])
    assert "Gen" in refusal and calls == []


def test_failed_admission_and_unbundled(monkeypatch):
    assert _run(monkeypatch, [], admitted=False)[0] == "bundle admission failed"
    assert _run(monkeypatch, [], bundled=False)[0] == "unbundled"


def test_resident_bundle_skips_admission_and_unknown_occupant_fails_closed(monkeypatch):
    assert _run(monkeypatch, ["Gen"], resident=True)[:2] == (None, [])
    refusal, calls, _ = _run(monkeypatch, ["Mystery"])
    assert "Mystery" in refusal and calls == []


def test_auto_evict_blockers_ignore_a_bundle_on_another_gpu(monkeypatch):
    spec = importlib.util.spec_from_file_location("candidate_gpm_evict_scope", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    bundles = {"A": {"bundle_name": "A"}, "B": {"bundle_name": "B"}}
    states = {g: SimpleNamespace(loaded_bundle=b, loaded_bundles=[b], worker_in_flight=0, oom_state="idle")
              for g, b in (("rx", "A"), ("mi", "B"))}
    monkeypatch.setattr(m, "_resolve_bundles", lambda: bundles)
    monkeypatch.setattr(m, "_get_all_gpu_states", lambda: states)
    monkeypatch.setattr(m, "_get_linked_bundles", lambda _n, bundles=None: [])
    monkeypatch.setattr(m, "_comfyui_admission_in_flight", lambda: 0)
    monkeypatch.setattr(m, "_bundle_physical_gpu_set", lambda n, bundles=None: {"rx"} if n == "A" else {"mi"})
    assert m._auto_evict_bundle_blockers("A", bundles) == []
    states["rx"].loaded_bundles = ["A", "B"]  # a second tenant on the SAME gpu still blocks
    assert m._auto_evict_bundle_blockers("A", bundles) == ["non_idle_bundle_loaded:B"]


def test_reasserting_a_resident_bundle_keeps_its_idle_clock(monkeypatch):
    """The 30 s reconcile re-asserts residents; it must not reset the idle TTL."""
    class State:
        loaded_bundle, loaded_bundles, last_activity_at = None, [], 0.0

        def set_loaded_bundles(self, names, source=""):
            self.loaded_bundles = list(names)

        def set_current_priority(self, priority, source=""):
            self.current_priority = priority

        def touch_activity(self, source=""):
            self.last_activity_at = 99.0

    spec = importlib.util.spec_from_file_location("candidate_gpm_touch", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    gs = State()
    monkeypatch.setattr(m, "_resolve_bundles", lambda: {"A": {"gpu_id": "rx", "priority": 2}})
    monkeypatch.setattr(m, "_get_linked_bundles", lambda _n, bundles=None: [])
    monkeypatch.setattr(m, "_get_gpu_state", lambda _g: gs)
    m._transition_linked_bundle_state("A", ["A"], source="load")
    assert gs.last_activity_at > 0  # a real load stamps
    gs.last_activity_at = 1.0
    m._transition_linked_bundle_state("A", ["A"], source="startup_reconcile")
    assert gs.last_activity_at == 1.0 and gs.loaded_bundles == ["A"]
