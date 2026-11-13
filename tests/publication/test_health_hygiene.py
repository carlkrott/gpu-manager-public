"""Health hygiene (audit 2026-11-12 C6): always-on units are not "idle",
disabled placeholders leave llm_readiness, old runtime observations are stale."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gpu_manager_semantics import _classify_intentionally_idle, snapshot_from_manager  # noqa: E402


def test_always_on_units_are_not_intentionally_idle():
    snap = snapshot_from_manager(
        configured_idle_service="",
        enabled_service_names=["fleet-llm-router.service", "comfyui.service"],
        disabled_service_names=["d1-mi50.service"],
        always_on_service_names=["fleet-llm-router.service"],
    )
    assert _classify_intentionally_idle(snap) == ["d1-mi50.service", "comfyui.service"]


def test_disabled_members_leave_llm_readiness(monkeypatch):
    spec = importlib.util.spec_from_file_location("gpm_hygiene", ROOT / "scripts" / "gpu-manager.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    row = {"fresh": True, "stale": False, "unknown": False, "backend_slots_total": 2, "accepting": True}
    payload = {"schema": "combined-gemma.broker-metric.v1", "queue": {"fresh": True, "depth": 0},
               "dispatch_loop": {"fencing_token": 1, "fail_counters": {"reservation_stale": 0}},
               "members": [{"name": "Qwen-Dual", **row, "backend_slots_total": 0}, {"name": "GLM", **row}]}

    async def fetch():
        return payload, None

    async def probe(*_a, **_kw):
        return {}

    monkeypatch.setattr(m, "_fetch_combined_gemma_visibility", fetch)
    monkeypatch.setattr(m, "_probe_configured_llm_service", probe)
    monkeypatch.setattr(m, "_reduce_llm_readiness", lambda **kw: kw["combined_router"])
    config = {"services": {"Qwen-Dual": {"enabled": False}, "GLM": {"enabled": True}}}
    assert list(asyncio.run(m._llm_readiness_snapshot(config))["members"]) == ["GLM"]


def test_old_runtime_observation_is_stale():
    import time
    from types import SimpleNamespace

    import runtime_controller as rc
    from runtime_state import RuntimeInstanceRecord

    now = time.time()
    records = [RuntimeInstanceRecord("old", "p", "f", observed_at=now - 5 * 86400),
               RuntimeInstanceRecord("new", "p", "f", observed_at=now - 5)]

    async def snapshots():
        return records

    fake = SimpleNamespace(store=SimpleNamespace(snapshots=snapshots))
    out = asyncio.run(rc.CheckedRuntimeController.health_summary(fake))
    flags = {i["instance_id"]: i["observation_stale"] for i in out["instances"]}
    assert flags == {"old": True, "new": False}


def test_service_definition_depth_checks():
    from gpu_manager_contracts import validate_service_definition

    assert validate_service_definition("GLM", {"enabled": True, "type": "llm_backend", "parallel": 2}) == []
    errors = validate_service_definition("GLM", {"enabled": "yes", "endpoint": 8080, "parallel": -1})
    assert errors == ["enabled must be true or false", "endpoint must be a string",
                      "parallel must be a non-negative integer"]


def test_llm_request_counters_survive_restart():
    spec = importlib.util.spec_from_file_location("gpm_counters", ROOT / "scripts" / "gpu-manager.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    before = m.LLMProxy()
    before.total_requests, before.backend_failures = 7, 2
    after = m.LLMProxy()
    after.restore_counters(before.counters())
    after.restore_counters({"total_requests": "x", "forward_attempts": -3})  # junk ignored
    metrics = after.status()["request_metrics"]
    assert (metrics["requests_received"], metrics["backend_failures"], metrics["forward_attempts"]) == (7, 2, 0)
    assert metrics["resets_on_restart"] is False


def test_runtime_profile_warning_only_for_gpu_engines():
    from gpu_manager_contracts import registry_legacy_findings

    unit = {"enabled": True, "type": "llm_backend", "systemd_unit": "x.service"}
    config = {"services": {"Cloud": dict(unit), "Local": {**unit, "gpu_id": "rx6950xt"}}}
    hits = [f["path"] for f in registry_legacy_findings(config) if f["code"] == "runtime_profile_missing"]
    assert hits == ["services.Local"]
