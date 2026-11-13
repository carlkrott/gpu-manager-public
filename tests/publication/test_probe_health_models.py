"""Audit B8: a non-llama.cpp server (Breccia: /health ok, /slots 404, /v1/models id
"breccia") is healthy under probe_profile=health_models; llama.cpp members unchanged."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gemma_broker.member_observer import MemberObserver  # noqa: E402

CFG = {"probe_profile": "health_models", "served_model": "breccia", "parallel": 4,
       "context_per_slot": 114688, "gpu_id": "rx6950xt", "model_path": "/m/breccia.gguf"}
BRECCIA = {"/health": {"status": "ok", "model": "breccia"},
           "/v1/models": {"object": "list", "data": [{"id": "breccia"}]}}


def _observe(cfg, routes, leases=1):
    seen: list[str] = []

    async def http(_cfg, path):
        seen.append(path)
        return (200, routes[path]) if path in routes else (404, None)

    async def systemd(_cfg):
        return {"systemd_active": True, "main_pid": 42}

    obs = MemberObserver({"B": cfg}, http_probe=http, systemd_probe=systemd,
                         dispatcher_state=lambda _n: (True, leases))
    return asyncio.run(obs.observe("B", now=1.0)), seen


def test_health_models_profile_is_healthy_without_slots():
    probes, seen = _observe(CFG, BRECCIA, leases=1)
    assert "/slots" not in seen
    assert probes["semantic_health"] is True and probes["model_resident"] is True
    assert (probes["backend_slots_total"], probes["backend_slots_busy"]) == (4, 1)
    assert probes["probe_error"] is None


def test_health_models_fails_closed_on_wrong_model():
    probes, _ = _observe({**CFG, "served_model": "other"}, BRECCIA)
    assert probes["semantic_health"] is False and probes["model_resident"] is False


def test_llama_cpp_member_without_slots_stays_unhealthy():
    cfg = {k: v for k, v in CFG.items() if k not in ("probe_profile", "served_model")}
    probes, seen = _observe(cfg, BRECCIA)
    assert "/slots" in seen and probes["semantic_health"] is False
