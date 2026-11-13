"""B9 (audit 2026-11-12): member edits in services.json reach the running broker."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from gemma_broker.member_cache import StrictMemberSnapshotCache  # noqa: E402

BLOCK = {"enabled": True, "candidate_mode": False, "ordered_members": ["A", "B"]}


def _svc(enabled=True, parallel=2):
    return {"enabled": enabled, "type": "llm_backend", "member_type": "openai_compatible",
            "endpoint": "http://127.0.0.1:1/v1/chat/completions", "model": "m",
            "parallel": parallel, "forward_timeout": 100}


def _module(monkeypatch):
    spec = importlib.util.spec_from_file_location("gpm_b9", ROOT / "scripts" / "gpu-manager.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = {"combined_gemma_broker": dict(BLOCK), "services": {"A": _svc(), "B": _svc()}}
    configs = {n: dict(v) for n, v in cfg["services"].items()}
    cache = StrictMemberSnapshotCache(configs)
    cache._snapshots["A"] = replace(cache._snapshots["A"], state_version=7)
    monkeypatch.setattr(m, "_services_config", cfg)
    monkeypatch.setattr("gemma_broker.config.BrokerConfig.from_dict", lambda *_a, **_k: None)
    monkeypatch.setattr("combined_gemma_broker_service.build_member_configs",
                        lambda services, _c: ({n: dict(services[n]) for n in BLOCK["ordered_members"]},
                                              {n: services[n]["endpoint"] for n in BLOCK["ordered_members"]}))
    m._combined_gemma_member_hot.update(configs=configs, endpoints={}, timeouts={}, block=dict(BLOCK),
                                        sig=m._combined_gemma_member_signature(cfg))
    return m, cfg, cache


def test_b9_disable_and_slots_hot_apply(monkeypatch):
    m, cfg, cache = _module(monkeypatch)
    cfg["services"]["A"] = _svc(enabled=False, parallel=4)
    asyncio.run(m._hot_apply_combined_gemma_member_config(cache))
    assert cache._configs["A"]["enabled"] is False
    snap = cache.snapshots()[0]
    assert (snap.accepting, snap.configured_slots, snap.state_version) == (False, 4, 8)
    assert m._combined_gemma_member_hot["timeouts"]["A"] == 160.0
    assert m._combined_gemma_broker_config_stale is None


def test_b9_member_list_change_is_reported_stale(monkeypatch):
    m, cfg, cache = _module(monkeypatch)
    cfg["combined_gemma_broker"] = {**BLOCK, "ordered_members": ["B", "A"]}
    asyncio.run(m._hot_apply_combined_gemma_member_config(cache))
    assert "restart" in m._combined_gemma_broker_config_stale
    assert list(cache._configs) == ["A", "B"]
