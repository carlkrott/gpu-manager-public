"""Idle-target selection, bounded backoff, disabled-member probes and queue drain.

Simulated clocks drive one minute of 2.5 s scheduler ticks; the probe-count and
enabled-down tests run against real aiohttp TestServers.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from idle_targets import DisabledMemberReporter, IdleTargetPlanner  # noqa: E402

TICK = 2.5
TICKS = int(60 / TICK)


def _config(mi50_enabled=False, fallback="aux-mi50", qwen_enabled=True):
    sched = {"idle_services": {"card0": "Gemma-RX", "card1": "Gemma-MI50"}}
    if fallback:
        sched["idle_fallback_services"] = {"card1": fallback}
    return {
        "scheduling": sched,
        "services": {
            "Gemma-RX": {"enabled": True, "systemd_unit": "rx.service", "port": 8101},
            "Gemma-MI50": {"enabled": mi50_enabled, "systemd_unit": "mi.service", "port": 8100},
            "aux-mi50": {"enabled": qwen_enabled, "systemd_unit": "q.service", "port": 8130},
        },
    }


def _planner(logs):
    return IdleTargetPlanner(logs.append, rng=lambda: 0.5)


def _minute(planner, cfg, fail=None):
    """Run one simulated minute; returns (attempts per gpu, ready-tick count)."""
    attempts: dict[str, int] = {}
    for tick in range(TICKS):
        now = tick * TICK
        for gpu, entry in planner.plan(cfg, now).items():
            if entry["service"] and entry["ready"]:
                attempts[gpu] = attempts.get(gpu, 0) + 1
                planner.report_result(gpu, not (fail and gpu in fail), now)
    return attempts


def test_unavailable_preferred_no_fallback_does_not_spin():
    logs: list[str] = []
    p = _planner(logs)
    attempts = _minute(p, _config(fallback=""))
    assert attempts == {"card0": 1 * TICKS}  # healthy gpu keeps being ready, card1 never tried
    assert "card1" not in attempts
    assert [l for l in logs if "card1" in l] == [
        "idle target card1: unavailable (preferred=Gemma-MI50 fallback=- not loadable)"
    ]


def test_qwen_fallback_selected_and_single_info_line():
    logs: list[str] = []
    p = _planner(logs)
    plan = p.plan(_config(), 0.0)
    assert plan["card1"] == {"service": "aux-mi50", "mode": "fallback", "ready": True}
    _minute(p, _config())
    assert len([l for l in logs if "card1" in l]) == 1


def test_missing_fallback_when_disabled_too():
    logs: list[str] = []
    p = _planner(logs)
    cfg = _config(qwen_enabled=False)
    assert p.plan(cfg, 0.0)["card1"]["service"] is None
    _minute(p, cfg)
    assert len([l for l in logs if "card1" in l]) == 1


def test_failing_restore_backs_off_exponentially_and_bounded():
    logs: list[str] = []
    p = _planner(logs)
    attempts = _minute(p, _config(), fail={"card1"})
    # 5s, 10s, 20s, ... : at most 4 tries in 60s instead of 24
    assert attempts["card1"] <= 4
    retries = [p.retry_at("card1")]
    assert retries[0] > 0
    # delay never exceeds the cap even after many failures
    for i in range(20):
        p.report_result("card1", False, 1000.0)
    assert p.retry_at("card1") - 1000.0 <= 300.0
    assert len([l for l in logs if "card1" in l]) == 1


def test_partial_restore_keeps_other_gpu_ready():
    logs: list[str] = []
    p = _planner(logs)
    attempts = _minute(p, _config(), fail={"card1"})
    assert attempts["card0"] == TICKS


def test_availability_change_resets_retry_state():
    logs: list[str] = []
    p = _planner(logs)
    cfg = _config()
    for t in range(3):
        p.plan(cfg, t)
        p.report_result("card1", False, t)
    assert p.plan(cfg, 3.0)["card1"]["ready"] is False
    cfg2 = _config(mi50_enabled=True)
    entry = p.plan(cfg2, 3.0)["card1"]
    assert entry == {"service": "Gemma-MI50", "mode": "preferred", "ready": True}


def test_member_reporter_logs_once():
    logs: list[str] = []
    r = DisabledMemberReporter(logs.append)
    for _ in range(TICKS):
        assert r.probeable("gemma-mi50", {"enabled": False}) is False
        assert r.probeable("gemma-rx", {"enabled": True}) is True
    assert logs == ["member gemma-mi50: disabled/absent; health probing skipped"]


@pytest.mark.asyncio
async def test_disabled_target_not_probed_and_enabled_down_visible():
    hits = {"up": 0, "down": 0, "disabled": 0}

    async def up(_):
        hits["up"] += 1
        return web.json_response({"status": "ok"})

    async def down(_):
        hits["down"] += 1
        return web.json_response({"status": "loading"}, status=503)

    async def disabled(_):
        hits["disabled"] += 1
        return web.json_response({"status": "ok"})

    servers = []
    for name, h in (("up", up), ("down", down), ("disabled", disabled)):
        app = web.Application()
        app.router.add_get("/health", h)
        s = TestServer(app)
        await s.start_server()
        servers.append((name, s))
    logs: list[str] = []
    reporter = DisabledMemberReporter(logs.append)
    members = {
        "up": {"enabled": True}, "down": {"enabled": True}, "disabled": {"enabled": False},
    }
    status = {}
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(TICKS):
                for name, s in servers:
                    if not reporter.probeable(name, members[name]):
                        status[name] = "disabled"
                        continue
                    async with session.get(s.make_url("/health")) as resp:
                        status[name] = "ok" if resp.status == 200 else "down"
    finally:
        for _, s in servers:
            await s.close()
    assert hits == {"up": TICKS, "down": TICKS, "disabled": 0}
    assert status == {"up": "ok", "down": "down", "disabled": "disabled"}
    assert len(logs) == 1


def _load():
    spec = importlib.util.spec_from_file_location("candidate_gpm_idle_health", ROOT / "scripts" / "gpu-manager.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _idle_module(monkeypatch, cfg):
    m = _load()
    monkeypatch.setattr(m, "_services_config", cfg, raising=False)
    monkeypatch.setattr(m, "_phase3_scheduling_config", lambda: {"scheduler_owns_load": True, "scheduler_dry_run_mode": True})
    monkeypatch.setattr(m, "_record_load_history", lambda d: None)
    monkeypatch.setattr(m, "queue_engine", object(), raising=False)
    monkeypatch.setattr(m, "_last_next_decision", {"next_decision": "idle", "idle_preferred": True}, raising=False)
    m._IDLE_PLANNER = IdleTargetPlanner(lambda msg: m.logger.info("%s", msg), rng=lambda: 0.5)
    return m


def test_idle_decisions_never_claim_and_log_once(monkeypatch, caplog):
    m = _idle_module(monkeypatch, _config())
    caplog.set_level("INFO", logger=m.logger.name)
    decisions = [m._scheduler_evaluate_load(i * TICK) for i in range(TICKS)]
    assert all(d["head_job_ids"] == [] and d["claims_worker"] is False for d in decisions)
    assert decisions[0]["action"] == "idle_load"
    assert decisions[0]["target_gpus"] == ["card0", "card1"]
    assert sum("evaluator.decision" in r.getMessage() for r in caplog.records) == 1


def test_idle_noop_when_no_loadable_target(monkeypatch, caplog):
    cfg = _config(fallback="")
    cfg["services"]["Gemma-RX"]["enabled"] = False
    m = _idle_module(monkeypatch, cfg)
    caplog.set_level("INFO", logger=m.logger.name)
    decisions = [m._scheduler_evaluate_load(i * TICK) for i in range(TICKS)]
    assert {d["action"] for d in decisions} == {"noop"}
    assert sum("evaluator.decision" in r.getMessage() for r in caplog.records) == 1


def test_queue_a_drains_before_higher_priority_b(monkeypatch):
    m = _load()
    monkeypatch.setattr(m, "_phase3_scheduling_config", lambda: {
        "scheduler_owns_load": True, "grace_period_seconds": 0,
        "priority_aging_interval_s": 60, "priority_aging_cap": 100,
    })
    prio = {"A": 1, "B": 9}
    monkeypatch.setattr(m, "_service_group_priority", lambda rg, g, t: prio[rg])
    monkeypatch.setattr(m, "_infer_active_service_group", lambda gs, groups: {"group": None})
    m._drain_cutoffs.clear()
    groups = {"A": {"queue_lag": 3, "oldest_wait_seconds": 0}}
    m._get_phase3_scheduler_state(groups, {}, {})
    cutoff_a = m._drain_cutoffs["A"]
    # B arrives with higher priority while A is still draining.
    groups["B"] = {"queue_lag": 1, "oldest_wait_seconds": 0}
    state = m._get_phase3_scheduler_state(groups, {}, {})
    assert state["next_decision"] == "demand:B"
    assert m._drain_cutoffs["A"] == cutoff_a  # A's cutoff preserved
    assert m._active_group_still_draining("B") is True  # B is held while A drains
    # aging: an old low-priority group competes after enough waiting
    groups["A"]["oldest_wait_seconds"] = 60 * 9
    state = m._get_phase3_scheduler_state(groups, {}, {})
    assert state["next_decision"] == "demand:A"
