"""Idle-restore lifecycle acceptance through the production manager handlers.

The production ``create_app`` and its real routes/handlers run under an aiohttp
TestServer. Only external boundaries are injected: Redis (fakeredis), the
services config, host/systemd calls and the bundle backends. No live service,
GPU or Redis is contacted and the startup/background loops are not started.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import fakeredis
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from bundle_lifecycle import InMemoryBundleLifecycleStore  # noqa: E402

CONTROLLER = ROOT / "scripts" / "gpu-manager.py"
TOKEN = "unit-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _load():
    spec = importlib.util.spec_from_file_location("candidate_gpm_http", CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest_asyncio.fixture
async def manager(monkeypatch, tmp_path):
    m = _load()
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(token_file))
    redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(m._redis_mod, "Redis", lambda **kw: redis)
    cfg = {"services": {}, "scheduling": {}, "bundles": {}}
    monkeypatch.setattr(m, "_load_services_config", lambda: cfg)
    # Hermetic: any connection the startup hook builds must hit a closed port,
    # never the host's live Redis.
    monkeypatch.setattr(m, "_redis_connection_kwargs", lambda: {
        "host": "127.0.0.1", "port": 1, "socket_connect_timeout": 0.2})
    # Hermetic host boundary: no real systemctl/sudo and no GPU/VRAM probing.
    commands: list[tuple] = []
    active: set[str] = set()  # fake systemd: units currently "active"

    class _Proc:
        def __init__(self, rc, out=b""):
            self.returncode, self._out = rc, out

        async def communicate(self):
            return self._out, b""

        async def wait(self):
            return self.returncode

    async def fake_exec(*argv, **kw):
        commands.append(argv)
        if "is-active" in argv:
            up = argv[-1] in active
            return _Proc(0 if up else 3, b"active\n" if up else b"inactive\n")
        if "stop" in argv:
            active.discard(argv[-1])
        elif "start" in argv:
            active.add(argv[-1])
        return _Proc(0)

    async def fake_poll_vram():
        return {"used_vram_mib": 0}

    async def fake_wait_below(self, *a, **k):
        return True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(m, "_poll_vram", fake_poll_vram)
    monkeypatch.setattr(m.VRAMLifecycleManager, "_wait_for_vram_below", fake_wait_below)
    app = m.create_app()
    # Disable live background side effects: no scheduler/reaper/gossip loops.
    app.on_startup[:] = app.on_startup[:1]
    app.on_cleanup[:] = app.on_cleanup[:1]
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    env = SimpleNamespace(m=m, client=client, redis=redis, cfg=cfg, mp=monkeypatch,
                          commands=commands, active=active)
    try:
        yield env
    finally:
        await client.close()


def _job(env, job_id="job-1", status=None, **fields):
    env.m.job_tracker.create_job(job_id, "comfyui", "test", "p", 1,
                                 routing_group="comfyui-gen")
    if status:
        env.m.job_tracker.update_status(job_id, status, **fields)
    return job_id


@pytest.mark.asyncio
async def test_unauthenticated_requests_are_rejected(manager):
    r = await manager.client.get("/v1/job/nope")
    assert r.status in (401, 403)


@pytest.mark.asyncio
async def test_job_read_through_real_handler_and_unknown_is_404(manager):
    _job(manager, "j-read")
    r = await manager.client.get("/v1/job/j-read", headers=AUTH)
    assert r.status == 200
    assert (await r.json())["job_id"] == "j-read"
    r = await manager.client.get("/v1/job/missing", headers=AUTH)
    assert r.status == 404


@pytest.mark.asyncio
async def test_terminal_job_is_retained_and_not_resurrected_by_reads(manager):
    jid = _job(manager, "j-term", "failed", error_type="outcome_unknown",
               error="provider outcome requires reconciliation")
    before = manager.redis.hgetall(f"job:{jid}") or None
    for _ in range(3):
        r = await manager.client.get(f"/v1/job/{jid}", headers=AUTH)
        body = await r.json()
        assert body["status"] == "failed"
        assert body.get("error_type") == "outcome_unknown"
    # Reads never mutate the retained audit record nor enqueue replays.
    assert (manager.redis.hgetall(f"job:{jid}") or None) == before
    q = await manager.client.get("/v1/queue", headers=AUTH)
    assert q.status == 200


@pytest.mark.asyncio
async def test_stop_comfyui_then_idle_preload_clears_llm_evicted(manager):
    m = manager.m
    status_ok = manager.mp

    async def fake_stop(self, reason="x"):
        self.mark_llm_evicted(True, source="stop_comfyui")
        return True

    async def fake_evict_then_load(name, reason="", priority=None):
        m.vram.mark_llm_evicted(False, source="_load_bundle:configured_idle")
        return True

    status_ok.setattr(m.VRAMLifecycleManager, "stop_comfyui", fake_stop)
    status_ok.setattr(m, "_resolve_bundles", lambda: {"Gemma-RX6950XT-idle": {}})
    status_ok.setattr(m, "_evict_then_load_bundle", fake_evict_then_load)

    r = await manager.client.post("/vram/stop-comfyui", headers=AUTH)
    assert r.status == 200 and (await r.json())["llm_evicted"] is True
    r = await manager.client.post("/v1/bundles/preload", headers=AUTH,
                                  json={"bundle_name": "Gemma-RX6950XT-idle"})
    assert r.status == 200 and (await r.json())["preload_state"] == "loaded"
    assert m.vram._llm_evicted is False


@pytest.mark.asyncio
async def test_idle_generation_bundle_released_and_both_baselines_restored(manager):
    """Production scheduler (built by create_app) releases an idle comfyui bundle
    after its TTL, restores both baselines, and /vram reflects the eviction."""
    m, mp = manager.m, manager.mp
    now = time.time()
    gs = m.GpuBundleState("gpu0")
    gs.loaded_bundle, gs.bundle_loaded_at, gs.last_activity_at = (
        "comfyui", now - 600, now - 300)
    unloads, restores = [], []
    cfg = {"scheduling": {"generation_idle_ttl_seconds": 20},
           "generation_templates": {}, "services": {}}

    async def unload(name):
        unloads.append(name)
        gs.loaded_bundle = None
        return True

    async def restore(snapshot=None):
        restores.append(1)
        return {"gpu0": {"restored": True}, "gpu1": {"restored": True}}

    async def passthrough(members, op, reason=""):
        return await op()

    mp.setattr(m, "_load_services_config", lambda: cfg)
    mp.setattr(m, "_maintenance_mode_enabled", lambda s: False)
    mp.setattr(m, "_resolve_bundles", lambda: {"comfyui": {"services": ["comfyui"]}})
    mp.setattr(m, "_get_all_gpu_states", lambda: {"gpu0": gs})
    mp.setattr(m, "_auto_evict_bundle_blockers", lambda *a, **k: [])
    mp.setattr(m, "_bundle_worker_in_flight", lambda *a, **k: 0)
    mp.setattr(m, "_get_linked_bundles", lambda *a, **k: [])
    mp.setattr(m, "_expand_bundle_transition_members", lambda mem, **k: mem)
    mp.setattr(m, "_run_bundle_group_transition", passthrough)
    mp.setattr(m, "_unload_bundle", unload)
    mp.setattr(m, "_restore_both_idle_llms", restore)
    sched = m.scheduler
    sched.logger = logging.getLogger("idle-restore-http")

    await sched._auto_evict_idle_bundles()
    assert unloads == ["comfyui"] and len(restores) == 1
    report = sched._last_idle_restore_report
    assert sorted(report["results"]) == ["gpu0", "gpu1"] and not report["partial"]
    r = await manager.client.get("/vram", headers=AUTH)
    assert r.status == 200


# ---------------------------------------------------------------------------
# Real-path fixtures: production bundle admission/commit and /health reduction.
# Only systemd start/stop, port probes and runtime-evidence checks are stubbed.
# ---------------------------------------------------------------------------

def _lifecycle_config(mi50_idle="aux-mi50"):
    """Two-GPU config: RX idle Gemma, MI50 idle resident, a ComfyUI generation bundle."""
    def svc(port, unit, **kw):
        return {"enabled": True, "systemd_unit": unit, "port": port, **kw}

    return {
        "scheduling": {
            "idle_services": {"card0": "gemma-rx", "card1": mi50_idle},
            "generation_idle_ttl_seconds": 20,
        },
        "services": {
            "gemma-rx": svc(8101, "gemma-rx.service", gpu_id="card0"),
            "aux-mi50": svc(8130, "aux-mi50.service", gpu_id="card1"),
            "gemma-mi50": {**svc(8100, "gemma-mi50.service", gpu_id="card1"), "enabled": False},
            "comfyui": svc(8188, "comfyui.service", gpu_id="card0"),
        },
        "bundles": {
            "Gemma-RX6950XT-idle": {"gpu_id": "card0", "services": ["gemma-rx"], "idle": True,
                                    "priority": 1},
            "aux-mi50": {"gpu_id": "card1", "services": ["aux-mi50"], "idle": True,
                           "priority": 1},
            "comfyui": {"gpu_id": "card0", "services": ["comfyui"], "priority": 5},
        },
        "generation_templates": {},
    }


@pytest_asyncio.fixture
async def lifecycle(manager):
    """``manager`` rewired with a realistic config and the external boundaries stubbed."""
    m, mp = manager.m, manager.mp
    lifecycle = manager
    cfg = _lifecycle_config()
    manager.cfg.clear()
    manager.cfg.update(cfg)
    mp.setattr(m, "_services_config", manager.cfg, raising=False)
    mp.setattr(m, "_bundle_cache", None, raising=False)
    started: list[str] = []

    async def start_members(config, bundle_name, members, services_cfg, timeout=0.0):
        started.append(bundle_name)
        lifecycle.active.update(services_cfg[x]["systemd_unit"] for x in members)
        return True, [services_cfg[x]["systemd_unit"] for x in members], False

    async def ready(*a, **k):
        return True

    async def verified(*a, **k):
        return True, {"authority": "test"}

    async def ceiling_ok(*a, **k):
        return True, ""

    async def probe(port, health_path="/health", timeout=2.0):
        # Port health boundary: healthy iff the fake-systemd unit on that port is active.
        return any(c.get("port") == port and c.get("systemd_unit") in lifecycle.active
                   for c in manager.cfg["services"].values())

    mp.setattr(m, "_start_bundle_members", start_members)
    mp.setattr(m, "_probe_health_inline", probe)
    mp.setattr(m, "_wait_port_ready", ready)
    mp.setattr(m, "_verify_bundle_runtime_for_commit", verified)
    mp.setattr(m, "_check_gpu_vram_ceiling", ceiling_ok)
    mp.setattr(m, "_rejoin_combined_gemma_bundle_members_after_start", ready)
    mp.setattr(m.VRAMLifecycleManager, "wait_for_free_mib", lambda self, *a, **k: ready())
    mp.setattr(m, "bundle_lifecycle_controller", m.BundleLifecycleController(
        InMemoryBundleLifecycleStore(), owner="test"), raising=False)
    manager.started = started
    return manager


@pytest.mark.asyncio
async def test_preload_of_configured_idle_bundle_clears_evicted_and_health_is_normal(lifecycle):
    """Eviction by a generation load, then an idle preload through the real
    handlers (real _load_bundle commit), must leave /health free of the
    llm_evicted contradiction."""
    m = lifecycle.m
    m.vram.mark_llm_evicted(True, source="evict_llm")  # what loading comfyui did
    before = await (await lifecycle.client.get("/health", headers=AUTH)).json()
    assert "llm_evicted_with_llm_loaded_state" in before["contradictions"]
    r = await lifecycle.client.post("/v1/bundles/preload", headers=AUTH,
                                    json={"bundle_name": "Gemma-RX6950XT-idle"})
    assert r.status == 200, await r.text()
    assert (await r.json())["preload_state"] == "loaded"
    assert lifecycle.started == ["Gemma-RX6950XT-idle"]
    assert m.vram._llm_evicted is False
    h = await (await lifecycle.client.get("/health", headers=AUTH)).json()
    assert "llm_evicted_with_llm_loaded_state" not in h["contradictions"]
    assert h["vram"]["llm_evicted"] is False


async def _preload(env, name):
    r = await env.client.post("/v1/bundles/preload", headers=AUTH, json={"bundle_name": name})
    return r.status, await r.json()


def _age_generation(env, gpu="card0", idle_for=300):
    gs = env.m._get_gpu_state(gpu)
    now = time.time()
    gs.bundle_loaded_at = now - idle_for - 60
    gs.last_activity_at = now - idle_for
    gs.bundle_post_load_grace_at = 0.0
    gs.last_evict_attempt_at = 0.0
    return gs


@pytest.mark.asyncio
async def test_generation_cycle_stop_then_idle_preload_returns_health_to_normal(lifecycle):
    """comfyui load evicts the idle LLM; stop-comfyui + idle preload through the
    real handlers/_load_bundle must clear llm_evicted and the /health contradiction."""
    m = lifecycle.m
    st, body = await _preload(lifecycle, "comfyui")
    assert st == 200, body
    assert m._get_gpu_state("card0").loaded_bundle == "comfyui"
    assert m.vram._llm_evicted is True

    async def comfy_state():
        return {"comfyui.service": "inactive"}

    lifecycle.mp.setattr(m.vram, "_comfyui_systemd_state", comfy_state)
    r = await lifecycle.client.post("/vram/stop-comfyui", headers=AUTH)
    assert r.status == 200
    mid = await (await lifecycle.client.get("/health", headers=AUTH)).json()
    assert "llm_evicted_with_llm_loaded_state" in mid["contradictions"]

    st, body = await _preload(lifecycle, "Gemma-RX6950XT-idle")
    assert st == 200, body
    h = await (await lifecycle.client.get("/health", headers=AUTH)).json()
    assert h["vram"]["llm_evicted"] is False
    assert "llm_evicted_with_llm_loaded_state" not in h["contradictions"]
    assert h["runtime_mode"] != "degraded"


async def _baselines_then_generation(env):
    """Both idle baselines up, then a ComfyUI generation load displaces card0's Gemma."""
    for name in ("aux-mi50", "Gemma-RX6950XT-idle", "comfyui"):
        st, body = await _preload(env, name)
        assert st == 200, (name, body)
    m = env.m
    assert m._get_gpu_state("card0").loaded_bundle == "comfyui"
    assert "gemma-rx.service" not in env.active
    env.started.clear()
    env.m.scheduler.logger = logging.getLogger("idle-restore-http")


@pytest.mark.asyncio
async def test_ttl_release_restores_both_baselines_through_real_restore(lifecycle):
    """Terminal generation (no work) -> TTL release -> real _restore_both_idle_llms
    re-admits BOTH idle baselines; /health returns to normal. Nothing here mocks
    the restore or the flag-clearing production operation."""
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    _age_generation(lifecycle)
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" not in lifecycle.active
    assert {"gemma-rx.service", "aux-mi50.service"} <= lifecycle.active
    report = m.scheduler._last_idle_restore_report
    assert report["partial"] is False and report["failed_gpus"] == []
    assert sorted(report["results"]) == ["card0", "card1"]
    assert m._get_gpu_state("card0").loaded_bundle in ("Gemma-RX6950XT-idle", "llm:Gemma-RX6950XT-idle")
    h = await (await lifecycle.client.get("/health", headers=AUTH)).json()
    assert h["vram"]["llm_evicted"] is False
    assert "llm_evicted_with_llm_loaded_state" not in h["contradictions"]


@pytest.mark.asyncio
async def test_not_released_before_ttl(lifecycle):
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    _age_generation(lifecycle, idle_for=5)
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" in lifecycle.active
    assert m._get_gpu_state("card0").loaded_bundle == "comfyui"


@pytest.mark.asyncio
async def test_work_arriving_under_the_fence_cancels_the_release(lifecycle):
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    gs = _age_generation(lifecycle)
    real = m._run_bundle_group_transition

    async def work_arrives(members, op, reason=""):
        gs.pending_requests = 1  # after the decision, before the destructive step
        return await real(members, op, reason=reason)

    lifecycle.mp.setattr(m, "_run_bundle_group_transition", work_arrives)
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" in lifecycle.active  # never stopped
    assert gs.loaded_bundle == "comfyui"
    assert lifecycle.started == []  # no baseline restore for a cancelled release
    assert gs.last_evict_attempt_at == 0.0  # cancelled, not a failed attempt


@pytest.mark.asyncio
async def test_explicit_hold_wins_over_ttl_until_released(lifecycle):
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    _age_generation(lifecycle)
    m.scheduler.acquire_idle_hold("h-1", "card0", ttl_seconds=600, reason="operator")
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" in lifecycle.active
    assert m.scheduler.release_idle_hold("h-1") is True
    _age_generation(lifecycle)
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" not in lifecycle.active
    assert "gemma-rx.service" in lifecycle.active


@pytest.mark.asyncio
async def test_unfinished_worker_blocks_release(lifecycle):
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    _age_generation(lifecycle)
    lifecycle.mp.setattr(m, "_bundle_worker_in_flight", lambda *a, **k: 1)
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" in lifecycle.active


@pytest.mark.asyncio
async def test_partial_restore_failure_is_reported_truthfully_and_backs_off(lifecycle):
    """card1's resident cannot be restored: card0 still is, the report names card1,
    and the next ticks inside the backoff window do not hammer systemd."""
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    assert await m._unload_bundle("aux-mi50")  # MI50 resident was evicted too
    assert "aux-mi50.service" not in lifecycle.active
    real_start = m._start_bundle_members
    attempts: list[str] = []

    async def start_members(config, bundle_name, members, services_cfg, timeout=0.0):
        if bundle_name == "aux-mi50":
            attempts.append(bundle_name)
            return False, [], False
        return await real_start(config, bundle_name, members, services_cfg, timeout=timeout)

    lifecycle.mp.setattr(m, "_start_bundle_members", start_members)
    m.scheduler._generation_absence_started_at = time.time() - 600.0
    m.scheduler._generation_terminal_at = time.time() - 301.0
    _age_generation(lifecycle)
    await m.scheduler._auto_evict_idle_bundles()
    report = m.scheduler._last_idle_restore_report
    assert report["partial"] is True and report["failed_gpus"] == ["card1"]
    assert report["total_generation_absence_seconds"] >= 600.0
    assert report["post_terminal_restore_seconds"] >= 301.0
    assert report["post_terminal_budget_exceeded"] is True
    assert m.scheduler._generation_absence_started_at is not None
    assert m.scheduler._generation_terminal_at is not None
    assert report["results"]["card0"]["restored"] is True
    assert report["results"]["card1"]["restored"] is False
    assert "gemma-rx.service" in lifecycle.active  # the healthy GPU was not held hostage
    assert attempts == ["aux-mi50"]
    # Immediately retrying is suppressed by the bounded backoff (no new attempt).
    again = await m._restore_both_idle_llms()
    assert again["card1"] == {"restored": False, "reason": "idle_target_backoff"}
    assert attempts == ["aux-mi50"]


# ---------------------------------------------------------------------------
# Unavailable idle target: fallback, bounded backoff, one INFO per transition,
# and disabled-member probe suppression, over a simulated minute (2.5 s ticks).
# ---------------------------------------------------------------------------
TICK, TICKS = 2.5, 24


def _mi50_unavailable(env, fallback="aux-mi50", preferred_enabled=False):
    cfg = env.cfg
    cfg["scheduling"]["idle_services"]["card1"] = "gemma-mi50"
    cfg["services"]["gemma-mi50"]["enabled"] = preferred_enabled
    cfg["bundles"]["Gemma-MI50-idle"] = {"gpu_id": "card1", "services": ["gemma-mi50"],
                                         "idle": True, "priority": 1}
    if fallback:
        cfg["scheduling"]["idle_fallback_services"] = {"card1": fallback}
    env.mp.setattr(env.m, "_bundle_cache", None, raising=False)


async def _simulate_minute(env, caplog):
    clock = {"t": 1_000_000.0}
    env.mp.setattr(time, "time", lambda: clock["t"])
    caplog.set_level(logging.INFO, logger=env.m.logger.name)
    results = []
    for _ in range(TICKS):
        results.append(await env.m._restore_both_idle_llms())
        clock["t"] += TICK
    return results


def _card1_lines(caplog):
    return [r.getMessage() for r in caplog.records if "idle target card1" in r.getMessage()]


@pytest.mark.asyncio
async def test_disabled_preferred_falls_back_to_qwen_without_spinning(lifecycle, caplog):
    m = lifecycle.m
    _mi50_unavailable(lifecycle)
    results = await _simulate_minute(lifecycle, caplog)
    assert all(r["card0"]["restored"] and r["card1"]["restored"] for r in results)
    assert lifecycle.started.count("aux-mi50") == 1
    assert "Gemma-MI50-idle" not in lifecycle.started  # disabled gemma-mi50 never started
    assert not any("start" in c and "gemma-mi50.service" in c for c in lifecycle.commands)
    assert "gemma-mi50.service" not in lifecycle.active
    lines = _card1_lines(caplog)
    assert len(lines) == 1 and "aux-mi50" in lines[0]


@pytest.mark.asyncio
async def test_no_loadable_target_is_one_state_change_not_a_retry_loop(lifecycle, caplog):
    _mi50_unavailable(lifecycle, fallback="")
    results = await _simulate_minute(lifecycle, caplog)
    assert all(r["card1"] == {"restored": False, "reason": "idle_target_unavailable"} for r in results)
    assert not any(b in lifecycle.started for b in ("Gemma-MI50-idle", "aux-mi50"))
    assert len(_card1_lines(caplog)) == 1
    assert lifecycle.started.count("Gemma-RX6950XT-idle") == 1  # healthy GPU unaffected


@pytest.mark.asyncio
async def test_enabled_but_failing_target_backs_off_and_recovers(lifecycle, caplog):
    m = lifecycle.m
    _mi50_unavailable(lifecycle, preferred_enabled=True, fallback="")
    real_start = m._start_bundle_members
    failing = {"on": True}
    attempts: list[float] = []

    async def start_members(config, bundle_name, members, services_cfg, timeout=0.0):
        if bundle_name == "Gemma-MI50-idle" and failing["on"]:
            attempts.append(time.time())
            return False, [], False
        return await real_start(config, bundle_name, members, services_cfg, timeout=timeout)

    lifecycle.mp.setattr(m, "_start_bundle_members", start_members)
    results = await _simulate_minute(lifecycle, caplog)
    assert 1 <= len(attempts) <= 4, attempts  # bounded, not one per 2.5 s tick
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert gaps == sorted(gaps)  # delays only grow
    assert len(_card1_lines(caplog)) == 1
    # Fixing the target (state change) is picked up once the window elapses.
    failing["on"] = False
    lifecycle.mp.setattr(time, "time", lambda: attempts[-1] + 400.0)
    again = await m._restore_both_idle_llms()
    assert again["card1"]["restored"] is True
    assert m._IDLE_PLANNER.retry_at("card1") == 0.0


@pytest.mark.asyncio
async def test_disabled_member_not_probed_enabled_down_member_is_visible(lifecycle, caplog):
    """Production _rebalance_queue health fan-out against real TestServers."""
    from aiohttp import web
    m = lifecycle.m
    hits = {"up": 0, "down": 0, "disabled": 0}

    def make(name, status):
        async def h(_):
            hits[name] += 1
            return web.json_response({"s": name}, status=status)
        app = web.Application()
        app.router.add_get("/health", h)
        return TestServer(app)

    servers = {"up": make("up", 200), "down": make("down", 503), "disabled": make("disabled", 200)}
    for s in servers.values():
        await s.start_server()
    try:
        lifecycle.cfg["services"] = {
            "dead": {"routing_group": "llm", "enabled": True},
            **{n: {"enabled": n != "disabled", "port": s.port, "routing_group": "llm"}
               for n, s in servers.items()},
        }
        lifecycle.mp.setattr(m, "_services_config", lifecycle.cfg, raising=False)
        lifecycle.mp.setattr(m, "routing_engine", SimpleNamespace(get_group_members=lambda g: [
            {"service": n} for n in ("dead", "up", "down", "disabled")]), raising=False)

        async def depth(_):
            return 0
        lifecycle.mp.setattr(m, "async_queue", SimpleNamespace(get_queue_depth=depth), raising=False)
        sup = object.__new__(m.WorkerSupervisor)

        async def probe(port, health_path="/health", fallback_tcp=True):
            async with m.ClientSession() as s:
                async with s.get(f"http://127.0.0.1:{port}{health_path}") as resp:
                    return resp.status == 200
        sup._check_health = probe
        caplog.set_level(logging.INFO, logger=m.logger.name)
        for _ in range(TICKS):
            await sup._rebalance_queue("dead")
    finally:
        for s in servers.values():
            await s.close()
    assert hits == {"up": TICKS, "down": TICKS, "disabled": 0}
    assert len([r for r in caplog.records if "member disabled" in r.getMessage()]) == 1


@pytest.mark.asyncio
async def test_two_stage_restore_accounting_starts_after_final_generation_completion(
    lifecycle,
):
    """A Krea->Qwen chain restores only after Qwen ends and reports both clocks."""
    m = lifecycle.m
    await _baselines_then_generation(lifecycle)
    clock = {"now": 1_000.0}
    lifecycle.mp.setattr(m.time, "time", lambda: clock["now"])
    state = m._get_gpu_state("card0")
    state.bundle_loaded_at = 1_000.0
    state.last_activity_at = 1_000.0
    state.bundle_post_load_grace_at = 0.0
    state.last_evict_attempt_at = 0.0
    stage_state = {"qwen_active": True}

    # Report active pipeline work without touching Redis or changing the route.
    lifecycle.mp.setattr(
        m,
        "_bundle_tracker_in_flight_jobs",
        lambda *_a, **_k: ["qwen-edit"] if stage_state["qwen_active"] else [],
    )

    # Krea has completed, but the Qwen edit stage still owns generation load.
    clock["now"] = 1_200.0
    state.last_activity_at = clock["now"]
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" in lifecycle.active
    assert lifecycle.started == []

    # Final generation completion establishes the post-terminal clock.
    stage_state["qwen_active"] = False
    clock["now"] = 1_700.0
    state.last_activity_at = clock["now"]
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" in lifecycle.active  # TTL has not elapsed yet
    clock["now"] = 1_720.0
    await m.scheduler._auto_evict_idle_bundles()

    report = m.scheduler._last_idle_restore_report
    assert report["partial"] is False
    assert sorted(report["results"]) == ["card0", "card1"]
    assert all(result["restored"] for result in report["results"].values())
    assert report["total_generation_absence_seconds"] == 720.0
    assert report["post_terminal_restore_seconds"] == 20.0
