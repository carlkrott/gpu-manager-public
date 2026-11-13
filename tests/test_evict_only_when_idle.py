"""Plan 4.7: an ``evict_only_when_idle`` bundle (strata-rx) loads only when the
GPU is idle and is never unloaded while it serves a request.

Drives the real gpu-manager facade (hyphenated file, loaded via
``spec_from_file_location``). GPU state, the bundle registry, the /slots probe
and the clock are replaced with fakes, so nothing sleeps for real, nothing
spawns a process and no GPU is touched.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FACADE = REPO / "scripts" / "gpu-manager.py"

STRATA = {
    "gpu_id": "rx6950xt", "idle": False, "priority": 3, "services": ["strata-rx"],
    "evict_only_when_idle": True, "idle_ttl_seconds": 900,
}
GEMMA_IDLE = {"gpu_id": "rx6950xt", "idle": True, "priority": 1, "services": ["Gemma-RX6950XT"]}
COMFYUI = {"gpu_id": "rx6950xt", "idle": False, "priority": 5, "services": ["comfyui"]}
LOWER = {"gpu_id": "rx6950xt", "idle": False, "priority": 2, "services": ["lower"]}
BUNDLES = {
    "strata-rx": STRATA, "Gemma-RX6950XT-idle": GEMMA_IDLE,
    "comfyui": COMFYUI, "lower": LOWER,
}


@pytest.fixture(scope="module")
def facade():
    scripts = str(REPO / "scripts")
    added = scripts not in sys.path
    if added:
        sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("gpu_manager_evict_only_when_idle", FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if added and scripts in sys.path:
            sys.path.remove(scripts)
    return module


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0
        self.sleeps = []

    def time(self):
        return self.now

    monotonic = time


def _fake_asyncio(real, clock):
    ns = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__")})

    async def sleep(delay=0, result=None):
        clock.now += float(delay or 0)
        clock.sleeps.append(delay)
        await real.sleep(0)
        return result

    ns.sleep = sleep
    return ns


def _gpu(loaded=None, loaded_bundles=(), state="idle", worker_in_flight=0, bundle_priority=0):
    return types.SimpleNamespace(
        loaded_bundle=loaded, loaded_bundles=list(loaded_bundles), state=state,
        worker_in_flight=worker_in_flight, bundle_priority=bundle_priority,
    )


@pytest.fixture
def rx(facade, monkeypatch):
    """One RX GPU with strata-rx resident; registry limited to BUNDLES."""
    states = {"rx6950xt": _gpu(loaded="strata-rx", bundle_priority=3)}
    monkeypatch.setattr(facade, "_gpu_states", states)
    monkeypatch.setattr(facade, "_get_gpu_state", lambda gid: states.get(gid))
    monkeypatch.setattr(facade, "_resolve_bundles", lambda: BUNDLES)
    monkeypatch.setattr(
        facade, "_bundle_physical_gpu_set",
        lambda name, bundles=None: {(bundles or BUNDLES)[name]["gpu_id"]},
    )
    monkeypatch.setattr(facade, "_collect_linked_gpus", lambda *a, **k: [])
    monkeypatch.setattr(facade, "scheduler", None, raising=False)

    async def no_resident(_gid):  # hermetic: never read live services.json / systemctl (Breccia)
        return None

    monkeypatch.setattr(facade, "_active_resident_unit", no_resident)
    return states


def _busy(answers):
    """Fake _bundle_busy: pops scripted answers, repeating the last one."""
    calls = []

    async def fake(bundle_name, bundles=None):
        calls.append(bundle_name)
        return answers.pop(0) if len(answers) > 1 else answers[0]

    fake.calls = calls
    return fake


# ── config ──────────────────────────────────────────────────────────────


def test_strata_rx_config_contract():
    cfg = json.loads((REPO / "config" / "services.json").read_text(encoding="utf-8"))
    bundle = cfg["bundles"]["strata-rx"]
    svc = cfg["services"]["strata-rx"]
    assert bundle["evict_only_when_idle"] is True
    assert bundle["idle"] is False and bundle["priority"] == 3
    assert bundle["gpu_id"] == "rx6950xt" and bundle["proxy_port"] is None
    assert bundle["resource_contract"] == "exclusive"
    assert svc["port"] == 8147 and svc["proxy_port"] is None
    assert svc["type"] == "llm_backend" and svc["routing_group"] == "strata-rx"
    assert set(svc["capabilities"]) == {"chat", "vision"}
    assert svc["mutex_group"] == cfg["services"]["Gemma-RX6950XT"]["mutex_group"]
    # Not a combined Gemma member.
    assert "broker_capabilities" not in svc
    gemma_group = json.dumps(cfg.get("routing_groups", {}).get("Gemma", {}))
    assert "strata-rx" not in gemma_group
    priorities = {name: b["priority"] for name, b in cfg["bundles"].items()}
    assert priorities["Gemma-RX6950XT-idle"] < 3 < priorities["comfyui"]


# ── pure helpers ────────────────────────────────────────────────────────


def test_refusal_rules(facade):
    refuse = facade._protected_occupant_refusal
    assert "idle bundle" in refuse("Gemma-RX6950XT-idle", GEMMA_IDLE, "strata-rx", STRATA)
    assert "priority 2 <= strata-rx priority 3" in refuse("lower", LOWER, "strata-rx", STRATA)
    assert refuse("same", dict(LOWER, priority=3), "strata-rx", STRATA)
    assert refuse("comfyui", COMFYUI, "strata-rx", STRATA) is None


def test_protected_bundles_on_gpu_strips_llm_prefix_and_skips_self(facade, rx):
    rx["rx6950xt"] = _gpu(loaded="llm:strata-rx", loaded_bundles=["comfyui", "strata-rx"])
    assert facade._protected_bundles_on_gpu("rx6950xt", BUNDLES) == ["strata-rx"]
    assert facade._protected_bundles_on_gpu("rx6950xt", BUNDLES, "strata-rx") == []
    assert facade._protected_bundles_on_gpu("mi50", BUNDLES) == []


class _Resp:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def json(self, content_type=None):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _session_factory(facade, outcome):
    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url):
            if isinstance(outcome, BaseException):
                raise outcome
            return _Resp(*outcome)

    return _Session


@pytest.mark.parametrize(
    "outcome, busy",
    [
        ((200, [{"id": 0, "is_processing": False}]), False),
        ((200, [{"id": 0, "is_processing": True}]), True),
        ((503, {"error": "loading"}), True),
        ((200, {"not": "a list"}), True),
        ((200, ValueError("bad json")), True),
        (asyncio.TimeoutError(), True),
    ],
)
def test_bundle_busy_probe_fails_closed(facade, rx, monkeypatch, outcome, busy):
    monkeypatch.setattr(facade, "_load_services_config", lambda: {"services": {"strata-rx": {"port": 8147}}})
    monkeypatch.setattr(facade, "capacity_tracker", types.SimpleNamespace(_active={}))
    monkeypatch.setattr(facade, "ClientSession", _session_factory(facade, outcome))
    assert asyncio.run(facade._bundle_busy("strata-rx", BUNDLES)) is busy


def test_bundle_busy_connection_refused_is_idle(facade, rx, monkeypatch):
    refused = facade.ClientConnectorError(
        types.SimpleNamespace(ssl=None, host="127.0.0.1", port=8147),
        OSError(111, "Connection refused"),
    )
    monkeypatch.setattr(facade, "_load_services_config", lambda: {"services": {"strata-rx": {"port": 8147}}})
    monkeypatch.setattr(facade, "capacity_tracker", types.SimpleNamespace(_active={}))
    monkeypatch.setattr(facade, "ClientSession", _session_factory(facade, refused))
    assert asyncio.run(facade._bundle_busy("strata-rx", BUNDLES)) is False


def test_bundle_busy_counts_in_flight_work_without_probing(facade, rx, monkeypatch):
    def no_http(*a, **k):
        raise AssertionError("must not probe /slots")

    monkeypatch.setattr(facade, "ClientSession", no_http)
    monkeypatch.setattr(facade, "_load_services_config", lambda: {"services": {"strata-rx": {"port": 8147}}})
    rx["rx6950xt"].worker_in_flight = 1
    assert asyncio.run(facade._bundle_busy("strata-rx", BUNDLES)) is True
    rx["rx6950xt"].worker_in_flight = 0
    monkeypatch.setattr(facade, "capacity_tracker", types.SimpleNamespace(_active={"strata-rx": 1}))
    assert asyncio.run(facade._bundle_busy("strata-rx", BUNDLES)) is True


# ── admission (_admit_bundle_backed_llm_service) ────────────────────────


def test_repeat_admission_of_resident_strata_short_circuits(facade, rx, monkeypatch):
    monkeypatch.setattr(facade, "_bundle_busy", _busy([True]))
    ok, bundle = asyncio.run(facade._admit_bundle_backed_llm_service("strata-rx", 3))
    assert ok is True and bundle["bundle_name"] == "strata-rx"


def test_strata_admission_refused_while_resident_occupant_busy(facade, rx, monkeypatch):
    # Another protected bundle on the GPU blocks strata's own cold load.
    other = dict(STRATA, services=["strata-b"], priority=1)
    bundles = dict(BUNDLES, **{"strata-b": other})
    monkeypatch.setattr(facade, "_resolve_bundles", lambda: bundles)
    rx["rx6950xt"] = _gpu(loaded="strata-b")
    monkeypatch.setattr(facade, "_bundle_busy", _busy([True]))
    ok, _ = asyncio.run(facade._admit_bundle_backed_llm_service("strata-rx", 3))
    assert ok is False


# ── evict-then-load (queue path / MCP preload) ──────────────────────────


def test_idle_bundle_never_evicts_strata(facade, rx, monkeypatch):
    busy = _busy([False])
    monkeypatch.setattr(facade, "_bundle_busy", busy)
    assert asyncio.run(facade._evict_then_load_bundle_unlocked("Gemma-RX6950XT-idle", reason="idle_restore")) is False
    assert busy.calls == []  # refused on policy before any probe


def test_lower_priority_bundle_never_evicts_strata(facade, rx, monkeypatch):
    monkeypatch.setattr(facade, "_bundle_busy", _busy([False]))
    assert asyncio.run(facade._evict_then_load_bundle_unlocked("lower")) is False


def test_worker_reserve_defers_while_strata_busy(facade, rx, monkeypatch):
    monkeypatch.setattr(facade, "_bundle_busy", _busy([True]))
    with pytest.raises(facade.BundleAdmissionDeferred):
        asyncio.run(facade._evict_then_load_bundle_unlocked("comfyui", reason="worker_reserve:comfyui"))


def test_queue_path_gives_up_after_wait_without_evicting(facade, rx, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(facade, "asyncio", _fake_asyncio(asyncio, clock))
    monkeypatch.setattr(facade, "time", types.SimpleNamespace(time=clock.time, monotonic=clock.monotonic))
    evicted = []

    class _Vram:
        async def evict_llm(self, *a, **k):
            evicted.append(a)
            return True

    monkeypatch.setattr(facade, "vram", _Vram())
    monkeypatch.setattr(facade, "_bundle_busy", _busy([True]))
    assert asyncio.run(facade._evict_then_load_bundle_unlocked("comfyui")) is False
    assert evicted == []
    assert sum(clock.sleeps) >= facade._EVICT_ONLY_WHEN_IDLE_WAIT_SECONDS


# ── evict_llm (every eviction path) ─────────────────────────────────────


def test_evict_llm_refuses_while_strata_busy(facade, monkeypatch):
    async def busy_probe(gpu_id, skip_bundle_name=None):
        return "strata-rx" if gpu_id == "rx6950xt" else None

    class Vram:
        calls = []

        async def evict_llm(self, **kw):
            self.calls.append(kw)
            return True

    monkeypatch.setattr(facade, "_protected_occupant_busy_on_gpu", busy_probe)
    vram = Vram()
    guarded = facade._evict_llm_unless_protected_busy
    assert asyncio.run(guarded(vram, reason="t", target_gpu="rx6950xt", admission_authorized=True)) is False
    assert vram.calls == []
    assert asyncio.run(guarded(vram, reason="t", target_gpu="mi50", admission_authorized=True)) is True
    assert vram.calls == [{"reason": "t", "target_gpu": "mi50", "skip_bundle_name": None,
                           "admission_authorized": True}]


def test_every_facade_eviction_goes_through_the_guard(facade):
    import ast
    tree = ast.parse(Path(facade.__file__).read_text(encoding="utf-8"))
    direct = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "evict_llm"
    ]
    guard = next(n for n in tree.body if getattr(n, "name", None) == "_evict_llm_unless_protected_busy")
    assert direct and all(guard.lineno <= ln <= guard.end_lineno for ln in direct)


# ── scheduler: idle restore and idle-TTL auto-evict ─────────────────────


def test_idle_restore_skips_gpu_holding_strata(facade, rx):
    assert facade._protected_bundles_on_gpu("rx6950xt", facade._resolve_bundles()) == ["strata-rx"]
    rx["rx6950xt"] = _gpu(loaded="Gemma-RX6950XT-idle")
    assert facade._protected_bundles_on_gpu("rx6950xt", facade._resolve_bundles()) == []
