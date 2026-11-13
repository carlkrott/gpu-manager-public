"""Plan 3.4: the idle swap/restart health wait polls the GPU's own idle service.

Live 2026-10-11 03:08-03:09: Gemma-RX6950XT-idle loaded (TX_VERIFY passed), then
the restart waited 90 s on the :8092 default port, declared the member "not
healthy after restart" and unloaded it. The restart lifecycle also had no step
template, so every step logged "unknown step".
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FACADE = REPO / "scripts" / "gpu-manager.py"

CONFIG = {
    "scheduling": {"idle_service": "", "idle_services": {"rx6950xt": "Gemma-RX6950XT"}},
    "services": {"Gemma-RX6950XT": {"port": 8094}},
}


@pytest.fixture(scope="module")
def facade():
    scripts = str(REPO / "scripts")
    added = scripts not in sys.path
    if added:
        sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("gpu_manager_idle_restart_health", FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if added and scripts in sys.path:
            sys.path.remove(scripts)
    return module


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, healthy_port):
        self.healthy_port = healthy_port
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if f":{self.healthy_port}/" in url:
            return _Resp(200, {"status": "ok"})
        raise ConnectionRefusedError(url)


class _Vram:
    def __init__(self, session):
        self.session = session


@pytest.fixture
def wired(facade, monkeypatch):
    session = _Session(healthy_port=8094)
    monkeypatch.setattr(facade, "_load_services_config", lambda: CONFIG)
    monkeypatch.setattr(facade, "vram", _Vram(session))

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(facade.asyncio, "sleep", no_sleep)
    return facade, session


def test_idle_health_wait_uses_the_gpu_idle_port(wired):
    facade, session = wired
    assert asyncio.run(facade._wait_for_idle_service_healthy("rx6950xt", timeout=5)) is True
    assert session.urls == ["http://localhost:8094/health"]


def test_gpu_less_port_lookup_still_falls_back_to_8092(wired):
    facade, _ = wired
    assert facade._get_idle_service_port() == 8092
    assert facade._get_idle_service_port("rx6950xt") == 8094


def test_unhealthy_idle_service_times_out_false(facade, monkeypatch):
    session = _Session(healthy_port=1)
    monkeypatch.setattr(facade, "_load_services_config", lambda: CONFIG)
    monkeypatch.setattr(facade, "vram", _Vram(session))
    clock = iter(range(0, 1000, 4))
    monkeypatch.setattr(facade.time, "time", lambda: float(next(clock)))

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(facade.asyncio, "sleep", no_sleep)
    assert asyncio.run(facade._wait_for_idle_service_healthy("rx6950xt", timeout=10)) is False
    assert session.urls and all(":8094/" in u for u in session.urls)


def test_swap_and_restart_paths_no_longer_call_the_gpu_less_wait():
    src = FACADE.read_text()
    assert "vram._wait_for_llm_healthy(" not in src
    assert src.count("await _wait_for_idle_service_healthy(\n") == 2


def test_restart_lifecycle_has_a_step_template(facade):
    ids = [s["id"] for s in facade.TRANSITION_STEPS["restart"]]
    assert ids == ["stop_old", "start_new", "wait_healthy"]
