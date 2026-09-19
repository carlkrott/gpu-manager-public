from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "candidate_gpu_manager_reaper_state", CONTROLLER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reaper_state_is_paused_during_maintenance():
    module = _load_controller()
    module.ORPHAN_REAPER_ENABLED = True
    module._services_config = {"scheduling": {"maintenance_mode": True}}
    module.scheduler = SimpleNamespace(
        _running=False,
        _last_orphan_reaper_at=90.0,
        _expected_pids=set(),
        _terminated_pids={},
    )

    state = module._orphan_reaper_status(now=100.0)

    assert state["state"] == "paused"
    assert state["reason"] == "maintenance_mode"
    assert state["last_tick_age_s"] == 10.0


def test_reaper_state_is_running_only_with_fresh_scheduler_tick():
    module = _load_controller()
    module.ORPHAN_REAPER_ENABLED = True
    module._services_config = {"scheduling": {"maintenance_mode": False}}
    module.scheduler = SimpleNamespace(
        _running=True,
        _last_orphan_reaper_at=99.0,
        _expected_pids=set(),
        _terminated_pids={},
    )

    state = module._orphan_reaper_status(now=100.0)

    assert state["state"] == "running"
    assert state["reason"] == "fresh_tick"


def test_reaper_state_is_disabled_when_configured_off():
    module = _load_controller()
    module.ORPHAN_REAPER_ENABLED = False
    module._services_config = {"scheduling": {"maintenance_mode": False}}
    module.scheduler = None

    state = module._orphan_reaper_status(now=100.0)

    assert state["state"] == "disabled"
    assert state["reason"] == "configuration_disabled"


def test_llm_health_skips_probe_without_registry_idle_service():
    module = _load_controller()
    module._load_services_config = lambda: {"services": {}, "scheduling": {}}
    proxy = module.LLMProxy()

    class ExplodingSession:
        def get(self, *_args, **_kwargs):
            raise AssertionError("unconfigured idle health must not probe a URL")

    proxy.session = ExplodingSession()

    assert asyncio.run(proxy.check_health()) is False


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return {"status": "ok"}


class _Session:
    closed = False

    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return _Response()


def test_llm_health_rejects_control_characters_in_registry_path():
    module = _load_controller()
    module._load_services_config = lambda: {
        "services": {
            "synthetic-idle": {
                "enabled": True,
                "port": 9123,
                "health_path": "/ready\r\nHost: injected.example",
            }
        },
        "scheduling": {"idle_service": "synthetic-idle"},
    }
    proxy = module.LLMProxy()
    session = _Session()
    proxy.session = session

    assert asyncio.run(proxy.check_health()) is False
    assert session.urls == []


def test_llm_health_uses_registry_idle_service_when_configured():
    module = _load_controller()
    module._load_services_config = lambda: {
        "services": {
            "synthetic-idle": {
                "enabled": True,
                "port": 9123,
                "health_path": "/ready",
            }
        },
        "scheduling": {"idle_service": "synthetic-idle"},
    }
    proxy = module.LLMProxy()
    session = _Session()
    proxy.session = session

    assert asyncio.run(proxy.check_health()) is True
    assert session.urls == ["http://127.0.0.1:9123/ready"]
