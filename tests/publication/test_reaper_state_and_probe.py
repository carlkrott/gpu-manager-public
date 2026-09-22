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


def test_maintenance_mode_dominates_scheduler_ownership_and_dispatch(monkeypatch):
    module = _load_controller()
    module._services_config = {
        "scheduling": {
            "maintenance_mode": True,
            "scheduler_owns_load": True,
            "scheduler_dry_run_mode": False,
            "proactive_scheduling_enabled": True,
        }
    }
    monkeypatch.setenv("GPU_MANAGER_SCHEDULER_OWNS_LOAD", "true")

    config = module._phase3_scheduling_config()

    assert config["maintenance_mode"] is True
    assert config["scheduler_owns_load"] is False
    assert config["scheduler_dry_run_mode"] is True
    assert config["proactive_scheduling_enabled"] is False


def test_invalid_maintenance_value_fails_safe(monkeypatch):
    module = _load_controller()
    monkeypatch.setenv("GPU_MANAGER_SCHEDULER_OWNS_LOAD", "true")
    monkeypatch.setattr(module, "ORPHAN_REAPER_ENABLED", True, raising=False)
    monkeypatch.setattr(
        module,
        "_get_loaded_runtime_snapshot",
        lambda: {"services": [], "bundles": []},
        raising=False,
    )
    monkeypatch.setattr(module, "_get_idle_service_name", lambda: "", raising=False)
    monkeypatch.setattr(module, "_gpu_states", {}, raising=False)
    monkeypatch.setattr(
        module,
        "vram",
        SimpleNamespace(
            gpu_state=SimpleNamespace(value="llm_loaded"),
            _llm_evicted=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "llm",
        SimpleNamespace(active_requests=0, queued_requests=0),
        raising=False,
    )

    for invalid in ("true", 1, None):
        module._services_config = {
            "scheduling": {
                "maintenance_mode": invalid,
                "scheduler_owns_load": True,
                "scheduler_dry_run_mode": False,
                "proactive_scheduling_enabled": True,
            }
        }
        monkeypatch.setattr(
            module,
            "_load_services_config",
            lambda: module._services_config,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "scheduler",
            SimpleNamespace(
                _running=True,
                _last_orphan_reaper_at=99.0,
                _expected_pids=set(),
                _terminated_pids={},
                _pinned_service="",
                _restoring=False,
            ),
            raising=False,
        )

        config = module._phase3_scheduling_config()
        reaper = module._orphan_reaper_status(now=100.0)
        semantics = module._canonical_runtime_semantics(
            {"available": False, "availability_sources": []}
        )

        assert config["maintenance_mode"] is True
        assert config["scheduler_owns_load"] is False
        assert config["scheduler_dry_run_mode"] is True
        assert config["proactive_scheduling_enabled"] is False
        assert reaper["state"] == "paused"
        assert reaper["reason"] == "maintenance_mode"
        assert semantics["runtime_mode"] == "maintenance"
        assert semantics["configured_intent"]["maintenance_mode"] is True


def test_service_recovery_observations_use_fail_closed_maintenance_normalizer():
    source = CONTROLLER.read_text(encoding="utf-8")

    assert 'maintenance_mode=bool(scheduling.get("maintenance_mode", False))' not in source


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


def test_per_gpu_idle_recovery_bypasses_legacy_global_restore(monkeypatch):
    module = _load_controller()
    config = {
        "services": {
            "synthetic-idle": {
                "enabled": True,
                "gpu_id": "gpu-a",
                "port": 9123,
                "systemd_unit": "synthetic-idle.service",
                "type": "llm_backend",
            }
        },
        "scheduling": {
            "maintenance_mode": False,
            "idle_service": "",
            "idle_services": {"gpu-a": "synthetic-idle"},
        },
        "gpu_devices": {"gpu-a": {}},
    }

    class DownSession:
        def get(self, *_args, **_kwargs):
            raise ConnectionError("synthetic idle service is stopped")

    class FakeVram:
        _llm_evicted = True
        gpu_state = module.GpuState.GPU_WORK

        async def _wait_for_llm_healthy(self, *, timeout):
            return True

        def mark_llm_evicted(self, value, *, source):
            self._llm_evicted = value

    calls: list[tuple[str, str]] = []
    decisions = iter(
        (
            SimpleNamespace(
                action=module.RecoveryAction.PROBE,
                next=module.RecoveryState(),
                reason="probe_required",
                emit_transition_log=False,
            ),
            SimpleNamespace(
                action=module.RecoveryAction.RESTART,
                next=module.RecoveryState(),
                reason="probe_failed",
                emit_transition_log=False,
            ),
        )
    )

    async def fake_generation_tenant(_gpu_id):
        return False, ""

    async def fake_transition(_members, operation, *, reason):
        return await operation()

    async def fake_unload(bundle_name):
        calls.append(("unload", bundle_name))
        return True

    async def fake_load(bundle_name, *, timeout):
        calls.append(("load", bundle_name))
        return True

    monkeypatch.setattr(module, "_load_services_config", lambda: config)
    monkeypatch.setattr(
        module,
        "_get_all_idle_services",
        lambda: [
            {
                "gpu_id": "gpu-a",
                "service_name": "synthetic-idle",
                "port": 9123,
                "systemd_unit": "synthetic-idle.service",
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_get_idle_bundle",
        lambda _gpu_id: {
            "bundle_name": "synthetic-idle-bundle",
            "idle": True,
            "services": ["synthetic-idle"],
        },
    )
    monkeypatch.setattr(
        module,
        "_resolve_bundles",
        lambda: {
            "synthetic-idle-bundle": {
                "idle": True,
                "services": ["synthetic-idle"],
            }
        },
    )
    monkeypatch.setattr(module, "_restore_after_gpu_work_blockers", lambda: [])
    monkeypatch.setattr(module, "_gpu_has_active_generation_tenant", fake_generation_tenant)
    monkeypatch.setattr(module, "_expand_bundle_transition_members", lambda roots, **_kwargs: roots)
    monkeypatch.setattr(module, "_run_bundle_group_transition", fake_transition)
    monkeypatch.setattr(module, "_unload_bundle", fake_unload)
    monkeypatch.setattr(module, "_load_bundle", fake_load)
    monkeypatch.setattr(module, "reduce_recovery", lambda *_args, **_kwargs: next(decisions))
    monkeypatch.setattr(module, "bundle_lifecycle_controller", None)
    monkeypatch.setattr(module, "_gpu_states", {"gpu-a": SimpleNamespace(state="idle")})
    monkeypatch.setattr(module, "llm", SimpleNamespace(active_requests=0))
    monkeypatch.setattr(module, "vram", FakeVram())

    scheduler = module.GPUScheduler(DownSession(), module.logger)
    scheduler._lifecycle_begin = lambda *_args, **_kwargs: None
    scheduler._lifecycle_step_begin = lambda *_args, **_kwargs: None
    scheduler._lifecycle_step_done = lambda *_args, **_kwargs: None
    scheduler._lifecycle_complete = lambda: None

    async def legacy_restore():
        calls.append(("legacy_restore", ""))

    scheduler._restore_after_gpu_work = legacy_restore

    asyncio.run(scheduler._ensure_idle_service())

    assert ("legacy_restore", "") not in calls
    assert calls == [
        ("unload", "synthetic-idle-bundle"),
        ("load", "synthetic-idle-bundle"),
    ]
