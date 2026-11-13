"""Regression coverage for routed LLM members without a local backend port."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
CONTROLLER_PATH = SCRIPTS / "gpu-manager.py"
SERVICES_PATH = ROOT / "config" / "services.json"


def _load_controller():
    scripts_path = str(SCRIPTS)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location("gpu_manager_portless_test", CONTROLLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _real_services():
    return json.loads(SERVICES_PATH.read_text())["services"]


def _request(service_name: str, *, route: str):
    headers = {"X-Backend": service_name} if route == "header" else {}
    query = {"service": service_name} if route == "query" else {}
    return SimpleNamespace(
        path="/v1/chat/completions",
        method="POST",
        headers=headers,
        query=query,
        can_read_body=False,
        read=lambda: asyncio.sleep(0, result=b""),
        app={},
    )


def _configure_safe_handler(monkeypatch, services):
    module = _load_controller()
    module._services_config = {"services": services}
    module._model_validator = None
    module.routing_engine = None
    module._scoped_gemma_request_is_authorized = lambda _request: False
    module._combined_gemma_ingress_service_name = lambda name: name

    async def no_combined_ingress(**_kwargs):
        return None

    module._maybe_handle_combined_gemma_ingress = no_combined_ingress
    module.llm.total_requests = 0
    forwarded = []
    activations = []
    gpu_effects = []

    async def activate(name):
        activations.append(name)
        return "http://localhost:8124"

    module._ensure_mutex_exclusive = activate
    module._get_gpu_state = lambda *_args, **_kwargs: gpu_effects.append("gpu_state")
    module._admission_gate = lambda *_args, **_kwargs: gpu_effects.append("admission")

    async def forward(method, path, body, headers, *, backend_url):
        forwarded.append((method, path, backend_url))
        from aiohttp import web
        return web.Response(status=200, text="forwarded")

    monkeypatch.setattr(module, "_forward_llm", forward)
    return module, forwarded, activations, gpu_effects


@pytest.mark.parametrize("service_name", ["Luna-6-Low", "Gemini-3.8-Flash-Low"])
@pytest.mark.parametrize("route", ["query", "header"])
def test_portless_cloud_members_are_rejected_cleanly(monkeypatch, service_name, route):
    real_services = _real_services()
    entry = copy.deepcopy(real_services[service_name])
    assert entry["enabled"] is True
    assert entry["type"] == "llm_backend"
    assert "port" not in entry
    assert "gpu_id" not in entry

    module, forwarded, activations, gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )
    response = asyncio.run(module.handle_llm_proxy(_request(service_name, route=route)))

    assert response.status == 422
    assert json.loads(response.text) == {
        "error": "This member has no local backend port; use the combined router with model=gemma / X-Backend Gemma and its existing member-pin interface.",
        "code": "llm_backend_requires_combined_route",
        "service": service_name,
    }
    assert not forwarded
    assert not activations
    assert not gpu_effects


def test_portless_mutex_member_is_rejected_before_activation(monkeypatch):
    service_name = "portless-mutex-test"
    entry = {"enabled": True, "type": "llm_backend", "mutex_group": "shared"}
    module, forwarded, activations, gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )

    response = asyncio.run(module.handle_llm_proxy(_request(service_name, route="query")))

    assert response.status == 422
    assert json.loads(response.text)["service"] == service_name
    assert not forwarded
    assert not activations
    assert not gpu_effects


def test_portless_routing_group_member_is_rejected(monkeypatch):
    group_name = "portless-group"
    service_name = "portless-routed-member"
    entry = {"enabled": True, "type": "llm_backend"}
    module, forwarded, activations, gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )
    module.routing_engine = SimpleNamespace(
        get_group_members=lambda name: [service_name] if name == group_name else [],
        find_service_for_group=lambda *_args, **_kwargs: service_name,
    )
    module._queue_depth_for_group = lambda _name: 0

    response = asyncio.run(
        module.handle_llm_proxy(_request(group_name, route="header"))
    )

    assert response.status == 422
    assert json.loads(response.text)["service"] == service_name
    assert not forwarded
    assert not activations
    assert not gpu_effects


def test_combined_ingress_response_precedes_portless_guard(monkeypatch):
    service_name = "Luna-6-Low"
    entry = copy.deepcopy(_real_services()[service_name])
    from aiohttp import web

    module, forwarded, activations, gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )

    async def combined_ingress(**_kwargs):
        return web.Response(status=202, text="brokered")

    module._maybe_handle_combined_gemma_ingress = combined_ingress
    response = asyncio.run(module.handle_llm_proxy(_request(service_name, route="query")))

    assert response.status == 202
    assert response.text == "brokered"
    assert not forwarded
    assert not activations
    assert not gpu_effects


def test_port_bearing_cpu_member_keeps_direct_forwarding(monkeypatch):
    service_name = "CPU-direct-test"
    entry = {"enabled": True, "type": "llm_backend", "port": 8123}
    module, forwarded, _activations, _gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )

    response = asyncio.run(module.handle_llm_proxy(_request(service_name, route="query")))

    assert response.status == 200
    assert forwarded == [("POST", "/v1/chat/completions", "http://localhost:8123")]


def test_port_bearing_mutex_member_keeps_activation_forwarding(monkeypatch):
    service_name = "CPU-mutex-test"
    entry = {"enabled": True, "type": "llm_backend", "port": 8124, "mutex_group": "shared"}
    module, forwarded, activations, _gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )
    response = asyncio.run(module.handle_llm_proxy(_request(service_name, route="query")))

    assert response.status == 200
    assert activations == [service_name]
    assert forwarded == [("POST", "/v1/chat/completions", "http://localhost:8124")]


@pytest.mark.parametrize("route", ["query", "header"])
def test_disabled_member_pin_is_rejected_not_rerouted(monkeypatch, route):
    # Audit A6: a pin to a disabled service must not fall through to the default LLM group.
    service_name = "disabled-pin-test"
    entry = {"enabled": False, "type": "llm_backend", "port": 8999, "gpu_id": "gpu0"}
    module, forwarded, activations, gpu_effects = _configure_safe_handler(
        monkeypatch, {service_name: entry}
    )

    response = asyncio.run(module.handle_llm_proxy(_request(service_name, route=route)))

    assert response.status == 409
    assert json.loads(response.text)["code"] == "service_disabled"
    assert not forwarded
    assert not activations
    assert not gpu_effects
