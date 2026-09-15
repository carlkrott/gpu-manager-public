from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from api_auth import (
    api_auth_middleware,
    bearer_matches,
    is_loopback_peer,
    request_is_authorized,
)
from combined_gemma_broker_service import create_standalone_app


ROOT = Path(__file__).parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


def _write_token(tmp_path: Path, value: str = "synthetic-publication-token") -> Path:
    path = tmp_path / "credentials" / "gpu-manager-api-token"
    path.parent.mkdir(parents=True)
    path.write_text(value, encoding="utf-8")
    return path


def _load_controller(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_gpu_manager_auth", CONTROLLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    services_path = tmp_path / "services.json"
    services_path.parent.mkdir(parents=True, exist_ok=True)
    services_path.write_text(json.dumps(module.empty_services_config()), encoding="utf-8")
    providers_path = tmp_path / "providers.json"
    providers_path.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    module.SERVICES_CONFIG_PATH = str(services_path)
    module.PROVIDERS_CONFIG_PATH = str(providers_path)

    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    fake_redis.connection_pool.connection_kwargs = {"decode_responses": True}
    module._redis_mod.Redis.return_value = fake_redis

    app = module.create_app()
    app.on_startup.clear()
    app.on_cleanup.clear()
    return app


def _fake_broker_service():
    service = MagicMock()
    service.health_live.return_value = {"status": "ok"}
    service.broker_metrics = AsyncMock(return_value={"metrics": "synthetic"})
    service.settings.registry = {"must_not_be_public": "synthetic"}
    return service


async def _request(app, method: str, path: str, headers: dict[str, str] | None = None):
    async with TestClient(TestServer(app)) as client:
        response = await client.request(method, path, headers=headers or {})
        return response.status, await response.text(), dict(response.headers)


def test_missing_token_rejects_loopback_and_sensitive_reads(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(tmp_path / "missing-token"))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)

    async def run():
        broker_status, _, _ = await _request(
            create_standalone_app(_fake_broker_service()), "GET", "/v1/registry/diagnostics"
        )
        assert broker_status == 503

    asyncio.run(run())


def test_bearer_requires_exact_constant_time_credential(monkeypatch, tmp_path):
    token_path = _write_token(tmp_path)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(token_path))

    request = SimpleNamespace(headers={"Authorization": "Bearer synthetic-publication-token"}, remote="127.0.0.1")
    wrong = SimpleNamespace(headers={"Authorization": "Bearer wrong"}, remote="127.0.0.1")
    assert request_is_authorized(request, allow_loopback=False)
    assert not request_is_authorized(wrong, allow_loopback=False)
    assert bearer_matches(request, "synthetic-publication-token")
    assert not bearer_matches(wrong, "synthetic-publication-token")


def test_loopback_compatibility_is_explicit_and_forwarded_headers_are_ignored(monkeypatch):
    monkeypatch.setenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", "1")
    loopback = SimpleNamespace(headers={}, remote="127.0.0.1")
    documentation_peer = SimpleNamespace(
        headers={"X-Forwarded-For": "127.0.0.1"}, remote="192.0.2.1"
    )
    assert is_loopback_peer(loopback)
    assert request_is_authorized(loopback, expected=None, allow_loopback=True)
    assert not is_loopback_peer(documentation_peer)
    assert not request_is_authorized(documentation_peer, expected=None, allow_loopback=True)


def test_options_is_metadata_only_and_does_not_reach_handler(monkeypatch):
    monkeypatch.delenv("GPU_MANAGER_API_TOKEN_FILE", raising=False)
    called = False

    async def handler(_request):
        nonlocal called
        called = True
        raise AssertionError("OPTIONS must not invoke the route handler")

    async def run():
        app = web.Application(middlewares=[api_auth_middleware])
        app.router.add_post("/sensitive", handler)
        async with TestClient(TestServer(app)) as client:
            response = await client.options("/sensitive")
            assert response.status == 204
            assert response.headers["Allow"] == "GET, HEAD, OPTIONS"

    asyncio.run(run())
    assert not called


def test_controller_factory_protects_get_head_post_and_exposes_only_liveness(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("GPU_MANAGER_API_TOKEN_FILE", raising=False)
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    app = _load_controller(monkeypatch, tmp_path)

    async def run():
        status, body, _ = await _request(app, "GET", "/health/live")
        assert status == 200 and body == '{"status": "ok"}'
        status, _, _ = await _request(app, "HEAD", "/health/live")
        assert status == 200
        status, _, _ = await _request(app, "GET", "/v1/stats")
        assert status == 503
        status, _, _ = await _request(app, "POST", "/v1/submit")
        assert status == 503
        status, body, _ = await _request(app, "GET", "/dashboard")
        assert status == 200 and "Memory-only dashboard credential flow" in body
        token_path = _write_token(tmp_path)
        monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(token_path))
        status, _, _ = await _request(
            app,
            "GET",
            "/dashboard",
            {"Authorization": "Bearer synthetic-publication-token"},
        )
        assert status == 200

    asyncio.run(run())


def test_standalone_broker_factory_accepts_valid_bearer_on_protected_routes(
    monkeypatch, tmp_path
):
    token_path = _write_token(tmp_path)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(token_path))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    headers = {"Authorization": "Bearer synthetic-publication-token"}

    broker = create_standalone_app(_fake_broker_service())

    async def run():
        broker_status, body, _ = await _request(
            broker, "GET", "/v1/registry/diagnostics", headers
        )
        assert broker_status == 200
        assert "must_not_be_public" in body
        status, _, _ = await _request(broker, "DELETE", "/v1/gemma/jobs/synthetic", headers)
        assert status != 401

    asyncio.run(run())
