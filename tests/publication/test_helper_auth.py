from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from native_task_client import native_host_has_unresolved
from native_task_host import NativeTaskHost
from runtime_host_client import HostRuntimeClientError, HostSupervisorRuntimeAdapter
from runtime_host_supervisor import create_app


TOKEN = "synthetic-helper-token"


def _profile():
    return {
        "schema_version": "runtime-profile.v1",
        "engine": "test-engine",
        "adapter": "test-adapter",
        "model_set": "synthetic-set",
        "capabilities": ["inspect"],
        "resource_requirements": {},
    }


def test_runtime_client_uses_dedicated_token_file_and_never_controller_token(tmp_path):
    token_file = tmp_path / "helper-token"
    token_file.write_text(TOKEN, encoding="utf-8")
    adapter = HostSupervisorRuntimeAdapter(
        profile_name="synthetic",
        socket_path=tmp_path / "supervisor.sock",
        service_token_file=token_file,
    )
    adapter.bind_transition("owner", 1)
    payload = adapter._request("inspect", "instance", _profile())
    assert adapter._helper_auth.headers() == {"Authorization": f"Bearer {TOKEN}"}
    assert "GPU_MANAGER_API_TOKEN" not in json.dumps(payload)


def test_runtime_client_rejects_configured_missing_token_without_connecting(tmp_path):
    adapter = HostSupervisorRuntimeAdapter(
        profile_name="synthetic",
        socket_path=tmp_path / "supervisor.sock",
        service_token_file=tmp_path / "missing-token",
    )
    adapter.bind_transition("owner", 1)
    with pytest.raises(HostRuntimeClientError, match="helper service token"):
        adapter._request("inspect", "instance", _profile())


def test_runtime_supervisor_rejects_wrong_bearer_and_accepts_injected_token():
    class Supervisor:
        async def execute(self, request):
            return {
                "schema_version": "runtime-host-action-result.v1",
                "status": "completed",
                "action": request["action"],
                "instance_id": request["instance_id"],
                "profile_fingerprint": request["profile_fingerprint"],
                "transition_owner": request["transition_owner"],
                "transition_fence": request["transition_fence"],
            }

    async def run():
        app = create_app(Supervisor(), service_token=TOKEN)
        async with TestClient(TestServer(app)) as client:
            payload = {
                "action": "inspect",
                "instance_id": "instance",
                "profile_fingerprint": "f" * 64,
                "transition_owner": "owner",
                "transition_fence": 1,
            }
            wrong = await client.post("/v1/runtime/inspect", json=payload,
                                     headers={"Authorization": "Bearer wrong"})
            assert wrong.status == 401
            assert TOKEN not in await wrong.text()
            valid = await client.post("/v1/runtime/inspect", json=payload,
                                     headers={"Authorization": f"Bearer {TOKEN}"})
            assert valid.status == 200

    asyncio.run(run())


def test_runtime_supervisor_missing_configured_token_fails_closed(tmp_path):
    class Supervisor:
        async def execute(self, _request):
            raise AssertionError("authentication must reject before execution")

    async def run():
        app = create_app(
            Supervisor(), service_token_file=tmp_path / "missing-helper-token"
        )
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/runtime/inspect", json={"action": "inspect"}
            )
            assert response.status == 503

    asyncio.run(run())


def test_native_host_health_requires_injected_token(tmp_path):
    async def run():
        host = NativeTaskHost(tmp_path / "state", "http://127.0.0.1:9999/submit", 1,
                              service_token=TOKEN)
        async with TestClient(TestServer(host.app())) as client:
            wrong = await client.get("/health", headers={"Authorization": "Bearer wrong"})
            assert wrong.status == 401
            valid = await client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
            assert valid.status == 200

    asyncio.run(run())


def test_native_client_sends_token_to_health_endpoint():
    seen = {}

    async def run():
        async def health(request):
            seen["authorization"] = request.headers.get("Authorization")
            return web.json_response({"status": "ok", "unresolved_tasks": []})

        app = web.Application()
        app.router.add_get("/health", health)
        async with TestServer(app) as server:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                assert not await native_host_has_unresolved(
                    session,
                    str(server.make_url("")),
                    service_token=TOKEN,
                )

    asyncio.run(run())
    assert seen["authorization"] == f"Bearer {TOKEN}"
