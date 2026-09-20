from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from native_task_client import native_host_has_unresolved
from native_task_host import NativeTaskHost
from runtime_host_client import (
    HelperCredentialError,
    HelperTokenAuth,
    HostRuntimeClientError,
    HostSupervisorRuntimeAdapter,
    resolve_required_service_credential,
)
from runtime_host_supervisor import create_app, create_unauthenticated_test_app


TOKEN = "synthetic-helper-token"


def _load_controller():
    root = Path(__file__).resolve().parents[2]
    source = root / "scripts" / "gpu-manager.py"
    spec = importlib.util.spec_from_file_location(
        "gpu_manager_distinct_helper_credentials", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_runtime_client_environment_uses_host_supervisor_credential_only(
    monkeypatch, tmp_path
):
    import runtime_host_client

    host_token = tmp_path / "host-supervisor-token"
    native_token = tmp_path / "native-task-token"
    host_token.write_text("host-supervisor-secret", encoding="utf-8")
    native_token.write_text("native-task-secret", encoding="utf-8")
    monkeypatch.setenv(
        "GPU_MANAGER_HOST_SUPERVISOR_TOKEN_FILE", str(host_token)
    )
    monkeypatch.setenv("GPU_MANAGER_NATIVE_TASK_TOKEN_FILE", str(native_token))
    monkeypatch.delenv("GPU_MANAGER_HELPER_TOKEN_FILE", raising=False)

    assert runtime_host_client.HOST_SUPERVISOR_TOKEN_FILE_ENV == (
        "GPU_MANAGER_HOST_SUPERVISOR_TOKEN_FILE"
    )
    assert runtime_host_client.NATIVE_TASK_TOKEN_FILE_ENV == (
        "GPU_MANAGER_NATIVE_TASK_TOKEN_FILE"
    )
    assert (
        runtime_host_client.HOST_SUPERVISOR_TOKEN_FILE_ENV
        != runtime_host_client.NATIVE_TASK_TOKEN_FILE_ENV
    )
    adapter = HostSupervisorRuntimeAdapter(
        profile_name="synthetic",
        socket_path=tmp_path / "supervisor.sock",
    )
    assert adapter._helper_auth.headers() == {
        "Authorization": "Bearer host-supervisor-secret"
    }


def test_legacy_shared_helper_credential_is_warned_and_ignored(
    monkeypatch, tmp_path, caplog
):
    legacy_token = tmp_path / "legacy-helper-token"
    legacy_token.write_text("legacy-shared-secret", encoding="utf-8")
    monkeypatch.setenv("GPU_MANAGER_HELPER_TOKEN_FILE", str(legacy_token))
    monkeypatch.delenv("GPU_MANAGER_HOST_SUPERVISOR_TOKEN_FILE", raising=False)

    with caplog.at_level("WARNING"):
        with pytest.raises(HelperCredentialError, match="helper service credential"):
            resolve_required_service_credential()

    assert "GPU_MANAGER_HELPER_TOKEN_FILE is deprecated and ignored" in caplog.text


def test_controller_native_helper_uses_native_task_credential_only(monkeypatch, tmp_path):
    host_token = tmp_path / "host-supervisor-token"
    native_token = tmp_path / "native-task-token"
    host_token.write_text("host-supervisor-secret", encoding="utf-8")
    native_token.write_text("native-task-secret", encoding="utf-8")
    monkeypatch.setenv(
        "GPU_MANAGER_HOST_SUPERVISOR_TOKEN_FILE", str(host_token)
    )
    monkeypatch.setenv("GPU_MANAGER_NATIVE_TASK_TOKEN_FILE", str(native_token))
    monkeypatch.delenv("GPU_MANAGER_HELPER_TOKEN_FILE", raising=False)
    controller = _load_controller()

    assert controller._required_native_helper_token_file() == str(native_token)


def test_helper_auth_scheme_is_case_insensitive_but_token_remains_exact():
    auth = HelperTokenAuth(service_token=TOKEN)
    request = type("Request", (), {"headers": {"Authorization": f"bEaReR {TOKEN}"}})()

    assert auth.authorized(request)


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


def test_runtime_supervisor_without_credential_fails_closed():
    with pytest.raises(HelperCredentialError, match="helper service credential"):
        create_app(object())


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


def test_native_host_submit_rejects_duplicate_json_keys_before_admission(tmp_path):
    async def run():
        host = NativeTaskHost(
            tmp_path / "state",
            "http://127.0.0.1:9999/submit",
            1,
            service_token=TOKEN,
        )
        async with TestClient(TestServer(host.app())) as client:
            response = await client.post(
                "/tasks/synthetic-task",
                data='{"prompt":"first","prompt":"second"}',
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Type": "application/json",
                },
            )
            assert response.status == 400
            payload = await response.json()
            assert payload == {
                "error": "invalid_json",
                "message": "valid JSON required",
            }
            assert not (host.root / "synthetic-task").exists()

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


@pytest.mark.parametrize("token_file_kind", ["missing", "empty", "unreadable"])
def test_required_service_credential_rejects_invalid_token_files(tmp_path, token_file_kind):
    token_file = tmp_path / "helper-token"
    if token_file_kind == "empty":
        token_file.write_text("   ", encoding="utf-8")
    elif token_file_kind == "unreadable":
        token_file.write_bytes(b"\xff")

    with pytest.raises(HelperCredentialError, match="helper service credential"):
        resolve_required_service_credential(service_token_file=token_file)


def test_runtime_supervisor_config_builder_requires_token_file(tmp_path):
    class Supervisor:
        async def execute(self, _request):
            raise AssertionError("authentication must be configured before serving")

    with pytest.raises(HelperCredentialError, match="helper service credential"):
        create_app(Supervisor(), service_token_file=tmp_path / "missing-helper-token")


@pytest.mark.parametrize("token_file_kind", ["missing", "empty", "unreadable"])
def test_runtime_supervisor_config_builder_rejects_invalid_token_files(
    tmp_path, token_file_kind
):
    token_file = tmp_path / "helper-token"
    if token_file_kind == "empty":
        token_file.write_text("", encoding="utf-8")
    elif token_file_kind == "unreadable":
        token_file.write_bytes(b"\xff")

    with pytest.raises(HelperCredentialError, match="helper service credential"):
        create_app(object(), service_token_file=token_file)



def test_runtime_supervisor_config_builder_accepts_valid_token_file(tmp_path):
    token_file = tmp_path / "helper-token"
    token_file.write_text(TOKEN, encoding="utf-8")

    app = create_app(object(), service_token_file=token_file)
    assert isinstance(app, web.Application)


def test_native_host_config_builder_requires_token_file(tmp_path):
    with pytest.raises(HelperCredentialError, match="helper service credential"):
        NativeTaskHost(
            tmp_path / "state",
            "http://127.0.0.1:9999/submit",
            1,
            service_token_file=tmp_path / "missing-helper-token",
        )

def test_native_host_without_credential_fails_closed(tmp_path):
    with pytest.raises(HelperCredentialError, match="helper service credential"):
        NativeTaskHost(tmp_path / "state", "http://127.0.0.1:9999/submit", 1)


@pytest.mark.parametrize("token_file_kind", ["missing", "empty", "unreadable"])
def test_native_host_config_builder_rejects_invalid_token_files(
    tmp_path, token_file_kind
):
    token_file = tmp_path / "helper-token"
    if token_file_kind == "empty":
        token_file.write_text("", encoding="utf-8")
    elif token_file_kind == "unreadable":
        token_file.write_bytes(b"\xff")

    with pytest.raises(HelperCredentialError, match="helper service credential"):
        NativeTaskHost(
            tmp_path / "state",
            "http://127.0.0.1:9999/submit",
            1,
            service_token_file=token_file,
        )



def test_native_host_config_builder_accepts_valid_token_file(tmp_path):
    token_file = tmp_path / "helper-token"
    token_file.write_text(TOKEN, encoding="utf-8")

    async def run():
        host = NativeTaskHost(
            tmp_path / "state",
            "http://127.0.0.1:9999/submit",
            1,
            service_token_file=token_file,
        )
        async with TestClient(TestServer(host.app())) as client:
            response = await client.get(
                "/health", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status == 200

    asyncio.run(run())


def test_unauthenticated_test_seams_are_explicit(tmp_path):
    app = create_unauthenticated_test_app(object())
    host = NativeTaskHost.for_unauthenticated_test_app(
        tmp_path / "state", "http://127.0.0.1:9999/submit", 1
    )

    assert isinstance(app, web.Application)
    assert host.helper_auth.available()
