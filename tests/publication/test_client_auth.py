from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
import pytest

import api_auth
from api_auth import ws_enforce_first_frame_auth


ROOT = Path(__file__).parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


def _load_controller():
    spec = importlib.util.spec_from_file_location("candidate_gpu_manager_clients", CONTROLLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dashboard_html() -> str:
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DASHBOARD_HTML"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError("DASHBOARD_HTML assignment not found")


def _write_token(tmp_path: Path) -> Path:
    path = tmp_path / "credentials" / "gpu-manager-api-token"
    path.parent.mkdir(parents=True)
    path.write_text("synthetic-publication-token", encoding="utf-8")
    return path


def _controller_app(module, tmp_path: Path):
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


class _TimeoutSocket:
    async def receive(self):
        raise asyncio.TimeoutError

    async def close(self, *, code):
        self.code = code


class _MessageSocket:
    def __init__(self, data):
        self.data = data
        self.code = None

    async def receive(self):
        return SimpleNamespace(type=WSMsgType.TEXT, data=self.data)

    async def close(self, *, code):
        self.code = code

    async def send_json(self, _payload):
        raise AssertionError("invalid handshake must not send application data")


def test_ws_handshake_timeout_and_oversized_frame_close_with_policy_violation(monkeypatch):
    monkeypatch.setattr(api_auth, "WS_AUTH_TIMEOUT_SECONDS", 0.01)

    async def run():
        timeout_socket = _TimeoutSocket()
        assert not await ws_enforce_first_frame_auth(
            timeout_socket, expected_token="synthetic"
        )
        assert timeout_socket.code == 1008

        oversized_socket = _MessageSocket("x" * (8 * 1024 + 1))
        assert not await ws_enforce_first_frame_auth(
            oversized_socket, expected_token="synthetic"
        )
        assert oversized_socket.code == 1008

    asyncio.run(run())


def test_controller_websocket_sends_no_state_before_auth_and_orders_auth_ok_first(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_write_token(tmp_path)))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    module = _load_controller()
    app = _controller_app(module, tmp_path / "controller")

    async def run():
        async with TestClient(TestServer(app)) as client:
            wrong = await client.ws_connect("/ws")
            with pytest.raises(asyncio.TimeoutError):
                await wrong.receive(timeout=0.1)
            await wrong.send_json({"type": "authenticate", "token": "wrong"})
            message = await wrong.receive(timeout=2)
            assert message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}
            assert wrong.close_code == 1008

            valid = await client.ws_connect("/ws")
            with pytest.raises(asyncio.TimeoutError):
                await valid.receive(timeout=0.1)
            await valid.send_json(
                {"type": "authenticate", "token": "synthetic-publication-token"}
            )
            auth_ok = await valid.receive(timeout=2)
            assert auth_ok.type == WSMsgType.TEXT
            assert json.loads(auth_ok.data) == {"type": "auth_ok"}
            full_state = await valid.receive(timeout=10)
            assert full_state.type == WSMsgType.TEXT
            assert json.loads(full_state.data)["type"] == "full_state"
            await valid.close()

    asyncio.run(run())


def test_dashboard_uses_memory_only_auth_and_authenticated_clients():
    html = _dashboard_html()
    assert "GPU Manager API token (memory only)" in html
    assert "window.fetch =" in html
    assert "Authorization" in html
    assert "type: 'authenticate'" in html
    assert "auth_ok" in html
    assert "localStorage" in html
    assert "localStorage.setItem('token'" not in html
    assert "localStorage.setItem('gpu_manager_token'" not in html
    assert "sessionStorage.setItem('token'" not in html
    assert "sessionStorage.setItem('gpu_manager_token'" not in html
    assert "document.cookie" not in html
    assert "?token=" not in html
    assert "URLSearchParams" not in html


def test_mcp_preload_uses_shared_bearer_policy_against_synthetic_server(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_write_token(tmp_path)))
    monkeypatch.delenv("GPU_MANAGER_API_URL", raising=False)
    monkeypatch.delenv("GPU_MANAGER_HTTP_URL", raising=False)
    module = _load_controller()
    module._MCP_HTTP_SESSION = None
    seen = {}

    async def handler(request):
        seen["authorization"] = request.headers.get("Authorization")
        return web.json_response({"ok": True})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/bundles/preload", handler)
        async with TestServer(app) as server:
            monkeypatch.setenv("GPU_MANAGER_API_URL", str(server.make_url("")))
            result = await module._mcp_preload_via_live_controller(
                {"bundle_name": "synthetic"}
            )
            assert result["_http_status"] == 200
            assert result["bundle"] == "synthetic"
        if module._MCP_HTTP_SESSION is not None:
            await module._MCP_HTTP_SESSION.close()

    asyncio.run(run())
    assert seen["authorization"] == "Bearer synthetic-publication-token"


def test_pipeline_transport_injects_api_token_only_for_own_loopback_origin(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_write_token(tmp_path)))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    module = _load_controller()
    seen: list[str | None] = []

    async def handler(request):
        seen.append(request.headers.get("Authorization"))
        return web.json_response({"ok": True})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        async with TestServer(app) as server, ClientSession() as session:
            transport = module._AiohttpTransport(session)
            module.LISTEN_PORT = server.port
            status, _, _ = await transport(
                {},
                "POST",
                str(server.make_url("/v1/chat/completions")),
                b"{}",
            )
            assert status == 200

            status, _, _ = await transport(
                {"Authorization": "Bearer caller-supplied"},
                "POST",
                str(server.make_url("/v1/chat/completions")),
                b"{}",
            )
            assert status == 200

            module.LISTEN_PORT = server.port + 1
            status, _, _ = await transport(
                {},
                "POST",
                str(server.make_url("/v1/chat/completions")),
                b"{}",
            )
            assert status == 200

    asyncio.run(run())
    assert seen == [
        "Bearer synthetic-publication-token",
        "Bearer caller-supplied",
        None,
    ]


def test_mcp_tool_names_are_unique_and_preload_mapping_is_present():
    module = _load_controller()
    tools = module._build_mcp_tools()["tools"]
    names = [tool["name"] for tool in tools]
    assert names
    assert len(names) == len(set(names))
    assert "preload_bundle" in names
    source_text = CONTROLLER.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    function_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_mcp_call_tool"
    )
    source = ast.get_source_segment(source_text, function_node)
    assert source and "_mcp_preload_via_live_controller" in source
