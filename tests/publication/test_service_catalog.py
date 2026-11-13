from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

from aiohttp.test_utils import TestClient, TestServer
import pytest

from api_auth import CATALOG_READ_TOKEN_FILE_ENV


ROOT = Path(__file__).parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


@pytest.fixture
def module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("candidate_gpu_manager_catalog", CONTROLLER)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    cliproxy = tmp_path / "cliproxy.yaml"
    cliproxy.write_text("validation:\n  max_message_length: 250000\n", encoding="utf-8")
    monkeypatch.setenv("GPU_MANAGER_CLIPROXY_CONFIG", str(cliproxy))
    return loaded


def _config(module) -> dict:
    config = module.empty_services_config()
    config["services"].update({
        "combined_gemma_broker": {"type": "embedded_broker", "enabled": True},
        "Cloud-Haiku": {"enabled": True, "model": "claude-haiku-5-5", "parallel": 4,
                        "context_per_slot": 200000},
        "Cloud-Luna": {"enabled": True, "model": "gpt-6-luna-high", "parallel": 2,
                       "context_per_slot": 114688, "max_input_chars": 400000},
        "Local-Qwen": {"enabled": True, "parallel": 2, "context_per_slot": 150000,
                       "broker_context_limit": 90000},
        "Disabled-Member": {"enabled": False, "parallel": 8, "context_per_slot": 1000},
        "comfyui": {"type": "comfyui", "enabled": True, "dependencies": ["Local-Qwen"]},
    })
    config["combined_gemma_broker"] = {
        "ordered_members": ["Cloud-Luna", "Cloud-Haiku", "Local-Qwen", "Disabled-Member"],
    }
    return config


def _metrics(walled: tuple[str, ...] = (), down: tuple[str, ...] = ()) -> dict:
    members = [
        {"name": name, "state": "unavailable" if name in down else "ready_accepting",
         "accepting": name not in down, "configured_slots": slots, "compatible_free_slots": slots}
        for name, slots in (("Cloud-Luna", 2), ("Cloud-Haiku", 4), ("Local-Qwen", 2))
    ]
    return {"members": members, "dispatch_loop": {"quota_walls": {n: {} for n in walled}}}


def _chat(catalog: dict) -> dict:
    return next(e for e in catalog["services"] if e["public_id"] == "llm.chat")


def test_wall_drops_healthy_slots_but_not_total(module):
    cap = module._cliproxy_max_input_chars()
    assert cap == 250000
    healthy = _chat(module._build_service_catalog(_config(module), _metrics(), cap))
    walled = _chat(module._build_service_catalog(_config(module), _metrics(walled=("Cloud-Haiku",)), cap))
    assert healthy["slots_total"] == walled["slots_total"] == 8  # disabled member excluded
    assert healthy["slots_healthy"] == healthy["recommended_concurrency"] == 8
    assert walled["slots_healthy"] == walled["recommended_concurrency"] == 4
    assert [m["walled"] for m in walled["members"] if m["name"] == "Cloud-Haiku"] == [True]


def test_removing_a_member_changes_catalog_version(module):
    config = _config(module)
    before = module._build_service_catalog(config, None)["catalog_version"]
    removed = copy.deepcopy(config)
    del removed["services"]["Local-Qwen"]
    removed["combined_gemma_broker"]["ordered_members"].remove("Local-Qwen")
    assert module._build_service_catalog(removed, None)["catalog_version"] != before
    assert module._build_service_catalog(copy.deepcopy(config), None)["catalog_version"] == before


def test_reordering_members_changes_catalog_version(module):
    config = _config(module)
    before = module._build_service_catalog(config, None)["catalog_version"]
    reordered = copy.deepcopy(config)
    reordered["combined_gemma_broker"]["ordered_members"] = [
        "Local-Qwen", "Cloud-Luna", "Cloud-Haiku", "Disabled-Member",
    ]
    assert module._build_service_catalog(reordered, None)["catalog_version"] != before


def test_effective_limits_are_min_over_healthy_members_only(module):
    cap = module._cliproxy_max_input_chars()
    limits = _chat(module._build_service_catalog(_config(module), _metrics(), cap))["limits"]
    # Qwen's broker_context_limit (90k) wins; Haiku's proxy cap (250k) beats Luna's 400k.
    assert limits == {"context_tokens": 90000, "max_input_chars": 250000,
                      "basis": "healthy", "complete": True}
    limits = _chat(module._build_service_catalog(
        _config(module), _metrics(down=("Local-Qwen",), walled=("Cloud-Haiku",)), cap))["limits"]
    assert limits["context_tokens"] == 114688 and limits["max_input_chars"] == 400000


def test_unknown_id_is_404_and_static_entries_project(module):
    catalog = module._build_service_catalog(_config(module), None)
    assert module._service_catalog_response(catalog, "nope.nope")[0] == 404
    status, body = module._service_catalog_response(catalog, "image.generate")
    assert status == 200 and body["constituents"] == [{"kind": "dependency", "name": "Local-Qwen"}]
    assert body["catalog_version"] == catalog["catalog_version"]


def test_display_name_defaults_to_combined_llm_and_metadata_wins(module):
    config = _config(module)
    catalog = module._build_service_catalog(config, None)
    assert _chat(catalog)["display_name"] == "Combined LLM"
    image = next(e for e in catalog["services"] if e["public_id"] == "image.generate")
    assert image["display_name"] == "comfyui"  # no display, no default: the service name
    config["services"]["combined_gemma_broker"]["metadata"] = {"display_name": "Chat"}
    assert _chat(module._build_service_catalog(config, None))["display_name"] == "Chat"


def _app(module, monkeypatch, tmp_path, config):
    services_path = tmp_path / "services.json"
    services_path.write_text(json.dumps(config), encoding="utf-8")
    module.SERVICES_CONFIG_PATH = str(services_path)
    providers_path = tmp_path / "providers.json"
    providers_path.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    module.PROVIDERS_CONFIG_PATH = str(providers_path)
    monkeypatch.setattr(module, "_load_services_config", lambda: copy.deepcopy(config))
    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    fake_redis.connection_pool.connection_kwargs = {"decode_responses": True}
    monkeypatch.setattr(module._redis_mod, "Redis", MagicMock(return_value=fake_redis))
    app = module.create_app()
    app.on_startup.clear()
    app.on_cleanup.clear()
    # /v1/routing needs an engine; the deprecation marker is what is checked.
    monkeypatch.setattr(module, "routing_engine", MagicMock())
    return app


def _token(path: Path, value: str) -> Path:
    path.write_text(value, encoding="ascii")
    path.chmod(0o600)
    return path


async def _calls(app, calls):
    out = []
    async with TestClient(TestServer(app)) as client:
        for method, path, token in calls:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            response = await client.request(method, path, headers=headers)
            body = await response.text()
            out.append((response.status, json.loads(body) if body.startswith("{") else body,
                        dict(response.headers)))
    return out


def test_catalog_read_scope_is_get_catalog_only(module, monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_token(tmp_path / "admin", "admin-synthetic")))
    monkeypatch.setenv(CATALOG_READ_TOKEN_FILE_ENV, str(_token(tmp_path / "cat", "catalog-synthetic")))
    app = _app(module, monkeypatch, tmp_path, _config(module))
    results = asyncio.run(_calls(app, [
        ("GET", "/v1/catalog", "catalog-synthetic"),
        ("GET", "/v1/catalog/llm.chat", "catalog-synthetic"),
        ("GET", "/v1/catalog/nope.nope", "catalog-synthetic"),
        ("GET", "/v1/routing", "catalog-synthetic"),
        ("GET", "/services", "catalog-synthetic"),
        ("POST", "/v1/catalog", "catalog-synthetic"),
        ("GET", "/v1/catalog?x=1", "catalog-synthetic"),
        ("GET", "/v1/catalog", "admin-synthetic"),
        ("GET", "/v1/catalog", None),
        ("GET", "/v1/routing", "admin-synthetic"),
    ]))
    statuses = [status for status, _, _ in results]
    assert statuses[:3] == [200, 200, 404]
    assert all(status in {401, 403} for status in statuses[3:7]), statuses
    assert statuses[7:9] == [200, 401]
    routing_status, routing, headers = results[9]
    assert routing_status == 200 and routing["deprecated"] is True and routing["see"] == "/v1/catalog"
    assert "Deprecation" in headers


def test_catalog_read_token_equal_to_admin_is_not_a_scope(module, monkeypatch, tmp_path):
    from api_auth import catalog_read_request_is_authorized

    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_token(tmp_path / "admin", "same-synthetic")))
    monkeypatch.setenv(CATALOG_READ_TOKEN_FILE_ENV, str(_token(tmp_path / "cat", "same-synthetic")))
    request = MagicMock(method="GET", path="/v1/catalog", raw_path="/v1/catalog", query_string="",
                        remote="127.0.0.1", headers={"Authorization": "Bearer same-synthetic"})
    assert not catalog_read_request_is_authorized(request)


def test_broker_unavailable_serves_static_catalog(module, monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_token(tmp_path / "admin", "admin-synthetic")))
    app = _app(module, monkeypatch, tmp_path, _config(module))
    [(status, body, _)] = asyncio.run(_calls(app, [("GET", "/v1/catalog", "admin-synthetic")]))
    assert status == 200 and body["live"] is False
    chat = _chat(body)
    assert chat["live"] is False and chat["slots_total"] == 8 and chat["slots_healthy"] is None
    assert chat["limits"]["basis"] == "configured"


def test_mcp_tool_returns_same_ids_as_http(module, monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_token(tmp_path / "admin", "admin-synthetic")))
    config = _config(module)
    app = _app(module, monkeypatch, tmp_path, config)
    monkeypatch.setattr(module._combined_gemma_api_proxy, "broker_metrics", lambda: _metrics())
    assert any(t["name"] == "get_service_catalog" for t in module._build_mcp_tools()["tools"])

    async def scenario():
        async with TestClient(TestServer(app)) as client:
            async def fake_http(method, path, body=None, **_):
                response = await client.request(
                    method, path, headers={"Authorization": "Bearer admin-synthetic"})
                return {**await response.json(), "_http_status": response.status}

            monkeypatch.setattr(module, "_mcp_http", fake_http)
            http = await (await client.get(
                "/v1/catalog", headers={"Authorization": "Bearer admin-synthetic"})).json()
            tool = await module._mcp_call_tool("get_service_catalog", {})
            one = await module._mcp_call_tool("get_service_catalog", {"public_id": "llm.chat"})

            async def unreachable(*_args, **_kwargs):
                return {"error": "HTTP API unreachable", "_http_status": 0}

            monkeypatch.setattr(module, "_mcp_http", unreachable)
            fallback = await module._mcp_call_tool("get_service_catalog", {})
            return http, tool, one, fallback

    http, tool, one, fallback = asyncio.run(scenario())
    ids = [e["public_id"] for e in http["services"]]
    assert ids == ["image.generate", "llm.chat"]
    assert http["live"] is True and _chat(http)["slots_healthy"] == 8
    assert [e["public_id"] for e in tool["services"]] == ids and "_http_status" not in tool
    assert one["public_id"] == "llm.chat"
    assert [e["public_id"] for e in fallback["services"]] == ids and fallback["live"] is False
