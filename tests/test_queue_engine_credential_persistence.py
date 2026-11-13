"""Regression coverage for queue credential-persistence boundaries."""
import asyncio
import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "scripts" / "gpu-manager.py"

from queue_engine import (
    AsyncRedisQueue,
    RedisQueue,
    scrub_persisted_credentials,
    strip_persisted_credential_headers,
)


SYNTHETIC_HEADERS = {
    "Authorization": "Bearer synthetic-test-token-not-a-real-credential",
    "pRoXy-AuThOrIzAtIoN": "synthetic-proxy-value",
    "Cookie": "synthetic-session=value",
    "X-API-KEY": "synthetic-api-key",
    "Content-Type": "application/json",
}


class FakeRedis:
    def __init__(self):
        self.payloads = []

    def xadd(self, _stream, payload):
        self.payloads.append(payload)
        return "1-0"


class FakeAsyncRedis:
    def __init__(self):
        self.payloads = []

    async def xadd(self, _stream, payload):
        self.payloads.append(payload)
        return "1-0"


def _sync_queue(monkeypatch):
    queue = RedisQueue.__new__(RedisQueue)
    queue.r = FakeRedis()
    queue.priority_lanes_enabled = False
    monkeypatch.setattr(queue, "_ensure_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue, "_stream", lambda *_args, **_kwargs: "queue:test")
    return queue


def _async_queue(monkeypatch):
    queue = AsyncRedisQueue.__new__(AsyncRedisQueue)
    queue.priority_lanes_enabled = False
    fake = FakeAsyncRedis()

    async def ensure_group(*_args, **_kwargs):
        return None

    async def get_conn():
        return fake

    monkeypatch.setattr(queue, "_ensure_group", ensure_group)
    monkeypatch.setattr(queue, "_get_conn", get_conn)
    monkeypatch.setattr(queue, "_stream", lambda *_args, **_kwargs: "queue:test")
    return queue, fake


def _decoded_headers(payload):
    value = payload["headers"]
    return json.loads(value) if isinstance(value, str) else value


def _load_controller():
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location("gpu_manager_auth_persistence_test", CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strip_all_credential_header_names_case_insensitively():
    scrubbed = strip_persisted_credential_headers(SYNTHETIC_HEADERS)
    assert scrubbed == {"Content-Type": "application/json"}
    assert SYNTHETIC_HEADERS["Authorization"].startswith("Bearer synthetic-")


def test_headers_field_accepts_json_encoded_mapping():
    result = scrub_persisted_credentials({"headers": json.dumps(SYNTHETIC_HEADERS)})
    assert _decoded_headers(result) == {"Content-Type": "application/json"}


def test_sync_enqueue_scrubs_mapping_before_serialization(monkeypatch):
    queue = _sync_queue(monkeypatch)
    assert queue.enqueue("test", {"headers": SYNTHETIC_HEADERS, "body": "kept"}) == "1-0"
    payload = queue.r.payloads[0]
    assert _decoded_headers(payload) == {"Content-Type": "application/json"}
    assert payload["body"] == "kept"


def test_sync_enqueue_scrubs_json_header_string(monkeypatch):
    queue = _sync_queue(monkeypatch)
    queue.enqueue("test", {"headers": json.dumps(SYNTHETIC_HEADERS)})
    assert _decoded_headers(queue.r.payloads[0]) == {"Content-Type": "application/json"}


def test_async_enqueue_scrubs_mapping_before_serialization(monkeypatch):
    queue, fake = _async_queue(monkeypatch)
    assert asyncio.run(queue.enqueue("test", {"headers": SYNTHETIC_HEADERS, "body": "kept"})) == "1-0"
    assert _decoded_headers(fake.payloads[0]) == {"Content-Type": "application/json"}
    assert fake.payloads[0]["body"] == "kept"


def test_async_enqueue_scrubs_json_header_string(monkeypatch):
    queue, fake = _async_queue(monkeypatch)
    asyncio.run(queue.enqueue("test", {"headers": json.dumps(SYNTHETIC_HEADERS)}))
    assert _decoded_headers(fake.payloads[0]) == {"Content-Type": "application/json"}


def test_gpu_manager_proxy_scrubber_removes_inbound_credentials():
    controller = _load_controller()
    scrubbed = controller._proxy_persisted_headers(SYNTHETIC_HEADERS)
    assert scrubbed == {"Content-Type": "application/json"}


def test_gpu_manager_proxy_scrubber_preserves_forwarding_and_backend_fields():
    controller = _load_controller()
    headers = {**SYNTHETIC_HEADERS, "X-Forwarded-For": "192.0.2.7"}
    scrubbed = controller._proxy_persisted_headers(headers)
    assert scrubbed["X-Forwarded-For"] == "192.0.2.7"
    job = scrub_persisted_credentials({"headers": headers, "api_key": "opaque-backend-reference"})
    assert job["api_key"] == "opaque-backend-reference"


def test_gpu_manager_cold_queue_path_uses_persistence_scrubber():
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    handler = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "handle_llm_proxy"
    )
    calls = [
        node for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_proxy_persisted_headers"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "headers"
    ]
    assert calls
