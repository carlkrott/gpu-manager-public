from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


PAYLOAD = {
    "schema": "combined-gemma.broker-metric.v1",
    "queue": {"depth": 0, "fresh": True},
    "members": [],
}


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "candidate_gpu_manager_embedded_visibility", CONTROLLER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_visibility_uses_in_process_proxy_without_http_session(monkeypatch):
    module = _load_controller()
    module._services_config = {
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {"mode": "embedded"},
    }
    module.session = None
    proxy = SimpleNamespace(broker_metrics=Mock(return_value=PAYLOAD))
    module._combined_gemma_api_proxy = proxy

    payload, error = asyncio.run(module._fetch_combined_gemma_visibility())

    assert payload is PAYLOAD
    assert error is None
    proxy.broker_metrics.assert_called_once_with()


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self, **_kwargs):
        return PAYLOAD


class _Session:
    closed = False

    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return _Response()


def test_external_visibility_retains_loopback_http_path(monkeypatch):
    module = _load_controller()
    module._services_config = {
        "combined_gemma_broker": {"enabled": True},
        "combined_gemma_deployment": {
            "mode": "external",
            "endpoint": "http://127.0.0.1:18095",
        },
    }
    session = _Session()
    module.session = session
    monkeypatch.setenv(
        "COMBINED_GEMMA_BROKER_METRICS_URL",
        "http://127.0.0.1:18095/v1/metrics/broker?durable=0",
    )

    payload, error = asyncio.run(module._fetch_combined_gemma_visibility())

    assert payload is PAYLOAD
    assert error is None
    assert session.urls == [
        "http://127.0.0.1:18095/v1/metrics/broker?durable=0"
    ]
