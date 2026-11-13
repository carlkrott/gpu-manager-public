"""Local vision QC must fail closed when a policy has no vision_url.

The CPU vision backend (127.0.0.1:8127) was retired 2026-10-10; vision now goes
through the fleet LLM router, so gpu-manager must never fall back to a default
endpoint.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import aiohttp
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("gpu_manager_vision_fail_closed", FACADE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy", [{}, {"vision_url": ""}, {"vision_url": "   "}])
def test_missing_vision_url_fails_closed_without_network(manager, monkeypatch, policy):
    def _no_session(*_a, **_kw):
        raise AssertionError("no HTTP session may be opened without vision_url")

    monkeypatch.setattr(aiohttp, "ClientSession", _no_session)
    with pytest.raises(RuntimeError, match="no vision_url"):
        asyncio.run(manager._analyze_image_with_local_vision(
            b"not-really-an-image", prompt="check", policy=policy,
        ))


def test_retired_cpu_vision_default_is_gone(manager):
    source = FACADE.read_text(encoding="utf-8")
    assert "127.0.0.1:8127" not in source
    assert "VISION_BACKEND_URL" not in source
    assert not hasattr(manager, "VISION_BACKEND_URL")
