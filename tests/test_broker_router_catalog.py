"""Router-served broker members are checked against the router's live /v1/models.

Replaces the hard-coded _FLEET_ROUTER_MODEL_IDS allowlist (2026-10-21 14:35 outage:
a new member model needed a broker code edit). Also covers broker_reasoning.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import combined_gemma_broker_service as broker

CONFIG = ROOT / "config" / "services.json"


def _config(tmp_path: Path, **member_changes) -> Path:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    for name, changes in member_changes.items():
        data["services"][name].update(changes)
    path = tmp_path / "services.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_new_router_model_needs_no_code_edit(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def catalog(url):
        calls.append(url)
        return frozenset({"gpt-6-luna", "gemini-3.8-flash", "MiniMax-M3.1-Flash-Preview", "claude-haiku-5-5"})

    monkeypatch.setattr(broker, "_fetch_router_models", catalog)
    path = _config(tmp_path, **{"MiniMax-M2.7": {"model": "MiniMax-M3.1-Flash-Preview"}})
    settings = broker.load_broker_settings(path)
    assert settings.member_configs["MiniMax-M2.7"]["model"] == "MiniMax-M3.1-Flash-Preview"
    assert calls == ["http://192.0.2.1:8646/v1/models"]  # fetched once for all router members


def test_unknown_model_disables_only_that_member(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(broker, "_fetch_router_models", lambda url: frozenset({"gpt-6-luna"}))
    settings = broker.load_broker_settings(_config(tmp_path))
    assert settings.member_configs["Gemini-3.8-Flash-Low"]["enabled"] is False


def test_unreachable_router_is_accepted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(broker, "_fetch_router_models", lambda url: None)
    path = _config(tmp_path, **{"Claude-Haiku-5.5": {"model": "claude-haiku-9-9"}})
    assert broker.load_broker_settings(path).member_configs["Claude-Haiku-5.5"]["model"] == "claude-haiku-9-9"


def test_fetch_router_models_reads_ids_and_fails_open() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({"object": "list", "data": [{"id": "a"}, {"id": "b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    try:
        assert broker._fetch_router_models(f"http://127.0.0.1:{server.server_port}/v1/models") == {"a", "b"}
    finally:
        server.server_close()
    assert broker._fetch_router_models("http://127.0.0.1:1/v1/models", timeout=1.0) is None


def test_member_reasoning_turns_thinking_on_and_keeps_answer_budget() -> None:
    reasoning = {"effort": "high", "budget_tokens": 1024}
    body = {"model": "gemma", "messages": [], "max_tokens": 24, "reasoning_effort": "minimal"}
    out = broker.apply_member_reasoning(body, reasoning)
    assert out["chat_template_kwargs"] == {"enable_thinking": True}
    assert out["reasoning_effort"] == "high"
    assert out["reasoning_budget_tokens"] == 1024
    assert out["max_tokens"] == 24 + 1024
    assert body["max_tokens"] == 24 and "chat_template_kwargs" not in body  # caller body untouched


def test_member_reasoning_respects_caller_opt_out_and_absent_config() -> None:
    off = {"messages": [], "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}
    assert broker.apply_member_reasoning(off, {"effort": "high", "budget_tokens": 1024}) is off
    plain = {"messages": []}
    assert broker.apply_member_reasoning(plain, None) is plain


def test_invalid_broker_reasoning_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(broker, "_fetch_router_models", lambda url: None)
    path = _config(tmp_path, **{"Gemma-MI50": {"broker_reasoning": {"effort": "max"}}})
    with pytest.raises(ValueError, match="MEMBER_BROKER_REASONING_INVALID:Gemma-MI50"):
        broker.load_broker_settings(path)


def test_broker_context_limit_caps_admission_not_probe(tmp_path, monkeypatch) -> None:
    from gemma_broker.member_cache import StrictMemberSnapshotCache
    from gemma_broker.readiness import reduce_member_readiness

    monkeypatch.setattr(broker, "_fetch_router_models", lambda url: None)
    settings = broker.load_broker_settings(
        _config(tmp_path, **{"Gemma-MI50": {"context_per_slot": 100000, "broker_context_limit": 16384}})
    )
    config = settings.member_configs["Gemma-MI50"]
    assert config["context_per_slot"] == 100000  # probe still matches the real n_ctx
    assert StrictMemberSnapshotCache._initial_snapshot("Gemma-MI50", config).context_per_slot == 16384
    assert reduce_member_readiness("Gemma-MI50", config, {}, now=0.0).context_per_slot == 16384


@pytest.mark.parametrize("limit", [0, 200000, "16k", True])
def test_invalid_broker_context_limit_rejected(tmp_path, monkeypatch, limit) -> None:
    monkeypatch.setattr(broker, "_fetch_router_models", lambda url: None)
    path = _config(tmp_path, **{"Gemma-MI50": {"context_per_slot": 100000, "broker_context_limit": limit}})
    with pytest.raises(ValueError, match="MEMBER_BROKER_CONTEXT_LIMIT_INVALID:Gemma-MI50"):
        broker.load_broker_settings(path)
