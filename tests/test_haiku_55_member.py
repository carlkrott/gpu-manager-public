"""Broker member Claude-Haiku-5.5 (replaces Claude-Haiku-4.5, live 2026-10-07).

The fleet router serves claude-haiku-5-5 and keeps claude-haiku-4-5 as a transition alias, so the
broker member is renamed in place: same endpoint, same order position, new model id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import combined_gemma_broker_service as broker

CONFIG = ROOT / "config" / "services.json"


def _data() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_member_renamed_in_order_and_object() -> None:
    data = _data()
    ordered = data["combined_gemma_broker"]["ordered_members"]
    assert "Claude-Haiku-5.5" in ordered
    assert not [n for n in ordered if n == "Claude-Haiku-4.5"]
    assert ordered.index("Claude-Haiku-5.5") == 3  # after Luna, Gemini, MiniMax; before the local Gemmas
    services = data["services"]
    assert "Claude-Haiku-4.5" not in services
    member = services["Claude-Haiku-5.5"]
    assert member["name"] == "Claude-Haiku-5.5"
    assert member["model"] == "claude-haiku-5-5"
    assert member["display"].startswith("Claude Haiku 5.5")


def test_no_haiku_4_5_text_left_in_config() -> None:
    assert "Haiku-4.5" not in CONFIG.read_text(encoding="utf-8")
    assert "Haiku 4.5" not in CONFIG.read_text(encoding="utf-8")


ROUTER_IDS = frozenset({"gpt-6-luna", "gemini-3.8-flash", "MiniMax-M2.7", "claude-haiku-5-5", "claude-haiku-4-5"})


@pytest.fixture(autouse=True)
def router_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # The router's /v1/models is the member-model catalog; pin it so the test is offline.
    monkeypatch.setattr(broker, "_fetch_router_models", lambda url: ROUTER_IDS)


def test_broker_loads_live_shaped_config(tmp_path: Path) -> None:
    path = tmp_path / "services.json"
    path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    settings = broker.load_broker_settings(path)
    assert "Claude-Haiku-5.5" in settings.member_configs
    assert settings.member_configs["Claude-Haiku-5.5"]["model"] == "claude-haiku-5-5"


def test_unknown_router_model_disables_member(tmp_path: Path) -> None:
    data = _data()
    data["services"]["Claude-Haiku-5.5"]["model"] = "claude-haiku-9-9"
    path = tmp_path / "services.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert broker.load_broker_settings(path).member_configs["Claude-Haiku-5.5"]["enabled"] is False
