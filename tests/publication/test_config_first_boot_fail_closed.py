"""Audit C4: a services.json that exists but never loaded must stop the HTTP server."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _load(monkeypatch, path):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("gpu_manager_c4_test", SCRIPTS / "gpu-manager.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SERVICES_CONFIG_PATH", str(path))
    module._services_config_cache = {}
    module._services_config_last_check = 0.0
    module._services_config_signature = ()
    return module


def test_bad_file_on_first_boot_fails_closed(monkeypatch, tmp_path):
    bad = tmp_path / "services.json"
    bad.write_text("{not json")
    assert _load(monkeypatch, bad)._first_boot_config_failed() is True


def test_missing_file_keeps_default_bootstrap(monkeypatch, tmp_path):
    assert _load(monkeypatch, tmp_path / "absent.json")._first_boot_config_failed() is False


def test_valid_file_serves(monkeypatch, tmp_path):
    good = tmp_path / "services.json"
    good.write_text(json.dumps(json.loads((ROOT / "config" / "services.json").read_text())))
    assert _load(monkeypatch, good)._first_boot_config_failed() is False


def test_c1_membership_drift_is_a_config_error(monkeypatch, tmp_path):
    module = _load(monkeypatch, tmp_path / "absent.json")
    config = json.loads((ROOT / "config" / "services.json").read_text())
    config["combined_gemma_broker"]["enabled"] = True
    ordered = config["combined_gemma_broker"]["ordered_members"]
    assert not [e for e in module._validate_config_for_observability(config)[0] if "ordered_members" in e]
    config["services"]["drift-member"] = {"enabled": True, "type": "llm_backend", "routing_group": "Gemma"}
    ordered.append("ghost-member")
    errors = module._validate_config_for_observability(config)[0]
    assert any("'drift-member' is enabled in routing_group 'Gemma'" in e for e in errors)
    assert any("'ghost-member' is not a configured service" in e for e in errors)
