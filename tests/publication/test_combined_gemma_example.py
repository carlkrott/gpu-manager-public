from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from combined_gemma_broker_service import build_member_configs, load_broker_settings
from gemma_broker.config import BrokerConfig, ConfigError
from gemma_broker.runtime import validate_candidate_config
from gpu_manager_contracts import validate_registry
from portable_defaults import empty_services_config


ROOT = Path(__file__).resolve().parents[2]
EMPTY_PATH = ROOT / "examples/empty/services.json"
COMBINED_PATH = ROOT / "examples/combined-gemma/services.json"
DEMO_PATH = ROOT / "examples/combined-gemma/demo.py"


def _combined_data() -> dict:
    return json.loads(COMBINED_PATH.read_text(encoding="utf-8"))


def _load_demo():
    spec = importlib.util.spec_from_file_location("combined_gemma_example_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_example_matches_portable_defaults() -> None:
    data = json.loads(EMPTY_PATH.read_text(encoding="utf-8"))
    assert data == empty_services_config()
    assert data["services"] == {}
    assert data["scheduling"]["queue_owner"] == "durable"
    assert validate_registry(data) == []


def test_combined_example_loads_through_real_broker_boundaries() -> None:
    data = _combined_data()
    settings = load_broker_settings(COMBINED_PATH)
    validate_candidate_config(settings.config)
    assert validate_registry(data) == []
    assert settings.config.candidate_mode is True
    assert settings.config.enabled is True
    assert settings.config.lifecycle_mutation_enabled is False
    assert settings.config.sse_enabled is False
    assert settings.config.redis_namespace == "qual:combined-gemma:example:"
    assert settings.config.ordered_members == (
        "member-primary",
        "member-secondary",
        "member-fallback",
    )

    member_configs, endpoints = build_member_configs(data["services"], settings.config)
    assert set(member_configs) == set(settings.config.ordered_members)
    assert all(member["enabled"] is False for member in member_configs.values())
    assert all(endpoint.startswith("http://127.0.0.1:") for endpoint in endpoints.values())


@pytest.mark.parametrize("timeout", [True, 0, -1, "600"])
def test_cloud_member_rejects_invalid_forward_timeout(timeout) -> None:
    data = _combined_data()
    block = data["combined_gemma_broker"]
    config = BrokerConfig.from_dict(block, candidate=block["candidate_mode"])
    member_name = config.ordered_members[0]
    data["services"][member_name] = {
        "enabled": False,
        "member_type": "openai_compatible",
        "endpoint": "http://provider.test/v1/chat/completions",
        "model": "synthetic-cloud-model",
        "forward_timeout": timeout,
    }

    with pytest.raises(ValueError, match="MEMBER_FORWARD_TIMEOUT_INVALID"):
        build_member_configs(data["services"], config)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"ordered_members": []}, "MEMBERS_NONEMPTY_REQUIRED"),
        ({"ordered_members": ["member-primary", "member-primary"]}, "DUPLICATE_MEMBERS"),
        ({"priority_values": {"interactive": 1, "normal": 10, "background": 20}}, "PRIORITY_VALUES_INVALID"),
        ({"sse_enabled": True}, "SSE_DISABLED_REQUIRED"),
        ({"lifecycle_mutation_enabled": True}, "CANDIDATE_LIFECYCLE_MUTATION_DISABLED_REQUIRED"),
        ({"redis_namespace": "prod:combined-gemma:example:"}, "CANDIDATE_REDIS_NAMESPACE_REQUIRED"),
        ({"unexpected": True}, "CONFIG_FIELDS_INVALID"),
    ],
)
def test_combined_example_contract_failures_are_closed(change, error) -> None:
    data = _combined_data()["combined_gemma_broker"]
    mutated = copy.deepcopy(data)
    mutated.update(change)
    with pytest.raises(ConfigError, match=error):
        validate_candidate_config(BrokerConfig.from_dict(mutated, candidate=True))


def test_combined_example_mode_mismatch_is_rejected() -> None:
    with pytest.raises(ConfigError, match="CANDIDATE_MODE_MISMATCH"):
        BrokerConfig.from_dict(_combined_data()["combined_gemma_broker"], candidate=False)


def test_demo_dry_run_and_mock_demo_are_bounded(capsys) -> None:
    demo = _load_demo()
    assert demo.main(["--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["status"] == "dry_run_valid"
    assert dry["network_or_redis_used"] is False
    assert dry["real_model_output"] is False

    assert demo.main(["--mock-demo"]) == 0
    mock = json.loads(capsys.readouterr().out)
    assert mock["status"] == "mock_demo_complete"
    assert mock["production_services_contacted"] is False
    assert mock["redis_contacted"] is False
    assert mock["real_model_output"] is False
    assert [event["event"] for event in mock["events"]] == [
        "health_rejection",
        "queued",
        "cancelled",
        "terminal",
    ]
    assert all(event["simulated"] is True for event in mock["events"])
    assert all(isinstance(port, int) and port > 0 for port in mock["fixture_loopback_ports"])


def test_demo_has_no_live_service_imports() -> None:
    source = DEMO_PATH.read_text(encoding="utf-8")
    assert "import redis" not in source
    assert "import aiohttp" not in source
    assert "subprocess" not in source


def test_examples_contain_no_host_or_credential_material() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (EMPTY_PATH, COMBINED_PATH, DEMO_PATH, DEMO_PATH.with_name("README.md"))
    )
    forbidden = (
        "/home/",
        "/mnt/",
        "/opt/",
        "/Users/",
        "PCI_ID=",
        "HF_TOKEN",
        "API_KEY=",
        "model_path",
        "download_url",
    )
    for marker in forbidden:
        assert marker not in text
