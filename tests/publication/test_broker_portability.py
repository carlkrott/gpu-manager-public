from __future__ import annotations

import pytest

from gemma_broker.config import BrokerConfig, ConfigError
from gemma_broker.runtime import validate_candidate_config
from combined_gemma_broker_service import load_broker_settings


def _config(**overrides):
    value = {
        "enabled": True,
        "ordered_members": ["member-primary", "member-secondary", "member-fallback"],
        "priority_values": {"interactive": 0, "normal": 10, "background": 20},
        "aging_interval_seconds": 60,
        "redis_namespace": "qual:combined-gemma:synthetic:",
        "leader_ttl_seconds": 15,
        "heartbeat_seconds": 2,
        "freshness_seconds": 10,
        "context_safety_margin": 256,
        "api_wait_timeout": 30,
        "eta_recent_window_seconds": 300,
        "eta_sparse_window_seconds": 3600,
        "eta_min_samples": 3,
        "lifecycle_mutation_enabled": False,
        "sse_enabled": False,
        "candidate_mode": True,
    }
    value.update(overrides)
    return value


def test_candidate_accepts_neutral_ordered_members():
    config = BrokerConfig.from_dict(_config(), candidate=True)
    validate_candidate_config(config)
    assert config.ordered_members == (
        "member-primary", "member-secondary", "member-fallback"
    )


def test_candidate_rejects_empty_or_duplicate_members():
    for members in ([], ["member-primary", "member-primary"]):
        with pytest.raises(ConfigError):
            BrokerConfig.from_dict(_config(ordered_members=members), candidate=True)


def test_broker_requires_explicit_config_path(monkeypatch):
    monkeypatch.delenv("GPU_MANAGER_CONFIG_PATH", raising=False)
    with pytest.raises(ValueError, match="CONFIG_PATH_REQUIRED"):
        load_broker_settings()


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"priority_values": {"interactive": 1, "normal": 10, "background": 20}}, "PRIORITY_VALUES_INVALID"),
        ({"sse_enabled": True}, "SSE_DISABLED_REQUIRED"),
        ({"redis_namespace": "prod:combined-gemma:synthetic:"}, "CANDIDATE_REDIS_NAMESPACE_REQUIRED"),
        ({"lifecycle_mutation_enabled": True}, "CANDIDATE_LIFECYCLE_MUTATION_DISABLED_REQUIRED"),
        ({"aging_interval_seconds": 0}, "POSITIVE_NUMBER_REQUIRED:aging_interval_seconds"),
    ],
)
def test_config_rejects_invalid_candidate_contracts(changes, error):
    with pytest.raises(ConfigError, match=error):
        BrokerConfig.from_dict(_config(**changes), candidate=True)


def test_config_rejects_mode_mismatch_and_unknown_fields():
    with pytest.raises(ConfigError, match="CANDIDATE_MODE_MISMATCH"):
        BrokerConfig.from_dict(_config(), candidate=False)
    invalid = _config()
    invalid["unexpected"] = True
    with pytest.raises(ConfigError, match="CONFIG_FIELDS_INVALID"):
        BrokerConfig.from_dict(invalid, candidate=True)


def test_production_namespace_is_required_for_non_candidate_mode():
    data = _config(
        candidate_mode=False,
        redis_namespace="qual:combined-gemma:synthetic:",
        lifecycle_mutation_enabled=True,
    )
    with pytest.raises(ConfigError, match="PRODUCTION_REDIS_NAMESPACE_REQUIRED"):
        BrokerConfig.from_dict(data, candidate=False)


def test_runtime_rejects_disabled_config_and_missing_namespace_colon():
    disabled = BrokerConfig.from_dict(_config(enabled=False), candidate=True)
    with pytest.raises(ConfigError, match="CANDIDATE_ENABLED_REQUIRED"):
        validate_candidate_config(disabled)
    no_colon = BrokerConfig.from_dict(
        _config(redis_namespace="qual:combined-gemma:synthetic"), candidate=True
    )
    with pytest.raises(ConfigError, match="CANDIDATE_REDIS_NAMESPACE_TRAILING_COLON_REQUIRED"):
        validate_candidate_config(no_colon)
