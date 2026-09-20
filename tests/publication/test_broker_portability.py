from __future__ import annotations

import pytest

from gemma_broker.compatibility import RequestRequirements
from gemma_broker.config import BrokerConfig, ConfigError
from gemma_broker.contracts import JobRecord, MemberSnapshot, MemberState
from gemma_broker.redis_store import RESERVE_JOB_LUA, RedisJobRepository
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


def test_redis_reservation_uses_separate_durable_leadership_clock():
    job_now = 123.5
    leadership_now = 1_789_943_218.5
    job = JobRecord.new(
        job_id="job-clock-domain",
        idempotency_key="idem-clock-domain",
        request_sha256="0" * 64,
        request_body={"messages": [{"role": "user", "content": "synthetic"}]},
        submitted_at=120.0,
        enqueue_sequence=1,
    )
    member = MemberSnapshot(
        name="member-primary",
        gpu_id=None,
        state=MemberState.READY_ACCEPTING,
        accepting=True,
        configured_slots=1,
        context_per_slot=4096,
        state_version=7,
        observed_at=job_now,
        compatible_free_slots=1,
        capabilities=("chat",),
    )
    requirements = RequestRequirements(
        input_tokens_estimate=8,
        max_output_tokens=16,
        required_capabilities=("chat",),
        estimate_source="synthetic",
    )

    class _RecordingRedis:
        def __init__(self):
            self.eval_calls = []

        def eval(self, *args):
            self.eval_calls.append(args)
            return 1

    class _Repository(RedisJobRepository):
        def peek_next(self, *, now):
            assert now == job_now
            return job

    client = _RecordingRedis()
    repository = _Repository(
        client,
        prefix="qual:combined-gemma:clock-domain:",
        leadership_clock=lambda: leadership_now,
    )

    reservation = repository.reserve_next(
        now=job_now,
        ordered_members=[member],
        requirements=requirements,
        leader_owner="synthetic-owner",
        fencing_token=7,
    )

    assert reservation is not None
    assert len(client.eval_calls) == 1
    call = client.eval_calls[0]
    assert call[11] == job_now
    assert call[-1] == leadership_now
    assert "leader_expiry <= tonumber(ARGV[12])" in RESERVE_JOB_LUA
    assert "'queue_claimed_at', ARGV[4]" in RESERVE_JOB_LUA
    assert "'reserved_at', ARGV[4]" in RESERVE_JOB_LUA
