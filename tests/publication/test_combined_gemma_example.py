from __future__ import annotations

import copy
import ast
import importlib.util
import json
from pathlib import Path

import pytest

import combined_gemma_broker_service
from combined_gemma_broker_service import build_member_configs, load_broker_settings
from gemma_broker.admission import AdmissionController
from gemma_broker.config import BrokerConfig, ConfigError
from gemma_broker.contracts import JobRecord, MemberSnapshot, MemberState
from gemma_broker.runtime import validate_candidate_config
from gpu_manager_contracts import validate_registry
from portable_defaults import empty_services_config


ROOT = Path(__file__).resolve().parents[2]
EMPTY_PATH = ROOT / "examples/empty/services.json"
COMBINED_PATH = ROOT / "examples/combined-gemma/services.json"
DEMO_PATH = ROOT / "examples/combined-gemma/demo.py"
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"
GROUP = "gemma-group"


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


ROUTER_SERVED = frozenset(
    {"gpt-6-luna", "gpt-5.6-luna", "gpt-5.3-codex-spark", "gemini-3.8-flash", "MiniMax-M2.7", "MiniMax-M2.5", "gemma"}
)


@pytest.fixture(autouse=True)
def _router_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Router members are validated against the router's live /v1/models; pin it offline.
    monkeypatch.setattr(combined_gemma_broker_service, "_fetch_router_models", lambda url: ROUTER_SERVED)


def _with_fleet_router_member(model: str) -> tuple[dict, BrokerConfig, str]:
    data = _combined_data()
    block = data["combined_gemma_broker"]
    config = BrokerConfig.from_dict(block, candidate=block["candidate_mode"])
    member_name = config.ordered_members[0]
    data["services"][member_name] = {
        "enabled": False,
        "member_type": "openai_compatible",
        "endpoint": "http://127.0.0.1:18646/v1/chat/completions",
        "model": model,
        "forward_timeout": 600,
        "systemd_unit": "fleet-llm-router.service",
        "models_url": "http://127.0.0.1:18646/v1/models",
    }
    return data, config, member_name


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-luna",
        "gpt-5.6-luna",
        "gpt-5.3-codex-spark",
        "gemini-3.8-flash",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "gemma",
    ],
)
def test_fleet_router_member_accepts_every_served_model_id(model: str) -> None:
    data, config, member_name = _with_fleet_router_member(model)

    members, _ = build_member_configs(data["services"], config)

    assert members[member_name]["model"] == model


@pytest.mark.parametrize(
    "model", ["not-a-proven-fleet-model", "gemini-3.8-flash-low", "gemini-general", "gpt-6"]
)
def test_fleet_router_member_with_unserved_model_is_disabled_not_fatal(model: str) -> None:
    data, config, member_name = _with_fleet_router_member(model)
    data["services"][member_name]["enabled"] = True

    members, _ = build_member_configs(data["services"], config)

    assert members[member_name]["enabled"] is False
    assert data["services"][member_name]["enabled"] is True  # input not mutated
    # Audit A7: the self-disable is visible in the member's snapshot, not only a log line.
    from gemma_broker.member_cache import StrictMemberSnapshotCache

    snap = StrictMemberSnapshotCache({member_name: members[member_name]}).snapshots()[0]
    assert snap.blockers == (f"self-disabled at broker build: router /v1/models lacks {model}",)


def test_model_id_is_only_checked_for_members_with_models_url() -> None:
    data, config, member_name = _with_fleet_router_member("some-other-endpoint-model")
    data["services"][member_name]["enabled"] = True
    del data["services"][member_name]["models_url"]

    members, _ = build_member_configs(data["services"], config)

    assert members[member_name]["model"] == "some-other-endpoint-model"
    assert members[member_name]["enabled"] is True


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


def _visibility_function():
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_combined_gemma_visibility"
    )
    namespace = {"_COMBINED_GEMMA_ROUTING_GROUP": GROUP}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CONTROLLER), "exec"), namespace)
    return namespace[function.name]


def _member(name, *, state="ready_accepting", accepting=True, free=2,
            fresh=True, stale=False, unknown=False, reason_codes=None):
    return {
        "name": name,
        "state": state,
        "accepting": accepting,
        "backend_slots_total": 4,
        "backend_slots_busy": 4 - free,
        "compatible_free_slots": free,
        "fresh": fresh,
        "stale": stale,
        "unknown": unknown,
        "observed_at": 100.0,
        "reason_codes": reason_codes or [],
    }


def _snapshot(members):
    names = [item["name"] for item in members]
    services = {name: {"routing_group": GROUP} for name in names}
    group = {"members": [{"service": name} for name in names], "can_accept": False}
    return services, {GROUP: group}, {
        "schema": "combined-gemma.broker-metric.v1",
        "queue": {"depth": 0, "fresh": True},
        "members": members,
    }


def test_live_capacity_survives_explicitly_systemd_inactive_member():
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950", free=2),
        _member(
            "Gemma-MI50", state="offline", accepting=False, free=0,
            fresh=False, stale=True,
            reason_codes=["MEMBER_SYSTEMD_INACTIVE"],
        ),
    ])

    _visibility_function()(services, groups, payload)

    assert groups[GROUP]["can_accept"] is True
    assert groups[GROUP]["slots_total"] == 4
    assert groups[GROUP]["slots_free"] == 2
    assert groups[GROUP]["capacity_unknown"] is False
    assert services["Gemma-RX6950"]["capacity_unknown"] is False
    assert services["Gemma-MI50"]["capacity_unknown"] is True


def test_live_unhealthy_systemd_inactive_shape_preserves_live_capacity():
    inactive = _member(
        "Gemma-MI50", state="unhealthy", accepting=False, free=0,
        fresh=False, stale=True,
        reason_codes=[
            "MEMBER_MODEL_NOT_RESIDENT",
            "MEMBER_SYSTEMD_INACTIVE",
            "MEMBER_PORT_DARK",
        ],
    )
    inactive.update({
        "state_version": 161445,
        "systemd_active": False,
        "backend_slots_total": None,
        "backend_slots_busy": None,
        "compatible_free_slots": None,
    })
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950XT", free=2),
        inactive,
    ])

    _visibility_function()(services, groups, payload)

    assert groups[GROUP]["can_accept"] is True
    assert groups[GROUP]["slots_free"] == 2
    assert groups[GROUP]["capacity_unknown"] is False
    assert services["Gemma-MI50"]["capacity_unknown"] is True


def test_fully_stale_group_remains_fail_closed():
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950", fresh=False, stale=True),
        _member(
            "Gemma-MI50", state="offline", accepting=False, free=0,
            fresh=False, stale=True,
            reason_codes=["MEMBER_SYSTEMD_INACTIVE"],
        ),
    ])

    _visibility_function()(services, groups, payload)

    assert groups[GROUP]["can_accept"] is False
    assert groups[GROUP]["capacity_unknown"] is True
    assert groups[GROUP]["slots_free"] is None


def _admission_member(name, *, state, accepting, free_slots, observed_at):
    return MemberSnapshot(
        name=name,
        gpu_id=name,
        state=state,
        accepting=accepting,
        configured_slots=2,
        context_per_slot=8192,
        state_version=1,
        observed_at=observed_at,
        compatible_free_slots=free_slots,
        capabilities=("chat",),
    )


def _admission_job():
    return JobRecord.new(
        job_id="job-1",
        idempotency_key="idem-1",
        request_sha256="0" * 64,
        request_body={"messages": [{"role": "user", "content": "hello"}]},
        submitted_at=100.0,
        enqueue_sequence=1,
        input_tokens_estimate=8,
        max_output_tokens=16,
        required_capabilities=("chat",),
    )


def _admission_controller():
    return AdmissionController(
        queue_cap=10,
        token_cap=1000,
        freshness_seconds=5.0,
        service_rate=1.0,
    )


def test_partial_visibility_capacity_is_also_admissible():
    live = _admission_member(
        "Gemma-RX6950", state=MemberState.READY_ACCEPTING, accepting=True,
        free_slots=2, observed_at=100.0,
    )
    inactive = _admission_member(
        "Gemma-MI50", state=MemberState.OFFLINE, accepting=False,
        free_slots=0, observed_at=80.0,
    )
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950", free=2),
        _member("Gemma-MI50", state="offline", accepting=False, free=0,
                fresh=False, stale=True,
                reason_codes=["MEMBER_SYSTEMD_INACTIVE"]),
    ])

    _visibility_function()(services, groups, payload)
    result = _admission_controller().evaluate(
        _admission_job(), [live, inactive], now=100.0,
        queue_depth=0, queued_tokens=0,
    )

    assert groups[GROUP]["can_accept"] is True
    assert result.admitted is True
    assert result.member_profiles == {
        "Gemma-RX6950": "eligible",
        "Gemma-MI50": "stale",
    }


def test_all_stale_visibility_and_admission_both_fail_closed():
    members = [
        _admission_member(
            name, state=MemberState.OFFLINE, accepting=False,
            free_slots=0, observed_at=80.0,
        )
        for name in ("Gemma-RX6950", "Gemma-MI50")
    ]
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950", fresh=False, stale=True),
        _member("Gemma-MI50", state="offline", accepting=False, free=0,
                fresh=False, stale=True,
                reason_codes=["MEMBER_SYSTEMD_INACTIVE"]),
    ])

    _visibility_function()(services, groups, payload)
    result = _admission_controller().evaluate(
        _admission_job(), members, now=100.0,
        queue_depth=0, queued_tokens=0,
    )

    assert groups[GROUP]["can_accept"] is False
    assert result.admitted is False
    assert result.reason_codes == ("capacity_stale",)


def test_fresh_busy_group_is_known_but_not_accepting():
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950", free=0),
    ])

    _visibility_function()(services, groups, payload)

    assert groups[GROUP]["can_accept"] is False
    assert groups[GROUP]["capacity_unknown"] is False
    assert groups[GROUP]["slots_free"] == 0


@pytest.mark.parametrize("mutate, error", [
    (lambda payload: payload.update(schema="wrong"), "invalid_broker_schema"),
    (lambda payload: payload["queue"].update(fresh=False), "invalid_or_stale_broker_snapshot"),
    (lambda payload: payload["members"].append(dict(payload["members"][0])), "duplicate_broker_member:Gemma-RX6950"),
    (lambda payload: payload["members"].clear(), "broker_member_set_mismatch"),
])
def test_malformed_or_incomplete_snapshot_stays_fail_closed(mutate, error):
    services, groups, payload = _snapshot([_member("Gemma-RX6950")])
    mutate(payload)

    _visibility_function()(services, groups, payload)

    assert groups[GROUP]["can_accept"] is False
    assert groups[GROUP]["capacity_unknown"] is True
    assert groups[GROUP]["visibility_error"] == error


def test_inactive_without_explicit_systemd_inactive_reason_is_not_excluded():
    services, groups, payload = _snapshot([
        _member("Gemma-RX6950", free=2),
        _member("Gemma-MI50", state="offline", accepting=False, free=0,
                fresh=False, stale=True),
    ])

    _visibility_function()(services, groups, payload)

    assert groups[GROUP]["can_accept"] is False
    assert groups[GROUP]["capacity_unknown"] is True
    assert groups[GROUP]["visibility_error"] == "invalid_or_stale_broker_member:Gemma-MI50"
