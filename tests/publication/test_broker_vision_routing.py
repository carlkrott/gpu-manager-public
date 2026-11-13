"""Image jobs need a vision member (plan 5.1).

Requests with image parts require the ``vision`` capability. Only members
tagged with ``broker_capabilities: ["vision"]`` (Gemini, GPU Gemma) may serve
them: Gemini first, and the job queues for GPU Gemma when Gemini is walled.
The reservation tests run the real Lua on a throwaway ``redis-server``.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import math
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import redis

from combined_gemma_broker_service import build_member_configs
from gemma_broker.api import APIError, DurableBrokerService
from gemma_broker.compatibility import (
    IMAGE_TOKEN_ALLOWANCE,
    build_requirements,
    evaluate,
    request_has_image,
)
from gemma_broker.config import BrokerConfig
from gemma_broker.contracts import JobState, MemberSnapshot, MemberState, ReasonCode
from gemma_broker.redis_store import RedisJobRepository, ReservationStale
from gemma_broker.runtime import NO_RESERVATION_BACKOFF_SECONDS, DispatchLoop

ROOT = Path(__file__).resolve().parents[2]
OWNER = "test-dispatcher"
# A realistically sized data URL (~200 KB): counted as text it would exceed
# every GPU member's context and the job could only ever go to Gemini.
DATA_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG" + b"\x00" * 150_000).decode()
IMAGE = {
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": DATA_URL}},
        {"type": "text", "text": "Return JSON only."},
    ]}],
    "max_tokens": 900,
}
CHAT = {"messages": [{"role": "user", "content": "hi"}]}
VISION = ("chat", "completion", "vision")
TEXT_ONLY = ("chat", "completion")


def _member(name, capabilities, *, accepting=True, free=1, context=8192):
    return MemberSnapshot(
        name=name, gpu_id=None,
        state=MemberState.READY_ACCEPTING if accepting else MemberState.UNHEALTHY,
        accepting=accepting, configured_slots=1, context_per_slot=context, state_version=1,
        observed_at=1.0, compatible_free_slots=free, capabilities=capabilities,
    )


# --------------------------------------------------------------- requirements

def test_image_parts_are_detected():
    assert request_has_image(IMAGE)
    assert request_has_image({"messages": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "https://example.invalid/a.png"}]}]})
    assert not request_has_image(CHAT)
    assert not request_has_image({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "data:image/png;base64,AAAA"}]}]})


def test_image_request_requires_vision_and_ignores_base64_bytes():
    req = build_requirements(IMAGE)
    assert req.required_capabilities == ("chat", "vision")
    assert len(DATA_URL) > 100_000
    assert IMAGE_TOKEN_ALLOWANCE <= req.input_tokens_estimate < IMAGE_TOKEN_ALLOWANCE + 1024
    assert build_requirements(CHAT).required_capabilities == ("chat",)
    assert build_requirements(IMAGE, required_capabilities=("chat", "vision")).required_capabilities == (
        "chat", "vision")


def test_large_text_body_uses_token_estimate_not_byte_count():
    """STEP 1(a): a ~250 KB body must be compatible with a 102400-context member.

    The old _utf8_upper_bound returned raw byte count, which is ~3.9x too high.
    The new _text_token_estimate divides by BYTES_PER_TOKEN_ESTIMATE (3) to get
    a conservative token count, so 250 KB text passes.
    """
    import json
    # Build a body whose JSON-encoded bytes total exactly 250 000.
    base = len(json.dumps([{"role": "user", "content": ""}], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    large_text_body = {"messages": [{"role": "user", "content": "x" * (250_000 - base)}], "max_tokens": 0}
    req = build_requirements(large_text_body)
    # ceil(250000/3) = 83334
    assert req.input_tokens_estimate == math.ceil(250_000 / 3), (
        f"Expected ~{math.ceil(250_000/3)}, got {req.input_tokens_estimate}"
    )
    assert req.estimate_source == "text_bytes_token_estimate"
    member_102k = _member("Gemma-102k", ("chat",), context=102_400)
    result = evaluate(req, member_102k)
    assert result.compatible, f"250 KB body must be compatible with 102400-context member: {result.reason_codes}"


def test_truly_oversized_body_still_rejected():
    """STEP 1(b): a body whose token estimate exceeds the member limit must be refused."""
    import json
    # ceil(400000/3) = 133334 + 1024 margin = 134358 > 102400
    base = len(json.dumps([{"role": "user", "content": ""}], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    large_text_body = {"messages": [{"role": "user", "content": "x" * (400_000 - base)}], "max_tokens": 0}
    req = build_requirements(large_text_body)
    member_102k = _member("Gemma-102k", ("chat",), context=102_400)
    result = evaluate(req, member_102k)
    assert not result.compatible
    assert "REQUEST_CONTEXT_INCOMPATIBLE" in str(result.reason_codes)


def test_text_only_member_is_incompatible_with_an_image_job():
    req = build_requirements(IMAGE)
    luna = evaluate(req, _member("Luna-6-Low", TEXT_ONLY))
    gpu = evaluate(req, _member("Gemma-RX6950XT", VISION))
    assert ReasonCode.REQUEST_CAPABILITY_INCOMPATIBLE in luna.reason_codes
    assert gpu.compatible, gpu.reason_codes


# --------------------------------------------------------------- member config

def _broker_config():
    import json
    data = json.loads((ROOT / "config/services.json").read_text(encoding="utf-8"))
    return data, BrokerConfig.from_dict(data["combined_gemma_broker"], candidate=False)


def test_shipped_config_tags_gemini_and_mi50_gemma_as_vision():
    data, config = _broker_config()
    members, _ = build_member_configs(data["services"], config)
    vision = {name for name, m in members.items() if "vision" in m["capabilities"]}
    # Gemma-RX6950XT runs -ub 128 (-ub 512 exceeds its VRAM ceiling); a Gemma 4
    # image takes up to 280 tokens and aborts the server there, so RX is text-only.
    assert vision == {"Gemini-3.8-Flash-Low", "Gemma-MI50"}
    assert all(m["capabilities"][:2] == TEXT_ONLY for m in members.values())
    assert all("broker_capabilities" not in m for m in members.values())


@pytest.mark.parametrize("bad", ["vision", [""], [1], {"vision": True}])
def test_invalid_broker_capabilities_are_rejected(bad):
    data, config = _broker_config()
    services = dict(data["services"])
    services["Luna-6-Low"] = {**services["Luna-6-Low"], "broker_capabilities": bad}
    with pytest.raises(ValueError, match="MEMBER_BROKER_CAPABILITIES_INVALID:Luna-6-Low"):
        build_member_configs(services, config)


def test_shipped_qc_policies_go_through_combined_gemma():
    data, _ = _broker_config()
    urls = []

    def walk(node):
        if isinstance(node, dict):
            if "vision_url" in node:
                urls.append((node["vision_url"], node.get("vision_model"), node.get("vision_timeout_s")))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    assert urls and all(
        u == ("http://127.0.0.1:8091/v1/chat/completions", "gemma", 330) for u in urls
    ), urls


# --------------------------------------------------------------- submit + reserve (real Lua)

@pytest.fixture(scope="module")
def redis_port(tmp_path_factory):
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server not installed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    workdir = tmp_path_factory.mktemp("redis")
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--bind", "127.0.0.1",
         "--save", "", "--appendonly", "no", "--dir", str(workdir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield port
    proc.terminate()
    proc.wait(timeout=10)


def _fleet(*, gemini_up=True, gpu_free=1):
    return [
        _member("Luna-6-Low", TEXT_ONLY),
        _member("Gemini-3.8-Flash-Low", VISION, accepting=gemini_up),
        _member("MiniMax-M2.7", TEXT_ONLY),
        _member("Gemma-RX6950XT", VISION, free=gpu_free),
        _member("Gemma-MI50", VISION, free=gpu_free),
    ]


@pytest.fixture
def broker(redis_port):
    client = redis.Redis(host="127.0.0.1", port=redis_port)
    client.flushall()
    repo = RedisJobRepository(client, prefix="vision:", leadership_clock=lambda: 2.0)
    state = {"members": [repo.put_member(m) for m in _fleet()]}
    service = DurableBrokerService(repo, members=lambda: state["members"], clock=lambda: 2.0)
    fence = repo.acquire_leadership(owner=OWNER, now=2.0, ttl_seconds=30.0)
    yield repo, state, service, fence
    client.close()


def _reserve(repo, members, fence):
    return repo.reserve_next(
        now=2.0, ordered_members=members, leader_owner=OWNER, fencing_token=fence,
        requirements_for=lambda job: replace(
            build_requirements(job.request_body), pinned_member=job.pinned_member),
    )


def test_image_job_goes_to_gemini_first_never_luna_or_gpu(broker):
    repo, state, service, fence = broker
    job = service.submit({"idempotency_key": "img-1", "request": IMAGE})
    assert job.required_capabilities == ("chat", "vision")
    reservation = _reserve(repo, state["members"], fence)
    assert reservation.member.name == "Gemini-3.8-Flash-Low"
    assert repo.lease_count("Gemma-RX6950XT") == repo.lease_count("Gemma-MI50") == 0


def _wall_gemini(repo, state):
    gemini = next(m for m in state["members"] if m.name == "Gemini-3.8-Flash-Low")
    walled = replace(gemini, state=MemberState.UNHEALTHY, accepting=False)
    repo.put_member(walled)
    state["members"] = [walled if m.name == gemini.name else m for m in state["members"]]


def test_image_job_stays_queued_when_gemini_walled_and_gpus_busy(broker):
    repo, state, service, fence = broker
    for gpu in ("Gemma-RX6950XT", "Gemma-MI50"):
        filler = service.submit({"idempotency_key": gpu, "request": CHAT, "pinned_member": gpu})
        assert _reserve(repo, state["members"], fence).job.job_id == filler.job_id
    _wall_gemini(repo, state)
    job = service.submit({"idempotency_key": "img-2", "request": IMAGE})
    # Redis holds the GPUs' full capacity; the dispatch loop backs off and retries.
    with pytest.raises(ReservationStale):
        _reserve(repo, state["members"], fence)
    assert repo.get(job.job_id).state is JobState.QUEUED
    assert repo.lease_count("Luna-6-Low") == repo.lease_count("MiniMax-M2.7") == 0


def _set_accepting(repo, state, name, accepting):
    member = next(m for m in state["members"] if m.name == name)
    updated = replace(
        member, accepting=accepting,
        state=MemberState.READY_ACCEPTING if accepting else MemberState.UNHEALTHY,
    )
    repo.put_member(updated)
    state["members"] = [updated if m.name == name else m for m in state["members"]]


def test_image_job_waits_for_evicted_gpus_then_the_same_job_takes_rx(broker):
    # Plan 5.1 literal: Gemini walled, both GPU Gemma members evicted for
    # generation. The job queues, never goes to a text member, and is served
    # by GPU Gemma once the RX rejoins.
    repo, state, service, fence = broker
    _wall_gemini(repo, state)
    for gpu in ("Gemma-RX6950XT", "Gemma-MI50"):
        _set_accepting(repo, state, gpu, False)
    job = service.submit({"idempotency_key": "img-5", "request": IMAGE})
    assert _reserve(repo, state["members"], fence) is None
    assert repo.get(job.job_id).state is JobState.QUEUED
    assert repo.lease_count("Luna-6-Low") == repo.lease_count("MiniMax-M2.7") == 0
    _set_accepting(repo, state, "Gemma-RX6950XT", True)
    reservation = _reserve(repo, state["members"], fence)
    assert (reservation.job.job_id, reservation.member.name) == (job.job_id, "Gemma-RX6950XT")


class _RecordingDispatcher:
    def __init__(self):
        self.reservations = []

    async def dispatch(self, reservation):
        self.reservations.append(reservation)
        return SimpleNamespace(state=JobState.COMPLETED, reason_code=None, error=None)


def test_dispatch_backs_off_when_image_job_waits_on_full_gpus(broker):
    # Every pass raises ReservationStale here; a 10 ms gate would hot-loop.
    repo, state, service, fence = broker
    for gpu in ("Gemma-RX6950XT", "Gemma-MI50"):
        service.submit({"idempotency_key": gpu, "request": CHAT, "pinned_member": gpu})
        _reserve(repo, state["members"], fence)
    _wall_gemini(repo, state)
    service.submit({"idempotency_key": "img-6", "request": IMAGE})
    dispatcher = _RecordingDispatcher()
    loop = DispatchLoop(
        repository=repo, transport=object(), members_provider=lambda: state["members"],
        dispatcher=dispatcher, owner=OWNER, leader_ttl_seconds=30.0, heartbeat_seconds=1.0,
        clock=lambda: 2.0, requirements_factory=build_requirements,
    )
    loop._fencing_token = fence
    before = time.monotonic()
    assert asyncio.run(loop._dispatch_one()) is False
    assert dispatcher.reservations == []
    assert loop._dispatch_one_fail_counters["reservation_stale"] == 1
    assert loop._reservation_retry_not_before_monotonic >= before + NO_RESERVATION_BACKOFF_SECONDS


def test_image_job_goes_to_free_gpu_when_gemini_walled(broker):
    repo, state, service, fence = broker
    _wall_gemini(repo, state)
    job = service.submit({"idempotency_key": "img-3", "request": IMAGE})
    reservation = _reserve(repo, state["members"], fence)
    assert (reservation.job.job_id, reservation.member.name) == (job.job_id, "Gemma-RX6950XT")


def test_text_job_is_unaffected_and_takes_luna(broker):
    repo, state, service, fence = broker
    service.submit({"idempotency_key": "txt", "request": CHAT})
    assert _reserve(repo, state["members"], fence).member.name == "Luna-6-Low"


def test_image_job_without_any_vision_member_is_refused_at_submit(broker):
    repo, state, service, fence = broker
    state["members"] = [_member("Luna-6-Low", TEXT_ONLY), _member("MiniMax-M2.7", TEXT_ONLY)]
    with pytest.raises(APIError) as err:
        service.submit({"idempotency_key": "img-4", "request": IMAGE})
    assert err.value.status == 422


# --------------------------------------------------------------- image QC self-call

@pytest.fixture
def manager(monkeypatch):
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("gpu_manager_vision_route", scripts / "gpu-manager.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qc_self_call_carries_token_and_gemma_backend(manager, monkeypatch):
    monkeypatch.setattr(manager, "_api_token", lambda: "tok-test")
    url = f"http://127.0.0.1:{manager.LISTEN_PORT}/v1/chat/completions"
    headers, redirects = manager._vision_request_headers(url, {})
    assert headers == {"X-Backend": "Gemma", "Authorization": "Bearer tok-test"}
    assert redirects is False


@pytest.mark.parametrize("url", [
    "http://192.0.2.1:8646/v1/chat/completions",
    "https://127.0.0.1:8091/v1/chat/completions",
    "http://127.0.0.1:8095/v1/chat/completions",
])
def test_qc_call_elsewhere_gets_no_token(manager, monkeypatch, url):
    monkeypatch.setattr(manager, "_api_token", lambda: "tok-test")
    assert manager._vision_request_headers(url, {}) == ({}, True)
