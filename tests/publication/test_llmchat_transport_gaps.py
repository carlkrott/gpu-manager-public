"""llm.chat transport gaps (audit 2026-11-12 A1, A2, A4).

A1: each member gets its own forward timeout, not the max over all members.
A2: fleet-llm-router sends 2xx headers and then may stream only an error
    event; that must fail before acceptance (requeue + cooldown), not finish
    as a success with no choices.
A4: 402 and a 429 with Retry-After are quota walls; Retry-After reaches the
    error string the quota-wall path parses.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gemma_broker.runtime import BufferedAiohttpTransport, DispatchLoop  # noqa: E402

classify = DispatchLoop._classify_preaccept_failure


def _member(name: str = "m") -> SimpleNamespace:
    return SimpleNamespace(name=name)


async def _send(handler, member=None) -> tuple[dict | Exception, list[str]]:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    accepted: list[str] = []
    async with TestServer(app) as server, ClientSession() as session:
        transport = BufferedAiohttpTransport(
            session=session,
            endpoint_for_member=lambda _m: str(server.make_url("/v1/chat/completions")),
        )
        try:
            result: dict | Exception = await transport.send(
                member or _member(), {"model": "x", "messages": []}, lambda: accepted.append("yes")
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            result = exc
    return result, accepted


def _sse_handler(events: list[dict]):
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for event in events:
            await resp.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    return handler


def test_a1_per_member_timeout_with_fallback():
    transport = BufferedAiohttpTransport(
        session=SimpleNamespace(),
        timeout=999.0,
        timeout_for_member=lambda m: {"fast": 120.0}.get(m.name),
    )
    assert transport._member_timeout(_member("fast")) == 120.0
    assert transport._member_timeout(_member("other")) == 999.0


def test_a2_error_only_stream_fails_before_acceptance():
    error = {"error": {"message": "gemini upstream 429 (quota_or_rate_limit)", "type": "upstream_error"}}
    result, accepted = asyncio.run(_send(_sse_handler([error])))
    assert isinstance(result, RuntimeError)
    assert str(result).startswith("TRANSPORT_STREAM_ERROR:")
    assert accepted == []
    assert classify(None, str(result)) == ("quota_wall", "quota_or_rate_limit")


def test_a2_plain_stream_error_is_transient():
    result, _ = asyncio.run(_send(_sse_handler([{"error": {"message": "connection reset"}}])))
    assert classify(None, str(result)) == ("transient", None)


def test_a2_real_chunk_accepts_once():
    chunk = {"id": "c", "object": "chat.completion.chunk", "model": "x",
             "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}
    done = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    result, accepted = asyncio.run(_send(_sse_handler([chunk, done])))
    assert accepted == ["yes"]
    assert result["choices"][0]["message"]["content"] == "hi"


def test_a4_retry_after_reaches_error_and_walls():
    async def handler(_request):
        return web.Response(status=429, text='{"error":"slow down"}', headers={"Retry-After": "42"})

    result, accepted = asyncio.run(_send(handler))
    assert accepted == []
    assert "Retry-After: 42" in str(result)
    assert classify(None, str(result)) == ("quota_wall", "retry_after")


@pytest.mark.parametrize(
    "error, expected",
    [
        ('TRANSPORT_HTTP_402:{"error":"payment required"}', ("quota_wall", "payment_required")),
        ('TRANSPORT_HTTP_429:{"error":"busy"}', ("transient", None)),
        ('TRANSPORT_HTTP_500:Retry-After: 5 boom', ("transient", None)),
    ],
)
def test_a4_classification(error, expected):
    assert classify(None, error) == expected


def test_a3_unpinned_submit_fails_fast_when_every_member_is_dead():
    from dataclasses import replace

    from gemma_broker.api import APIError, DurableBrokerService
    from gemma_broker.contracts import MemberState
    from gemma_broker.redis_store import InMemoryJobRepository
    from test_broker_pin_503_regressions import CHAT, _member

    now = [100.0]
    live = _member("GLM-5.3-Flash")
    dead = replace(live, state=MemberState.UNHEALTHY, accepting=False, observed_at=99.0)
    members = [replace(dead, observed_at=0.0)]  # boot: never probed
    service = DurableBrokerService(InMemoryJobRepository(), members=lambda: members, clock=lambda: now[0])

    def submit(key):
        return service.submit({"idempotency_key": key, "request": CHAT})

    submit("boot")  # unprobed members still queue
    members[:] = [dead]
    submit("blip")  # inside the grace window (router safe-restart) still queues
    now[0] += service.DEAD_GRACE_SECONDS
    with pytest.raises(APIError) as err:
        submit("outage")
    assert (err.value.status, err.value.code) == (503, "no_live_member")
    members[:] = [live]
    submit("recovered")
    assert service._all_dead_since is None


def test_a5_snapshot_not_refreshed_is_served_unhealthy():
    from gemma_broker.contracts import MemberState
    from gemma_broker.member_cache import STALE_AFTER_SECONDS, StrictMemberSnapshotCache
    from test_broker_pin_503_regressions import CPU, _probes

    mono = [0.0]
    cache = StrictMemberSnapshotCache({"m": CPU}, freshness_ms=3000, monotonic=lambda: mono[0])
    assert cache.snapshots()[0].state is MemberState.UNHEALTHY  # boot: never observed
    probes = _probes(error=None, observed_at=100.0, semantic=True, backend_slots_total=2, backend_slots_busy=0)
    cache.update("m", probes, now=100.0)
    assert cache.snapshots()[0].accepting
    mono[0] += STALE_AFTER_SECONDS + 1  # refresh loop wedged
    stale = cache.snapshots()[0]
    assert (stale.state, stale.accepting, stale.compatible_free_slots) == (MemberState.UNHEALTHY, False, 0)
    assert "stale" in stale.blockers[-1]
    cache.update("m", probes, now=100.0)
    assert cache.snapshots()[0].accepting


def test_a8_configured_context_safety_margin_reaches_admission():
    from gemma_broker.api import APIError, DurableBrokerService
    from gemma_broker.redis_store import InMemoryJobRepository
    from test_broker_pin_503_regressions import _member

    members = [_member("GLM-5.3-Flash")]  # 4096 tokens per slot
    request = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 3300}

    def submit(margin):
        kw = {} if margin is None else {"safety_margin_tokens": margin}
        service = DurableBrokerService(InMemoryJobRepository(), members=lambda: members, clock=lambda: 100.0, **kw)
        return service.submit({"idempotency_key": f"k{margin}", "request": request})

    with pytest.raises(APIError) as err:
        submit(None)  # built-in 1024 margin: 3300 + prompt + 1024 > 4096
    assert err.value.status == 422
    submit(512)  # services.json context_safety_margin=512 now admits it
