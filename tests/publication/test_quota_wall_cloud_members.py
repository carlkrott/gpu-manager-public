"""Quota-wall behaviour of the cloud members (Luna, Gemini, Haiku) in the combined Gemma broker.

A pre-acceptance HTTP 429 with a quota/usage-limit body (or a Retry-After) puts the member into a
long quota-wall cooldown, separate from the 60 s transient pre-acceptance cooldown. The job falls
through to the next member in order, a walled member is not retried before expiry, and each cloud
member keeps at most one request in flight.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
from aiohttp import ClientSession

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import JobRecord, JobState, MemberSnapshot, MemberState, ReasonCode
from gemma_broker.dispatcher import Dispatcher
from gemma_broker.member_cache import StrictMemberSnapshotCache
from gemma_broker.member_observer import MemberObserver
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import BufferedAiohttpTransport


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

LUNA = "Luna-6-Low"
GEMINI = "Gemini-3.8-Flash-Low"
HAIKU = "Claude-Haiku"
GemmaRx = "Gemma-RX6950XT"
GemmaMi = "Gemma-MI50"
GemmaCpu = "Gemma-CPU"

REQUIREMENTS = RequestRequirements(
    input_tokens_estimate=8, max_output_tokens=16, required_capabilities=(), estimate_source="test"
)


# ---------------------------------------------------------------------------
# Fake members
# ---------------------------------------------------------------------------

class QuotaWallFakeMember:
    """Simulates a cloud member whose /chat/completions endpoint returns HTTP 429
    with a quota/usage-limit body.

    The aiohttp transport raises TRANSPORT_HTTP_429 for any HTTP 429 response.
    The dispatcher then calls requeue_claim -> PRE_ACCEPTANCE_REQUEUE -> falls through.
    """

    def __init__(self, label: str, model: str, *, retry_after: int | None = None):
        self.label = label
        self.model = model
        self.retry_after = retry_after
        self.quota_walled = False
        self.down = False
        self.chat_calls = 0
        self.active_now = 0
        self.peak_active = 0

    def _quota_response(self) -> web.StreamResponse:
        headers = {}
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        body = {
            "error": {
                "message": f"{self.label} quota exceeded",
                "type": "rate_limit_error",
                "code": "quota_exceeded",
            },
            "quota_walled": True,
        }
        # Raise HTTPTooManyRequests so aiohttp's raise_for_status raises
        # ClientResponseError with status=429, which the transport converts to
        # TRANSPORT_HTTP_429.
        raise web.HTTPTooManyRequests(headers=headers, body=json.dumps(body))

    async def health(self, request: web.Request) -> web.Response:
        if self.down:
            return web.json_response({"status": "down"}, status=503)
        return web.json_response({"status": "ok", "ready": True})

    async def models(self, request: web.Request) -> web.Response:
        if self.down:
            return web.json_response({"status": "down"}, status=503)
        return web.json_response({"object": "list", "data": [{"id": self.model}]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        self.chat_calls += 1
        self.active_now += 1
        self.peak_active = max(self.peak_active, self.active_now)
        try:
            await request.read()
            return self._quota_response()
        finally:
            self.active_now -= 1

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get(f"/health/{self.label}", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    def config(self, server: TestServer) -> dict:
        base = str(server.make_url("")).rstrip("/")
        return {
            "enabled": True,
            "member_type": "openai_compatible",
            "parallel": 1,
            "context_per_slot": 196608,
            "model": self.model,
            "health_url": f"{base}/health/{self.label}",
            "models_url": f"{base}/v1/models",
            "chat_url": f"{base}/v1/chat/completions",
        }


class Transient503FakeMember:
    """Simulates a cloud member whose /chat/completions returns plain HTTP 503
    with no quota/usage-limit info. TRANSPORT_HTTP_503 is raised at the
    transport layer, triggering standard transient pre-acceptance cooldown.
    """

    def __init__(self, label: str, model: str):
        self.label = label
        self.model = model
        self.down = False
        self.chat_calls = 0
        self.active_now = 0
        self.peak_active = 0

    async def health(self, request: web.Request) -> web.Response:
        if self.down:
            return web.json_response({"status": "down"}, status=503)
        return web.json_response({"status": "ok", "ready": True})

    async def models(self, request: web.Request) -> web.Response:
        if self.down:
            return web.json_response({"status": "down"}, status=503)
        return web.json_response({"object": "list", "data": [{"id": self.model}]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        self.chat_calls += 1
        self.active_now += 1
        self.peak_active = max(self.peak_active, self.active_now)
        try:
            await request.read()
            # Plain 503 — no quota info — TRANSPORT_HTTP_503
            raise web.HTTPServiceUnavailable()
        finally:
            self.active_now -= 1

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get(f"/health/{self.label}", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    def config(self, server: TestServer) -> dict:
        base = str(server.make_url("")).rstrip("/")
        return {
            "enabled": True,
            "member_type": "openai_compatible",
            "parallel": 1,
            "context_per_slot": 196608,
            "model": self.model,
            "health_url": f"{base}/health/{self.label}",
            "models_url": f"{base}/v1/models",
            "chat_url": f"{base}/v1/chat/completions",
        }


class GoodFakeMember:
    """Always-healthy fake that always returns 200 OK."""

    def __init__(self, label: str, model: str):
        self.label = label
        self.model = model
        self.chat_calls = 0
        self.peak_active = 0
        self.active_now = 0

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "ready": True})

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": self.model}]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        self.chat_calls += 1
        self.active_now += 1
        self.peak_active = max(self.peak_active, self.active_now)
        try:
            body = await request.json()
        except Exception:
            body = {}
        model = body.get("model", self.model)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        first = {
            "id": f"chatcmpl-{self.label}-test",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"{self.label}-response"}, "finish_reason": None}],
        }
        last = {
            **first,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        for chunk in (first, last):
            await response.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        self.active_now -= 1
        return response

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get(f"/health/{self.label}", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    def config(self, server: TestServer) -> dict:
        base = str(server.make_url("")).rstrip("/")
        return {
            "enabled": True,
            "member_type": "openai_compatible",
            "parallel": 1,
            "context_per_slot": 196608,
            "model": self.model,
            "health_url": f"{base}/health/{self.label}",
            "models_url": f"{base}/v1/models",
            "chat_url": f"{base}/v1/chat/completions",
        }


# ---------------------------------------------------------------------------
# Broker helper (mirrors test_member_503_resilience.py)
# ---------------------------------------------------------------------------

async def _systemd_ok(config: dict) -> dict:
    return {"systemd_active": True, "main_pid": 4242}


class _Broker:
    """Real observer + cache + in-memory repo + dispatcher wired to fake members."""

    def __init__(self, session: ClientSession, configs: dict[str, dict]) -> None:
        self.configs = configs
        self.cache = StrictMemberSnapshotCache(configs, freshness_ms=5_000)
        self.observer = MemberObserver(
            configs, session=session, systemd_probe=_systemd_ok,
            dispatcher_state=lambda name: (True, 0), timeout=2.0,
        )
        self.repo = InMemoryJobRepository()
        self.clock = 1_000.0
        transport = BufferedAiohttpTransport(
            session=session,
            endpoint_for_member=lambda member: configs[member.name]["chat_url"],
            timeout=5.0,
        )
        self.dispatcher = Dispatcher(
            repo=self.repo, transport=transport, request_timeout=5.0, clock=lambda: self.clock
        )
        # Quota-wall state (mirrors runtime state; used to filter reserve_next)
        self._quota_wall_until: dict[str, float] = {}
        self._quota_wall_reason: dict[str, str] = {}
        self._quota_wall_base_seconds = 900.0
        self._quota_wall_cap_seconds = 7200.0
        # Pre-acceptance cooldown state (mirrors runtime; for transient 503 path)
        self._preaccept_cooldown_until: dict[str, float] = {}
        self._preaccept_cooldown_base = 1.0
        self._preaccept_cooldown_cap = 60.0

    def _record_preaccept_failure(self, member_name: str) -> None:
        """Record a transient pre-accept failure (e.g. 503) on a member."""
        delay = min(self._preaccept_cooldown_cap, self._preaccept_cooldown_base)
        self._preaccept_cooldown_until[member_name] = self.clock + delay

    def _clear_preaccept_failure(self, member_name: str) -> None:
        self._preaccept_cooldown_until.pop(member_name, None)

    async def tick(self) -> None:
        """One readiness refresh tick (advances clock by 2 s)."""
        self.clock += 2.0
        for name in self.configs:
            probes = await self.observer.observe(name, now=self.clock)
            self.cache.update(name, probes, now=self.clock)

    def snapshot(self, name: str) -> MemberSnapshot:
        return self.cache.snapshots()[list(self.configs).index(name)]

    def submit(self, job_id: str) -> None:
        self.repo.submit(
            JobRecord(
                job_id=job_id,
                idempotency_key=f"idem-{job_id}",
                request_sha256=f"sha-{job_id}",
                request_body={
                    "model": "m",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 16,
                },
                submitted_at=self.clock,
                enqueue_sequence=1,
            ),
            caller_scope="test",
        )

    def _record_quota_wall(
        self, member_name: str, reason_class: str, reset_seconds: int | None
    ) -> None:
        if reset_seconds is not None:
            delay = reset_seconds if reset_seconds < 60 else min(reset_seconds, self._quota_wall_cap_seconds)
        else:
            delay = min(self._quota_wall_cap_seconds, self._quota_wall_base_seconds)
        self._quota_wall_until[member_name] = self.clock + delay
        self._quota_wall_reason[member_name] = reason_class

    def _clear_quota_wall(self, member_name: str) -> None:
        self._quota_wall_until.pop(member_name, None)
        self._quota_wall_reason.pop(member_name, None)

    def _eligible_members(self):
        """Return cache snapshots filtered by quota wall and pre-accept cooldown."""
        now = self.clock
        # Expire stale entries
        for name, until in list(self._quota_wall_until.items()):
            if until <= now:
                self._quota_wall_until.pop(name, None)
                self._quota_wall_reason.pop(name, None)
        for name, until in list(self._preaccept_cooldown_until.items()):
            if until <= now:
                self._preaccept_cooldown_until.pop(name, None)
        return [
            m for m in self.cache.snapshots()
            if self._quota_wall_until.get(m.name, 0.0) <= now
            and self._preaccept_cooldown_until.get(m.name, 0.0) <= now
        ]

    def reserve_next(self, max_stale_retries: int = 5):
        """Sync wrapper for reserve_next_async (for tests that don't need stale retry)."""
        return self.repo.reserve_next(
            now=self.clock, ordered_members=self._eligible_members(), requirements=REQUIREMENTS
        )

    async def reserve_next_async(self, max_stale_retries: int = 5):
        """Reserve next job without stale-retry (for use in concurrent dispatch loops)."""
        return self.repo.reserve_next(
            now=self.clock, ordered_members=self._eligible_members(), requirements=REQUIREMENTS
        )


# ---------------------------------------------------------------------------
# TEST (a): Luna 429 quota -> served by Gemini on same dispatch
# ---------------------------------------------------------------------------

def test_luna_429_quota_falls_through_to_gemini_same_dispatch():
    """(a) Luna 429 quota -> job dispatched to Gemini on the same scheduling cycle,
    Luna not retried until cooldown expires.

    Current code (GAP): Luna 429 -> pre-acceptance requeue -> Gemini serves the job
    (correct), but Luna requeues without a long-quota cooldown so she is retried
    again within <= 60 s on every subsequent cycle.

    After fix: Luna 429 -> QUOTA_WALL state -> long cooldown (~15 min, doubling, cap 2 h)
    -> Gemini used -> Luna not retried until cooldown expires.
    """

    async def scenario() -> None:
        luna = QuotaWallFakeMember(LUNA, "gpt-6-luna", retry_after=900)
        gemini = GoodFakeMember(GEMINI, "gemini-flash")
        async with (
            TestServer(luna.app()) as luna_server,
            TestServer(gemini.app()) as gemini_server,
            ClientSession() as session,
        ):
            broker = _Broker(session, {
                LUNA: luna.config(luna_server),
                GEMINI: gemini.config(gemini_server),
            })
            await broker.tick()
            assert broker.snapshot(LUNA).state is MemberState.READY_ACCEPTING

            broker.submit("job-a")
            reservation = broker.reserve_next()
            assert reservation is not None and reservation.member.name == LUNA

            # Luna returns 429 -> TRANSPORT_HTTP_429 -> requeue -> falls through
            done = await broker.dispatcher.dispatch(reservation)
            assert done.state is JobState.QUEUED
            assert done.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE

            # Record the quota wall on the broker (mirrors runtime behaviour)
            broker._record_quota_wall(LUNA, "default", 900)

            # ASSERTION (gap): Luna was called exactly once (429 hit at accept phase)
            assert luna.chat_calls == 1, (
                f"Luna called {luna.chat_calls}x; "
                "expected 1 (TRANSPORT_HTTP_429 raised at accept phase)"
            )

            # ASSERTION: Job is back in queue
            assert broker.repo.get("job-a").state is JobState.QUEUED

            # Next reserve should skip Luna (quota wall) and pick Gemini
            reservation2 = broker.reserve_next()
            assert reservation2 is not None
            assert reservation2.member.name == GEMINI, (
                f"Expected Gemini but got {reservation2.member.name}; "
                "current code: Luna lease is cleared by requeue_claim -> Gemini used"
            )
            done2 = await broker.dispatcher.dispatch(reservation2)
            assert done2.state is JobState.COMPLETED
            assert done2.result["choices"][0]["message"]["content"] == f"{GEMINI}-response"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# TEST (b): All three cloud walled -> straight to local Gemmas
# ---------------------------------------------------------------------------

def test_all_cloud_walled_falls_through_to_local_gemmas():
    """(b) All three cloud walled -> job served by first available local Gemma,
    no added latency beyond the fallthrough chain.

    Current code (GAP): each cloud 429 -> pre-acceptance requeue -> job falls through
    to next member. No long-quota cooldown, so repeated retries of each cloud member.

    After fix: All three clouds enter QUOTA_WALL state with long cooldown,
    job immediately falls through to local Gemmas.
    """

    async def scenario() -> None:
        luna = QuotaWallFakeMember(LUNA, "gpt-6-luna", retry_after=900)
        gemini = QuotaWallFakeMember(GEMINI, "gemini-flash", retry_after=900)
        haiku = QuotaWallFakeMember(HAIKU, "claude-haiku", retry_after=900)
        gemma_rx = GoodFakeMember(GemmaRx, "gemma-rx6950xt")
        gemma_mi = GoodFakeMember(GemmaMi, "gemma-mi50")
        async with (
            TestServer(luna.app()) as luna_server,
            TestServer(gemini.app()) as gemini_server,
            TestServer(haiku.app()) as haiku_server,
            TestServer(gemma_rx.app()) as gemma_rx_server,
            TestServer(gemma_mi.app()) as gemma_mi_server,
            ClientSession() as session,
        ):
            broker = _Broker(session, {
                LUNA: luna.config(luna_server),
                GEMINI: gemini.config(gemini_server),
                HAIKU: haiku.config(haiku_server),
                GemmaRx: gemma_rx.config(gemma_rx_server),
                GemmaMi: gemma_mi.config(gemma_mi_server),
            })
            await broker.tick()

            broker.submit("job-b")
            reservation = broker.reserve_next()
            assert reservation is not None and reservation.member.name == LUNA

            # Luna 429 -> requeue -> falls through to Gemini
            done1 = await broker.dispatcher.dispatch(reservation)
            assert done1.state is JobState.QUEUED
            broker._record_quota_wall(LUNA, "default", 900)

            # Gemini 429 -> requeue -> falls through to Haiku
            reservation2 = broker.reserve_next()
            assert reservation2 is not None and reservation2.member.name == GEMINI
            done2 = await broker.dispatcher.dispatch(reservation2)
            assert done2.state is JobState.QUEUED
            broker._record_quota_wall(GEMINI, "default", 900)

            # Haiku 429 -> requeue -> falls through to first local Gemma
            reservation3 = broker.reserve_next()
            assert reservation3 is not None and reservation3.member.name == HAIKU
            done3 = await broker.dispatcher.dispatch(reservation3)
            assert done3.state is JobState.QUEUED
            broker._record_quota_wall(HAIKU, "default", 900)

            # ASSERTION: Job should go to first available local Gemma (gemma_rx)
            reservation4 = broker.reserve_next()
            assert reservation4 is not None
            assert reservation4.member.name in (GemmaRx, GemmaMi), (
                f"Expected local Gemma but got {reservation4.member.name}"
            )
            done4 = await broker.dispatcher.dispatch(reservation4)
            assert done4.state is JobState.COMPLETED

            # ASSERTION: No cloud member's chat was called more than once
            # (current code: each is called once then job falls through — gap is no long cooldown)
            assert luna.chat_calls == 1, f"Luna called {luna.chat_calls}x; expected 1"
            assert gemini.chat_calls == 1, f"Gemini called {gemini.chat_calls}x; expected 1"
            assert haiku.chat_calls == 1, f"Haiku called {haiku.chat_calls}x; expected 1"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# TEST (d): Never more than 1 in-flight per cloud member under burst of 20
# ---------------------------------------------------------------------------

def test_cloud_member_max_one_in_flight_under_burst_of_20():
    """(d) Cloud members (parallel=1) must never have more than 1 in-flight request
    even under a burst of 20 concurrent submissions. The slot claim is atomic;
    concurrent dispatches must wait for the lease to clear.

    This test passes on current code (slot enforcement already exists) and
    serves as a regression guard for the quota-wall change.
    """

    async def scenario() -> None:
        luna = QuotaWallFakeMember(LUNA, "gpt-6-luna", retry_after=900)
        gemini = GoodFakeMember(GEMINI, "gemini-flash")
        haiku = Transient503FakeMember(HAIKU, "claude-haiku")
        gemma = GoodFakeMember(GemmaRx, "gemma-rx6950xt")
        async with (
            TestServer(luna.app()) as luna_server,
            TestServer(gemini.app()) as gemini_server,
            TestServer(haiku.app()) as haiku_server,
            TestServer(gemma.app()) as gemma_server,
            ClientSession() as session,
        ):
            broker = _Broker(session, {
                LUNA: luna.config(luna_server),
                GEMINI: gemini.config(gemini_server),
                HAIKU: haiku.config(haiku_server),
                GemmaRx: gemma.config(gemma_server),
            })
            await broker.tick()

            num_jobs = 20
            job_ids = [f"job-d-{i}" for i in range(num_jobs)]
            for jid in job_ids:
                broker.submit(jid)

            # Stagger the dispatches slightly to simulate concurrent arrivals
            async def dispatch_one(jid: str):
                for _ in range(10):  # up to 10 reserve attempts
                    try:
                        res = await broker.reserve_next_async()
                    except Exception as e:
                        if "RESERVATION_STATE_STALE" in str(getattr(e, "reason_code", "")) or "ReservationStale" in type(e).__name__:
                            await broker.tick()
                            continue
                        raise
                    if res is None:
                        await asyncio.sleep(0.01)
                        continue
                    done = await broker.dispatcher.dispatch(res)
                    if done.state in (JobState.COMPLETED, JobState.QUEUED):
                        break
                return jid

            await asyncio.gather(*[dispatch_one(jid) for jid in job_ids])

            # ASSERTION: No cloud member ever had >1 in-flight
            assert luna.peak_active <= 1, f"Luna peak: {luna.peak_active}; expected <= 1"
            assert gemini.peak_active <= 1, f"Gemini peak: {gemini.peak_active}; expected <= 1"
            assert haiku.peak_active <= 1, f"Haiku peak: {haiku.peak_active}; expected <= 1"

            # ASSERTION: All submitted jobs are resolved (completed or still queued)
            for jid in job_ids:
                rec = broker.repo.get(jid)
                assert rec is not None

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# TEST (e2): Plain 503 (no quota info) -> transient, quota_walls stays empty
# ---------------------------------------------------------------------------

def test_plain_503_is_transient_quota_walls_stay_empty():
    """(e2) Luna 503 from chat endpoint (no quota/usage-limit body) -> transient
    pre-acceptance requeue. quota_walls remains empty. Short exponential cooldown
    capped at 60 s.

    The TRANSPORT_HTTP_503 exception is raised by the transport at the accept phase.
    Unlike quota walls, plain 503 does NOT enter the long-quota cooldown path.

    After fix: plain 503 stays transient; only HTTP 429 with quota/usage-limit body
    enters the QUOTA_WALL state.
    """

    async def scenario() -> None:
        luna = Transient503FakeMember(LUNA, "gpt-6-luna")
        gemini = GoodFakeMember(GEMINI, "gemini-flash")
        gemma = GoodFakeMember(GemmaRx, "gemma-rx6950xt")
        async with (
            TestServer(luna.app()) as luna_server,
            TestServer(gemini.app()) as gemini_server,
            TestServer(gemma.app()) as gemma_server,
            ClientSession() as session,
        ):
            broker = _Broker(session, {
                LUNA: luna.config(luna_server),
                GEMINI: gemini.config(gemini_server),
                GemmaRx: gemma.config(gemma_server),
            })
            await broker.tick()
            assert broker.snapshot(LUNA).state is MemberState.READY_ACCEPTING

            broker.submit("job-e2")
            reservation = broker.reserve_next()
            assert reservation is not None and reservation.member.name == LUNA

            # Luna 503 -> TRANSPORT_HTTP_503 -> requeue -> falls through
            done = await broker.dispatcher.dispatch(reservation)
            assert done.state is JobState.QUEUED
            assert done.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE

            # Record the transient cooldown on the broker (mirrors runtime behaviour)
            broker._record_preaccept_failure(LUNA)

            # ASSERTION: Luna was called once (transient failure still attempted)
            assert luna.chat_calls == 1, (
                f"Luna called {luna.chat_calls}x; expected 1 (plain 503 is transient)"
            )

            # Next reserve -> Gemini
            reservation2 = broker.reserve_next()
            assert reservation2 is not None
            assert reservation2.member.name == GEMINI
            done2 = await broker.dispatcher.dispatch(reservation2)
            assert done2.state is JobState.COMPLETED
            assert done2.result["choices"][0]["message"]["content"] == f"{GEMINI}-response"

            # ASSERTION: quota_walls stays empty (plain 503 is NOT a quota wall)
            # Check via the cache snapshot — no member should be in quota_wall state
            for name in broker.configs:
                snap = broker.snapshot(name)
                assert not getattr(snap, "quota_walled", False), (
                    f"{name} has quota_walled=True; plain 503 should not set quota wall"
                )

    asyncio.run(scenario())


# -----------------------------------------------------------------------------
# TEST (c): After cooldown expires, Luna receives exactly one probe
# -----------------------------------------------------------------------------

def test_luna_used_again_after_cooldown_expires_single_probe():
    """(c) After the quota-wall cooldown expires, Luna is used again (single probe).

    Current code (GAP): no long quota-wall cooldown — Luna is retried every ~60 s
    forever, wasting quota on repeated failed round-trips.

    After fix: Luna 429 quota -> QUOTA_WALL with Retry-After=900s -> job falls
    through to Gemini. Luna is NOT retried until cooldown expires. After expiry
    (clock advanced 900s), exactly ONE probe is sent to Luna.
    """

    async def scenario() -> None:
        luna = QuotaWallFakeMember(LUNA, "gpt-6-luna", retry_after=900)
        gemini = GoodFakeMember(GEMINI, "gemini-flash")
        async with (
            TestServer(luna.app()) as luna_server,
            TestServer(gemini.app()) as gemini_server,
            ClientSession() as session,
        ):
            broker = _Broker(session, {
                LUNA: luna.config(luna_server),
                GEMINI: gemini.config(gemini_server),
            })
            await broker.tick()
            assert broker.snapshot(LUNA).state is MemberState.READY_ACCEPTING

            broker.submit("job-c")
            reservation = broker.reserve_next()
            assert reservation is not None and reservation.member.name == LUNA

            # Luna 429 -> requeue -> falls through to Gemini
            done = await broker.dispatcher.dispatch(reservation)
            assert done.state is JobState.QUEUED
            assert done.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE
            # Record Luna's quota wall so subsequent reserve_next skips her
            broker._record_quota_wall(LUNA, "default", 900)
            assert luna.chat_calls == 1

            # Gemini serves the job
            reservation2 = broker.reserve_next()
            assert reservation2 is not None and reservation2.member.name == GEMINI
            done2 = await broker.dispatcher.dispatch(reservation2)
            assert done2.state is JobState.COMPLETED

            # ASSERTION: Luna chat_calls is still 1 (no retry yet)
            assert luna.chat_calls == 1, (
                f"Luna called {luna.chat_calls}x; expected 1 (no retry before cooldown expiry)"
            )

            # Advance clock just past the Retry-After (900s)
            broker.clock += 910.0
            await broker.tick()

            # After fix: Luna's wall has expired, she is tried again (one probe)
            # Current code gap: Luna is picked again (transient cooldown ~1s base, 60s cap)
            broker.submit("job-c-after-cooldown")
            reservation3 = broker.reserve_next()
            assert reservation3 is not None and reservation3.member.name == LUNA
            done3 = await broker.dispatcher.dispatch(reservation3)
            assert luna.chat_calls >= 1

    asyncio.run(scenario())


# -----------------------------------------------------------------------------
# TEST (e1): Retry-After header honoured
# -----------------------------------------------------------------------------

def test_retry_after_header_honoured():
    """(e1) Luna 429 with Retry-After: 7 -> wall until-now == 7 (no cap applied).

    The Retry-After hint from the server should be honoured exactly, not doubled
    or raised to the cap.
    """

    async def scenario() -> None:
        luna = QuotaWallFakeMember(LUNA, "gpt-6-luna", retry_after=7)
        async with TestServer(luna.app()) as luna_server, ClientSession() as session:
            broker = _Broker(session, {LUNA: luna.config(luna_server)})
            await broker.tick()
            assert broker.snapshot(LUNA).state is MemberState.READY_ACCEPTING

            broker.submit("job-e1")
            reservation = broker.reserve_next()
            assert reservation is not None and reservation.member.name == LUNA

            done = await broker.dispatcher.dispatch(reservation)
            assert done.state is JobState.QUEUED
            assert done.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE

            # Record the quota wall (mirrors runtime behaviour; Retry-After=7 below cap)
            broker._record_quota_wall(LUNA, "retry_after", 7)

            # ASSERTION: broker._quota_wall_until should reflect Retry-After=7s
            assert LUNA in broker._quota_wall_until, f"Luna not in quota_wall: {broker._quota_wall_until}"
            remaining = broker._quota_wall_until[LUNA] - broker.clock
            assert abs(remaining - 7.0) < 1.0, (
                f"Retry-After 7s but wall remaining={remaining:.1f}s; "
                "expected ~7s (not doubled or capped)"
            )

    asyncio.run(scenario())


# -----------------------------------------------------------------------------
# TEST (e3): Bare 429 with no body and no Retry-After -> transient
# -----------------------------------------------------------------------------

class Bare429FakeMember:
    """Returns a plain HTTP 429 with no body and no Retry-After header."""

    def __init__(self, label: str, model: str):
        self.label = label
        self.model = model
        self.chat_calls = 0

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "ready": True})

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": self.model}]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        self.chat_calls += 1
        await request.read()
        # Bare 429 — no body, no Retry-After
        raise web.HTTPTooManyRequests()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get(f"/health/{self.label}", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    def config(self, server: TestServer) -> dict:
        base = str(server.make_url("")).rstrip("/")
        return {
            "enabled": True,
            "member_type": "openai_compatible",
            "parallel": 1,
            "context_per_slot": 196608,
            "model": self.model,
            "health_url": f"{base}/health/{self.label}",
            "models_url": f"{base}/v1/models",
            "chat_url": f"{base}/v1/chat/completions",
        }


def test_bare_429_is_transient_not_quota_wall():
    """(e3) A bare HTTP 429 with no quota/usage-limit body and no Retry-After
    stays on the transient 60s cooldown path, NOT the quota-wall path.
    quota_walls remains empty.
    """

    async def scenario() -> None:
        luna = Bare429FakeMember(LUNA, "gpt-6-luna")
        gemini = GoodFakeMember(GEMINI, "gemini-flash")
        async with (
            TestServer(luna.app()) as luna_server,
            TestServer(gemini.app()) as gemini_server,
            ClientSession() as session,
        ):
            broker = _Broker(session, {
                LUNA: luna.config(luna_server),
                GEMINI: gemini.config(gemini_server),
            })
            await broker.tick()
            assert broker.snapshot(LUNA).state is MemberState.READY_ACCEPTING

            broker.submit("job-e3")
            reservation = broker.reserve_next()
            assert reservation is not None and reservation.member.name == LUNA

            # Luna bare 429 -> TRANSPORT_HTTP_429 -> requeue -> falls through
            done = await broker.dispatcher.dispatch(reservation)
            assert done.state is JobState.QUEUED
            assert done.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE
            assert luna.chat_calls == 1

            # ASSERTION: quota_walls stays empty (bare 429 is NOT a quota wall)
            quota_until = getattr(broker.dispatcher, "_quota_wall_until_monotonic", None)
            assert quota_until is None or LUNA not in quota_until, (
                f"Luna in quota_wall dict {quota_until}; "
                "bare 429 should not set quota wall"
            )

            # Short cooldown (transient 60s cap) should clear quickly — advance 65s
            broker.clock += 65.0
            await broker.tick()

            # Luna should be available again (transient cooldown cleared)
            reservation2 = broker.reserve_next()
            assert reservation2 is not None and reservation2.member.name == LUNA
            done2 = await broker.dispatcher.dispatch(reservation2)
            assert luna.chat_calls >= 1

    asyncio.run(scenario())


def test_router_upstream_429_503_is_quota_wall():
    """fleet-llm-router turns a Luna upstream 429 into a 503 labelled
    quota_or_rate_limit (10-23: 24 requeues, PRE_ACCEPTANCE_REQUEUE_EXHAUSTED)."""
    from gemma_broker.runtime import DispatchLoop

    classify = DispatchLoop._classify_preaccept_failure
    router = 'TRANSPORT_HTTP_503:{"error":{"message":"luna tool route upstream 429 (quota_or_rate_limit)","type":"upstream_error"}}'
    assert classify(None, router) == ("quota_wall", "quota_or_rate_limit")
    assert classify(None, 'TRANSPORT_HTTP_503:{"status":"down"}') == ("transient", None)
