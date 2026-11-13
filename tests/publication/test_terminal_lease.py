"""Handler-level proof that hung/lost work ends typed and leases reach zero.

Backends are real aiohttp TestServers; no GPU, Redis or live service is used.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from comfyui_liveness import (
    LOST_COMPLETION_ERROR_TYPE,
    LostCompletionTracker,
    PromptLiveness,
    lost_completion_grace_s,
    probe_prompt,
)
from gemma_broker.compatibility import RequestRequirements
from gemma_broker.contracts import (
    JobRecord,
    JobState,
    MemberSnapshot,
    MemberState,
    ReasonCode,
)
from gemma_broker.dispatcher import Dispatcher
from gemma_broker.redis_store import (
    InMemoryJobRepository,
    LeadershipLost,
    RepositoryUnavailable,
)


def run(coro):
    return asyncio.run(coro)


# ── ComfyUI lost-completion ──────────────────────────────────────────────────

def _comfy_app(history: dict, running: list, pending: list) -> web.Application:
    async def hist(request):
        pid = request.match_info["pid"]
        return web.json_response({pid: history[pid]} if pid in history else {})

    async def queue(request):
        return web.json_response({"queue_running": running, "queue_pending": pending})

    app = web.Application()
    app.router.add_get("/history/{pid}", hist)
    app.router.add_get("/queue", queue)
    return app


async def _probe(history, running, pending, pid="p1"):
    server = TestServer(_comfy_app(history, running, pending))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            return await probe_prompt(session, str(server.make_url("")).rstrip("/"), pid)
    finally:
        await server.close()


def test_probe_classifies_history_queue_and_absent():
    assert run(_probe({"p1": {"status": {}}}, [], [])) is PromptLiveness.PRESENT
    assert run(_probe({}, [[1, "p1", {}]], [])) is PromptLiveness.ACTIVE
    assert run(_probe({}, [], [[2, "p1", {}]])) is PromptLiveness.ACTIVE
    assert run(_probe({}, [], [])) is PromptLiveness.ABSENT


def test_probe_unreachable_backend():
    async def go():
        server = TestServer(_comfy_app({}, [], []))
        await server.start_server()
        base = str(server.make_url("")).rstrip("/")
        await server.close()
        async with aiohttp.ClientSession() as session:
            return await probe_prompt(session, base, "p1")

    assert run(go()) is PromptLiveness.UNREACHABLE


def test_absent_prompt_is_lost_only_after_grace_and_active_work_is_protected():
    tracker = LostCompletionTracker(grace_s=60)
    tracker.observe(PromptLiveness.ABSENT, 100.0)
    assert tracker.lost_reason(159.0) is None
    assert "prompt_absent" in tracker.lost_reason(161.0)
    # Real running work resets the clock: never declared lost.
    tracker.observe(PromptLiveness.ACTIVE, 170.0)
    assert tracker.lost_reason(10_000.0) is None
    tracker.observe(PromptLiveness.UNREACHABLE, 200.0)
    assert "backend_unreachable" in tracker.lost_reason(261.0)


def test_grace_is_clamped_and_error_type_is_typed():
    assert lost_completion_grace_s({}) == 120
    assert lost_completion_grace_s({"lost_completion_grace_s": 1}) == 30
    assert lost_completion_grace_s({"lost_completion_grace_s": 10**9}) == 1800
    assert LOST_COMPLETION_ERROR_TYPE == "lost_completion"


# ── Gemma dispatcher lease reconciliation ────────────────────────────────────

def _member() -> MemberSnapshot:
    return MemberSnapshot(
        name="m1", gpu_id="gpu0", state=MemberState.READY_ACCEPTING,
        accepting=True, configured_slots=2, context_per_slot=4096,
        state_version=1, observed_at=1.0, compatible_free_slots=2,
        capabilities=("chat",),
    )


def _reserve(repo: InMemoryJobRepository, job_id: str):
    member = repo.put_member(_member()) if not repo.get_member("m1") else repo.get_member("m1")
    repo.submit(
        JobRecord.new(
            job_id=job_id, idempotency_key=f"i-{job_id}", request_sha256="0" * 64,
            request_body={"messages": [{"role": "user", "content": "x"}]},
            submitted_at=1.0, enqueue_sequence=1, required_capabilities=("chat",),
        ),
        caller_scope="t",
    )
    return repo.reserve_next(
        now=2.0, ordered_members=[member],
        requirements=RequestRequirements(8, 16, ("chat",), "t"),
    )


class _Backend:
    def __init__(self, status=200):
        self.status = status
        self.server = TestServer(self._app())

    def _app(self):
        async def chat(request):
            await request.read()
            return web.json_response({"ok": True}, status=self.status)

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        return app


class _Transport:
    def __init__(self, base, session, raises=None):
        self.base, self.session, self.raises = base, session, raises

    async def send(self, member, body, on_accepted):
        if self.raises:
            on_accepted()
            raise self.raises
        async with self.session.post(f"{self.base}/v1/chat/completions", json=body) as r:
            on_accepted()
            return await r.json()


async def _scenario(fn):
    backend = _Backend()
    await backend.server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            base = str(backend.server.make_url("")).rstrip("/")
            return await fn(base, session)
    finally:
        await backend.server.close()


def _dispatcher(repo, transport):
    return Dispatcher(
        repo=repo, transport=transport, request_timeout=5.0,
        clock=lambda: 3.0, terminal_write_delay=0.0,
    )


def _outage_on_terminal(repo, times):
    original = repo.replace_job
    state = {"left": times}

    def flaky(job):
        if job.state in {JobState.COMPLETED, JobState.OUTCOME_UNKNOWN} and state["left"]:
            state["left"] -= 1
            raise RepositoryUnavailable("redis_state_unknown")
        return original(job)

    repo.replace_job = flaky


def test_completed_releases_lease_and_duplicate_dispatch_is_stale():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "j1")
        d = _dispatcher(repo, _Transport(base, session))
        out = await d.dispatch(res)
        assert out.state is JobState.COMPLETED
        assert repo.lease_count("m1") == 0
        # Duplicate completion must not release anything twice.
        try:
            await d.dispatch(res)
        except Exception as exc:  # LEASE_NOT_OWNED surfaces, never a second release
            assert "LEASE_NOT_OWNED" in str(exc)
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


def test_transient_outage_on_terminal_write_is_retried_and_lease_freed():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "j2")
        _outage_on_terminal(repo, 2)
        out = await _dispatcher(repo, _Transport(base, session)).dispatch(res)
        assert out.state is JobState.COMPLETED
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


def test_persistent_outage_is_reconciled_to_outcome_unknown_without_replay():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "j3")
        _outage_on_terminal(repo, 10)
        d = _dispatcher(repo, _Transport(base, session))
        try:
            await d.dispatch(res)
            raise AssertionError("outage must propagate")
        except RepositoryUnavailable:
            pass
        assert repo.lease_count("m1") == 1 and len(d.unreleased) == 1
        assert repo.get("j3").state is JobState.ACCEPTED
        # Still down: nothing released, nothing lost.
        repo.replace_job = type(repo).replace_job.__get__(repo)
        _outage_on_terminal(repo, 10)
        assert await d.reconcile_unreleased() == []
        assert len(d.unreleased) == 1
        # Repository recovers: accepted work becomes OUTCOME_UNKNOWN, lease freed.
        repo.replace_job = type(repo).replace_job.__get__(repo)
        assert await d.reconcile_unreleased() == [res.job.current_attempt_id]
        job = repo.get("j3")
        assert job.state is JobState.OUTCOME_UNKNOWN
        assert job.reason_code is ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN
        assert repo.lease_count("m1") == 0 and not d.unreleased
        assert await d.reconcile_unreleased() == []  # exactly once

    run(_scenario(fn))


def test_leadership_loss_does_not_release_and_successor_lease_is_protected():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "j4")
        original = repo.replace_job

        def lost(job):
            if job.state is JobState.COMPLETED:
                raise LeadershipLost("fenced")
            return original(job)

        repo.replace_job = lost
        d = _dispatcher(repo, _Transport(base, session))
        try:
            await d.dispatch(res)
            raise AssertionError("leadership loss must propagate")
        except LeadershipLost:
            pass
        # A successor owns reconciliation; this dispatcher must not touch it.
        assert repo.lease_count("m1") == 1 and not d.unreleased
        # A newer attempt on the same member keeps its lease through reconcile.
        repo.add_member_lease("m1", "successor-attempt")
        d.unreleased["stale-attempt"] = res
        repo.replace_job = original
        await d.reconcile_unreleased()
        assert "successor-attempt" in repo._leases["m1"]

    run(_scenario(fn))


def test_cancel_after_acceptance_releases_lease():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "j5")

        class Cancelling(_Transport):
            async def send(self, member, body, on_accepted):
                on_accepted()
                repo.replace_job(replace(repo.get("j5"), cancel_requested=True))
                return {"ok": True}

        out = await _dispatcher(repo, Cancelling(base, session)).dispatch(res)
        assert out.state is JobState.CANCELLED
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


def test_post_acceptance_transport_error_is_outcome_unknown_and_released():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "j6")
        out = await _dispatcher(
            repo, _Transport(base, session, raises=RuntimeError("boom"))
        ).dispatch(res)
        assert out.state is JobState.OUTCOME_UNKNOWN
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


# ── Dispatch-exception lease convergence ─────────────────────────────────────

def _fail_once(repo, name, exc):
    original = getattr(repo, name)
    state = {"left": 1}

    def flaky(*args, **kwargs):
        if state["left"]:
            state["left"] -= 1
            raise exc
        return original(*args, **kwargs)

    setattr(repo, name, flaky)


def test_pre_acceptance_backend_error_requeues_and_frees_lease():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "d1")

        class Refusing(_Transport):
            async def send(self, member, body, on_accepted):
                raise ConnectionError("refused before acceptance")

        out = await _dispatcher(repo, Refusing(base, session)).dispatch(res)
        assert out.state is JobState.QUEUED
        assert out.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


def test_cancel_before_acceptance_frees_lease():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "d2")
        repo.replace_job(replace(repo.get("d2"), cancel_requested=True))
        out = await _dispatcher(repo, _Transport(base, session)).dispatch(res)
        assert out.state is JobState.CANCELLED
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


def test_outage_on_pre_acceptance_requeue_converges_to_queued_zero_leases():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "d3")

        class Refusing(_Transport):
            async def send(self, member, body, on_accepted):
                raise ConnectionError("refused before acceptance")

        _fail_once(repo, "requeue_claim", RepositoryUnavailable("redis_state_unknown"))
        d = _dispatcher(repo, Refusing(base, session))
        try:
            await d.dispatch(res)
            raise AssertionError("outage must propagate")
        except RepositoryUnavailable:
            pass
        assert repo.lease_count("m1") == 1 and len(d.unreleased) == 1
        assert await d.reconcile_unreleased() == [res.job.current_attempt_id]
        assert repo.get("d3").state is JobState.QUEUED
        assert repo.lease_count("m1") == 0 and not d.unreleased
        assert await d.reconcile_unreleased() == []

    run(_scenario(fn))


def test_outage_recording_acceptance_is_outcome_unknown_never_replayed():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "d4")
        original = repo.replace_job

        def lost_acceptance(job):
            if job.state is JobState.ACCEPTED:
                raise RepositoryUnavailable("redis_state_unknown")
            return original(job)

        repo.replace_job = lost_acceptance
        d = _dispatcher(repo, _Transport(base, session))
        try:
            await d.dispatch(res)
            raise AssertionError("outage must propagate")
        except RepositoryUnavailable:
            pass
        assert repo.lease_count("m1") == 1 and len(d.unreleased) == 1
        repo.replace_job = original
        assert await d.reconcile_unreleased() == [res.job.current_attempt_id]
        job = repo.get("d4")
        assert job.state is JobState.OUTCOME_UNKNOWN
        assert job.reason_code is ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN
        assert repo.lease_count("m1") == 0 and not d.unreleased

    run(_scenario(fn))


def test_outage_on_cancel_before_acceptance_converges_to_cancelled():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "d5")
        repo.replace_job(replace(repo.get("d5"), cancel_requested=True))
        _fail_once(repo, "replace_job", RepositoryUnavailable("redis_state_unknown"))
        d = _dispatcher(repo, _Transport(base, session))
        try:
            await d.dispatch(res)
            raise AssertionError("outage must propagate")
        except RepositoryUnavailable:
            pass
        assert repo.lease_count("m1") == 1
        await d.reconcile_unreleased()
        assert repo.get("d5").state is JobState.CANCELLED
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


def test_leadership_loss_before_acceptance_is_left_to_the_successor_fence():
    async def fn(base, session):
        repo = InMemoryJobRepository()
        res = _reserve(repo, "d6")
        _fail_once(repo, "replace_job", LeadershipLost("fenced"))
        d = _dispatcher(repo, _Transport(base, session))
        try:
            await d.dispatch(res)
            raise AssertionError("leadership loss must propagate")
        except LeadershipLost:
            pass
        # Not released or tracked here: a successor owns recovery, and its
        # stale-claim recovery (requeue_claim) frees the lease exactly once.
        assert repo.lease_count("m1") == 1 and not d.unreleased
        repo.requeue_claim(
            repo.get("d6"), reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE, terminal_at=9.0
        )
        assert repo.lease_count("m1") == 0

    run(_scenario(fn))


# ── burst release and idle zero-lease acceptance ─────────────────────────────

def test_idle_member_holds_zero_leases_and_empty_queue_reserves_nothing():
    repo = InMemoryJobRepository()
    member = repo.put_member(_member())
    assert repo.lease_count("m1") == 0
    got = repo.reserve_next(
        now=2.0, ordered_members=[member],
        requirements=RequestRequirements(8, 16, ("chat",), "t"),
    )
    assert got is None
    assert repo.lease_count("m1") == 0


def test_burst_of_concurrent_terminals_releases_every_lease():
    """Concurrent completions, a transport failure and a terminal-write outage
    in one burst all converge to zero leases, repeatedly, with no replay."""
    async def fn(base, session):
        repo = InMemoryJobRepository()
        for round_no in range(6):
            reservations = [
                _reserve(repo, f"b{round_no}a"),
                _reserve(repo, f"b{round_no}b"),
            ]
            assert all(r is not None for r in reservations)
            assert repo.lease_count("m1") == 2
            if round_no % 2:
                _outage_on_terminal(repo, 1)
            transports = [
                _Transport(base, session),
                _Transport(base, session, raises=aiohttp.ClientConnectionError("reset"))
                if round_no % 3 == 0 else _Transport(base, session),
            ]
            outcomes = await asyncio.gather(
                *(
                    _dispatcher(repo, t).dispatch(r)
                    for t, r in zip(transports, reservations)
                )
            )
            terminal = {JobState.COMPLETED, JobState.OUTCOME_UNKNOWN, JobState.FAILED, JobState.CANCELLED}
            assert all(o.state in terminal for o in outcomes), [o.state for o in outcomes]
            assert repo.lease_count("m1") == 0, f"round {round_no} leaked a lease"

    run(_scenario(fn))
