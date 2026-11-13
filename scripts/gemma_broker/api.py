"""Durable asynchronous API for the combined Gemma broker."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from threading import Lock
from typing import Callable
from uuid import uuid4

from aiohttp import web

from .compatibility import build_requirements, evaluate, normalize_chat_request
from .contracts import AttemptRecord, ContractError, JobRecord, JobState, MemberState, Priority, ReasonCode
from .redis_store import RepositoryUnavailable


class APIError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class DurableBrokerService:
    ALLOWED_SUBMIT_FIELDS = frozenset(
        {"idempotency_key", "request", "priority", "caller_scope", "pinned_member"}
    )

    def __init__(
        self,
        repository,
        *,
        members: Callable[[], list],
        clock: Callable[[], float],
        safety_margin_tokens: int = 1024,
    ) -> None:
        self.repository = repository
        self.safety_margin_tokens = int(safety_margin_tokens)
        self.members = members
        self.clock = clock
        self.member_admission: Callable[[str], bool] | None = None
        self.member_capacity: Callable[[str], bool] | None = None
        self._sequence = 0
        self._lock = Lock()
        self._all_dead_since: float | None = None

    #: An outage of every member shorter than this (a router safe-restart) still queues.
    DEAD_GRACE_SECONDS = 30.0
    _DEAD_STATES = frozenset({MemberState.UNHEALTHY, MemberState.FAULTED, MemberState.OFFLINE})

    def _fail_fast_if_all_dead(self, requirements, candidates) -> None:
        """503 instead of queueing for up to api_wait_timeout when nothing can serve.

        Only observed members count (observed_at 0 = no probe yet after boot).
        """
        compatible = [m for m in candidates if evaluate(requirements, m, safety_margin_tokens=self.safety_margin_tokens).compatible]
        if not all(m.state in self._DEAD_STATES and m.observed_at > 0 for m in compatible):
            self._all_dead_since = None
            return
        now = self.clock()
        if self._all_dead_since is None:
            self._all_dead_since = now
        if now - self._all_dead_since >= self.DEAD_GRACE_SECONDS:
            raise APIError(503, "no_live_member", "every compatible member is unhealthy")

    #: The only reason ``reduce_member_readiness`` reports ``ready_busy``.
    #: Capacity exhaustion is steady-state load, not a lifecycle transition.
    CAPACITY_ONLY_REASONS = frozenset({ReasonCode.MEMBER_NO_CAPACITY.value})

    def _pinned_admits(self, pinned, member_name: str) -> bool:
        """Decide whether an observed pinned member may take a job right now.

        ``accepting``/``state`` on the observer snapshot fold capacity in with
        lifecycle health, and an embedded observer republishes only on its own
        refresh tick. A slot released moments earlier is therefore invisible
        for up to one interval, and a pinned submit landing in that window was
        refused even though the durable mirror -- the authority the reservation
        Lua reads -- had already proved the slot free.

        So a refusal that is *capacity-only* is re-read from the durable mirror
        via ``member_capacity``. Every other refusal, real saturation included,
        still answers 503 immediately: no wait, no tick dependency.
        """
        state = getattr(pinned.state, "value", pinned.state)
        if pinned.accepting and state == "ready_accepting":
            return True
        reasons = {getattr(code, "value", code) for code in (pinned.reason_codes or ())}
        if state != "ready_busy" or reasons != set(self.CAPACITY_ONLY_REASONS):
            return False
        if self.member_capacity is None:
            return False
        try:
            return bool(self.member_capacity(member_name))
        except RepositoryUnavailable:
            return False

    @staticmethod
    def _request_hash(request_body: dict) -> str:
        encoded = json.dumps(
            request_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def submit(self, payload: dict, *, kv_scope: dict | None = None) -> JobRecord:
        if not isinstance(payload, dict):
            raise APIError(400, "invalid_json", "JSON object required")
        unknown = sorted(set(payload) - self.ALLOWED_SUBMIT_FIELDS)
        if unknown:
            raise APIError(400, "unknown_fields", ",".join(unknown))
        idem = payload.get("idempotency_key")
        request_body = payload.get("request")
        if not isinstance(idem, str) or not idem or not isinstance(request_body, dict):
            raise APIError(400, "invalid_request", "idempotency_key and request are required")
        request_body = normalize_chat_request(request_body)  # returns a copy
        # The KV affinity scope is server-only: a client-sent value is always dropped, and only the
        # in-process compat path (verified principal) may pass one as a keyword argument.
        request_body.pop("_kv_session_id", None)
        if kv_scope:
            request_body["_kv_session_id"] = dict(kv_scope)
        messages = request_body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise APIError(400, "invalid_request", "messages or prompt is required")
        try:
            priority = Priority(int(payload.get("priority", Priority.NORMAL)))
        except (TypeError, ValueError):
            raise APIError(400, "priority_invalid", "priority must be 0, 10, or 20")
        hint_raw = request_body.get("input_tokens_hint")
        try:
            input_tokens_hint = None if hint_raw is None else int(hint_raw)
        except (TypeError, ValueError) as exc:
            raise APIError(
                400,
                "invalid_request",
                "input_tokens_hint must be a non-negative integer",
            ) from exc
        if isinstance(hint_raw, bool) or (
            input_tokens_hint is not None and input_tokens_hint < 0
        ):
            raise APIError(
                400,
                "invalid_request",
                "input_tokens_hint must be a non-negative integer",
            )
        requirements = build_requirements(
            request_body,
            required_capabilities=("chat",),
            input_tokens_hint=input_tokens_hint,
        )
        candidates = self.members()
        pinned_member = payload.get("pinned_member")
        if pinned_member is not None:
            if not isinstance(pinned_member, str) or not pinned_member:
                raise APIError(400, "invalid_request", "pinned_member must be a member name")
            if pinned_member not in {member.name for member in candidates}:
                raise APIError(404, "unknown_member", pinned_member)
            requirements = replace(requirements, pinned_member=pinned_member)
            pinned = next(member for member in candidates if member.name == pinned_member)
            if not evaluate(requirements, pinned, safety_margin_tokens=self.safety_margin_tokens).compatible:
                if not set(requirements.required_capabilities).issubset(set(pinned.capabilities)):
                    raise APIError(503, "pinned_member_unavailable", "pinned member has no compatible route")
                raise APIError(422, "no_eligible_member", ReasonCode.NO_ELIGIBLE_MEMBER.value)
            try:
                admission_ok = self.member_admission is None or self.member_admission(pinned_member)
            except RepositoryUnavailable as exc:
                raise APIError(503, "redis_state_unknown", str(exc)) from exc
            if not admission_ok or not self._pinned_admits(pinned, pinned_member):
                raise APIError(503, "pinned_member_unavailable", "pinned member is not currently dispatchable")
        if not any(evaluate(requirements, member, safety_margin_tokens=self.safety_margin_tokens).compatible for member in candidates):
            raise APIError(422, "no_eligible_member", ReasonCode.NO_ELIGIBLE_MEMBER.value)
        if pinned_member is None:
            self._fail_fast_if_all_dead(requirements, candidates)
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        job = JobRecord.new(
            job_id=str(uuid4()),
            idempotency_key=idem,
            request_sha256=self._request_hash(request_body),
            request_body=request_body,
            submitted_at=self.clock(),
            enqueue_sequence=sequence,
            base_priority=priority,
            input_tokens_estimate=requirements.input_tokens_estimate,
            max_output_tokens=requirements.max_output_tokens,
            required_capabilities=requirements.required_capabilities,
            pinned_member=pinned_member,
        )
        try:
            return self.repository.submit(
                job, caller_scope=str(payload.get("caller_scope", "api"))
            )
        except RepositoryUnavailable as exc:
            raise APIError(503, "redis_state_unknown", str(exc)) from exc
        except ContractError as exc:
            if ReasonCode.IDEMPOTENCY_KEY_REUSED.value in str(exc):
                raise APIError(409, "idempotency_key_reused", str(exc)) from exc
            raise APIError(400, "contract_error", str(exc)) from exc

    def status(self, job_id: str) -> JobRecord:
        try:
            job = self.repository.get(job_id)
        except RepositoryUnavailable as exc:
            raise APIError(503, "redis_state_unknown", str(exc)) from exc
        if job is None:
            raise APIError(404, "job_not_found", job_id)
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self.status(job_id)
        if job.accepted_boundary_crossed or job.state in {JobState.ACCEPTED, JobState.IN_FLIGHT}:
            raise APIError(
                409,
                "cancel_unsafe_after_acceptance",
                ReasonCode.CANCEL_UNSAFE_AFTER_ACCEPTANCE.value,
            )
        if job.state is not JobState.QUEUED:
            return job
        try:
            return self.repository.cancel_queued(job_id, completed_at=self.clock())
        except RepositoryUnavailable as exc:
            raise APIError(503, "redis_state_unknown", str(exc)) from exc

    def complete_for_test(self, job_id: str, result: dict) -> JobRecord:
        current = self.status(job_id)
        return self.repository.replace_job(
            replace(
                current,
                state=JobState.COMPLETED,
                result=result,
                completed_at=self.clock(),
                state_version=current.state_version + 1,
            )
        )

    def accept_for_test(self, job_id: str) -> JobRecord:
        current = self.status(job_id)
        return self.repository.replace_job(
            replace(
                current,
                state=JobState.ACCEPTED,
                accepted_boundary_crossed=True,
                state_version=current.state_version + 1,
            )
        )


def _terminal(job: JobRecord) -> bool:
    return job.state in {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.OUTCOME_UNKNOWN,
    }


def _status_body(job: JobRecord, attempt: AttemptRecord | None = None) -> dict:
    body = job.to_dict()
    body["terminal"] = _terminal(job)
    if attempt is None:
        body["attempt"] = None
    else:
        attempt_body = attempt.to_dict()
        attempt_body.pop("backend_url", None)
        body["attempt"] = attempt_body
    return body


def _current_attempt(service, job: JobRecord) -> AttemptRecord | None:
    if not job.current_attempt_id:
        return None
    repository = getattr(service, "repository", None)
    getter = getattr(repository, "get_attempt", None)
    if not callable(getter):
        raise RepositoryUnavailable("current attempt repository unavailable")
    attempt = getter(job.current_attempt_id)
    if attempt is None:
        raise RepositoryUnavailable(
            f"current attempt record missing: {job.current_attempt_id}"
        )
    if not isinstance(attempt, AttemptRecord):
        raise RepositoryUnavailable(
            f"current attempt record invalid: {job.current_attempt_id}"
        )
    return attempt


def _error(exc: APIError) -> web.Response:
    return web.json_response(
        {"error": exc.message, "code": exc.code}, status=exc.status
    )


def register_api_routes(app: web.Application, service) -> None:
    async def submit(request: web.Request) -> web.Response:
        try:
            try:
                payload = await request.json()
            except Exception as exc:
                raise APIError(400, "invalid_json", "valid JSON required") from exc
            job = service.submit(payload)
            status_url = f"/v1/gemma/jobs/{job.job_id}"
            cancel_url = status_url
            response = web.json_response(
                {
                    "job_id": job.job_id,
                    "state": job.state.value,
                    "status_url": status_url,
                    "cancel_url": cancel_url,
                    "next_attempt_id": None,
                },
                status=202,
            )
            response.headers["Location"] = status_url
            return response
        except APIError as exc:
            return _error(exc)

    async def status(request: web.Request) -> web.Response:
        try:
            job = service.status(request.match_info["job_id"])
            return web.json_response(_status_body(job, _current_attempt(service, job)))
        except APIError as exc:
            return _error(exc)
        except RepositoryUnavailable as exc:
            return web.json_response(
                {"error": str(exc), "code": "redis_state_unknown"}, status=503
            )

    async def cancel(request: web.Request) -> web.Response:
        try:
            job = service.cancel(request.match_info["job_id"])
            return web.json_response(_status_body(job, _current_attempt(service, job)))
        except APIError as exc:
            return _error(exc)
        except RepositoryUnavailable as exc:
            return web.json_response(
                {"error": str(exc), "code": "redis_state_unknown"}, status=503
            )

    async def observability(request: web.Request) -> web.Response:
        handlers = {
            "/health/live": "health_live",
            "/health/ready": "health_ready",
            "/health/dispatch": "health_dispatch",
            "/v1/metrics/broker": "broker_metrics",
        }
        method_name = handlers.get(request.path)
        method = getattr(service, method_name, None) if method_name else None
        if not callable(method):
            return web.json_response(
                {"error": "broker observability unavailable", "code": "broker_unavailable"},
                status=503,
            )
        try:
            result = method()
            if isinstance(result, tuple) and len(result) == 2:
                body, status = result
            else:
                body, status = result, 200
            return web.json_response(body, status=int(status))
        except APIError as exc:
            return _error(exc)
        except RepositoryUnavailable as exc:
            return web.json_response(
                {"error": str(exc), "code": "redis_state_unknown"}, status=503
            )
        except Exception as exc:
            return web.json_response(
                {"error": str(exc), "code": "broker_observability_error"}, status=503
            )

    app.router.add_post("/v1/gemma/jobs", submit)
    app.router.add_get("/v1/gemma/jobs/{job_id}", status)
    app.router.add_delete("/v1/gemma/jobs/{job_id}", cancel)
    app.router.add_get("/health/live", observability)
    app.router.add_get("/health/ready", observability)
    app.router.add_get("/health/dispatch", observability)
    app.router.add_get("/v1/metrics/broker", observability)


def create_api_app(service: DurableBrokerService) -> web.Application:
    app = web.Application()
    register_api_routes(app, service)
    return app
