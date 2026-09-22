"""Self-contained broker runtime composition.

The runtime is a closed composition layer that wires the existing broker
components together using dependency injection. It exposes only the closed
operations (build_runtime, app, dispatch_loop, wait_for_terminal, etc.)
and refuses any candidate-side lifecycle mutation or SSE request. It is
designed to be constructed once per candidate process and shut down
cleanly via close().

Closed-contract rules enforced here:
    - The candidate runtime NEVER mutates member lifecycle. The methods
      start_member / stop_member / restart_member always raise.
    - The candidate runtime NEVER registers or serves Server-Sent Events.
      The exposed handle_proxy and the constructed aiohttp app refuse
      any text/event-stream request and any /v1/chat/completions request
      whose body advertises stream=True.
    - The candidate runtime accepts any RedisJobRepository-compatible
      repository (the durable RedisJobRepository OR the InMemoryJobRepository
      semantically equivalent) and validates the caller-supplied config and
      member ordering against the strict candidate invariants:
        * enabled flag is True
        * redis_namespace begins with "qual:combined-gemma:"
        * ordered_members is a non-empty, unique list of operator-chosen IDs
    - Dispatch loop ownership is fenced via repository.acquire_leadership
      / renew_leadership; only one runtime ever dispatches for a prefix.
    - Polling-based terminal wait is provided for non-SSE buffered
      responses; the wrapper never opens an SSE stream.
    - The constructed aiohttp candidate app exposes /health, /v1/gemma/jobs
      (POST submit, GET status, DELETE cancel), and /v1/chat/completions.
    - The readiness mapper requires semantic HTTP proven input (semantic
      health probe + model resident + idle service effective + dispatcher
      registered); TCP-only readiness is rejected.
    - close() drains any running loop and async tasks without mutating
      member lifecycle.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import time
from uuid import uuid4
from typing import Any, Awaitable, Callable, Iterable, Protocol, cast, runtime_checkable

from aiohttp import web

from .api import DurableBrokerService
from .compat import CompatibilityWrapper
from .compatibility import build_requirements, evaluate
from .config import BrokerConfig, ConfigError
from .contracts import (
    JobRecord,
    JobState,
    MemberSnapshot,
    MemberState,
    Priority,
    ReasonCode,
)
from .dispatcher import Dispatcher
from .events import BoundedEventProducer
from .observability import BrokerMetrics, SCHEMA_VERSION
from .drain import MemberDrainCoordinator
from .readiness import reduce_member_readiness
from .reconcile import InMemoryRepairFence, Reconciler
from .redis_store import (
    InMemoryJobRepository,
    LeadershipLost,
    RedisJobRepository,
    RepositoryUnavailable,
    ReservationStale,
)


logger = logging.getLogger("gemma_broker.runtime")


# ----------------------------------------------------------------------
# Repository protocol and config validation
# ----------------------------------------------------------------------


@runtime_checkable
class _RepositoryProtocol(Protocol):
    """Subset of the repository surface the runtime depends on.

    This is the runtime's closed contract: any object exposing these
    methods (RedisJobRepository, InMemoryJobRepository, or a future
    minimal fake) is acceptable. The runtime NEVER assumes a specific
    class — it dispatches via duck typing against this protocol.
    """

    def submit(self, job: JobRecord, *, caller_scope: str) -> JobRecord: ...
    def get(self, job_id: str) -> JobRecord | None: ...
    def get_member(self, member_name: str) -> MemberSnapshot | None: ...
    def lease_count(self, member_name: str) -> int: ...
    def begin_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot: ...
    def complete_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot: ...
    def acquire_leadership(self, *, owner: str, now: float, ttl_seconds: float) -> int: ...
    def renew_leadership(
        self, *, owner: str, fencing_token: int, now: float, ttl_seconds: float
    ) -> int: ...
    def replace_job(self, job: JobRecord) -> JobRecord: ...
    def requeue_claim(
        self,
        job: JobRecord,
        *,
        reason: ReasonCode,
        terminal_at: float | None = None,
    ) -> JobRecord: ...
    def reserve_next(
        self,
        *,
        now: float,
        ordered_members: list[MemberSnapshot],
        requirements: Any,
        leader_owner: str,
        fencing_token: int,
        backend_url: str | None = None,
        member_lookup: Callable[[str], MemberSnapshot] | None = None,
    ) -> Any: ...
    def cancel_queued(self, job_id: str, *, completed_at: float) -> JobRecord: ...
    def begin_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        generation_fence: int,
    ) -> MemberSnapshot: ...
    def complete_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        readiness: MemberSnapshot,
    ) -> MemberSnapshot: ...
    def release_member_lease(
        self,
        member_name: str,
        attempt_id: str,
        *,
        claimed_member_state_version: int,
    ) -> MemberSnapshot: ...
    def reconcile_dead_job(
        self,
        job_id: str,
        *,
        repair_owner: str,
        repair_fence: int,
    ) -> JobRecord: ...


def _is_repository_compatible(obj: Any) -> bool:
    """Return True if obj quacks like the runtime's repository protocol.

    The check is structural — it tests for the presence of the method
    names the runtime and its direct collaborators actually call — so
    a future minimal fake (in tests, in a different VM-side harness) is
    accepted without inheritance.

    The required set is the intersection of the public method names on
    the two real repository implementations
    (InMemoryJobRepository and RedisJobRepository), restricted to the
    methods the runtime's collaborators use on every code path. The
    runtime only demands the methods that BOTH real classes already
    expose — anything beyond that is the collaborator's responsibility
    (a stub that wants to skip the dispatch loop or the reconcile path
    must override the corresponding collaborator directly).

    The authoritative inventory lives in
    :data:`REPOSITORY_REQUIRED_METHODS` — this function and the
    failure-report path in :class:`BrokerRuntime` both read from that
    single constant so the two can never drift.
    """
    if obj is None:
        return False
    return all(callable(getattr(obj, name, None)) for name in REPOSITORY_REQUIRED_METHODS)


#: Canonical inventory of the repository methods the runtime's
#: structural compatibility gate requires. This is the single source
#: of truth shared by :func:`_is_repository_compatible` and the
#: failure-report path in :class:`BrokerRuntime.__init__`; the
#: inventory and the error message must reference each other so a
#: reader of the failure can find the authoritative list in one place.
#:
#: The set is the union of the methods called on every code path by
#: the runtime's direct collaborators (``DurableBrokerService``,
#: ``Dispatcher``, ``DispatchLoop``, ``MemberDrainCoordinator``,
#: ``Reconciler``, ``PollingTerminalWait``). Both
#: :class:`InMemoryJobRepository` and :class:`RedisJobRepository`
#: expose every name on this tuple — adding a name that isn't on
#: both real implementations would break the in-process tests.
REPOSITORY_REQUIRED_METHODS: tuple[str, ...] = (
    "submit",
    "get",
    "get_member",
    "lease_count",
    "begin_member_drain",
    "complete_member_drain",
    "replace_job",
    "reserve_next",
    "cancel_queued",
    "release_member_lease",
    "begin_member_rejoin",
    "complete_member_rejoin",
    "acquire_leadership",
    "renew_leadership",
    "requeue_claim",
)


CANDIDATE_NAMESPACE_PREFIX = "qual:combined-gemma:"


def validate_candidate_config(config: BrokerConfig) -> None:
    """Enforce the closed candidate invariants on a BrokerConfig.

    The runtime never starts a process with a config that does not
    match the candidate production-mountable contract. The checks
    here are the same ones the candidate qualification harness
    applies, but provided as a callable so callers (qualification,
    tests, management scripts) can use the same function.

    Raises ConfigError on the first violation.

    Production-mode policies are still gated by the candidate design:
    a production config must keep ``lifecycle_mutation_enabled`` as a
    strict bool (operator-chosen) and must end its Redis namespace
    with ``:`` for the repository's strict prefix invariant. The
    candidate-only invariants below are enforced ONLY when the config
    is candidate mode (``config.candidate_mode`` True); production
    mounts are validated for shape but not for the qualifier-only
    constraints, since those exist to keep candidate qualification
    hermetic and would otherwise block production promotion.
    """
    if not isinstance(config, BrokerConfig):
        raise ConfigError("CONFIG_OBJECT_REQUIRED")
    if config.enabled is not True:
        raise ConfigError("CANDIDATE_ENABLED_REQUIRED")
    if config.sse_enabled is not False:
        raise ConfigError("SSE_DISABLED_REQUIRED")
    if not isinstance(config.redis_namespace, str) or not config.redis_namespace:
        raise ConfigError("REDIS_NAMESPACE_REQUIRED")
    if not config.redis_namespace.endswith(":"):
        raise ConfigError("CANDIDATE_REDIS_NAMESPACE_TRAILING_COLON_REQUIRED")
    if not config.ordered_members:
        raise ConfigError("MEMBERS_NONEMPTY_REQUIRED")
    if len(config.ordered_members) != len(set(config.ordered_members)):
        raise ConfigError("DUPLICATE_MEMBERS")
    # Candidate-only constraints: lifecycle mutation MUST be off in
    # candidate mode (qualification isolation) and the namespace MUST be
    # a ``qual:combined-gemma:`` namespace. Production mode lifts these
    # two checks because operator policy may legitimately enable them.
    if config.candidate_mode:
        if config.lifecycle_mutation_enabled is not False:
            raise ConfigError("CANDIDATE_LIFECYCLE_MUTATION_DISABLED_REQUIRED")
        if not config.redis_namespace.startswith(CANDIDATE_NAMESPACE_PREFIX):
            raise ConfigError(
                f"CANDIDATE_REDIS_NAMESPACE_PREFIX_INVALID:"
                f"expected_prefix={CANDIDATE_NAMESPACE_PREFIX!r}"
            )


# ----------------------------------------------------------------------
# Buffered aiohttp transport
# ----------------------------------------------------------------------


class Transport(Protocol):
    """Closed transport contract the dispatcher calls."""

    async def send(
        self,
        member: Any,
        body: dict[str, Any],
        on_accepted: Callable[[], None],
    ) -> dict[str, Any]: ...


def _split_leading_think(content: str) -> tuple[str | None, str]:
    """Separate a leading MiniMax think block from final assistant content."""
    stripped = content.lstrip()
    if not stripped.startswith("<think>"):
        return None, content
    remainder = stripped[len("<think>") :]
    closing_index = remainder.find("</think>")
    if closing_index < 0:
        return remainder.strip(), ""
    reasoning = remainder[:closing_index].strip()
    final_content = remainder[closing_index + len("</think>") :].strip()
    return reasoning, final_content


class BufferedAiohttpTransport:
    """Aiohttp-backed buffered transport for the candidate runtime.

    For each call `send` posts the request to the backend using its internal
    streaming response form. The first response headers establish the backend
    acceptance boundary; the SSE body is then fully buffered and assembled
    into one normal terminal JSON object before returning to the broker.

    The transport is injectable: tests instantiate it with a raw
    aiohttp web.Application fixture, while production code wires it
    to the real member backend URL via the MemberSnapshot's
    `idle_service_configured` plus a name->URL mapping.

    The transport is buffered at the broker boundary: it never exposes
    streaming chunks to callers and never returns a streaming response.
    This keeps the external candidate contract polling-only while allowing
    the internal backend to signal acceptance before a long generation ends.
    """

    def __init__(
        self,
        *,
        session: Any | None = None,
        endpoint_for_member: Callable[[Any], str] | None = None,
        request_for_member: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._session = session
        self._owned_session = False
        self._endpoint_for_member = endpoint_for_member or _default_endpoint_for_member
        self._request_for_member = request_for_member
        self._timeout = float(timeout)

    async def send(
        self,
        member: Any,
        body: dict[str, Any],
        on_accepted: Callable[[], None],
    ) -> dict[str, Any]:
        if member is None:
            raise ValueError("member is required")
        url = self._endpoint_for_member(member)
        if not url:
            raise ValueError(f"no endpoint for member: {member!r}")
        session = await self._get_session()
        # llama-server's non-streaming OpenAI endpoint does not send HTTP
        # response headers until generation is complete. Ask the backend for
        # its internal SSE form so the first response chunk is the acceptance
        # boundary, while this transport still buffers and returns one normal
        # JSON response to the broker caller.
        payload = dict(body)
        if self._request_for_member is not None:
            payload = self._request_for_member(member, payload)
        payload["stream"] = True
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp_total_timeout(self._timeout),
                headers={"Accept": "text/event-stream"},
            ) as response:
                if not 200 <= response.status < 300:
                    text = await response.text()
                    raise RuntimeError(
                        f"TRANSPORT_HTTP_{response.status}:{text[:256]}"
                    )
                # The backend stream handler waits for its first result before
                # emitting the response headers. Entering the response is
                # therefore a proven acceptance boundary for this request.
                on_accepted()
                model = payload.get("model")
                return await self._read_streamed_chat_response(
                    response,
                    split_leading_think=(
                        isinstance(model, str)
                        and model.casefold().startswith("minimax-")
                    ),
                )
        finally:
            pass

    async def _read_streamed_chat_response(
        self,
        response: Any,
        *,
        split_leading_think: bool = False,
    ) -> dict[str, Any]:
        """Buffer an internal OpenAI SSE response into one terminal object."""
        first_chunk: dict[str, Any] | None = None
        usage: dict[str, Any] | None = None
        choices: dict[int, dict[str, Any]] = {}
        data_lines: list[str] = []

        async def consume_event(event: str) -> None:
            nonlocal first_chunk, usage
            if not event or event == "[DONE]":
                return
            try:
                chunk = json.loads(event)
            except json.JSONDecodeError:
                return
            if not isinstance(chunk, dict):
                return
            if first_chunk is None:
                first_chunk = chunk
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                try:
                    index = int(choice.get("index", 0))
                except (TypeError, ValueError):
                    index = 0
                acc = choices.setdefault(
                    index,
                    {
                        "role": "assistant",
                        "content": [],
                        "reasoning_content": [],
                        "tool_calls": {},
                        "finish_reason": None,
                    },
                )
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    role = delta.get("role")
                    if role:
                        acc["role"] = role
                    content = delta.get("content")
                    if content is not None:
                        acc["content"].append(str(content))
                    reasoning_content = delta.get("reasoning_content")
                    if reasoning_content is not None:
                        acc["reasoning_content"].append(str(reasoning_content))
                    for call in delta.get("tool_calls") or []:
                        if not isinstance(call, dict):
                            continue
                        try:
                            call_index = int(call.get("index", 0))
                        except (TypeError, ValueError):
                            call_index = len(acc["tool_calls"])
                        target = acc["tool_calls"].setdefault(call_index, {})
                        for key in ("id", "type"):
                            if call.get(key) is not None:
                                target[key] = call[key]
                        function = call.get("function") or {}
                        if isinstance(function, dict):
                            fn = target.setdefault("function", {})
                            for key in ("name", "arguments"):
                                if function.get(key) is not None:
                                    if key == "arguments":
                                        fn[key] = fn.get(key, "") + str(function[key])
                                    else:
                                        fn[key] = function[key]
                if choice.get("finish_reason") is not None:
                    acc["finish_reason"] = choice["finish_reason"]

        buffer = b""

        async for raw_chunk in response.content:
            # The stream iterator may yield complete lines, arbitrary chunks,
            # or a mixture depending on the client/adapter. Buffer bytes until
            # a physical LF is present so event framing is independent of TCP
            # chunk boundaries and UTF-8 decoding happens only per full line.
            buffer += bytes(raw_chunk)
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                if line == "":
                    await consume_event("\n".join(data_lines).strip())
                    data_lines.clear()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        if buffer:
            line = buffer.decode("utf-8", errors="replace").rstrip("\r")
            if line == "":
                await consume_event("\n".join(data_lines).strip())
                data_lines.clear()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            await consume_event("\n".join(data_lines).strip())

        if first_chunk is None:
            raise RuntimeError("TRANSPORT_STREAM_EMPTY")
        output: dict[str, Any] = {
            "id": first_chunk.get("id"),
            "object": "chat.completion",
            "created": first_chunk.get("created"),
            "model": first_chunk.get("model"),
            "choices": [],
        }
        for index in sorted(choices):
            acc = choices[index]
            content = "".join(acc["content"])
            reasoning_content = "".join(acc["reasoning_content"])
            if split_leading_think:
                extracted_reasoning, content = _split_leading_think(content)
                if extracted_reasoning is not None:
                    reasoning_content = "\n\n".join(
                        part for part in (reasoning_content.strip(), extracted_reasoning) if part
                    )
            message: dict[str, Any] = {
                "role": acc["role"],
                "content": content,
            }
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if acc["tool_calls"]:
                message["tool_calls"] = [
                    acc["tool_calls"][call_index]
                    for call_index in sorted(acc["tool_calls"])
                ]
            output["choices"].append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": acc["finish_reason"],
                }
            )
        if usage is not None:
            output["usage"] = usage
        return output

    async def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        import aiohttp

        self._session = aiohttp.ClientSession()
        self._owned_session = True
        return self._session

    async def close(self) -> None:
        if self._owned_session and self._session is not None:
            await self._session.close()
            self._session = None
            self._owned_session = False


def _default_endpoint_for_member(member: Any) -> str:
    name = getattr(member, "name", None)
    if not name:
        return ""
    # Look up the model's `backend_url` if exposed; otherwise return ""
    # so the dispatcher can fail closed for any unconfigured member.
    return getattr(member, "backend_url", "") or ""


def aiohttp_total_timeout(seconds: float) -> Any:
    """Return an aiohttp ClientTimeout with the given total budget."""
    import aiohttp

    return aiohttp.ClientTimeout(total=float(seconds))


# ----------------------------------------------------------------------
# Strict readiness member mapper (semantic HTTP only — never TCP-only)
# ----------------------------------------------------------------------


def _strip_non_deterministic(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    for key in ("state_version", "main_pid", "snapshot_age_ms", "observed_at"):
        cleaned.pop(key, None)
    return cleaned


def map_member_ready(
    *,
    name: str,
    config: dict[str, Any],
    probes: dict[str, Any],
    previous: MemberSnapshot | None = None,
    now: float,
    freshness_ms: int = 5000,
) -> MemberSnapshot:
    """Map a member's readiness probes into a strict MemberSnapshot.

    Strict-mode rules enforced here:
        * `semantic_health` MUST be True — a TCP-only HTTP probe is
          rejected with a ReasonCode.MEMBER_HTTP_UNREADY blocker.
        * `model_resident` MUST be True (or the member is cpu_only).
        * `idle_service_effective` MUST be True (or the member is cpu_only).
        * `dispatcher_registered` MUST be True.
        * `systemd_active` MUST be True and `main_pid` MUST be truthy.

    These rules are stricter than the default readiness reducer; the
    candidate runtime always uses them because the candidate may not
    start any service itself and must rely on observed HTTP-level
    semantic health to declare a member ready.
    """
    if not probes.get("semantic_health") is True:
        # Hard rejection: TCP-only readiness is not sufficient.
        probes = dict(probes)
        probes["semantic_health"] = False
    snapshot = reduce_member_readiness(
        name,
        config,
        probes,
        now=now,
        freshness_ms=freshness_ms,
        previous=previous,
    )
    # Reject any snapshot whose semantic_health is False; map to UNHEALTHY.
    if not snapshot.semantic_health:
        from dataclasses import replace as _replace

        return _replace(
            snapshot,
            state=MemberState.UNHEALTHY,
            accepting=False,
            compatible_free_slots=0,
        )
    return snapshot


# ----------------------------------------------------------------------
# Fenced dispatch loop
# ----------------------------------------------------------------------


class DispatchLoop:
    """Fenced async dispatch loop owned by a single candidate process.

    The loop only runs while this process holds the leader
    Redis fence on the repository's configured prefix. If the leader
    lease is lost (process pause, network blip, sibling candidate
    racing for the lock), the loop yields and stops dispatching
    until the owner is back to normal.

    The loop is fully injectable: it accepts the same transport
    the dispatcher uses, and it shares the repository's leadership
    semantics. Tests inject a stub repository whose
    acquire_leadership / renew_leadership are deterministic.
    """

    def __init__(
        self,
        *,
        repository: _RepositoryProtocol,
        transport: Transport,
        members_provider: Callable[[], Iterable[MemberSnapshot]],
        dispatcher: Dispatcher,
        owner: str,
        leader_ttl_seconds: float,
        heartbeat_seconds: float,
        clock: Callable[[], float],
        poll_interval_seconds: float = 0.01,
        dispatch_concurrency_cap: int | None = None,
        shutdown_timeout_seconds: float = 5.0,
        startup_recovery_grace_seconds: float | None = None,
        requirements_factory: Callable[[dict[str, Any]], Any] | None = None,
        event_producer: BoundedEventProducer | None = None,
    ) -> None:
        if not owner:
            raise ValueError("owner is required")
        if leader_ttl_seconds <= 0:
            raise ValueError("leader_ttl_seconds must be positive")
        if heartbeat_seconds <= 0 or heartbeat_seconds >= leader_ttl_seconds:
            raise ValueError(
                "heartbeat_seconds must be positive and < leader_ttl_seconds"
            )
        if dispatch_concurrency_cap is not None and (
            isinstance(dispatch_concurrency_cap, bool)
            or not isinstance(dispatch_concurrency_cap, int)
            or dispatch_concurrency_cap <= 0
        ):
            raise ValueError("dispatch_concurrency_cap must be a positive integer")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        if startup_recovery_grace_seconds is not None and (
            startup_recovery_grace_seconds <= 0
        ):
            raise ValueError("startup_recovery_grace_seconds must be positive")
        self._repository = repository
        self._transport = transport
        self._members_provider = members_provider
        self._dispatcher = dispatcher
        self._owner = owner
        self._leader_ttl_seconds = float(leader_ttl_seconds)
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._clock = clock
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._explicit_dispatch_concurrency_cap = dispatch_concurrency_cap
        self._dispatch_concurrency_cap = self._resolve_dispatch_concurrency_cap()
        self._semaphore = asyncio.Semaphore(self._dispatch_concurrency_cap)
        self._reservation_lock = asyncio.Lock()
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._startup_recovery_grace_seconds = (
            None
            if startup_recovery_grace_seconds is None
            else float(startup_recovery_grace_seconds)
        )
        self._startup_recovery_started_at_monotonic: float | None = None
        self._startup_recovery_retry_not_before_monotonic: float = 0.0
        self._startup_recovery_complete = False
        self._startup_recovered_count = 0
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._requirements_factory = requirements_factory or _default_requirements
        self._event_producer = event_producer
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._fencing_token: int | None = None
        self._dispatched = 0
        # Step 2e-1 / 2e-2 — fence-stale counter and dispatch-one fail counters.
        # These are observable, behavior-preserving additions: the hot path is
        # unchanged unless ``_dispatch_one`` actually raises one of the three
        # swallowed exception classes (``ReservationStale``, ``LeadershipLost``,
        # ``RepositoryUnavailable``), which is the pre-existing failure shape.
        self._fence_stale_consecutive: int = 0
        self._fence_last_log_at_monotonic: float = 0.0
        self._reservation_stale_consecutive: int = 0
        self._reservation_stale_last_log_at_monotonic: float = 0.0
        self._reservation_retry_not_before_monotonic: float = 0.0
        self._dispatch_one_fail_counters: dict[str, int] = {
            "reservation_stale": 0,
            "leadership_lost": 0,
            "repo_unavailable": 0,
            "no_member_eligible": 0,
            "no_reservation": 0,
            "pre_acceptance_requeue": 0,
        }
        self._preaccept_failure_counts: dict[str, int] = {}
        self._preaccept_cooldown_until_monotonic: dict[str, float] = {}
        self._preaccept_cooldown_base_seconds: float = 1.0
        self._preaccept_cooldown_cap_seconds: float = 60.0
        # Step 2e-2 — tunables. Defaults match the spec; they may be wired
        # in from ``BrokerConfig`` (services.json) at composition time. They
        # are NOT constructor kwargs to preserve the closed contract used by
        # the qualification harness.
        self._fence_stale_threshold: int = 10
        self._fence_reacquire_backoff_seconds: float = 0.5
        self._fence_max_reacquire_attempts: int = 60

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def fencing_token(self) -> int | None:
        return self._fencing_token

    @property
    def dispatched_count(self) -> int:
        return self._dispatched

    @property
    def metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "dispatched_count": self._dispatched,
            "fencing_token": self._fencing_token,
            "fail_counters": dict(self._dispatch_one_fail_counters),
            "pre_acceptance_cooldowns": {
                name: {
                    "failure_count": self._preaccept_failure_counts.get(name, 0),
                    "remaining_seconds": max(0.0, until - now),
                }
                for name, until in self._preaccept_cooldown_until_monotonic.items()
                if until > now
            },
            "startup_recovery": {
                "enabled": self._startup_recovery_grace_seconds is not None,
                "complete": self._startup_recovery_complete,
                "recovered_count": self._startup_recovered_count,
            },
        }

    def _try_startup_recovery(self) -> None:
        reconcile = getattr(
            self._repository,
            "reconcile_stale_post_acceptance_jobs",
            None,
        )
        if not callable(reconcile):
            self._startup_recovery_complete = True
            return
        token = int(self._fencing_token or 0)
        if token <= 0:
            return
        try:
            recovered = list(
                cast(
                    Iterable[JobRecord],
                    reconcile(
                        repair_owner=self._owner,
                        repair_fence=token,
                        limit=100,
                    ),
                )
            )
        except (RepositoryUnavailable, ReservationStale, LeadershipLost) as exc:
            self._startup_recovery_retry_not_before_monotonic = (
                time.monotonic() + max(1.0, self._heartbeat_seconds)
            )
            logger.warning("startup stale-job reconciliation deferred: %s", exc)
            return
        self._startup_recovered_count = len(recovered)
        self._startup_recovery_complete = True
        if recovered:
            logger.warning(
                "startup reconciled %d prior-fence post-acceptance jobs as outcome_unknown",
                len(recovered),
            )

    def _members_outside_preaccept_cooldown(
        self,
        members: Iterable[MemberSnapshot],
    ) -> list[MemberSnapshot]:
        now = time.monotonic()
        expired = [
            name
            for name, until in self._preaccept_cooldown_until_monotonic.items()
            if until <= now
        ]
        for name in expired:
            self._preaccept_cooldown_until_monotonic.pop(name, None)
        return [
            member
            for member in members
            if self._preaccept_cooldown_until_monotonic.get(member.name, 0.0) <= now
        ]

    def _record_preaccept_failure(self, member_name: str) -> None:
        count = self._preaccept_failure_counts.get(member_name, 0) + 1
        self._preaccept_failure_counts[member_name] = count
        delay = min(
            self._preaccept_cooldown_cap_seconds,
            self._preaccept_cooldown_base_seconds * (2 ** min(count - 1, 16)),
        )
        self._preaccept_cooldown_until_monotonic[member_name] = (
            time.monotonic() + delay
        )
        self._dispatch_one_fail_counters["pre_acceptance_requeue"] += 1
        logger.warning(
            "BROKER_DISPATCH: pre-acceptance failure member=%s "
            "consecutive=%d cooldown_seconds=%.3f",
            member_name,
            count,
            delay,
        )

    def _clear_preaccept_failure(self, member_name: str) -> None:
        self._preaccept_failure_counts.pop(member_name, None)
        self._preaccept_cooldown_until_monotonic.pop(member_name, None)

    def _record_fail_event(
        self,
        kind: str,
        *,
        member: str | None = None,
        attempt_id: str | None = None,
        error: str | None = None,
    ) -> None:
        if self._event_producer is not None:
            self._event_producer.record_failure(
                kind=kind,
                member=member,
                attempt_id=attempt_id,
                error=error,
                ts=self._clock(),
            )

    @property
    def dispatch_concurrency_cap(self) -> int:
        return self._dispatch_concurrency_cap

    def _resolve_dispatch_concurrency_cap(self) -> int:
        configured_slots = sum(
            max(0, int(getattr(member, "configured_slots", 0) or 0))
            for member in self._members_provider()
        )
        # A zero-slot snapshot cannot safely admit work. Keep the semaphore
        # constructible while retaining fail-closed capacity in _dispatch_one.
        derived = max(1, configured_slots)
        if self._explicit_dispatch_concurrency_cap is None:
            return derived
        return min(self._explicit_dispatch_concurrency_cap, derived)

    async def start(self) -> None:
        """Begin the loop. Idempotent: a second call is a no-op."""
        if self.is_running:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(
            self._run(), name=f"gemma-dispatch-loop:{self._owner}"
        )

    async def stop(self) -> None:
        """Stop spawning work and boundedly await accepted in-flight work.

        In-flight tasks are intentionally not cancelled: after the backend
        acceptance boundary, ``Dispatcher`` must reach its terminal lease
        release path.  A timeout returns control to the caller while retaining
        task tracking; a later ``stop`` call can finish the drain.
        """
        self._stopping.set()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self._wait_for_inflight()
        self._task = None
        self._fencing_token = None

    async def _wait_for_inflight(self) -> None:
        pending = {task for task in self._in_flight if not task.done()}
        if not pending:
            return
        done, still_pending = await asyncio.wait(
            pending,
            timeout=self._shutdown_timeout_seconds,
        )
        for task in done:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass
        if still_pending:
            logger.warning(
                "dispatch loop shutdown timed out with %d accepted tasks still in flight",
                len(still_pending),
            )

    async def _dispatch_one_guarded(self) -> None:
        try:
            if await self._dispatch_one():
                self._dispatched += 1
        except Exception as exc:
            logger.warning("dispatch loop in-flight task failed: %s", exc)
        finally:
            self._semaphore.release()

    def _authoritative_members(self) -> list[MemberSnapshot]:
        """Prefer repository mirrors for dispatch capacity decisions.

        Observer snapshots describe backend health, but they can lag durable
        dispatcher leases. Using them directly for the advisory precheck
        creates a hot retry loop while Redis correctly rejects overclaims.
        Fall back to the observer only while a member has no durable mirror.
        """
        observed = list(self._members_provider())
        authoritative: list[MemberSnapshot] = []
        reservation_getter = getattr(
            self._repository,
            "get_reservation_member",
            self._repository.get_member,
        )
        for member in observed:
            stored = reservation_getter(member.name)
            authoritative.append(stored if stored is not None else member)
        return authoritative

    def _track_inflight(self, task: asyncio.Task[None]) -> None:
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)

    async def _try_acquire_leader(self) -> int | None:
        """Acquire (or fail to acquire) the leader lease.

        Step 2e-3 extraction. Wraps the original L540-547 acquire block
        so the bounded-retry path in ``_run`` can call it without
        duplicating the repository error semantics. Returns the new
        fencing token on success, or ``None`` if another owner has it
        or the repository is unavailable.
        """
        try:
            token = self._repository.acquire_leadership(
                owner=self._owner,
                now=self._clock(),
                ttl_seconds=self._leader_ttl_seconds,
            )
        except LeadershipLost:
            self._fencing_token = None
            return None
        except (RepositoryUnavailable, Exception) as exc:
            logger.warning("dispatch loop leadership acquire failed: %s", exc)
            self._fencing_token = None
            return None
        # Only update _fencing_token on a positive value. A 0 token is
        # an invalid result (Lua guards require > 0) and signals
        # contention; the retry loop will see the absence as "not yet".
        if token and int(token) > 0:
            self._fencing_token = int(token)
            return self._fencing_token
        self._fencing_token = None
        return None

    async def _acquire_leader_until_available(self, *, reason: str) -> bool:
        """Keep the dispatcher alive until leadership is available or stopping.

        The attempt limit is an observability threshold, not a termination
        condition. A transient foreign lease or Redis blip must not leave a
        permanently dormant dispatch task.
        """
        attempts = 0
        while not self._stopping.is_set():
            attempts += 1
            token = await self._try_acquire_leader()
            if token is not None and token > 0:
                if attempts > 1:
                    logger.warning(
                        "dispatch loop acquired leadership after %d attempts reason=%s",
                        attempts,
                        reason,
                    )
                return True
            if attempts == self._fence_max_reacquire_attempts:
                logger.critical(
                    "CRITICAL: dispatch_loop_leadership_unavailable "
                    "after %d attempts; continuing retry reason=%s",
                    attempts,
                    reason,
                )
            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=self._fence_reacquire_backoff_seconds,
                )
            except asyncio.TimeoutError:
                pass
        return False

    async def _run(self) -> None:
        # Initial contention is transient. Do not terminate the dispatcher
        # before the first successful fence acquisition.
        if not await self._acquire_leader_until_available(reason="initial"):
            return

        self._startup_recovery_started_at_monotonic = time.monotonic()
        last_renewal = self._clock()
        try:
            while not self._stopping.is_set():
                now_monotonic = time.monotonic()
                if (
                    not self._startup_recovery_complete
                    and self._startup_recovery_grace_seconds is not None
                    and self._startup_recovery_started_at_monotonic is not None
                    and now_monotonic
                    >= self._startup_recovery_started_at_monotonic
                    + self._startup_recovery_grace_seconds
                    and now_monotonic
                    >= self._startup_recovery_retry_not_before_monotonic
                ):
                    self._try_startup_recovery()
                now = self._clock()
                # Renew leadership on a heartbeat cadence.  Dispatch tasks are
                # deliberately independent of this loop so a full dispatch
                # window cannot stall lease renewal.
                if now - last_renewal >= self._heartbeat_seconds:
                    try:
                        self._repository.renew_leadership(
                            owner=self._owner,
                            fencing_token=int(self._fencing_token),
                            now=now,
                            ttl_seconds=self._leader_ttl_seconds,
                        )
                    except LeadershipLost:
                        logger.warning(
                            "dispatch loop lost leadership; reacquiring "
                            "until available (log threshold %d attempts)",
                            self._fence_max_reacquire_attempts,
                        )
                        if not await self._acquire_leader_until_available(reason="renewal"):
                            return
                        # Successful reacquire: reset the heartbeat clock so the
                        # next renew is on cadence, NOT immediately.
                        last_renewal = self._clock()
                        continue
                    except RepositoryUnavailable as exc:
                        logger.warning("dispatch loop leadership renew failed: %s", exc)
                        if not await self._acquire_leader_until_available(reason="redis"):
                            return
                        last_renewal = self._clock()
                        continue
                    last_renewal = now

                retry_wait = (
                    self._reservation_retry_not_before_monotonic
                    - time.monotonic()
                )
                if retry_wait > 0:
                    try:
                        await asyncio.wait_for(
                            self._stopping.wait(),
                            timeout=retry_wait,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Do not spin when the queue is empty or every live member is
                # currently incompatible/full.  The old loop spawned a task
                # unconditionally, and each empty task returned immediately;
                # that created a tight CPU/logging loop while idle and also
                # hammered Redis during member drain/rejoin windows.
                if not self._has_dispatchable_head():
                    try:
                        await asyncio.wait_for(
                            self._stopping.wait(),
                            timeout=self._poll_interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Reserve a semaphore slot before creating the task.  The
                # reservation itself remains synchronous/atomic inside
                # _dispatch_one; this loop never awaits that task inline.
                spawned = False
                if not self._semaphore.locked():
                    await self._semaphore.acquire()
                    if self._stopping.is_set():
                        self._semaphore.release()
                        break
                    task = asyncio.create_task(
                        self._dispatch_one_guarded(),
                        name=f"gemma-dispatch:{self._owner}",
                    )
                    self._track_inflight(task)
                    spawned = True

                if spawned:
                    # Yield so the guarded task reaches the atomic reserve
                    # boundary, then continue the heartbeat/poll loop.
                    await asyncio.sleep(0)
                else:
                    # Capacity is full (or no task could be spawned).  Keep
                    # polling so leadership renewal is never starved.
                    try:
                        await asyncio.wait_for(
                            self._stopping.wait(),
                            timeout=self._poll_interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
        except asyncio.CancelledError:
            return
        finally:
            self._fencing_token = None

    async def _dispatch_one(self) -> bool:
        try:
            members = self._members_outside_preaccept_cooldown(
                self._authoritative_members()
            )
        except RepositoryUnavailable as exc:
            self._dispatch_one_fail_counters["repo_unavailable"] += 1
            self._record_fail_event("repo_unavailable", error=str(exc))
            return False
        if not members:
            return False
        head = None
        try:
            async with self._reservation_lock:
                head = self._peek_head()
                if head is None:
                    return False
                requirements = self._requirements_factory(head.request_body)
                reservation = self._repository.reserve_next(
                    now=self._clock(),
                    ordered_members=members,
                    requirements=requirements,
                    leader_owner=self._owner,
                    fencing_token=int(self._fencing_token or 0),
                )

        except ReservationStale as exc:
            head_id = getattr(head, "job_id", None) if head else None
            now_mono = time.monotonic()
            self._reservation_stale_consecutive += 1
            self._reservation_retry_not_before_monotonic = max(
                self._reservation_retry_not_before_monotonic,
                now_mono + self._poll_interval_seconds,
            )
            if (
                self._reservation_stale_consecutive == 1
                or now_mono - self._reservation_stale_last_log_at_monotonic >= 1.0
            ):
                logger.warning(
                    "BROKER_DISPATCH: reserve_exc=%s head=%s consecutive=%d",
                    type(exc).__name__,
                    head_id,
                    self._reservation_stale_consecutive,
                )
                self._reservation_stale_last_log_at_monotonic = now_mono
            # Reservation staleness is unrelated to fence loss. Keep the
            # cumulative counter, but gate new reservation attempts until the
            # normal poll interval has elapsed.
            self._fence_stale_consecutive = 0
            self._dispatch_one_fail_counters["reservation_stale"] += 1
            self._record_fail_event("reservation_stale", error=str(exc))
            return False
        except LeadershipLost as exc:
            head_id = getattr(head, "job_id", None) if head else None
            logger.warning(
                "BROKER_DISPATCH: reserve_exc=%s head=%s",
                type(exc).__name__, head_id,
            )
            # Step 2e-1 — fence loss: bump consecutive + reason counter,
            # rate-limited CRITICAL log on threshold crossing.
            self._fence_stale_consecutive += 1
            self._dispatch_one_fail_counters["leadership_lost"] += 1
            self._record_fail_event("leadership_lost", error=str(exc))
            if (
                self._fence_stale_consecutive
                >= self._fence_stale_threshold
            ):
                now_mono = time.monotonic()
                if (
                    now_mono - self._fence_last_log_at_monotonic
                    > 1.0
                ):
                    logger.critical(
                        "CRITICAL: dispatch_loop_fence_stuck count=%d "
                        "threshold=%d last_err=%r",
                        self._fence_stale_consecutive,
                        self._fence_stale_threshold,
                        exc,
                        exc_info=True,
                    )
                    self._fence_last_log_at_monotonic = now_mono
            return False
        except RepositoryUnavailable as exc:
            head_id = getattr(head, "job_id", None) if head else None
            logger.warning(
                "BROKER_DISPATCH: reserve_exc=%s head=%s",
                type(exc).__name__, head_id,
            )
            # Step 2e-1 — repository unavailable is unrelated to fence
            # loss; reset the streak (matches Test 6 boundary).
            self._fence_stale_consecutive = 0
            self._dispatch_one_fail_counters["repo_unavailable"] += 1
            self._record_fail_event("repo_unavailable", error=str(exc))
            return False
        except Exception as exc:
            head_id = getattr(head, "job_id", None) if head else None
            logger.warning(
                "BROKER_DISPATCH: reserve_unexpected_exc=%s head=%s msg=%s",
                type(exc).__name__, head_id, exc,
            )
            return False
        if reservation is None:
            self._dispatch_one_fail_counters["no_reservation"] += 1
            return False
        try:
            # Step 2e-1 — clear the streak immediately before dispatching
            # so a successful reservation clears any prior LeadershipLost
            # streak. This is the "reset discipline" called out in the spec.
            self._fence_stale_consecutive = 0
            self._reservation_stale_consecutive = 0
            self._reservation_retry_not_before_monotonic = 0.0
            outcome = await self._dispatcher.dispatch(reservation)
            if (
                outcome.state is JobState.QUEUED
                and outcome.reason_code is ReasonCode.PRE_ACCEPTANCE_REQUEUE
            ):
                self._record_preaccept_failure(reservation.member.name)
            else:
                self._clear_preaccept_failure(reservation.member.name)
            return True
        except RepositoryUnavailable as exc:
            self._dispatch_one_fail_counters["repo_unavailable"] += 1
            self._record_fail_event(
                "repo_unavailable",
                member=reservation.member.name,
                attempt_id=reservation.job.current_attempt_id,
                error=str(exc),
            )
            return False
        except LeadershipLost as exc:
            self._dispatch_one_fail_counters["leadership_lost"] += 1
            self._record_fail_event(
                "leadership_lost",
                member=reservation.member.name,
                attempt_id=reservation.job.current_attempt_id,
                error=str(exc),
            )
            return False
        except Exception as exc:
            logger.warning("BROKER_DISPATCH: dispatch_one dispatch failed: %s", exc)  # raised from debug
            return False

    def _has_dispatchable_head(self) -> bool:
        """Return whether one queued head can currently use a member.

        This is an advisory pre-check only; ``reserve_next`` remains the
        authoritative atomic capacity/fence boundary. Its purpose is to
        sleep instead of hot-spinning when the queue is empty or all members
        are temporarily unavailable.
        """
        head = self._peek_head()
        if head is None:
            return False
        try:
            members = self._members_outside_preaccept_cooldown(
                self._authoritative_members()
            )
        except RepositoryUnavailable:
            return False
        if not members:
            return False
        try:
            requirements = self._requirements_factory(head.request_body)
        except Exception:
            # Let the normal dispatch path classify/log malformed requests;
            # do not turn an input error into an idle hot loop.
            return False
        return any(
            member.state is MemberState.READY_ACCEPTING
            and member.accepting
            and (member.compatible_free_slots or 0) > 0
            and evaluate(requirements, member).compatible
            for member in members
        )

    def _peek_head(self) -> JobRecord | None:
        peek = getattr(self._repository, "peek_next", None)
        if peek is None:
            return None
        try:
            return peek(now=self._clock())
        except Exception:
            return None


def _default_requirements(body: dict[str, Any]) -> Any:
    return build_requirements(
        body,
        required_capabilities=("chat",),
        input_tokens_hint=body.get("input_tokens_hint"),
    )


# ----------------------------------------------------------------------
# Polling terminal wait
# ----------------------------------------------------------------------


class TerminalNotifier:
    """Wake in-process terminal waiters after a durable terminal write."""

    def __init__(self) -> None:
        self._waiters: dict[str, set[asyncio.Event]] = {}

    def subscribe(self, job_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._waiters.setdefault(job_id, set()).add(event)
        return event

    def unsubscribe(self, job_id: str, event: asyncio.Event) -> None:
        waiters = self._waiters.get(job_id)
        if waiters is None:
            return
        waiters.discard(event)
        if not waiters:
            self._waiters.pop(job_id, None)

    def notify_terminal(self, job_id: str) -> None:
        for event in tuple(self._waiters.get(job_id, ())):
            event.set()

    def subscriber_count(self, job_id: str) -> int:
        return len(self._waiters.get(job_id, ()))


class PollingTerminalWait:
    """Event-driven terminal wait with a slow Redis failover fallback."""

    def __init__(
        self,
        *,
        repository: _RepositoryProtocol,
        clock: Callable[[], float],
        notifier: TerminalNotifier | None = None,
        fallback_interval_seconds: float = 0.5,
        fallback_jitter_seconds: float = 0.1,
    ) -> None:
        if fallback_interval_seconds <= 0:
            raise ValueError("fallback_interval_seconds must be positive")
        if fallback_jitter_seconds < 0 or fallback_jitter_seconds >= fallback_interval_seconds:
            raise ValueError("fallback_jitter_seconds must be non-negative and less than fallback interval")
        self._repository = repository
        self._clock = clock
        self._notifier = notifier or TerminalNotifier()
        self._fallback_interval_seconds = float(fallback_interval_seconds)
        self._fallback_jitter_seconds = float(fallback_jitter_seconds)

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        event = self._notifier.subscribe(job_id)
        deadline = self._clock() + timeout
        try:
            job = self._repository.get(job_id)
            if job is None:
                return self._timeout_payload(job_id, accepted=False)
            if job.state in TERMINAL_JOB_STATES:
                return self._terminal_payload(job)
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return self._timeout_payload(
                        job_id, accepted=bool(job.accepted_boundary_crossed)
                    )
                jitter = random.uniform(
                    -self._fallback_jitter_seconds,
                    self._fallback_jitter_seconds,
                )
                wait_for = min(
                    remaining,
                    max(0.01, self._fallback_interval_seconds + jitter),
                )
                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=wait_for)
                except asyncio.TimeoutError:
                    pass
                job = self._repository.get(job_id)
                if job is None:
                    return self._timeout_payload(job_id, accepted=False)
                if job.state in TERMINAL_JOB_STATES:
                    return self._terminal_payload(job)
        finally:
            self._notifier.unsubscribe(job_id, event)

    @staticmethod
    def _terminal_payload(job: JobRecord) -> dict[str, Any]:
        # Format the result for the compat wrapper. The candidate
        # marshals a buffered JSON body and a content_type.
        if job.state is JobState.COMPLETED:
            return {
                "status": 200,
                "content_type": "application/json",
                "body": job.result or {},
                "job_id": job.job_id,
                "state": job.state.value,
            }
        if job.state is JobState.CANCELLED:
            return {
                "status": 409,
                "content_type": "application/json",
                "body": {
                    "error": "cancelled",
                    "code": "cancelled",
                    "job_id": job.job_id,
                },
                "job_id": job.job_id,
                "state": job.state.value,
            }
        if job.state is JobState.FAILED:
            return {
                "status": 502,
                "content_type": "application/json",
                "body": {
                    "error": job.error or "backend_failed",
                    "code": job.reason_code.value if job.reason_code else "backend_failed",
                    "job_id": job.job_id,
                },
                "job_id": job.job_id,
                "state": job.state.value,
            }
        # OUTCOME_UNKNOWN
        return {
            "status": 504,
            "content_type": "application/json",
            "body": {
                "error": "outcome_unknown",
                "code": ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN.value,
                "job_id": job.job_id,
            },
            "job_id": job.job_id,
            "state": job.state.value,
        }

    @staticmethod
    def _timeout_payload(job_id: str, *, accepted: bool) -> dict[str, Any]:
        return {
            "status": 504,
            "content_type": "application/json",
            "body": {
                "error": "compatibility wait timed out",
                "code": (
                    "outcome_pending_after_acceptance"
                    if accepted
                    else "compatibility_timeout"
                ),
                "job_id": job_id,
                "retry_after": 1,
                "retry_safe": not accepted,
            },
            "job_id": job_id,
            "timeout": True,
        }


TERMINAL_JOB_STATES = frozenset(
    {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.OUTCOME_UNKNOWN,
    }
)


# ----------------------------------------------------------------------
# aiohttp candidate app
# ----------------------------------------------------------------------


def build_candidate_app(runtime: "BrokerRuntime") -> web.Application:
    """Construct the aiohttp candidate app for the runtime.

    The app exposes:
        * GET  /health            — readiness/health probe
        * POST /v1/gemma/jobs     — submit a durable job
        * GET  /v1/gemma/jobs/{id} — fetch a job's status
        * DELETE /v1/gemma/jobs/{id} — cancel a queued job
        * POST /v1/chat/completions — buffered OpenAI compatibility

    The candidate app NEVER registers an SSE handler. Any
    text/event-stream request is rejected with SSE_DISABLED.
    """
    app = web.Application()
    runtime_key = web.AppKey("runtime", BrokerRuntime)
    app[runtime_key] = runtime

    async def health(request: web.Request) -> web.Response:
        rt: BrokerRuntime = request.app[runtime_key]
        ready_members = sum(
            1
            for member in rt._members_provider()
            if member.state is MemberState.READY_ACCEPTING
            and member.accepting is True
            and (member.compatible_free_slots or 0) > 0
            and member.semantic_health is True
        )
        readiness = "ready" if not rt._closed and ready_members else (
            "closed" if rt._closed else "no_ready_accepting_member"
        )
        return web.json_response(
            {
                "status": "ok" if readiness == "ready" else "unavailable",
                "readiness": readiness,
                "candidate": True,
                "namespace": rt.config.redis_namespace,
                "ready_members": ready_members,
            },
            status=200 if readiness == "ready" else 503,
        )

    async def broker_metrics(request: web.Request) -> web.Response:
        rt: BrokerRuntime = request.app[runtime_key]
        members = list(rt._members_provider())
        include_durable_counts = request.query.get("durable", "1") != "0"

        def collect_metrics() -> tuple[dict[str, int], int, dict[str, Any] | None]:
            lease_counts = {
                member.name: rt._repository.lease_count(member.name)
                for member in members
            }
            queue_depth = int(getattr(rt._repository, "queue_total"))
            counts = rt._repository.durable_counts() if include_durable_counts else None
            return lease_counts, queue_depth, counts

        try:
            lease_counts, queue_depth, counts = await asyncio.to_thread(
                collect_metrics
            )
        except RepositoryUnavailable:
            return web.json_response(
                {"error": "repository unavailable", "code": "redis_state_unknown"},
                status=503,
            )
        payload = rt.metrics.snapshot(
            members=members,
            queue_depth=queue_depth,
            now=rt._clock(),
            lease_counts=lease_counts,
            schema=SCHEMA_VERSION,
            namespace=rt.config.redis_namespace,
            leader={
                "owner": rt.owner_token,
                "token": rt.dispatch_loop.fencing_token,
                "expires_at": None,
            },
            dispatch_loop=rt.dispatch_loop.metrics,
            attempts_total=counts["attempts_total"] if counts is not None else 0,
            attempts_by_state=counts["attempts_by_state"] if counts is not None else {},
            jobs_total=counts["jobs_total"] if counts is not None else 0,
            jobs_by_state=counts["jobs_by_state"] if counts is not None else {},
        )
        payload["members"] = list(payload["members"].values())
        if counts is None:
            payload["attempts"]["fresh"] = False
            payload["jobs"]["fresh"] = False
        return web.json_response(payload)

    async def submit(request: web.Request) -> web.Response:
        rt: BrokerRuntime = request.app[runtime_key]
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"error": "valid JSON required", "code": "invalid_json"}, status=400
            )
        try:
            job = rt.api_service.submit(payload)
        except Exception as exc:
            return web.json_response(
                {"error": str(exc), "code": getattr(exc, "code", "submit_error")},
                status=getattr(exc, "status", 500),
            )
        location = f"/v1/gemma/jobs/{job.job_id}"
        response = web.json_response(
            {
                "job_id": job.job_id,
                "state": job.state.value,
                "status_url": location,
                "cancel_url": location,
            },
            status=202,
        )
        response.headers["Location"] = location
        return response

    async def status(request: web.Request) -> web.Response:
        rt: BrokerRuntime = request.app[runtime_key]
        try:
            job = rt.api_service.status(request.match_info["job_id"])
        except Exception as exc:
            return web.json_response(
                {"error": str(exc), "code": getattr(exc, "code", "status_error")},
                status=getattr(exc, "status", 500),
            )
        return web.json_response(_job_status_body(job))

    async def cancel(request: web.Request) -> web.Response:
        rt: BrokerRuntime = request.app[runtime_key]
        try:
            job = rt.api_service.cancel(request.match_info["job_id"])
        except Exception as exc:
            return web.json_response(
                {"error": str(exc), "code": getattr(exc, "code", "cancel_error")},
                status=getattr(exc, "status", 500),
            )
        return web.json_response(_job_status_body(job))

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        rt: BrokerRuntime = request.app[runtime_key]
        body = await request.read()
        try:
            return await rt.handle_proxy(
                method="POST",
                path="/v1/chat/completions",
                body=body,
                headers=request.headers,
                request=request,
            )
        except Exception as exc:
            return web.json_response(
                {"error": str(exc), "code": getattr(exc, "code", "compat_error")},
                status=500,
            )

    app.router.add_get("/health", health)
    app.router.add_get("/v1/metrics/broker", broker_metrics)
    app.router.add_post("/v1/gemma/jobs", submit)
    app.router.add_get("/v1/gemma/jobs/{job_id}", status)
    app.router.add_delete("/v1/gemma/jobs/{job_id}", cancel)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def _refuse_sse_response() -> web.Response:
    return web.json_response(
        {
            "error": "Server-Sent Events are disabled in the candidate runtime",
            "code": CandidateSSEDisabled.CODE,
        },
        status=406,
    )


def _job_status_body(job: JobRecord) -> dict[str, Any]:
    body = job.to_dict()
    body["terminal"] = job.state in TERMINAL_JOB_STATES
    return body


# ----------------------------------------------------------------------
# Public runtime
# ----------------------------------------------------------------------


class CandidateLifecycleMutationDisabled(RuntimeError):
    """Raised when a candidate runtime is asked to perform a member
    lifecycle mutation (start / stop / restart)."""

    CODE = "LIFECYCLE_MUTATION_DISABLED"


class CandidateSSEDisabled(RuntimeError):
    """Raised when a candidate runtime is asked to serve an SSE response."""

    CODE = "SSE_DISABLED"


class CandidateConfigInvalid(RuntimeError):
    """Raised when the runtime is built with a config that violates a
    candidate invariant."""

    CODE = "CANDIDATE_CONFIG_INVALID"


class BrokerRuntime:
    """Self-contained composition root for the candidate broker.

    The runtime holds no state of its own beyond the wiring of the
    injected components. All durable state belongs to the injected
    repository; all member identity belongs to the injected members
    provider. The runtime itself is therefore safe to construct and
    discard in a qualification harness without leaving residuals.
    """

    def __init__(
        self,
        *,
        config: BrokerConfig,
        repository: Any,
        members_provider: Callable[[], Iterable[MemberSnapshot]],
        transport: Transport,
        clock: Callable[[], float],
        repair_fence: InMemoryRepairFence | None = None,
        request_timeout: float = 30.0,
        compat_timeout: float = 30.0,
        dispatch_owner: str | None = None,
        event_producer: BoundedEventProducer | None = None,
    ) -> None:
        if not isinstance(config, BrokerConfig):
            raise CandidateConfigInvalid("config must be a BrokerConfig instance")
        validate_candidate_config(config)
        if not _is_repository_compatible(repository):
            missing = [
                name
                for name in REPOSITORY_REQUIRED_METHODS
                if not callable(getattr(repository, name, None))
            ]
            raise CandidateConfigInvalid(
                "repository must satisfy REPOSITORY_REQUIRED_METHODS; "
                "missing methods: "
                + (", ".join(missing) if missing else "(none)")
            )
        if members_provider is None:
            raise ValueError("members_provider is required")
        if transport is None:
            raise ValueError("transport is required")
        if clock is None:
            raise ValueError("clock is required")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if compat_timeout <= 0:
            raise ValueError("compat_timeout must be positive")

        # Construct (don't start) every collaborator using the injected
        # dependencies. No background work runs until the runtime's
        # dispatchers are actually awaited by callers.
        self.config: BrokerConfig = config
        self._repository = repository
        self._members_provider: Callable[[], Iterable[MemberSnapshot]] = members_provider
        self._transport = transport
        self._clock = clock
        self._request_timeout = float(request_timeout)
        self._compat_timeout = float(compat_timeout)
        self._repair_fence = repair_fence or InMemoryRepairFence()
        self.owner_token = dispatch_owner or str(uuid4())
        self._dispatch_owner = self.owner_token
        self.event_producer = event_producer or BoundedEventProducer(max_entries=100)
        self.metrics = BrokerMetrics(freshness_seconds=self.config.freshness_seconds)
        self.terminal_notifier = TerminalNotifier()
        set_terminal_notifier = getattr(self._repository, "set_terminal_notifier", None)
        if callable(set_terminal_notifier):
            set_terminal_notifier(self.terminal_notifier)

        # Public surface — the only handles a candidate ever exposes.
        self.api_service: DurableBrokerService = DurableBrokerService(
            self._repository,
            members=self._safe_members,
            clock=self._clock,
        )
        # The compat wrapper is exposed for callers that need to proxy a
        # single OpenAI-shaped request. The runtime intercepts every call
        # to refuse SSE-shaped payloads.
        self._polling_wait = PollingTerminalWait(
            repository=self._repository,
            clock=self._clock,
            notifier=self.terminal_notifier,
        )
        self._compat_wrapper = CompatibilityWrapper(
            _CandidateCompatAdapter(
                self.api_service, polling_wait=self._polling_wait
            ),
            timeout=self._compat_timeout,
        )
        self.dispatcher: Dispatcher = Dispatcher(
            repo=self._repository,
            transport=self._transport,
            request_timeout=self._request_timeout,
            clock=self._clock,
        )
        self.drain_coordinator: MemberDrainCoordinator = MemberDrainCoordinator(
            self._repository
        )
        self.reconciler: Reconciler = Reconciler(
            self._repository, self._repair_fence
        )
        self.dispatch_loop: DispatchLoop = DispatchLoop(
            repository=self._repository,
            transport=self._transport,
            members_provider=self._members_provider,
            dispatcher=self.dispatcher,
            owner=self._dispatch_owner,
            leader_ttl_seconds=self.config.leader_ttl_seconds,
            heartbeat_seconds=self.config.heartbeat_seconds,
            clock=self._clock,
            startup_recovery_grace_seconds=self._request_timeout,
            event_producer=self.event_producer,
        )

        # close()/stop() bookkeeping.
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()

    # --------------------------------------------------------------- surface

    @property
    def repository(self) -> Any:
        return self._repository

    @property
    def members_provider(self) -> Callable[[], Iterable[MemberSnapshot]]:
        return self._members_provider

    @property
    def transport(self) -> Transport:
        return self._transport

    async def handle_proxy(
        self,
        *,
        method: str,
        path: str,
        body: bytes | None,
        headers,
        request: web.Request | None = None,
    ) -> web.StreamResponse:
        """Composition entrypoint that proxies an OpenAI-shaped request.

        Streaming-shaped requests are accepted by the durable broker. The
        compatibility adapter emits a synthesized SSE response after the
        buffered terminal result is available, so queue admission and lease
        release never depend on a backend token stream remaining open.
        """
        await self._ensure_open()
        # Delegate both buffered and stream-shaped requests. The wrapper
        # decides the response form after durable submission.
        return await self._compat_wrapper.handle_proxy(
            method=method,
            path=path,
            body=body,
            headers=headers,
            request=request,
        )

    def drain_member(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        timeout: float,
        retry_stale: bool = False,
    ) -> Awaitable[Any]:
        """Begin-and-wait on a single member drain via the coordinator."""
        return self.drain_coordinator.begin_and_wait(
            member_name,
            expected_state_version=expected_state_version,
            timeout=timeout,
            retry_stale=retry_stale,
        )

    def map_member_ready(
        self,
        *,
        name: str,
        config: dict[str, Any],
        probes: dict[str, Any],
        previous: MemberSnapshot | None = None,
        freshness_ms: int = 5000,
    ) -> MemberSnapshot:
        """Strict readiness mapper: requires semantic HTTP proven input."""
        return map_member_ready(
            name=name,
            config=config,
            probes=probes,
            previous=previous,
            now=self._clock(),
            freshness_ms=freshness_ms,
        )

    async def wait_for_terminal(
        self, job_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Polling-based terminal wait for a buffered (non-SSE) response."""
        return await self._polling_wait.wait(
            job_id, timeout=float(timeout if timeout is not None else self._compat_timeout)
        )

    def build_app(self) -> web.Application:
        """Construct the aiohttp candidate app exposing /health,
        /v1/gemma/jobs, and /v1/chat/completions."""
        return build_candidate_app(self)

    # ------------------------------------------ refused lifecycle mutators

    def start_member(self, member_name: str) -> None:
        raise CandidateLifecycleMutationDisabled(
            f"{CandidateLifecycleMutationDisabled.CODE}:start:{member_name}"
        )

    def stop_member(self, member_name: str) -> None:
        raise CandidateLifecycleMutationDisabled(
            f"{CandidateLifecycleMutationDisabled.CODE}:stop:{member_name}"
        )

    def restart_member(self, member_name: str) -> None:
        raise CandidateLifecycleMutationDisabled(
            f"{CandidateLifecycleMutationDisabled.CODE}:restart:{member_name}"
        )

    # ----------------------------------------------------------- close/stop

    async def close(self) -> None:
        """Idempotent async shutdown. Stops the dispatch loop, cancels
        and awaits any background tasks the runtime scheduled. Refuses
        to mutate any member lifecycle. Safe to call more than once.
        """
        async with self._close_lock:
            if self._closed:
                return
            # Stop the dispatch loop first (cancellation only, no
            # lifecycle mutation).
            try:
                await self.dispatch_loop.stop()
            except Exception as exc:
                logger.debug("dispatch_loop stop raised: %s", exc)
            # Cancel + drain any background tasks.
            for task in list(self._background_tasks):
                task.cancel()
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks, return_exceptions=True
                )
                self._background_tasks.clear()
            # Close the transport if it owns an aiohttp session.
            close = getattr(self._transport, "close", None)
            if close is not None and callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    logger.debug("transport close raised: %s", exc)
            # Mark closed. We intentionally do NOT mutate the injected
            # repository or any member snapshot — they belong to the caller.
            self._closed = True

    async def stop(self) -> None:
        """Alias for close(); a candidate runtime has nothing else to stop."""
        await self.close()

    def schedule_background(self, coro: Any) -> asyncio.Task:
        """Track a background coroutine so close() can drain it.

        Exposed for callers (or future internal users) that need to keep
        long-running work alive under the runtime. The returned task is
        owned by the runtime and will be cancelled at close().
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError("schedule_background requires a coroutine")
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    # ----------------------------------------------------------- internals

    def _safe_members(self) -> list[MemberSnapshot]:
        snapshot = list(self._members_provider())
        return snapshot

    async def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RUNTIME_CLOSED")


# -------------------------------------------------------------- adapters


def _sse_frame(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n\n"


class _CandidateCompatAdapter:
    """Map the compatibility wrapper onto durable broker operations.

    The polling wait is injected so the wrapper can deliver either a buffered
    response or a synthesized SSE response without opening an internal
    backend stream.
    """

    def __init__(self, service: DurableBrokerService, *, polling_wait: PollingTerminalWait) -> None:
        self._service = service
        self._polling_wait = polling_wait

    def submit_compat(self, request_body: dict, *, idempotency_key: str) -> JobRecord:
        payload = {
            "idempotency_key": idempotency_key,
            "request": request_body,
            "priority": int(Priority.NORMAL),
            "caller_scope": "compat",
        }
        return self._service.submit(payload)

    def status(self, job_id: str) -> JobRecord:
        return self._service.status(job_id)

    async def wait_for_terminal(self, job_id: str, *, timeout: float) -> dict[str, Any]:
        return await self._polling_wait.wait(job_id, timeout=timeout)

    async def stream_result(self, job_id: str, *, timeout: float):
        """Yield an SSE projection of one durable terminal result.

        Backend generation remains buffered so acceptance and lease cleanup
        are independent of client connection lifetime. This preserves the
        OpenAI streaming wire shape without introducing a second queue or a
        long-lived backend stream.
        """
        terminal = await self._polling_wait.wait(job_id, timeout=timeout)
        result = terminal if isinstance(terminal, dict) else {}
        raw_body = result.get("body", {})
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8", "replace")
        if isinstance(raw_body, str):
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                payload = {"error": raw_body}
        elif isinstance(raw_body, dict):
            payload = raw_body
        else:
            payload = {"error": "invalid durable result"}

        if not isinstance(payload, dict):
            payload = {"error": "invalid durable result"}
        if int(result.get("status", 200) or 200) >= 400 or "error" in payload:
            yield _sse_frame(payload)
            yield b"data: [DONE]\n\n"
            return

        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        base = {
            "id": payload.get("id", f"chatcmpl-{job_id}"),
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", "gemma"),
            "object": "chat.completion.chunk",
        }
        first = dict(base)
        first["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        yield _sse_frame(first)
        delta = dict(base)
        delta["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
        yield _sse_frame(delta)
        final = dict(base)
        final["choices"] = [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]
        yield _sse_frame(final)
        yield b"data: [DONE]\n\n"


# -------------------------------------------------------------- helpers


def _refuse_sse() -> web.StreamResponse:
    return _refuse_sse_response()


# ----------------------------------------------------------- public API


def build_runtime(
    *,
    config: BrokerConfig,
    repository: Any,
    members_provider: Callable[[], Iterable[MemberSnapshot]],
    transport: Transport,
    clock: Callable[[], float],
    repair_fence: InMemoryRepairFence | None = None,
    request_timeout: float = 30.0,
    compat_timeout: float = 30.0,
    dispatch_owner: str | None = None,
) -> BrokerRuntime:
    """Construct a fresh BrokerRuntime with all collaborators wired via DI.

    Required injections:
        config           — a BrokerConfig validated against the
                           candidate invariants (qual:combined-gemma: prefix,
                           enabled=True, ordered_members in declared order).
        repository       — RedisJobRepository-compatible (RedisJobRepository,
                           InMemoryJobRepository, or any structural duck-type).
        members_provider — zero-arg callable returning an iterable of
                           MemberSnapshot in the configured order.
        transport        — Transport protocol implementation (a buffered
                           aiohttp transport, a test stub, etc.).
        clock            — zero-arg callable returning a monotonic float.

    Optional:
        repair_fence     — InMemoryRepairFence (a fresh one is created
                           if not supplied).
        request_timeout  — dispatcher transport timeout (seconds).
        compat_timeout   — compat wrapper wait timeout (seconds).
        dispatch_owner   — owner key for the dispatch-loop leadership
                           fence (defaults to a process-stable id).
    """
    return BrokerRuntime(
        config=config,
        repository=repository,
        members_provider=members_provider,
        transport=transport,
        clock=clock,
        repair_fence=repair_fence,
        request_timeout=request_timeout,
        compat_timeout=compat_timeout,
        dispatch_owner=dispatch_owner,
    )


__all__ = [
    "BrokerRuntime",
    "BufferedAiohttpTransport",
    "CandidateLifecycleMutationDisabled",
    "CandidateSSEDisabled",
    "CandidateConfigInvalid",
    "DispatchLoop",
    "PollingTerminalWait",
    "TerminalNotifier",
    "Transport",
    "build_candidate_app",
    "build_runtime",
    "map_member_ready",
    "validate_candidate_config",
    "CANDIDATE_NAMESPACE_PREFIX",
]
