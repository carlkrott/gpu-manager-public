"""Standalone Combined Gemma broker process.

This module is the production composition root.  It owns only broker state and
observation; GPUManager remains responsible for lifecycle mutations.  Importing
this module has no network, Redis, systemd, or process side effects.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field, replace
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping

import aiohttp
from aiohttp import web

from gemma_broker.api import APIError, _current_attempt, _status_body
from gemma_broker.config import BrokerConfig
from gemma_broker.contracts import MemberState
from gemma_broker.drain import MemberDrainCoordinator, MemberRejoinCoordinator
from gemma_broker.member_cache import StrictMemberSnapshotCache
from gemma_broker.member_observer import MemberObserver
from gemma_broker.redis_store import (
    RedisJobRepository,
    RepositoryUnavailable,
    ReservationStale,
)
from gemma_broker.runtime import BufferedAiohttpTransport, build_runtime
from gpu_manager_contracts import registry_summary
from api_auth import api_auth_middleware


LOGGER = logging.getLogger("combined_gemma_broker_service")
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"



@dataclass(frozen=True, slots=True)
class BrokerSettings:
    config: BrokerConfig
    member_configs: dict[str, dict[str, Any]]
    endpoints: dict[str, str]
    registry: dict[str, Any] = field(default_factory=dict)


def build_member_configs(
    raw_services: Mapping[str, Any], config: BrokerConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Validate and normalize member service config without mutating input."""
    members: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, str] = {}
    for name in config.ordered_members:
        raw = raw_services.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"MEMBER_CONFIG_INVALID:{name}")
        # A disabled GPU member remains part of broker topology.  Readiness
        # reduces it to a config-disabled blocker and the CPU member can still
        # serve; rejecting it here incorrectly makes the whole broker unloadable.
        if type(raw.get("enabled")) is not bool:
            raise ValueError(f"MEMBER_ENABLED_INVALID:{name}")
        member = dict(raw)
        if member.get("member_type") == "openai_compatible":
            endpoint = member.get("endpoint")
            model = member.get("model")
            if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
                raise ValueError(f"MEMBER_ENDPOINT_INVALID:{name}")
            if not isinstance(model, str) or not model:
                raise ValueError(f"MEMBER_MODEL_INVALID:{name}")
            member["cpu_only"] = True
            member["idle_service_configured"] = None
            member["capabilities"] = ("chat", "completion")
            members[name] = member
            endpoints[name] = endpoint
            continue
        port = member.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError(f"MEMBER_PORT_INVALID:{name}")
        timeout = member.get("forward_timeout", 600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"MEMBER_FORWARD_TIMEOUT_INVALID:{name}")
        member["idle_service_configured"] = None if member.get("cpu_only") else name
        member["capabilities"] = ("chat", "completion")
        members[name] = member
        endpoints[name] = f"http://127.0.0.1:{port}/v1/chat/completions"
    return members, endpoints


def load_broker_settings(path: str | os.PathLike[str] | None = None) -> BrokerSettings:
    """Load broker settings from an explicitly supplied path or env var."""
    selected_path = path or os.environ.get("GPU_MANAGER_CONFIG_PATH")
    if not selected_path:
        raise ValueError("CONFIG_PATH_REQUIRED")
    source = Path(selected_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("SERVICES_CONFIG_OBJECT_REQUIRED")
    block = data.get("combined_gemma_broker")
    if not isinstance(block, dict):
        raise ValueError("COMBINED_GEMMA_BROKER_BLOCK_MISSING")
    candidate = block.get("candidate_mode")
    if type(candidate) is not bool:
        raise ValueError("COMBINED_GEMMA_BROKER_CANDIDATE_MODE_BOOL_REQUIRED")
    broker_config = BrokerConfig.from_dict(block, candidate=candidate)
    raw_services = data.get("services")
    if not isinstance(raw_services, dict):
        raise ValueError("SERVICES_BLOCK_REQUIRED")
    member_configs, endpoints = build_member_configs(raw_services, broker_config)
    return BrokerSettings(
        broker_config,
        member_configs,
        endpoints,
        registry=registry_summary(data, owner="broker"),
    )


class StandaloneBrokerService:
    """Own the runtime, cache, observation loop, and broker HTTP facade."""

    def __init__(
        self,
        *,
        settings: BrokerSettings,
        runtime,
        cache: StrictMemberSnapshotCache,
        observer: MemberObserver,
        session: aiohttp.ClientSession,
        redis_client: Any,
        refresh_interval: float = 2.0,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.cache = cache
        self.observer = observer
        self.session = session
        self.redis_client = redis_client
        self.refresh_interval = float(refresh_interval)
        self.drain_coordinator = MemberDrainCoordinator(runtime.repository)
        self.rejoin_coordinator = MemberRejoinCoordinator(runtime.repository)
        self._refresh_task: asyncio.Task | None = None
        self._closed = False

    @property
    def repository(self):
        return self.runtime.repository

    def members(self):
        return self.cache.snapshots()

    async def refresh_once(self) -> None:
        now = asyncio.get_running_loop().time()
        evidence = await self.observer.observe_all(now=now)
        for name in self.settings.config.ordered_members:
            snapshot = self.cache.update(name, evidence[name], now=now)
            try:
                persisted = self.repository.get_member(name)
            except RepositoryUnavailable:
                # A malformed generation fence must not prevent the broker from
                # booting.  The repository-side recovery is CAS/fail-closed and
                # refuses any record whose other fields, state, or leases are not
                # safe for repair.
                persisted = self.repository.repair_malformed_member_generation_fence(name)
                LOGGER.warning("repaired malformed member generation fence for %s", name)
            if (
                persisted is not None
                and persisted.state is MemberState.JOINING
                and snapshot.state is MemberState.READY_ACCEPTING
                and snapshot.accepting
                and not snapshot.blockers
                and not snapshot.reason_codes
                and self.repository.lease_count(name) == 0
            ):
                # A broker/controller restart can strand a member in JOINING
                # after the runtime has become fully ready. Completing the
                # existing fenced transition is safe and idempotent; OFFLINE,
                # DRAINING, FAULTED, disabled and unhealthy members never enter
                # this branch. A concurrent lifecycle mutation wins by CAS.
                try:
                    snapshot = self.rejoin_coordinator.rejoin(
                        name,
                        expected_state_version=persisted.state_version,
                        generation_fence=persisted.generation_fence,
                        readiness=snapshot,
                    )
                    persisted = snapshot
                    self.cache._snapshots[name] = snapshot
                    LOGGER.warning(
                        "completed healthy persisted member rejoin member=%s "
                        "state_version=%s",
                        name,
                        snapshot.state_version,
                    )
                except ReservationStale:
                    persisted = self.repository.get_member(name)
                    if persisted is None:
                        raise
            if persisted is not None and persisted.state in {
                MemberState.DRAINING,
                MemberState.OFFLINE,
                MemberState.JOINING,
                MemberState.FAULTED,
            }:
                # These are controller-owned lifecycle states.  Probe evidence
                # may refresh liveness details, but it must not erase a drain,
                # offline fence, or rejoin while that operation is in flight.
                # In particular, publishing READY while a drain is waiting on
                # live leases makes complete_member_drain fail with
                # DRAIN_INVALID_STATE and breaks non-preemptive GPU handoff.
                # FAULTED is also a controller-owned quarantine: a healthy
                # probe alone must not clear it, otherwise a transient probe
                # would reopen a member that still needs an explicit recovery
                # decision and generation-fenced rejoin.
                blocker = {
                    MemberState.DRAINING: "member drain in progress",
                    MemberState.OFFLINE: "member rejoin required",
                    MemberState.JOINING: "member rejoin in progress",
                    MemberState.FAULTED: "member faulted; explicit rejoin required",
                }[persisted.state]
                snapshot = replace(
                    snapshot,
                    state=persisted.state,
                    accepting=False,
                    state_version=persisted.state_version,
                    generation_fence=persisted.generation_fence,
                    compatible_free_slots=0,
                    blockers=tuple(
                        dict.fromkeys((*snapshot.blockers, blocker))
                    ),
                )
                self.cache._snapshots[name] = snapshot
            elif persisted is not None and (
                snapshot.state_version < persisted.state_version
                or snapshot.generation_fence < persisted.generation_fence
            ):
                # Probe evidence has no lifecycle authority and commonly carries
                # process-local version defaults.  Once a drain/rejoin advances
                # either durable CAS fence, observation may refresh readiness but
                # must never regress those lifecycle versions.
                snapshot = replace(
                    snapshot,
                    state_version=max(
                        snapshot.state_version,
                        persisted.state_version,
                    ),
                    generation_fence=max(
                        snapshot.generation_fence,
                        persisted.generation_fence,
                    ),
                )
                self.cache._snapshots[name] = snapshot
            # The standalone process is the sole effective-member-hash writer.
            # Cache the lease-adjusted durable result, not the pre-Lua observer
            # projection, so health and reservation see the same capacity.
            self.cache._snapshots[name] = self.repository.put_member(snapshot)

    async def _refresh_loop(self) -> None:
        while not self._closed:
            try:
                await self.refresh_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("member observation refresh failed")
            await asyncio.sleep(self.refresh_interval)

    async def start(self) -> None:
        await self.runtime.dispatch_loop.start()
        await self.refresh_once()
        self._refresh_task = asyncio.create_task(self._refresh_loop(), name="combined-gemma-observer")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)
            self._refresh_task = None
        await self.runtime.close()
        close_redis = getattr(self.redis_client, "close", None)
        if callable(close_redis):
            result = close_redis()
            if asyncio.iscoroutine(result):
                await result
        if not self.session.closed:
            await self.session.close()

    # Durable API facade
    def submit(self, payload: dict):
        return self.runtime.api_service.submit(payload)

    def status(self, job_id: str):
        return self.runtime.api_service.status(job_id)

    def cancel(self, job_id: str):
        return self.runtime.api_service.cancel(job_id)

    async def handle_proxy(self, **kwargs):
        return await self.runtime.handle_proxy(**kwargs)

    def health_live(self) -> dict:
        return {"status": "ok"}

    def health_ready(self):
        members = list(self.members())
        ready = [
            m for m in members
            if m.accepting and m.semantic_health and (m.compatible_free_slots or 0) > 0
        ]
        body = {
            "status": "ok" if not self._closed and ready else "unavailable",
            "readiness": "ready" if not self._closed and ready else "no_ready_accepting_member",
            "broker": "combined_gemma",
            "ready_members": len(ready),
            "members": [m.to_dict() for m in members],
            "registry": dict(self.settings.registry),
        }
        return body, 200 if body["status"] == "ok" else 503

    def health_dispatch(self):
        loop = self.runtime.dispatch_loop
        body = {
            "status": "ok" if loop.is_running else "unavailable",
            "broker": "combined_gemma",
            "dispatch_loop": loop.metrics,
            "owner": self.runtime.owner_token,
        }
        return body, 200 if loop.is_running else 503

    async def broker_metrics(self, *, include_durable_counts: bool = True) -> dict:
        from gemma_broker.observability import SCHEMA_VERSION

        members = list(self.members())

        def collect_metrics():
            counts = self.repository.durable_counts() if include_durable_counts else None
            queue_depth = int(self.repository.queue_total)
            lease_counts = {
                member.name: self.repository.lease_count(member.name)
                for member in members
            }
            return counts, queue_depth, lease_counts

        counts, queue_depth, lease_counts = await asyncio.to_thread(collect_metrics)
        payload = self.runtime.metrics.snapshot(
            members=members,
            queue_depth=queue_depth,
            now=asyncio.get_running_loop().time(),
            lease_counts=lease_counts,
            schema=SCHEMA_VERSION,
            namespace=self.settings.config.redis_namespace,
            leader={
                "owner": self.runtime.owner_token,
                "token": self.runtime.dispatch_loop.fencing_token,
                "expires_at": None,
            },
            dispatch_loop=self.runtime.dispatch_loop.metrics,
            attempts_total=counts["attempts_total"] if counts is not None else 0,
            attempts_by_state=counts["attempts_by_state"] if counts is not None else {},
            jobs_total=counts["jobs_total"] if counts is not None else 0,
            jobs_by_state=counts["jobs_by_state"] if counts is not None else {},
        )
        payload["members"] = list(payload["members"].values())
        if counts is None:
            payload["attempts"]["fresh"] = False
            payload["jobs"]["fresh"] = False
        return payload

    async def drain(self, name: str, *, expected_state_version: int, timeout: float, retry_stale: bool = False):
        return await self.runtime.drain_member(
            name,
            expected_state_version=expected_state_version,
            timeout=timeout,
            retry_stale=retry_stale,
        )


async def create_standalone_service(
    settings: BrokerSettings,
    *,
    redis_url: str = DEFAULT_REDIS_URL,
    refresh_interval: float = 2.0,
) -> StandaloneBrokerService:
    """Construct and start a standalone service; callers own its shutdown."""
    import redis

    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=603))
    repository = RedisJobRepository(
        redis_client,
        prefix=settings.config.redis_namespace,
        aging_seconds=settings.config.aging_interval_seconds,
        leadership_clock=time.time,
    )
    recovered = repository.recover_stale_pre_acceptance_claims(
        now=asyncio.get_running_loop().time(), max_age_seconds=120.0
    )
    if recovered:
        LOGGER.warning("requeued %d stale pre-acceptance claims", len(recovered))
    cache = StrictMemberSnapshotCache(
        settings.member_configs,
        freshness_ms=int(settings.config.freshness_seconds * 1000),
    )
    runtime_ref: dict[str, Any] = {}

    def dispatcher_state(member_name: str | None = None) -> tuple[bool, int]:
        runtime = runtime_ref.get("runtime")
        if runtime is None:
            return False, 0
        leases = repository.lease_count(member_name) if member_name else 0
        return bool(runtime.dispatch_loop.is_running), leases

    observer = MemberObserver(
        settings.member_configs,
        session=session,
        dispatcher_state=dispatcher_state,
        timeout=min(3.0, settings.config.freshness_seconds),
    )
    timeout = max(
        float(settings.config.api_wait_timeout),
        max(float(c.get("forward_timeout", 600)) for c in settings.member_configs.values()),
    ) + 60.0
    transport = BufferedAiohttpTransport(
        session=session,
        endpoint_for_member=lambda member: settings.endpoints.get(member.name, ""),
        request_for_member=lambda member, body: {
            **body,
            "model": settings.member_configs.get(member.name, {}).get(
                "model", body.get("model", "gemma")
            ),
        },
        timeout=timeout,
    )
    runtime = build_runtime(
        config=settings.config,
        repository=repository,
        members_provider=cache.snapshots,
        transport=transport,
        clock=asyncio.get_running_loop().time,
        request_timeout=timeout,
        compat_timeout=timeout,
    )
    runtime_ref["runtime"] = runtime
    service = StandaloneBrokerService(
        settings=settings,
        runtime=runtime,
        cache=cache,
        observer=observer,
        session=session,
        redis_client=redis_client,
        refresh_interval=refresh_interval,
    )
    try:
        await service.start()
    except Exception:
        await service.close()
        raise
    return service


def _error(exc: Exception) -> web.Response:
    return web.json_response(
        {"error": str(exc), "code": getattr(exc, "code", type(exc).__name__)},
        status=int(getattr(exc, "status", 500)),
    )


def create_standalone_app(service: StandaloneBrokerService) -> web.Application:
    app = web.Application(middlewares=[api_auth_middleware])

    async def submit(request: web.Request):
        try:
            job = service.submit(await request.json())
            location = f"/v1/gemma/jobs/{job.job_id}"
            response = web.json_response(
                {"job_id": job.job_id, "state": job.state.value, "status_url": location, "cancel_url": location, "next_attempt_id": None},
                status=202,
            )
            response.headers["Location"] = location
            return response
        except Exception as exc:
            return _error(exc)

    async def status(request: web.Request):
        try:
            job = service.status(request.match_info["job_id"])
            return web.json_response(_status_body(job, _current_attempt(service, job)))
        except Exception as exc:
            return _error(exc)

    async def cancel(request: web.Request):
        try:
            job = service.cancel(request.match_info["job_id"])
            return web.json_response(job.to_dict())
        except Exception as exc:
            return _error(exc)

    async def chat(request: web.Request):
        try:
            return await service.handle_proxy(
                method="POST", path="/v1/chat/completions", body=await request.read(),
                headers=request.headers, request=request,
            )
        except Exception as exc:
            return _error(exc)

    async def health(request: web.Request):
        try:
            body, code = service.health_ready()
            return web.json_response(body, status=code)
        except Exception as exc:
            return _error(exc)

    async def observability(request: web.Request):
        try:
            if request.path == "/health/live":
                result = service.health_live()
            elif request.path == "/health/ready" or request.path == "/health":
                result = service.health_ready()
            elif request.path == "/health/dispatch":
                result = service.health_dispatch()
            else:
                result = await service.broker_metrics(
                    include_durable_counts=request.query.get("durable", "1") != "0"
                )
            if isinstance(result, tuple):
                return web.json_response(result[0], status=result[1])
            return web.json_response(result)
        except Exception as exc:
            return _error(exc)

    async def registry_diagnostics(request: web.Request):
        return web.json_response(dict(service.settings.registry))

    async def drain(request: web.Request):
        name = request.match_info["name"]
        try:
            current = service.repository.get_member(name)
            if current is None:
                raise APIError(404, "unknown_member", name)
            body = await request.json() if request.can_read_body else {}
            has_explicit_version = "expected_state_version" in body
            expected = body.get("expected_state_version", current.state_version)
            timeout = float(body.get("timeout", 60.0))
            result = await service.drain(
                name,
                expected_state_version=int(expected),
                timeout=timeout,
                retry_stale=not has_explicit_version,
            )
            response = {"member": result.member.to_dict(), "drained": result.drained, "lease_count": result.lease_count, "blockers": list(result.blockers)}
            return web.json_response(response, status=200 if result.drained else 409)
        except Exception as exc:
            return _error(exc)

    async def rejoin(request: web.Request):
        name = request.match_info["name"]
        try:
            current = service.repository.get_member(name)
            if current is None:
                raise APIError(404, "unknown_member", name)
            body = await request.json() if request.can_read_body else {}
            expected = int(body.get("expected_state_version", current.state_version))
            generation = int(body["generation_fence"])
            evidence = await service.observer.observe(name, now=asyncio.get_running_loop().time())
            readiness = service.cache.update(name, evidence, now=asyncio.get_running_loop().time())
            updated = service.rejoin_coordinator.rejoin(
                name, expected_state_version=expected,
                generation_fence=generation, readiness=readiness,
            )
            service.repository.put_member(updated)
            return web.json_response({"member": updated.to_dict(), "rejoined": True})
        except Exception as exc:
            return _error(exc)

    async def repair_generation_fence(request: web.Request):
        name = request.match_info["name"]
        try:
            body = await request.json() if request.can_read_body else {}
            if "expected_state_version" not in body:
                raise ValueError("expected_state_version is required")
            repaired = service.repository.repair_member_generation_fence(
                name,
                expected_state_version=int(body["expected_state_version"]),
            )
            return web.json_response(
                {"member": repaired.to_dict(), "generation_fence_repaired": True}
            )
        except Exception as exc:
            return _error(exc)

    app.router.add_get("/health", observability)
    app.router.add_get("/health/live", observability)
    app.router.add_get("/health/ready", observability)
    app.router.add_get("/health/dispatch", observability)
    app.router.add_get("/v1/metrics/broker", observability)
    app.router.add_get("/v1/registry/diagnostics", registry_diagnostics)
    app.router.add_post("/v1/gemma/jobs", submit)
    app.router.add_get("/v1/gemma/jobs/{job_id}", status)
    app.router.add_delete("/v1/gemma/jobs/{job_id}", cancel)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/gemma/members/{name}/drain", drain)
    app.router.add_post("/v1/gemma/members/{name}/rejoin", rejoin)
    app.router.add_post(
        "/v1/gemma/members/{name}/repair-generation-fence",
        repair_generation_fence,
    )
    return app


async def serve(args: argparse.Namespace) -> None:
    settings = load_broker_settings(args.config)
    service = await create_standalone_service(
        settings, redis_url=args.redis_url, refresh_interval=args.refresh_interval
    )
    runner = web.AppRunner(create_standalone_app(service), access_log=LOGGER)
    await runner.setup()
    site = web.TCPSite(runner, host=args.host, port=args.port)
    await site.start()
    LOGGER.info("combined Gemma broker listening on %s:%d", args.host, args.port)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await service.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--redis-url-file", type=Path, default=None)
    parser.add_argument("--host", default=os.environ.get("COMBINED_GEMMA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("COMBINED_GEMMA_PORT", "18095")))
    parser.add_argument("--refresh-interval", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.redis_url_file is not None:
        args.redis_url = args.redis_url_file.read_text(encoding="utf-8").strip()
    if not args.redis_url:
        args.redis_url = os.environ.get("COMBINED_GEMMA_REDIS_URL", DEFAULT_REDIS_URL)
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(serve(parse_args(argv)))


if __name__ == "__main__":
    main()
