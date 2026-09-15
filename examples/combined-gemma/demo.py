#!/usr/bin/env python3
"""Validate and exercise the neutral Combined Gemma example.

The default mode is schema validation only. ``--mock-demo`` runs the real
candidate BrokerRuntime with an in-memory repository, injected member snapshots
and a no-op transport. It creates only short-lived loopback fixture sockets on
OS-assigned ports; it never contacts Redis, systemd, a model, or a production
service.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from combined_gemma_broker_service import build_member_configs, load_broker_settings
from gemma_broker.api import APIError
from gemma_broker.contracts import MemberSnapshot, MemberState
from gemma_broker.redis_store import InMemoryJobRepository
from gemma_broker.runtime import BrokerRuntime, validate_candidate_config
from gpu_manager_contracts import validate_registry

EXAMPLE_PATH = Path(__file__).with_name("services.json")


class SimulatedTransport:
    """Transport contract used only by the fixture runtime."""

    async def send(self, member, body, on_accepted):  # pragma: no cover - dispatch is not started
        on_accepted()
        return {"simulated": True, "status": "completed"}


def _snapshot(name: str, *, healthy: bool) -> MemberSnapshot:
    return MemberSnapshot(
        name=name,
        gpu_id=None,
        state=MemberState.READY_ACCEPTING if healthy else MemberState.UNHEALTHY,
        accepting=healthy,
        configured_slots=1,
        context_per_slot=8192,
        state_version=1,
        observed_at=0.0,
        generation_fence=1,
        backend_slots_total=1,
        backend_slots_busy=0,
        compatible_free_slots=1 if healthy else 0,
        capabilities=("chat",) if healthy else (),
        semantic_health=healthy,
        model_resident=healthy,
        dispatcher_registered=healthy,
        blockers=() if healthy else ("simulated_health_rejection",),
    )


def _payload(key: str) -> dict[str, Any]:
    return {
        "idempotency_key": key,
        "caller_scope": "example-demo",
        "request": {
            "messages": [{"role": "user", "content": "simulated fixture request"}],
            "max_tokens": 16,
        },
    }


def _validate_example() -> tuple[Any, dict[str, Any], dict[str, str]]:
    data = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    errors = validate_registry(data)
    if errors:
        raise ValueError("registry validation failed: " + "; ".join(errors))
    settings = load_broker_settings(EXAMPLE_PATH)
    validate_candidate_config(settings.config)
    member_configs, endpoints = build_member_configs(data["services"], settings.config)
    if any(member.get("enabled") is not False for member in member_configs.values()):
        raise ValueError("example members must all be disabled")
    return settings, member_configs, endpoints


def dry_run() -> dict[str, Any]:
    settings, member_configs, endpoints = _validate_example()
    return {
        "status": "dry_run_valid",
        "config": "examples/combined-gemma/services.json",
        "candidate_mode": settings.config.candidate_mode,
        "redis_namespace": settings.config.redis_namespace,
        "ordered_members": list(settings.config.ordered_members),
        "member_count": len(member_configs),
        "member_endpoints": endpoints,
        "all_members_disabled": all(not item["enabled"] for item in member_configs.values()),
        "real_model_output": False,
        "network_or_redis_used": False,
    }


def mock_demo() -> dict[str, Any]:
    settings, _, _ = _validate_example()
    fixture_sockets: list[socket.socket] = []
    fixture_ports: list[int] = []
    try:
        for _ in settings.config.ordered_members:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            fixture_sockets.append(listener)
            fixture_ports.append(int(listener.getsockname()[1]))

        current_members = {
            "value": [_snapshot(name, healthy=False) for name in settings.config.ordered_members]
        }
        clock_value = 0.0

        def clock() -> float:
            nonlocal clock_value
            clock_value += 1.0
            return clock_value

        runtime = BrokerRuntime(
            config=settings.config,
            repository=InMemoryJobRepository(aging_seconds=settings.config.aging_interval_seconds),
            members_provider=lambda: current_members["value"],
            transport=SimulatedTransport(),
            clock=clock,
            request_timeout=settings.config.api_wait_timeout,
            compat_timeout=settings.config.api_wait_timeout,
        )
        events: list[dict[str, Any]] = []
        try:
            try:
                runtime.api_service.submit(_payload("health-rejection"))
            except APIError as exc:
                events.append({"event": "health_rejection", "code": exc.code, "simulated": True})

            current_members["value"] = [
                _snapshot(name, healthy=True) for name in settings.config.ordered_members
            ]
            queued = runtime.api_service.submit(_payload("cancel-me"))
            events.append({"event": "queued", "state": queued.state.value, "simulated": True})
            cancelled = runtime.api_service.cancel(queued.job_id)
            events.append({"event": "cancelled", "state": cancelled.state.value, "simulated": True})

            terminal = runtime.api_service.submit(_payload("terminal-me"))
            completed = runtime.api_service.complete_for_test(
                terminal.job_id,
                {"simulated": True, "output": "no model output generated"},
            )
            events.append({"event": "terminal", "state": completed.state.value, "simulated": True})
        finally:
            asyncio.run(runtime.close())

        return {
            "status": "mock_demo_complete",
            "ordered_members": list(settings.config.ordered_members),
            "fixture_loopback_ports": fixture_ports,
            "events": events,
            "real_model_output": False,
            "production_services_contacted": False,
            "redis_contacted": False,
        }
    finally:
        for listener in fixture_sockets:
            listener.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="validate only (the default)")
    modes.add_argument("--mock-demo", action="store_true", help="run the bounded fixture runtime")
    args = parser.parse_args(argv)
    try:
        result = mock_demo() if args.mock_demo else dry_run()
    except (OSError, ValueError, APIError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
