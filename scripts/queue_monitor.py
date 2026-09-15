#!/usr/bin/env python3
"""Sample GPU Manager's read-only semantic health and queue telemetry.

This replaces direct Valkey, journal and hardware scraping. It discovers
current services from the controller, keeps unavailable data distinct from a
valid zero, and writes to stdout unless an explicit JSONL destination is given.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class MonitorError(RuntimeError):
    """A telemetry endpoint could not be sampled safely."""


def _api_root(value: str) -> str:
    root = value.strip().rstrip("/")
    parsed = urlsplit(root)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("API URL must be an explicit HTTP(S) origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("API URL may not contain credentials or query data")
    return root


def _get_json(api_root: str, path: str, timeout: float) -> Mapping[str, Any]:
    request = Request(api_root + path, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise MonitorError(f"{path} exceeded the response limit")
    except HTTPError as exc:
        raise MonitorError(f"{path} returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise MonitorError(f"{path} is unavailable: {type(exc).__name__}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MonitorError(f"{path} returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise MonitorError(f"{path} did not return an object")
    return value


def _sample(api_root: str, *, timeout: float, window: int) -> dict[str, Any]:
    results: dict[str, Mapping[str, Any]] = {}
    errors: dict[str, str] = {}
    endpoints = {
        "health": "/health",
        "service_groups": "/v1/service-groups",
        "metrics": "/v1/metrics/summary?" + urlencode({"window": window}),
    }
    for name, path in endpoints.items():
        try:
            results[name] = _get_json(api_root, path, timeout)
        except MonitorError as exc:
            errors[name] = str(exc)

    health = results.get("health", {})
    registry = health.get("registry") if isinstance(health, Mapping) else None
    checked = health.get("checked_runtimes") if isinstance(health, Mapping) else None
    service_groups = results.get("service_groups", {})
    raw_groups = (
        service_groups.get("groups", {})
        if isinstance(service_groups, Mapping)
        else {}
    )
    groups = {}
    if isinstance(raw_groups, Mapping):
        for name, group in raw_groups.items():
            if not isinstance(group, Mapping):
                continue
            queue_error = group.get("queue_lag_error")
            queue_depth = group.get("queue_lag")
            queue_known = group.get("queue_lag_known")
            queue_reason = group.get("queue_lag_reason")
            if queue_error:
                queue_state = "unavailable"
            elif queue_reason in {
                "external_queue_authority",
                "external_combined_gemma_broker",
            }:
                queue_state = "delegated"
            elif queue_known is False or queue_depth is None:
                queue_state = "unavailable"
            else:
                queue_state = "observed"
            groups[str(name)] = {
                "type": group.get("type"),
                "queue_depth": queue_depth,
                "queue_state": queue_state,
                "queue_reason": queue_reason,
                "oldest_wait_seconds": group.get("oldest_wait_seconds"),
                "pool": group.get("pool_summary"),
            }
            if queue_error:
                errors[f"queue:{name}"] = str(queue_error)[:300]
            elif queue_state == "unavailable":
                errors[f"queue:{name}"] = str(
                    queue_reason or "queue depth unavailable"
                )[:300]

    metrics = results.get("metrics", {})
    metric_total = (
        int(metrics.get("total_completed", 0) or 0)
        + int(metrics.get("total_failed", 0) or 0)
        if isinstance(metrics, Mapping)
        else 0
    )
    registry_errors = (
        registry.get("validation_errors", [])
        if isinstance(registry, Mapping)
        else ["registry telemetry unavailable"]
    )
    if not isinstance(registry_errors, list):
        registry_errors = ["registry validation response is malformed"]

    status = "ok"
    if "health" in errors:
        status = "unavailable"
    elif (
        errors
        or registry_errors
        or (
            isinstance(checked, Mapping)
            and checked.get("status") in {"blocked", "unavailable"}
        )
    ):
        status = "degraded"

    return {
        "schema_version": "gpu-manager-semantic-monitor.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "controller": {
            "llm_available": health.get("llm_available")
            if isinstance(health, Mapping)
            else None,
            "maintenance_mode": health.get("maintenance_mode")
            if isinstance(health, Mapping)
            else None,
            "registry_fingerprint": registry.get("fingerprint")
            if isinstance(registry, Mapping)
            else None,
            "registry_validation_error_count": len(registry_errors),
            "checked_runtime_status": checked.get("status")
            if isinstance(checked, Mapping)
            else None,
        },
        "queues": groups,
        "metrics": {
            "window_seconds": window,
            "sample_count": metrics.get("sample_count", metric_total)
            if isinstance(metrics, Mapping)
            else 0,
            "sample_state": metrics.get(
                "sample_state", "observed" if metric_total else "no_samples"
            )
            if isinstance(metrics, Mapping)
            else "unavailable",
            "total_completed": metrics.get("total_completed")
            if isinstance(metrics, Mapping)
            else None,
            "total_failed": metrics.get("total_failed")
            if isinstance(metrics, Mapping)
            else None,
            "success_rate": metrics.get("success_rate")
            if isinstance(metrics, Mapping) and metric_total
            else None,
        },
        "errors": errors,
    }


def _append_jsonl(path: Path, line: str) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        type=_api_root,
        default=_api_root(
            os.environ.get("GPU_MANAGER_API_URL", "http://127.0.0.1:8091")
        ),
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-degraded",
        action="store_true",
        help="return non-zero for degraded/unavailable samples",
    )
    args = parser.parse_args(argv)
    if not 0.1 <= args.timeout <= 30:
        parser.error("--timeout must be between 0.1 and 30 seconds")
    if not 1 <= args.window <= 86400:
        parser.error("--window must be between 1 and 86400 seconds")

    record = _sample(args.api_url, timeout=args.timeout, window=args.window)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        sys.stdout.write(line)
    else:
        try:
            _append_jsonl(args.output, line)
        except OSError as exc:
            print(f"queue monitor output failed: {type(exc).__name__}", file=sys.stderr)
            return 2
    if args.fail_on_degraded:
        return {"ok": 0, "degraded": 1, "unavailable": 2}[record["status"]]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
