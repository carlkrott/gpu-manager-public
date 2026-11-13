#!/usr/bin/env python3
"""Read-only authenticated rollout preflight for the OOM-inflight candidate."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROUTES = (
    "/health",
    "/v1/queue",
    "/v1/workers",
    "/v1/metrics/broker",
)
TERMINAL_JOB_STATES = {"completed", "failed", "cancelled"}


class PreflightError(RuntimeError):
    """A required read-only observation did not prove the rollout gate."""


def in_disallowed_window(now: datetime) -> bool:
    """Deployment is barred locally from 03:15 through 03:55 inclusive."""
    minute = now.hour * 60 + now.minute
    return 3 * 60 + 15 <= minute <= 3 * 60 + 55


def _zero_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PreflightError(f"{label} is missing or not a non-negative integer")
    if value != 0:
        raise PreflightError(f"{label} is nonzero ({value})")
    return value


def validate_observations(payloads: dict[str, Any], job_id: str | None = None) -> dict[str, int | str]:
    """Validate all authenticated route bodies without retaining sensitive fields."""
    for route in ROUTES:
        if route not in payloads or not isinstance(payloads[route], dict):
            raise PreflightError(f"authenticated GET {route} returned no JSON object")
        if "error" in payloads[route]:
            raise PreflightError(f"authenticated GET {route} returned an API error")

    health = payloads["/health"]
    _zero_int(health.get("active_requests"), "/health active_requests")
    _zero_int(health.get("queued_requests"), "/health queued_requests")

    queue = payloads["/v1/queue"]
    services = queue.get("services")
    if not isinstance(services, dict) or not services:
        raise PreflightError("/v1/queue services is missing or empty")
    for name, status in services.items():
        if not isinstance(status, dict):
            raise PreflightError("/v1/queue contains a malformed service row")
        _zero_int(status.get("queue_depth"), f"/v1/queue {name} queue_depth")
        _zero_int(status.get("slots_active"), f"/v1/queue {name} slots_active")

    workers = payloads["/v1/workers"]
    pools = workers.get("pools")
    if not isinstance(pools, list):
        raise PreflightError("/v1/workers pools is missing or malformed")
    active_by_pool = 0
    for pool in pools:
        if not isinstance(pool, dict):
            raise PreflightError("/v1/workers contains a malformed pool row")
        active_by_pool += _zero_int(pool.get("slots_active"), "/v1/workers pool slots_active")
    _zero_int(workers.get("total_active"), "/v1/workers total_active")
    if workers["total_active"] != active_by_pool:
        raise PreflightError("/v1/workers aggregate active count does not match pools")

    job_status: str | None = None
    if job_id is not None:
        job = payloads.get(f"/v1/job/{job_id}")
        if not isinstance(job, dict) or job.get("error") is not None:
            raise PreflightError("selected image job readback is missing or returned an API error")
        if job.get("service_name") != "comfyui" or job.get("status") not in TERMINAL_JOB_STATES:
            raise PreflightError("selected image job readback is non-terminal or invalid")
        job_status = job["status"]

    broker = payloads["/v1/metrics/broker"]
    if broker.get("schema") != "combined-gemma.broker-metric.v1":
        raise PreflightError("broker metric schema mismatch")
    broker_queue = broker.get("queue")
    jobs = broker.get("jobs")
    members = broker.get("members")
    if not isinstance(broker_queue, dict) or broker_queue.get("fresh") is not True:
        raise PreflightError("broker queue observation is not fresh")
    _zero_int(broker_queue.get("depth"), "broker queue depth")
    if not isinstance(jobs, dict):
        raise PreflightError("broker jobs observation is missing")
    _zero_int(jobs.get("depth"), "broker jobs depth")
    _zero_int(jobs.get("total"), "broker jobs total")
    by_state = jobs.get("by_state")
    if not isinstance(by_state, dict):
        raise PreflightError("broker jobs by_state is missing or malformed")
    for state, count in by_state.items():
        _zero_int(count, f"broker jobs by_state[{state}]")
    if not isinstance(members, list) or not members:
        raise PreflightError("broker member lease observations are missing")
    for member in members:
        if not isinstance(member, dict) or member.get("fresh") is not True:
            raise PreflightError("broker member lease observation is stale or malformed")
        _zero_int(member.get("live_lease_count"), "broker member live_lease_count")
        _zero_int(member.get("dispatcher_leases"), "broker member dispatcher_leases")

    return {
        "queue_services": len(services),
        "worker_pools": len(pools),
        "broker_members": len(members),
        "job_status": job_status or "not_requested",
    }


def active_local_lease_count(database: Path, now_epoch: float | None = None) -> int:
    """Count active local leases using SQLite read-only mode and expiry only."""
    if database.is_symlink() or not database.is_file():
        raise PreflightError("local lease database is missing or unsafe")
    uri = database.resolve(strict=True).as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=3)
        try:
            connection.execute("PRAGMA query_only=ON")
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if "leases" not in tables:
                raise PreflightError("local lease database has no leases table")
            columns = {row[1] for row in connection.execute('PRAGMA table_info("leases")')}
            if "expires" not in columns:
                raise PreflightError("local lease table has no expiry column")
            now = time.time() if now_epoch is None else now_epoch
            row = connection.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN typeof(expires) NOT IN ('integer','real') THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN typeof(expires) IN ('integer','real') AND expires > ? THEN 1 ELSE 0 END) "
                "FROM leases",
                (now,),
            ).fetchone()
            if row is None or row[1] not in (0, None):
                raise PreflightError("local lease expiry values are not numeric")
            return int(row[2] or 0)
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise PreflightError("local lease database could not be read-only queried") from exc


def backend_unit(services_path: Path) -> str:
    if services_path.is_symlink() or not services_path.is_file():
        raise PreflightError("services registry is missing or unsafe")
    try:
        data = json.loads(services_path.read_text(encoding="utf-8"))
        unit = data["services"]["comfyui"]["systemd_unit"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PreflightError("cannot resolve comfyui systemd unit from registry") from exc
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9@_.-]+\.service", unit):
        raise PreflightError("registry-derived comfyui systemd unit is unsafe")
    return unit


def privileged_unit_read(unit: str) -> dict[str, str]:
    command = [
        "sudo", "-n", "systemctl", "show", unit,
        "-p", "ActiveState", "-p", "SubState", "-p", "Result",
        "-p", "ExecMainStatus", "--no-pager",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("privileged read-only backend unit observation failed") from exc
    if completed.returncode != 0:
        raise PreflightError("privileged read-only backend unit observation was refused")
    values = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    required = {"ActiveState", "SubState", "Result", "ExecMainStatus"}
    if not required.issubset(values):
        raise PreflightError("privileged backend unit observation is incomplete")
    if not re.fullmatch(r"-?[0-9]+", values["ExecMainStatus"]):
        raise PreflightError("backend ExecMainStatus is malformed")
    return {key: values[key] for key in sorted(required)}


def _load_token(auth_source_dir: Path, credential_file: Path) -> str:
    source = str(auth_source_dir.resolve(strict=True))
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from api_auth import configured_token
        token = configured_token({"GPU_MANAGER_API_TOKEN_FILE": str(credential_file)})
    except Exception as exc:
        raise PreflightError("existing API credential loader is unavailable") from exc
    if not token:
        raise PreflightError("existing API credential loader could not read its systemd credential")
    return token


def _read_route(base_url: str, route: str, token: str) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + route,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=8) as response:
            if response.status != 200:
                raise PreflightError(f"authenticated GET {route} returned HTTP {response.status}")
            body = response.read(4 * 1024 * 1024 + 1)
    except HTTPError as exc:
        raise PreflightError(f"authenticated GET {route} returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise PreflightError(f"authenticated GET {route} failed") from exc
    if len(body) > 4 * 1024 * 1024:
        raise PreflightError(f"authenticated GET {route} response exceeded the size bound")
    try:
        result = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"authenticated GET {route} returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise PreflightError(f"authenticated GET {route} returned a non-object")
    return result


def _load_fixture(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError("test response fixture could not be read") from exc
    if not isinstance(value, dict):
        raise PreflightError("test response fixture is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-source-dir", type=Path, required=True)
    parser.add_argument("--services-json", type=Path, required=True)
    parser.add_argument("--lease-db", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, default=Path(
        os.environ.get("GPU_MANAGER_API_TOKEN_FILE", "/run/credentials/gpu-manager.service/gpu-manager-api-token")
    ))
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--job-id", help="optional job id whose terminal ComfyUI status must be verified")
    parser.add_argument("--responses-fixture", type=Path)
    parser.add_argument("--now", help="fixture-only local ISO timestamp; requires --responses-fixture")
    args = parser.parse_args()

    try:
        if args.now is not None and args.responses_fixture is None:
            raise PreflightError("--now is accepted only with --responses-fixture")
        if args.job_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_id):
            raise PreflightError("--job-id contains unsafe characters")
        now = datetime.fromisoformat(args.now) if args.now else datetime.now()
        if in_disallowed_window(now):
            raise PreflightError("deployment is inside the prohibited local 03:15-03:55 window")
        if args.responses_fixture is not None:
            payloads = _load_fixture(args.responses_fixture)
        else:
            token = _load_token(args.auth_source_dir, args.credential_file)
            routes = ROUTES + ((f"/v1/job/{args.job_id}",) if args.job_id is not None else ())
            payloads = {route: _read_route(args.base_url, route, token) for route in routes}
        if args.responses_fixture is not None and args.job_id is not None:
            route = f"/v1/job/{args.job_id}"
            if route not in payloads:
                raise PreflightError(f"selected job fixture is missing route {route}")
        counts = validate_observations(payloads, args.job_id)
        lease_count = active_local_lease_count(args.lease_db)
        if lease_count != 0:
            raise PreflightError(f"local active lease count is nonzero ({lease_count})")
        unit = backend_unit(args.services_json)
        unit_state = privileged_unit_read(unit)
        if unit_state["ActiveState"] != "inactive":
            raise PreflightError(f"backend unit {unit} is not inactive")
        job_message = (
            f"; selected image job={counts['job_status']}" if args.job_id is not None else ""
        )
        print(
            "oom-inflight-preflight: read-only gates passed; "
            f"{counts['queue_services']} queue services idle; "
            f"{counts['worker_pools']} worker pools idle; "
            f"{counts['broker_members']} broker members lease-free; "
            f"local active leases=0{job_message}; "
            f"backend={unit} {unit_state['ActiveState']}/{unit_state['SubState']}; "
            "outside prohibited local window; no writes performed"
        )
        return 0
    except PreflightError as exc:
        print(f"oom-inflight-preflight: REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
