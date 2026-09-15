#!/usr/bin/env python3
"""Priority/idempotency bridge for the pinned MiniMax two-stage runner.

This process is selected by ``HERMES_MINIMAX_STOCK_RUNNER`` inside the checked
MiniMax adapter. It does not accept user configuration. Deployment-owned
environment values identify the already hash-verified upstream runner and the
durable parent job whose priority its two leaf submissions must inherit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing checked adapter control: {name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_upstream():
    path = Path(_required("GPU_MANAGER_MINIMAX_PINNED_GENERATION_RUNNER"))
    expected = _required("GPU_MANAGER_MINIMAX_PINNED_GENERATION_RUNNER_SHA256")
    if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError("pinned MiniMax generation runner identity mismatch")
    spec = importlib.util.spec_from_file_location(
        "gpu_manager_minimax_upstream_generation", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned MiniMax generation runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_parent_contract(module) -> None:
    parent_job_id = _required("GPU_MANAGER_MINIMAX_PARENT_JOB_ID")
    priority = int(_required("GPU_MANAGER_MINIMAX_PARENT_PRIORITY"))
    if not 0 <= priority <= 100:
        raise RuntimeError("MiniMax parent priority must be between 0 and 100")
    priority_class = _required("GPU_MANAGER_MINIMAX_PARENT_PRIORITY_CLASS")
    if priority_class not in {"interactive", "normal", "background"}:
        raise RuntimeError("MiniMax parent priority class is invalid")

    original_request_body = module._request_body

    def request_body(generation_type, params, harness, candidate, timeout):
        body = original_request_body(
            generation_type, params, harness, candidate, timeout
        )
        body["priority"] = priority
        body["priority_class"] = priority_class
        body.setdefault("metadata", {})["parent_job_id"] = parent_job_id
        body["metadata"]["priority_source"] = "parent"
        return body

    def request(path: str, *, body: dict | None = None):
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if path == "/v1/submit/generation" and isinstance(body, dict):
            stage_type = str(body.get("type") or "unknown")
            token = hashlib.sha256(
                f"minimax-music3:{parent_job_id}:{stage_type}".encode("utf-8")
            ).hexdigest()
            headers["Idempotency-Key"] = token
        req = urllib.request.Request(
            module.MANAGER + path,
            data=encoded,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"error": raw}
            status = exc.code
        # The upstream runner requires 202 for a submission receipt. A replay
        # of an active leaf is the same unambiguous receipt and remains safe to
        # poll, even though terminal replays use HTTP 200 at the public API.
        if (
            path == "/v1/submit/generation"
            and status == 200
            and isinstance(payload, dict)
            and payload.get("idempotent_replay") is True
            and payload.get("job_id")
        ):
            status = 202
        return status, payload

    module._request_body = request_body
    module._request = request


def main(argv: list[str] | None = None) -> int:
    upstream = _load_upstream()
    _install_parent_contract(upstream)
    return int(upstream.main(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
