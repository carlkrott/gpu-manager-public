#!/usr/bin/env python3
"""Durable HTTP ownership for a synchronous native generation runtime.

This service survives GPU Manager restarts. Only its configured upstream may
be called; task IDs identify immutable requests. A host crash after forwarding
leaves an explicit unknown outcome, never an automatic replay.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from aiohttp import ClientSession, ClientTimeout, web

from runtime_host_client import (
    HelperTokenAuth,
    resolve_required_service_credential,
)


def atomic_write(path: Path, content: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".commit-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class NativeTaskHost:
    def __init__(
        self,
        root: Path,
        upstream: str,
        timeout: float,
        *,
        service_token: str | None = None,
        service_token_file: str | os.PathLike[str] | None = None,
        allow_unauthenticated_test_app: bool = False,
    ):
        """Construct the host; production construction requires a credential.

        ``allow_unauthenticated_test_app`` is reserved for explicit neutral
        in-process tests and must never be used by the CLI entrypoint.
        """
        if allow_unauthenticated_test_app:
            if service_token is not None or service_token_file is not None:
                raise ValueError("unauthenticated test app cannot receive credentials")
            helper_auth = HelperTokenAuth(use_environment=False)
        else:
            credential = resolve_required_service_credential(
                service_token=service_token, service_token_file=service_token_file
            )
            helper_auth = HelperTokenAuth(service_token=credential)
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner_lock = (self.root / ".owner.lock").open("a+")
        fcntl.flock(self.owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.upstream = upstream
        self.timeout = timeout
        self.helper_auth = helper_auth
        self.tasks: dict[str, asyncio.Task] = {}
        self.admission = asyncio.Lock()
        self.session: ClientSession | None = None
        # A previous process may have forwarded these. It cannot prove that
        # the engine stopped, so they must not be resubmitted or called failed.
        for path in self.root.glob("*/state.json"):
            record = json.loads(path.read_text())
            if record["status"] in {"accepted", "in_flight"}:
                record.update(status="outcome_unknown", error="native_task_host_restarted")
                self.save(path.parent, record)

    @classmethod
    def for_unauthenticated_test_app(
        cls, root: Path, upstream: str, timeout: float
    ) -> "NativeTaskHost":
        """Construct a neutral unauthenticated host for in-process tests only."""
        return cls(
            root,
            upstream,
            timeout,
            allow_unauthenticated_test_app=True,
        )

    async def runtime_pid(self) -> int | None:
        process = await asyncio.create_subprocess_exec(
            "systemctl", "show", "audiocpp-minimax-music3-candidate.service",
            "--property=MainPID", "--value", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        # Unavailable observation is not evidence that the accepting process
        # exited. A successful MainPID=0 observation does establish inactivity.
        return int(output.strip()) if process.returncode == 0 and output.strip().isdigit() else None

    async def reconcile_record(self, task_id: str) -> dict:
        record = self.record(task_id)
        if record["status"] == "outcome_unknown" and record.get("forwarded_runtime_pid"):
            current_pid = await self.runtime_pid()
            if current_pid is not None and current_pid != record["forwarded_runtime_pid"]:
                # The exact process accepting the synchronous request exited.
                # There is no recoverable response and no job on the replacement
                # process. Preserve this failed identity; never rerun it.
                raw = json.dumps({"error": "native_runtime_exited_before_response",
                                  "task_id": task_id}).encode()
                atomic_write(self.directory(task_id) / "response.json", raw)
                record.update(status="failed", upstream_status=503,
                              response_sha256=hashlib.sha256(raw).hexdigest(),
                              response_bytes=len(raw), completed_at=time.time(),
                              reconciled_runtime_pid=current_pid,
                              error="native_runtime_exited_before_response")
                self.save(self.directory(task_id), record)
        return record

    @staticmethod
    def save(directory: Path, record: dict) -> None:
        record["updated_at"] = time.time()
        atomic_write(directory / "state.json", json.dumps(record, sort_keys=True).encode())

    def directory(self, task_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", task_id):
            raise web.HTTPBadRequest(text="invalid task identity")
        return self.root / task_id

    def record(self, task_id: str) -> dict:
        path = self.directory(task_id) / "state.json"
        if not path.is_file():
            raise web.HTTPNotFound(text="native task not found")
        return json.loads(path.read_text())

    async def submit(self, request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        directory = self.directory(task_id)
        payload = await request.json()
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="JSON object required")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(body).hexdigest()
        async with self.admission:
            if (directory / "state.json").is_file():
                record = self.record(task_id)
                if record["request_sha256"] != digest:
                    raise web.HTTPConflict(text="task identity belongs to a different request")
                return web.json_response(record)
            # One serial native engine. Unknown outcomes retain ownership;
            # admitting another job could overlap an unobserved generation.
            for path in self.root.glob("*/state.json"):
                if json.loads(path.read_text())["status"] in {"accepted", "in_flight", "outcome_unknown"}:
                    raise web.HTTPConflict(text="native runtime has an unresolved task")
            directory.mkdir(mode=0o700, exist_ok=True)
            atomic_write(directory / "request.json", body)
            record = {"task_id": task_id, "request_sha256": digest,
                      "status": "accepted", "accepted_at": time.time(),
                      "upstream_submit_count": 0}
            self.save(directory, record)
            task = asyncio.create_task(self.execute(task_id, body))
            self.tasks[task_id] = task
            task.add_done_callback(lambda _task: self.tasks.pop(task_id, None))
            return web.json_response(record, status=202)

    async def execute(self, task_id: str, body: bytes) -> None:
        directory = self.directory(task_id)
        record = self.record(task_id)
        record.update(status="in_flight", upstream_submit_count=1, forwarded_at=time.time(),
                      forwarded_runtime_pid=await self.runtime_pid())
        self.save(directory, record)
        try:
            assert self.session is not None
            async with self.session.post(
                self.upstream, data=body, headers={"Content-Type": "application/json"},
                timeout=ClientTimeout(total=self.timeout, sock_connect=5),
            ) as response:
                raw = await response.read()
                atomic_write(directory / "response.json", raw)
                record.update(status="completed" if 200 <= response.status < 300 else "failed",
                              upstream_status=response.status,
                              response_sha256=hashlib.sha256(raw).hexdigest(),
                              response_bytes=len(raw), completed_at=time.time())
        except (Exception, asyncio.CancelledError) as exc:
            # No returned response means neither success nor failure of the
            # generation is established. Keep its identity and exclusive slot.
            record.update(status="outcome_unknown", error=type(exc).__name__)
        self.save(directory, record)

    async def status(self, request: web.Request) -> web.Response:
        return web.json_response(await self.reconcile_record(request.match_info["task_id"]))

    async def result(self, request: web.Request) -> web.StreamResponse:
        task_id = request.match_info["task_id"]
        record = self.record(task_id)
        if record["status"] not in {"completed", "failed"}:
            return web.json_response(record, status=409)
        return web.FileResponse(self.directory(task_id) / "response.json",
                                headers={"Content-Type": "application/json"})

    async def cancel(self, request: web.Request) -> web.Response:
        # audio.cpp has no task cancellation contract. Never report an active
        # task cancelled or kill a shared engine on a client's DELETE request.
        record = self.record(request.match_info["task_id"])
        return web.json_response({**record, "cancelled": False,
                                  "reason": "native runtime cannot cancel an accepted task"}, status=409)

    async def health(self, _request: web.Request) -> web.Response:
        unresolved = [json.loads(p.read_text())["task_id"] for p in self.root.glob("*/state.json")
                      if json.loads(p.read_text())["status"] in {"accepted", "in_flight", "outcome_unknown"}]
        return web.json_response({"status": "ok", "unresolved_tasks": unresolved})

    async def lifecycle(self, _app):
        async with ClientSession() as session:
            self.session = session
            yield
            # A normal host stop waits for its owned requests; the controller
            # does not stop this unit when switching models or restarting.
            if self.tasks:
                await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)

    def app(self) -> web.Application:
        @web.middleware
        async def helper_auth_middleware(request: web.Request, handler):
            if not self.helper_auth.available():
                return web.json_response(
                    {"error": "helper service authentication is unavailable"},
                    status=503,
                    headers={"Cache-Control": "no-store"},
                )
            if not self.helper_auth.authorized(request):
                return web.json_response(
                    {"error": "helper service authentication required"},
                    status=401,
                    headers={
                        "Cache-Control": "no-store",
                        "WWW-Authenticate": "Bearer",
                    },
                )
            return await handler(request)

        app = web.Application(
            client_max_size=1024 * 1024, middlewares=[helper_auth_middleware]
        )
        app.cleanup_ctx.append(self.lifecycle)
        app.router.add_get("/health", self.health)
        app.router.add_post("/tasks/{task_id}", self.submit)
        app.router.add_get("/tasks/{task_id}", self.status)
        app.router.add_get("/tasks/{task_id}/result", self.result)
        app.router.add_delete("/tasks/{task_id}", self.cancel)
        return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8133/v1/tasks/run")
    parser.add_argument("--port", type=int, default=8134)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    host = NativeTaskHost(
        args.state_dir,
        args.upstream,
        args.timeout,
        service_token_file=args.token_file,
    )
    # Loopback only: this is a controller adapter, never public generation ingress.
    web.run_app(host.app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
