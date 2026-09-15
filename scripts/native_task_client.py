"""Reconnectable controller client for the native task owner."""
from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
import json

from aiohttp import ClientError, ClientTimeout


class NativeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body

    async def read(self) -> bytes:
        return self.body


class NativeObservationError(RuntimeError):
    """Missing or inconsistent task-owner evidence, not native failure."""


@asynccontextmanager
async def native_task_request(session, *, host_url, task_id, body, on_state, admission_started=False):
    """Admit once by immutable identity, then poll until a proved outcome.

    Cancellation disconnects this observer only. The native task host remains
    the execution owner and the queue can reconnect with the same task ID.
    """
    task_url = host_url.rstrip("/") + "/tasks/" + task_id
    accepted = bool(admission_started)
    request_sha = hashlib.sha256(json.dumps(json.loads(body), sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    timeout = ClientTimeout(total=15, sock_connect=3)
    def observe(record):
        try:
            on_state(record)
            return True
        except Exception:
            return False

    while True:
        try:
            # Before replaying even an idempotent admission, inspect its
            # durable identity. An earlier lost response may already own it.
            async with session.get(task_url, timeout=timeout) as response:
                if response.status == 404 and not accepted:
                    record = None
                elif response.status == 200:
                    record = await response.json()
                    accepted = True
                else:
                    raise NativeObservationError(f"native_task_status_http_{response.status}")
            if record is None:
                # Persist the crossing of the submission boundary before the
                # POST. After restart, a missing host record is unknown, not
                # permission to generate again from scratch.
                if not observe({'status': 'submit_intent', 'task_id': task_id}):
                    raise NativeObservationError('native task submit intent could not be persisted')
                accepted = True
                async with session.post(task_url, data=body,
                                        headers={"Content-Type": "application/json"},
                                        timeout=timeout) as response:
                    if response.status not in {200, 202}:
                        message = (await response.text())[:250]
                        raise NativeObservationError(f"native_task_admission_http_{response.status}: {message}")
                    record = await response.json()
                    accepted = True
            if record.get("request_sha256") != request_sha:
                raise NativeObservationError("native_task_request_identity_mismatch")
            state = record.get("status", "outcome_unknown")
            observe(record)
            if state in {"completed", "failed"}:
                async with session.get(task_url + "/result", timeout=timeout) as response:
                    if response.status != 200:
                        raise NativeObservationError(f"native_task_result_http_{response.status}")
                    raw = await response.read()
                if hashlib.sha256(raw).hexdigest() != record.get("response_sha256"):
                    raise NativeObservationError("native_task_response_identity_mismatch")
                yield NativeResponse(int(record["upstream_status"]), raw)
                return
            if state not in {"accepted", "in_flight", "outcome_unknown"}:
                raise NativeObservationError(f"native_task_invalid_state:{state}")
        except NativeObservationError as exc:
            # Missing/corrupt host receipts and identity mismatches are not
            # native generation failures. Hold ownership for reconciliation.
            observe({'status': 'outcome_unknown', 'task_id': task_id, 'reason': str(exc)[:250]})
            accepted = True
        except (ClientError, asyncio.TimeoutError):
            # A transport outage is not a generation failure. Keep the worker's
            # resource claim and the durable handle while trying to reconnect.
            observe({"status": "observation_unavailable", "task_id": task_id})
        await asyncio.sleep(2)


async def native_host_has_unresolved(session, host_url: str) -> bool:
    """Fail closed before unloading an engine owned by the task host."""
    try:
        async with session.get(host_url.rstrip("/") + "/health",
                               timeout=ClientTimeout(total=5)) as response:
            if response.status != 200:
                return True
            data = await response.json()
        return data.get("status") != "ok" or not isinstance(data.get("unresolved_tasks"), list) or bool(data["unresolved_tasks"])
    except (ClientError, asyncio.TimeoutError, ValueError):
        return True
