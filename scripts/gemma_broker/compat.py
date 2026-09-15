"""Synchronous OpenAI compatibility wrapper over the durable broker."""
from __future__ import annotations

import hashlib
import json
import os
from urllib.parse import parse_qs

from aiohttp import web


def _parse_bool(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _wants_async(*, path: str, headers, request: web.Request | None) -> bool:
    explicit_wait = headers.get("X-Compat-Wait") or headers.get("x-compat-wait")
    parsed_wait = _parse_bool(explicit_wait)
    if parsed_wait is not None:
        return not parsed_wait

    prefer = headers.get("Prefer") or headers.get("prefer") or ""
    if any(part.strip().split(";", 1)[0].lower() == "respond-async" for part in prefer.split(",")):
        return True

    query = getattr(request, "query", None)
    if query is None:
        _, _, raw_query = path.partition("?")
        query = parse_qs(raw_query)
    wait_value = query.get("wait") if hasattr(query, "get") else None
    if isinstance(wait_value, (list, tuple)):
        wait_value = wait_value[-1] if wait_value else None
    return _parse_bool(wait_value) is False


class CompatibilityWrapper:
    def __init__(self, service, *, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.service = service
        self.timeout = timeout
        # Per-request X-Compat-Timeout is clamped to [min_timeout, max_timeout].
        # max reuses the configured default as the upper bound; this
        # prevents a client from holding a worker indefinitely by sending
        # an absurd header value. Operators can tune via env vars.
        self.max_timeout = float(os.environ.get("COMPAT_TIMEOUT_MAX", "600"))
        self.min_timeout = float(os.environ.get("COMPAT_TIMEOUT_MIN", "1"))

    async def handle_proxy(
        self,
        *,
        method: str,
        path: str,
        body: bytes | None,
        headers,
        request: web.Request | None = None,
    ) -> web.StreamResponse:
        route_path = path.partition("?")[0]
        if method != "POST" or route_path != "/v1/chat/completions":
            return web.json_response(
                {"error": "unsupported compatibility route", "code": "unsupported_route"},
                status=404,
            )
        try:
            request_body = json.loads(body or b"{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return web.json_response(
                {"error": "valid JSON required", "code": "invalid_json"}, status=400
            )
        if not isinstance(request_body, dict):
            return web.json_response(
                {"error": "JSON object required", "code": "invalid_request"}, status=400
            )
        idempotency_key = headers.get("Idempotency-Key")
        if not idempotency_key:
            idempotency_key = "compat-" + hashlib.sha256(body or b"{}").hexdigest()
        # Per-request X-Compat-Timeout header. Falls back to self.timeout
        # when missing or invalid. Out-of-range values are clamped; a
        # negative / non-numeric / non-finite value is treated as missing
        # so a misconfigured client cannot cause a divide-by-zero or
        # infinite-wait downstream. Parsing BEFORE submit_compat so a
        # bad header never consumes an idempotency slot.
        import math as _math
        header_val = headers.get("X-Compat-Timeout") or headers.get("x-compat-timeout")
        effective_timeout = self.timeout
        if header_val is not None:
            try:
                requested = float(header_val)
            except (TypeError, ValueError):
                requested = self.timeout
            if not _math.isfinite(requested) or requested <= 0:
                requested = self.timeout
            effective_timeout = min(
                self.max_timeout, max(self.min_timeout, requested)
            )
        wait_for_result = not _wants_async(path=path, headers=headers, request=request)
        if not wait_for_result and request_body.get("stream") is True:
            return web.json_response(
                {
                    "error": "async compatibility requests cannot use stream=true",
                    "code": "async_stream_unsupported",
                },
                status=400,
            )
        # Translate broker APIError into a structured HTTP response so the
        # client gets a 4xx/5xx with body instead of aiohttp's bare
        # ``500 Server got itself in trouble``. Without this,
        # ``compat_wait timed out`` and ``no_eligible_member`` would all
        # surface as opaque 500s.
        try:
            from gemma_broker.api import APIError as _BrokerAPIError
        except ImportError:  # pragma: no cover - api is always importable in production
            _BrokerAPIError = None  # type: ignore[assignment]
        try:
            job = self.service.submit_compat(
                request_body, idempotency_key=idempotency_key
            )
        except Exception as exc:
            # APIError is the broker's structured submission error. Map it
            # to its declared HTTP status and stable reason code. Anything
            # else is an unexpected class — return 500 with a code, not a
            # bare aiohttp "Server got itself in trouble".
            if _BrokerAPIError is not None and isinstance(exc, _BrokerAPIError):
                http_status = int(getattr(exc, "status", 500) or 500)
                http_status = 400 if http_status < 400 else http_status
                code = str(getattr(exc, "code", "compat_error") or "compat_error")
                detail = str(exc) if exc else "broker api error"
                return web.json_response(
                    {"error": detail, "code": code}, status=http_status,
                )
            return web.json_response(
                {
                    "error": f"compat submit unexpected: {type(exc).__name__}: {exc!r}",
                    "code": "compat_submit_unexpected",
                },
                status=500,
            )
        if not wait_for_result:
            status_url = f"/v1/gemma/jobs/{job.job_id}"
            state = getattr(job, "state", "queued")
            state = getattr(state, "value", state)
            response = web.json_response(
                {
                    "job_id": job.job_id,
                    "state": str(state),
                    "status_url": status_url,
                    "cancel_url": status_url,
                },
                status=202,
            )
            response.headers["Location"] = status_url
            response.headers["X-Job-ID"] = job.job_id
            return response
        if request_body.get("stream") is True:
            if request is None:
                chunks = [
                    chunk
                    async for chunk in self.service.stream_result(
                        job.job_id, timeout=effective_timeout
                    )
                ]
                return web.Response(
                    body=b"".join(chunks),
                    content_type="text/event-stream",
                    headers={"X-Job-ID": job.job_id},
                )
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Job-ID": job.job_id,
                },
            )
            await response.prepare(request)
            async for chunk in self.service.stream_result(
                job.job_id, timeout=effective_timeout
            ):
                await response.write(chunk)
            await response.write_eof()
            return response
        try:
            terminal = await self.service.wait_for_terminal(
                job.job_id, timeout=effective_timeout
            )
        except TimeoutError:
            latest = self.service.status(job.job_id)
            accepted = bool(latest.accepted_boundary_crossed)
            return web.json_response(
                {
                    "error": "compatibility wait timed out",
                    "code": (
                        "outcome_pending_after_acceptance"
                        if accepted
                        else "compatibility_timeout"
                    ),
                    "job_id": job.job_id,
                    "status_url": f"/v1/gemma/jobs/{job.job_id}",
                    "retry_after": max(1, int(effective_timeout)),
                    "retry_safe": not accepted,
                    "timeout_used": effective_timeout,
                },
                status=504,
                headers={"X-Job-ID": job.job_id},
            )
        # `wait_for_terminal` returns the terminal payload dict directly
        # (per the runtime's ``_terminal_payload`` shape); it is the ``result``
        # shape the compat wrapper validates downstream.
        result = terminal if isinstance(terminal, dict) else {}
        if not {
            "status",
            "body",
            "content_type",
        }.issubset(result):
            return web.json_response(
                {"error": "invalid durable result", "code": "result_contract_invalid"},
                status=502,
            )
        response_body = result["body"]
        if isinstance(response_body, (dict, list)):
            encoded = json.dumps(
                response_body, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        elif isinstance(response_body, str):
            encoded = response_body.encode("utf-8")
        else:
            encoded = bytes(response_body)
        latest = self.service.status(job.job_id)
        response_headers = {"X-Job-ID": job.job_id}
        selected_member = getattr(latest, "selected_member", None)
        if selected_member:
            response_headers["X-Selected-Member"] = str(selected_member)
        attempt_id = getattr(latest, "current_attempt_id", None)
        if attempt_id:
            response_headers["X-Attempt-ID"] = str(attempt_id)
        return web.Response(
            body=encoded,
            status=int(result["status"]),
            content_type=str(result["content_type"]).split(";", 1)[0],
            headers=response_headers,
        )
