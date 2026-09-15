"""Small, fail-closed HTTP authentication policy shared by public services."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from collections.abc import Mapping
from typing import Any

from aiohttp import web


PUBLIC_LIVENESS_PATHS = frozenset({"/health/live"})
PUBLIC_BOOTSTRAP_PATHS = frozenset({"/dashboard"})
WEBSOCKET_AUTH_PATH = "/ws"
LOOPBACK_COMPATIBILITY_ENV = "GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK"
TOKEN_FILE_ENV = "GPU_MANAGER_API_TOKEN_FILE"
CREDENTIALS_DIRECTORY_ENV = "CREDENTIALS_DIRECTORY"
TOKEN_FILE_NAME = "gpu-manager-api-token"
WS_AUTH_FRAME_TYPE = "authenticate"
WS_AUTH_TIMEOUT_SECONDS = 5.0
WS_MAX_AUTH_FRAME_BYTES = 8 * 1024
WS_CLOSE_POLICY_VIOLATION = 1008


def configured_token(environ: Mapping[str, str] | None = None) -> str | None:
    """Read a bounded token from an explicit file or systemd credentials dir."""
    env = os.environ if environ is None else environ
    path = env.get(TOKEN_FILE_ENV)
    if not path:
        credential_dir = env.get(CREDENTIALS_DIRECTORY_ENV)
        if credential_dir:
            path = os.path.join(credential_dir, TOKEN_FILE_NAME)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            token = handle.read(4097).strip()
    except (OSError, UnicodeError):
        return None
    if not token or len(token) > 4096:
        return None
    return token


def loopback_compatibility_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the explicitly requested legacy loopback bypass is on."""
    env = os.environ if environ is None else environ
    return env.get(LOOPBACK_COMPATIBILITY_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_loopback_peer(request: Any) -> bool:
    """Recognize only literal loopback peer addresses; never forwarded headers."""
    return getattr(request, "remote", None) in {"127.0.0.1", "::1"}


def bearer_matches(request: Any, expected: str | None) -> bool:
    """Validate the exact Bearer credential without exposing token material."""
    if expected is None:
        return False
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return False
    provided = authorization[len("Bearer ") :]
    return bool(provided) and hmac.compare_digest(provided, expected)


async def ws_enforce_first_frame_auth(ws: Any, expected_token: str | None = None) -> bool:
    """Require a bounded authenticate frame before any WebSocket data."""
    expected = configured_token() if expected_token is None else expected_token
    try:
        message = await asyncio.wait_for(
            ws.receive(), timeout=WS_AUTH_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        await ws.close(code=WS_CLOSE_POLICY_VIOLATION)
        return False
    if getattr(message, "type", None) != web.WSMsgType.TEXT:
        await ws.close(code=WS_CLOSE_POLICY_VIOLATION)
        return False
    raw = str(getattr(message, "data", ""))
    if len(raw.encode("utf-8")) > WS_MAX_AUTH_FRAME_BYTES:
        await ws.close(code=WS_CLOSE_POLICY_VIOLATION)
        return False
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        await ws.close(code=WS_CLOSE_POLICY_VIOLATION)
        return False
    token = payload.get("token") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("type") != WS_AUTH_FRAME_TYPE
        or not isinstance(token, str)
        or expected is None
        or not token
        or not hmac.compare_digest(token, expected)
    ):
        await ws.close(code=WS_CLOSE_POLICY_VIOLATION)
        return False
    await ws.send_json({"type": "auth_ok"})
    return True


def request_is_authorized(
    request: Any,
    *,
    expected: str | None = None,
    allow_loopback: bool | None = None,
) -> bool:
    """Authorize a protected request using bearer auth or explicit compatibility."""
    if allow_loopback is None:
        allow_loopback = loopback_compatibility_enabled()
    if allow_loopback and is_loopback_peer(request):
        return True
    if expected is None:
        expected = configured_token()
    return bearer_matches(request, expected)


def _auth_failure() -> web.Response:
    expected = configured_token()
    if expected is None:
        return web.json_response(
            {
                "error": "external API authentication is not configured",
                "code": "api_auth_not_configured",
            },
            status=503,
            headers={"Cache-Control": "no-store"},
        )
    return web.json_response(
        {"error": "authentication required", "code": "api_auth_required"},
        status=401,
        headers={
            "Cache-Control": "no-store",
            "WWW-Authenticate": "Bearer",
        },
    )


@web.middleware
async def api_auth_middleware(request: web.Request, handler):
    """Protect every route except the exact liveness response and OPTIONS metadata."""
    if request.method == "OPTIONS":
        return web.Response(
            status=204,
            headers={
                "Allow": "GET, HEAD, OPTIONS",
                "Cache-Control": "no-store",
            },
        )
    if (
        request.method == "GET"
        and request.path == WEBSOCKET_AUTH_PATH
        and request.headers.get("Upgrade", "").lower() == "websocket"
    ):
        return await handler(request)
    if request.method in {"GET", "HEAD"} and (
        request.path in PUBLIC_LIVENESS_PATHS
        or request.path in PUBLIC_BOOTSTRAP_PATHS
    ):
        return await handler(request)
    if request_is_authorized(request):
        return await handler(request)
    return _auth_failure()


__all__ = [
    "CREDENTIALS_DIRECTORY_ENV",
    "LOOPBACK_COMPATIBILITY_ENV",
    "PUBLIC_BOOTSTRAP_PATHS",
    "PUBLIC_LIVENESS_PATHS",
    "TOKEN_FILE_ENV",
    "TOKEN_FILE_NAME",
    "WEBSOCKET_AUTH_PATH",
    "WS_AUTH_FRAME_TYPE",
    "WS_AUTH_TIMEOUT_SECONDS",
    "WS_CLOSE_POLICY_VIOLATION",
    "WS_MAX_AUTH_FRAME_BYTES",
    "api_auth_middleware",
    "bearer_matches",
    "configured_token",
    "is_loopback_peer",
    "loopback_compatibility_enabled",
    "request_is_authorized",
    "ws_enforce_first_frame_auth",
]
