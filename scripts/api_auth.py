"""Small, fail-closed HTTP authentication policy shared by public services."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from typing import Any

from aiohttp import streams, web


PUBLIC_LIVENESS_PATHS = frozenset({"/health/live"})
PUBLIC_BOOTSTRAP_PATHS = frozenset({"/dashboard"})
WEBSOCKET_AUTH_PATH = "/ws"
LOOPBACK_COMPATIBILITY_ENV = "GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK"
TOKEN_FILE_ENV = "GPU_MANAGER_API_TOKEN_FILE"
CREDENTIALS_DIRECTORY_ENV = "CREDENTIALS_DIRECTORY"
TOKEN_FILE_NAME = "gpu-manager-api-token"
KDA_GEMMA_TOKEN_FILE_ENV = "GPU_MANAGER_KDA_GEMMA_TOKEN_FILE"
HONCHO_GEMMA_TOKEN_FILE_ENV = "GPU_MANAGER_HONCHO_GEMMA_TOKEN_FILE"
CATALOG_READ_TOKEN_FILE_ENV = "GPU_MANAGER_CATALOG_READ_TOKEN_FILE"
_CATALOG_READ_PATH = re.compile(r"/v1/catalog(?:/[a-z0-9][a-z0-9._-]*)?")
# Directory (0700, same UID) holding one 0600 token file per catalog client,
# named exactly as the client identity, e.g. ``zeroclaw-msi``.
CATALOG_CLIENT_TOKENS_DIR_ENV = "GPU_MANAGER_CATALOG_CLIENT_TOKENS_DIR"
CATALOG_CLIENTS = frozenset({
    "zeroclaw-msi", "zeroclaw-pi", "zeroclaw-macbook", "honcho", "t3", "hermes", "pibook",
})
_CATALOG_SUBSCRIBE_PATH = re.compile(r"/v1/catalog/[a-z0-9][a-z0-9._-]*/subscribe")
_CATALOG_HEARTBEAT_PATH = re.compile(r"/v1/catalog/subscriptions/[0-9a-f]{32}/heartbeat")
_CATALOG_SUBSCRIPTION_PATH = re.compile(r"/v1/catalog/subscriptions/[0-9a-f]{32}")
KDA_GEMMA_SCOPED_GATE_APP_KEY = "gpu_manager_kda_gemma_scoped_gate"
# Callable returning the Gemma routing group's member names (controller config,
# never a client header) that the scoped token may additionally pin.
KDA_GEMMA_MEMBER_PINS_APP_KEY = "gpu_manager_kda_gemma_member_pins"
# Callable returning ``(model, member)``: a scoped POST /v1/chat/completions that
# sends NO X-Backend (a client that cannot add headers) and whose body names
# ``model`` is pinned server-side to ``member``. Controller config, never client input.
KDA_GEMMA_DEFAULT_PIN_APP_KEY = "gpu_manager_kda_gemma_default_pin"
# The logical Gemma routing-group name. It names a routing group rather than a
# broker member, so it never pins a submission (gemma_broker/compat.py maps a
# backend that casefolds to "gemma" back to pinned_member=None).
KDA_GEMMA_ROUTING_GROUP = "Gemma"
# Phase 6 (catalog plan): clients may name the combined service by its catalog id.
# Flag-off by default; when on, the auth middleware rewrites these chat-body models
# to "gemma" once, before every check and before any handler forwards the body.
LLM_CHAT_ALIAS_ENV = "GPU_MANAGER_LLM_CHAT_ALIAS"
LLM_CHAT_MODEL_ALIASES = frozenset({"llm.chat", "combined-llm"})
WS_AUTH_FRAME_TYPE = "authenticate"
WS_AUTH_TIMEOUT_SECONDS = 5.0
WS_MAX_AUTH_FRAME_BYTES = 8 * 1024
WS_CLOSE_POLICY_VIOLATION = 1008
NODE_AUTH_METHOD_PATHS = frozenset({
    ("POST", "/v1/nodes/register"),
    ("POST", "/v1/nodes/heartbeat"),
    ("POST", "/v1/nodes/drain"),
})
NODE_FORWARDED_LLM_PATHS = frozenset({
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/responses"),
})


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


def llm_chat_alias_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether chat-body model aliases (``llm.chat``) map to ``gemma``."""
    env = os.environ if environ is None else environ
    return env.get(LLM_CHAT_ALIAS_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def canonical_llm_chat_body(body: bytes) -> bytes:
    """Return ``body`` with an aliased ``model`` rewritten to ``gemma``.

    Every other body (flag off, not JSON, not an object, non-string or unknown
    model) is returned as the very same object, so it stays byte-identical.
    """
    if not body or not llm_chat_alias_enabled():
        return body
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body
    model = payload.get("model") if isinstance(payload, dict) else None
    if not (isinstance(model, str) and model.casefold() in LLM_CHAT_MODEL_ALIASES):
        return body
    payload["model"] = "gemma"
    return json.dumps(payload).encode()


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


def _configured_kda_gemma_token() -> str | None:
    """Read a dedicated same-UID credential without following symlinks."""
    path = os.environ.get(KDA_GEMMA_TOKEN_FILE_ENV)
    if not path or not os.path.isabs(path):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(fd)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                    or metadata.st_mode & 0o077 or metadata.st_nlink != 1
                    or not 0 < metadata.st_size <= 8192):
                return None
            raw = os.read(fd, 8193)
        finally:
            os.close(fd)
    except OSError:
        return None
    if len(raw) != metadata.st_size:
        return None
    token = raw.removesuffix(b"\n")
    if not token or any(byte < 33 or byte > 126 for byte in token):
        return None
    return token.decode("ascii")


def _configured_honcho_gemma_token() -> str | None:
    """Read Honcho's dedicated credential with the same strict file policy as KDA."""
    return _read_scoped_token_file(HONCHO_GEMMA_TOKEN_FILE_ENV)


def _read_scoped_token_file(env_name: str) -> str | None:
    """Read a scoped same-UID credential file without following symlinks."""
    return _read_token_path(os.environ.get(env_name))


def _read_token_path(path: str | None) -> str | None:
    if not path or not os.path.isabs(path):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(fd)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                    or metadata.st_mode & 0o077 or metadata.st_nlink != 1
                    or not 0 < metadata.st_size <= 8192):
                return None
            raw = os.read(fd, 8193)
        finally:
            os.close(fd)
    except OSError:
        return None
    if len(raw) != metadata.st_size:
        return None
    token = raw.removesuffix(b"\n")
    if not token or any(byte < 33 or byte > 126 for byte in token):
        return None
    return token.decode("ascii")


def _scoped_default_pin(request: Any) -> tuple[str, str] | None:
    """Controller-configured (model, target) default pin.

    The target is a real member or the logical routing group; the group keeps the
    submission unpinned so the broker picks any compatible member.
    """
    provider = request.app.get(KDA_GEMMA_DEFAULT_PIN_APP_KEY)
    pin = provider() if callable(provider) else None
    if not (isinstance(pin, tuple) and len(pin) == 2 and all(isinstance(v, str) and v for v in pin)):
        return None
    members_provider = request.app.get(KDA_GEMMA_MEMBER_PINS_APP_KEY)
    members = members_provider() if callable(members_provider) else ()
    if pin[1].casefold() == KDA_GEMMA_ROUTING_GROUP.casefold():
        return pin if members else None
    return pin if pin[1] in set(members) else None


def kda_gemma_backend_is_unpinned(backend: Any) -> bool:
    """True when an X-Backend value leaves a submission UNPINNED.

    ``gemma_broker.compat`` maps a backend whose name casefolds to the logical
    routing group back to ``pinned_member=None``, so any compatible member may
    serve the request. This is the single source of truth for that rule, shared
    with the compat layer so the two can never disagree about which names select
    a routing group rather than a member.
    """
    return not isinstance(backend, str) or backend.casefold() == KDA_GEMMA_ROUTING_GROUP.casefold()


def scoped_gemma_pin_is_unpinned(request: Any) -> bool:
    """True when a scoped request's X-Backend leaves its submission UNPINNED.

    A missing header obviously leaves it unpinned, and so does the logical
    routing-group name: it selects a routing group, not a broker member, so
    ``gemma_broker.compat`` maps it back to ``pinned_member=None`` and any
    compatible member may serve the request. Both cases must therefore be held to
    the same model scope as an explicit member pin.
    """
    try:
        backend = request.headers.get("X-Backend")
    except AttributeError:
        return True
    # Fail closed: an unreadable or missing pin cannot be treated as a member pin.
    return kda_gemma_backend_is_unpinned(backend)


def scoped_gemma_default_member(request: Any, body: bytes | None) -> str | None:
    """Member for an already-authorized scoped request that would not pin itself.

    Returns ``None`` (caller must reject) unless the JSON body's ``model`` is the
    controller's default-pin model or its ``gemma`` alias; the member never comes
    from the client.
    """
    try:
        if not scoped_gemma_pin_is_unpinned(request):
            return None
        default_pin = _scoped_default_pin(request)
        if default_pin is None or not body:
            return None
        payload = json.loads(body)
    except (AttributeError, TypeError, ValueError):
        return None
    model, member = default_pin
    requested_model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(requested_model, str) and requested_model.casefold() in (model.casefold(), "gemma"):
        return member
    return None


def scoped_gemma_request_is_authorized(request: Any) -> bool:
    """Admit only KDA's loopback logical-Gemma probes and review request."""
    try:
        gate = request.app.get(KDA_GEMMA_SCOPED_GATE_APP_KEY)
        if not callable(gate) or not gate() or not is_loopback_peer(request):
            return False
        path = request.path
        if request.query_string or request.raw_path != path:
            return False
        if (request.method, path) not in {
            ("GET", "/health"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        }:
            return False
        backend = request.headers.get("X-Backend")
        if backend is None:
            # Header-less client: admitted only for the chat route, and only when the
            # controller names a default member; the handler enforces the model.
            default_pin = _scoped_default_pin(request)
            if default_pin is None or request.method != "POST" or path != "/v1/chat/completions":
                return False
        elif not kda_gemma_backend_is_unpinned(backend):
            members_provider = request.app.get(KDA_GEMMA_MEMBER_PINS_APP_KEY)
            members = members_provider() if callable(members_provider) else ()
            if not isinstance(backend, str) or backend not in set(members):
                return False
        token = _configured_kda_gemma_token()
        return bool(token and token != configured_token() and bearer_matches(request, token))
    except (AttributeError, TypeError, ValueError):
        return False


def honcho_gemma_body_is_in_scope(request: Any, body: bytes | None) -> bool:
    """True when a chat body names logical ``gemma`` and routes UNPINNED.

    Shared by every admission path that must accept Honcho's chat body, so the
    model and effective-routing rules cannot drift between the controller proxy
    and the standalone broker. An ``X-Backend`` that names a broker member (or
    anything else that is not the logical routing group) is a pin and is
    refused: Honcho's scope is the group, never a member.
    """
    try:
        if not kda_gemma_backend_is_unpinned(request.headers.get("X-Backend")):
            return False
        if not body:
            return False
        payload = json.loads(body)
        model = payload.get("model") if isinstance(payload, dict) else None
        return isinstance(model, str) and model.casefold() == "gemma"
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return False


def honcho_gemma_request_is_authorized(request: Any, body: bytes | None = None) -> bool:
    """Admit Honcho's distinct loopback-only logical Gemma and metrics scope."""
    try:
        gate = request.app.get(KDA_GEMMA_SCOPED_GATE_APP_KEY)
        if not callable(gate) or not gate() or not is_loopback_peer(request):
            return False
        path, query = request.path, request.query_string
        if request.raw_path != path + ("?" + query if query else ""):
            return False
        if request.method == "GET" and path in {"/health", "/v1/models"} and not query:
            pass
        elif request.method == "GET" and path == "/v1/metrics/broker" and query in {"", "durable=0"}:
            pass
        elif request.method == "POST" and path == "/v1/chat/completions" and not query:
            if not honcho_gemma_body_is_in_scope(request, body):
                return False
        else:
            return False
        token = _configured_honcho_gemma_token()
        kda_token = _configured_kda_gemma_token()
        return bool(token and token != configured_token() and token != kda_token
                    and bearer_matches(request, token))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return False


def catalog_read_request_is_authorized(request: Any) -> bool:
    """Admit the read-only catalog.read scope: loopback GET/HEAD /v1/catalog[/{id}]."""
    try:
        if request.method not in {"GET", "HEAD"} or not is_loopback_peer(request):
            return False
        if (request.query_string or request.raw_path != request.path
                or not _CATALOG_READ_PATH.fullmatch(request.path)):
            return False
        token = _read_scoped_token_file(CATALOG_READ_TOKEN_FILE_ENV)
        others = {configured_token(), _configured_kda_gemma_token(), _configured_honcho_gemma_token()}
        return bool(token and token not in others and bearer_matches(request, token))
    except (AttributeError, TypeError, ValueError):
        return False


def catalog_client_tokens() -> dict[str, str]:
    """Map client identity -> token from the per-client token directory.

    Fail closed: unset, relative, symlinked, foreign-owned or group/world
    accessible directories yield no identities. A token equal to another
    principal's token, or shared by two clients, identifies nobody.
    """
    directory = os.environ.get(CATALOG_CLIENT_TOKENS_DIR_ENV)
    if not directory or not os.path.isabs(directory):
        return {}
    try:
        metadata = os.lstat(directory)
    except OSError:
        return {}
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077):
        return {}
    tokens = {name: _read_token_path(os.path.join(directory, name)) for name in CATALOG_CLIENTS}
    tokens = {name: token for name, token in tokens.items() if token}
    others = {configured_token(), _configured_kda_gemma_token(), _configured_honcho_gemma_token(),
              _read_scoped_token_file(CATALOG_READ_TOKEN_FILE_ENV)}
    values = list(tokens.values())
    return {name: token for name, token in tokens.items()
            if token not in others and values.count(token) == 1}


def catalog_client_identity(request: Any) -> str | None:
    """The catalog client a loopback request's bearer identifies, else None."""
    try:
        if not is_loopback_peer(request):
            return None
        matches = [name for name, token in catalog_client_tokens().items()
                   if bearer_matches(request, token)]
        return matches[0] if len(matches) == 1 else None
    except (AttributeError, TypeError, ValueError):
        return None


def catalog_client_request_is_authorized(request: Any) -> bool:
    """Admit a catalog client: catalog read plus subscribe/heartbeat/release.

    Lease ownership is checked by the handlers, which resolve the identity from
    the bearer again (never from the body).
    """
    try:
        if request.query_string or request.raw_path != request.path:
            return False
        method, path = request.method, request.path
        if not ((method in {"GET", "HEAD"} and _CATALOG_READ_PATH.fullmatch(path))
                or (method == "POST" and (_CATALOG_SUBSCRIBE_PATH.fullmatch(path)
                                          or _CATALOG_HEARTBEAT_PATH.fullmatch(path)))
                or (method == "DELETE" and _CATALOG_SUBSCRIPTION_PATH.fullmatch(path))):
            return False
        return catalog_client_identity(request) is not None
    except (AttributeError, TypeError, ValueError):
        return False


def _honcho_may_need_chat_body(request: Any) -> bool:
    """True when this request can only be admitted by the Honcho chat scope.

    Only the Honcho principal decides admission from the request body, so only
    its own bearer may trigger the body read. ``BaseRequest.read()`` drains the
    payload and leaves ``can_read_body`` False, so an eager read for every
    Bearer caller would strip the body from every other principal's handler.
    """
    if request.method != "POST" or request.path != "/v1/chat/completions":
        return False
    if not request.headers.get("Authorization", "").startswith("Bearer "):
        return False
    token = _configured_honcho_gemma_token()
    return bool(token and bearer_matches(request, token))


async def _read_chat_body_preserving_request(request: Any) -> tuple[bytes, bool]:
    """Read a chat body for admission and leave it readable by the handler.

    Returns ``(body, aliased)``: an ``llm.chat`` model alias is canonicalized here,
    once, so admission and the route handler see the same canonical bytes.
    aiohttp exposes no public way to rewind a consumed payload, so an equivalent
    in-memory stream is armed over the same bytes.
    """
    if not request.can_read_body:
        return b"", False
    raw = await request.read()
    body = canonical_llm_chat_body(raw)
    # read() caches its bytes; the handler's own read() must return the canonical ones.
    request._read_bytes = body
    original = getattr(request, "_payload", None)
    protocol = getattr(original, "_protocol", None)
    if protocol is not None:
        replay = streams.StreamReader(
            protocol, getattr(original, "_limit", 2**16),
            loop=getattr(original, "_loop", None),
        )
        replay.feed_data(body)
        replay.feed_eof()
        request._payload = replay
    return body, body is not raw


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
    body: bytes | None = None,
) -> bool:
    """Authorize a protected request using bearer auth or explicit compatibility."""
    if allow_loopback is None:
        allow_loopback = loopback_compatibility_enabled()
    if allow_loopback and is_loopback_peer(request):
        return True
    if expected is not None:
        return bearer_matches(request, expected)
    return (bearer_matches(request, configured_token())
            or scoped_gemma_request_is_authorized(request)
            or honcho_gemma_request_is_authorized(request, body)
            or catalog_read_request_is_authorized(request)
            or catalog_client_request_is_authorized(request))


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
    # Node protocol routes have their own principal-specific bearer check in
    # NodeAPI. Do not require a second, unrelated controller API token there.
    if ((request.method, request.path) in NODE_AUTH_METHOD_PATHS
            and request.app.get("gpu_manager_node_api") is not None):
        return await handler(request)
    if ((request.method, request.path) in NODE_FORWARDED_LLM_PATHS
            and request.headers.get("X-Node-Forwarded") == "1"):
        node_api = request.app.get("gpu_manager_node_api")
        if node_api is not None and await node_api.authorize_forwarded(request):
            return await handler(request)
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
    body: bytes | None = None
    aliased = False
    if _honcho_may_need_chat_body(request):
        try:
            body, aliased = await _read_chat_body_preserving_request(request)
        except (ValueError, asyncio.TimeoutError):
            return _auth_failure()
    if (request.method == "GET" and request.path == "/v1/metrics/broker"
            and request.query_string in {"", "durable=0"}
            and request.raw_path == request.path + ("?" + request.query_string if request.query_string else "")
            and is_loopback_peer(request)):
        kda_token = _configured_kda_gemma_token()
        if (kda_token and bearer_matches(request, kda_token)
                and not bearer_matches(request, configured_token())):
            return web.json_response(
                {"error": "KDA scoped token cannot access broker metrics", "code": "scope_forbidden"},
                status=403, headers={"Cache-Control": "no-store"},
            )
    if request_is_authorized(request, body=body):
        if (body is None and llm_chat_alias_enabled()
                and request.method == "POST" and request.path == "/v1/chat/completions"):
            try:
                body, aliased = await _read_chat_body_preserving_request(request)
            except (ValueError, asyncio.TimeoutError):
                return _auth_failure()
        if aliased and not (kda_gemma_backend_is_unpinned(request.headers.get("X-Backend"))
                            and kda_gemma_backend_is_unpinned(request.query.get("service"))):
            # llm.chat names the logical combined service; it never pins a member.
            return web.json_response(
                {"error": "model alias llm.chat cannot be pinned to a member",
                 "code": "llm_chat_alias_pinned"},
                status=403, headers={"Cache-Control": "no-store"},
            )
        return await handler(request)
    return _auth_failure()


__all__ = [
    "CATALOG_CLIENTS",
    "CATALOG_CLIENT_TOKENS_DIR_ENV",
    "CATALOG_READ_TOKEN_FILE_ENV",
    "CREDENTIALS_DIRECTORY_ENV",
    "KDA_GEMMA_DEFAULT_PIN_APP_KEY",
    "KDA_GEMMA_MEMBER_PINS_APP_KEY",
    "KDA_GEMMA_ROUTING_GROUP",
    "KDA_GEMMA_SCOPED_GATE_APP_KEY",
    "KDA_GEMMA_TOKEN_FILE_ENV",
    "LLM_CHAT_ALIAS_ENV",
    "LLM_CHAT_MODEL_ALIASES",
    "HONCHO_GEMMA_TOKEN_FILE_ENV",
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
    "canonical_llm_chat_body",
    "catalog_client_identity",
    "catalog_client_request_is_authorized",
    "catalog_client_tokens",
    "catalog_read_request_is_authorized",
    "configured_token",
    "is_loopback_peer",
    "honcho_gemma_request_is_authorized",
    "honcho_gemma_body_is_in_scope",
    "kda_gemma_backend_is_unpinned",
    "llm_chat_alias_enabled",
    "loopback_compatibility_enabled",
    "request_is_authorized",
    "scoped_gemma_default_member",
    "scoped_gemma_pin_is_unpinned",
    "scoped_gemma_request_is_authorized",
    "ws_enforce_first_frame_auth",
]
