"""Phase 6 gate: chat-body model ``llm.chat`` / ``combined-llm`` is ``gemma``.

The alias is canonicalized once, in api_auth_middleware, before every scoped
check and before any handler forwards the body. These tests drive the real
middleware and prove what the broker receives. Synthetic tokens only.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from api_auth import (
    HONCHO_GEMMA_TOKEN_FILE_ENV,
    KDA_GEMMA_DEFAULT_PIN_APP_KEY,
    KDA_GEMMA_MEMBER_PINS_APP_KEY,
    KDA_GEMMA_SCOPED_GATE_APP_KEY,
    LLM_CHAT_ALIAS_ENV,
    api_auth_middleware,
    canonical_llm_chat_body,
    kda_gemma_backend_is_unpinned,
)
from combined_gemma_broker_service import create_standalone_app
from gemma_broker.api import DurableBrokerService
from gemma_broker.compat import CompatibilityWrapper
from gemma_broker.redis_store import InMemoryJobRepository
from test_scoped_gate_member_pins import _member, controller  # noqa: F401 (fixture)

ADMIN = "synthetic-admin-token"
KDA = "synthetic-scoped-token"
HONCHO = "synthetic-honcho-token"
MEMBERS = ("Gemma-CPU", "Qwen-Dual")
ASYNC = {"Prefer": "respond-async"}


@pytest.fixture
def tokens(tmp_path, monkeypatch):
    for name, value, env in (
        ("admin", ADMIN, "GPU_MANAGER_API_TOKEN_FILE"),
        ("kda", KDA, "GPU_MANAGER_KDA_GEMMA_TOKEN_FILE"),
        ("honcho", HONCHO, HONCHO_GEMMA_TOKEN_FILE_ENV),
    ):
        path = tmp_path / name
        path.write_text(value)
        path.chmod(0o600)
        monkeypatch.setenv(env, str(path))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    monkeypatch.delenv(LLM_CHAT_ALIAS_ENV, raising=False)


@pytest.fixture
def flag_on(monkeypatch, tokens):
    monkeypatch.setenv(LLM_CHAT_ALIAS_ENV, "1")


def _body(model) -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()


# --- unit: the canonicalizer itself ---

def test_canonical_body_flag_off_is_identity(tokens):
    raw = _body("llm.chat")
    assert canonical_llm_chat_body(raw) is raw


@pytest.mark.parametrize("model", ["llm.chat", "LLM.Chat", "combined-llm", "Combined-LLM", "COMBINED-LLM"])
def test_canonical_body_rewrites_aliases(flag_on, model):
    out = json.loads(canonical_llm_chat_body(_body(model)))
    assert out["model"] == "gemma" and out["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("raw", [
    _body("gemma"), _body("llm.chatx"), _body("gemma2"), _body("llm.chat "), _body("combined llm"),
    _body(42), _body(None), _body(["llm.chat"]), _body({"id": "llm.chat"}),
    b"", b"{", b"[]", b'"llm.chat"', b"\xff\xfe", b'{"messages":[]}',
])
def test_canonical_body_leaves_everything_else_byte_identical(flag_on, raw):
    assert canonical_llm_chat_body(raw) is raw


def test_canonical_body_duplicate_model_key_follows_the_parser(flag_on):
    # json.loads keeps the last key, as every downstream parser in gm does; the
    # rewrite emits a single model key, so no later parser can see a different one.
    out = canonical_llm_chat_body(b'{"model":"gemma","model":"llm.chat"}')
    assert out == b'{"model": "gemma"}'
    raw = b'{"model":"llm.chat","model":"gpt-6-luna"}'
    assert canonical_llm_chat_body(raw) is raw


def test_unpinned_backend_set_is_not_widened():
    for backend in (None, "Gemma", "gemma", "GEMMA"):
        assert kda_gemma_backend_is_unpinned(backend)
    for backend in ("llm.chat", "LLM.Chat", "combined-llm", "Combined LLM", "Qwen-Dual", ""):
        assert not kda_gemma_backend_is_unpinned(backend)


# --- controller: real middleware -> real handler -> real compat -> real broker ---

class _CaptureCompat:
    def __init__(self, inner):
        self.inner, self.bodies = inner, []

    def submit_compat(self, body, *, idempotency_key, pinned_member=None, **_kw):
        self.bodies.append((dict(body), pinned_member))
        payload = {"idempotency_key": idempotency_key, "request": body}
        if pinned_member:
            payload["pinned_member"] = pinned_member
        return self.inner.submit(payload)


def _controller_call(controller, monkeypatch, *, token, model=None, raw=None, backend=None):
    monkeypatch.setattr(controller, "_services_config", {
        "services": {"Gemma-CPU": {"enabled": True, "type": "llm_backend", "port": 18099,
                                   "routing_group": "Gemma"}},
        "combined_gemma_broker": {"enabled": True, "ordered_members": list(MEMBERS)},
    })
    repo = InMemoryJobRepository()
    members = [repo.put_member(_member(n, i)) for i, n in enumerate(MEMBERS)]
    capture = _CaptureCompat(DurableBrokerService(repo, members=lambda: members, clock=lambda: 2.0))
    forward = AsyncMock(return_value=web.json_response({"direct": True}))
    monkeypatch.setattr(controller, "_combined_gemma_broker", CompatibilityWrapper(capture, timeout=5.0))
    monkeypatch.setattr(controller, "_combined_gemma_deployment_config", lambda: {"mode": "embedded"})
    monkeypatch.setattr(controller, "_forward_llm", forward)
    monkeypatch.setattr(controller, "_model_validator", None)

    async def go():
        app = web.Application(middlewares=[api_auth_middleware])
        app[KDA_GEMMA_SCOPED_GATE_APP_KEY] = lambda: True
        app[KDA_GEMMA_MEMBER_PINS_APP_KEY] = lambda: frozenset(MEMBERS)
        app[KDA_GEMMA_DEFAULT_PIN_APP_KEY] = lambda: controller._SCOPED_DEFAULT_PIN
        app.router.add_post("/v1/chat/completions", controller.handle_llm_proxy)
        headers = {"Authorization": f"Bearer {token}", **ASYNC}
        if backend is not None:
            headers["X-Backend"] = backend
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions",
                                         data=raw if raw is not None else _body(model), headers=headers)
            return response.status, await response.json()

    status, payload = asyncio.run(go())
    return status, payload, capture.bodies, forward


@pytest.mark.parametrize("token,backend", [
    (HONCHO, None), (HONCHO, "Gemma"), (KDA, None), (KDA, "gemma"), (ADMIN, "Gemma"),
])
@pytest.mark.parametrize("model", ["llm.chat", "LLM.Chat", "combined-llm", "Combined-LLM"])
def test_flag_on_alias_reaches_broker_as_gemma_unpinned(controller, monkeypatch, flag_on, token, backend, model):
    status, payload, bodies, forward = _controller_call(
        controller, monkeypatch, token=token, model=model, backend=backend)
    assert status == 202, payload
    forward.assert_not_called()
    assert [(b["model"], pin) for b, pin in bodies] == [("gemma", None)]


@pytest.mark.parametrize("token", [HONCHO, KDA])
@pytest.mark.parametrize("model", ["llm.chat", "Combined-LLM"])
def test_flag_on_alias_with_member_pin_is_refused(controller, monkeypatch, flag_on, token, model):
    # Qwen-Dual IS a configured member, so the KDA refusal comes from the alias
    # rule, not from the member-set check.
    status, payload, bodies, forward = _controller_call(
        controller, monkeypatch, token=token, model=model, backend="Qwen-Dual")
    assert status == (401 if token == HONCHO else 403), payload
    if token == KDA:
        assert payload["code"] == "llm_chat_alias_pinned"
    assert bodies == [] and not forward.called


def test_flag_on_admin_alias_with_member_pin_is_refused(controller, monkeypatch, flag_on):
    status, payload, bodies, forward = _controller_call(
        controller, monkeypatch, token=ADMIN, model="llm.chat", backend="Gemma-CPU")
    assert status == 403 and payload["code"] == "llm_chat_alias_pinned"
    assert bodies == [] and not forward.called


def test_flag_on_gemma_with_member_pin_is_unchanged_for_kda(controller, monkeypatch, flag_on):
    status, payload, bodies, _ = _controller_call(
        controller, monkeypatch, token=KDA, model="gemma", backend="Qwen-Dual")
    assert status == 202, payload
    assert [(b["model"], pin) for b, pin in bodies] == [("gemma", "Qwen-Dual")]


@pytest.mark.parametrize("token", [HONCHO, KDA])
@pytest.mark.parametrize("backend", ["llm.chat", "LLM.Chat", "combined-llm", "Combined LLM"])
def test_flag_on_alias_as_x_backend_is_not_a_routing_group(controller, monkeypatch, flag_on, token, backend):
    status, _, bodies, forward = _controller_call(
        controller, monkeypatch, token=token, model="gemma", backend=backend)
    assert status == 401
    assert bodies == [] and not forward.called


@pytest.mark.parametrize("token", [HONCHO, KDA])
@pytest.mark.parametrize("model", ["llm.chatx", "gemma2", "llm.chat ", "combined llm", "", 42, None, ["llm.chat"], {"m": 1}])
def test_flag_on_unknown_or_non_string_model_is_refused(controller, monkeypatch, flag_on, token, model):
    status, _, bodies, forward = _controller_call(controller, monkeypatch, token=token, model=model)
    assert status == (401 if token == HONCHO else 403)
    assert bodies == [] and not forward.called


@pytest.mark.parametrize("token,expected", [(HONCHO, 401), (KDA, 403)])
def test_flag_off_alias_is_refused_as_today(controller, monkeypatch, tokens, token, expected):
    status, _, bodies, forward = _controller_call(controller, monkeypatch, token=token, model="llm.chat")
    assert status == expected
    assert bodies == [] and not forward.called


def test_flag_off_kda_member_pin_alias_passes_through_unchanged(controller, monkeypatch, tokens):
    # Today's behaviour, kept byte-for-byte: a KDA member pin is not model-checked.
    status, _, bodies, _ = _controller_call(
        controller, monkeypatch, token=KDA, model="llm.chat", backend="Qwen-Dual")
    assert status == 202
    assert [(b["model"], pin) for b, pin in bodies] == [("llm.chat", "Qwen-Dual")]


# --- standalone broker app: the same middleware feeds handle_proxy ---

def _standalone_call(*, token, raw, backend=None):
    service = MagicMock()
    service.settings.config.ordered_members = MEMBERS
    service.settings.config.enabled = True
    service.settings.registry = {}
    service.handle_proxy = AsyncMock(side_effect=lambda **_kw: web.json_response({"ok": True}))
    app = create_standalone_app(service)

    async def go():
        headers = {"Authorization": f"Bearer {token}"}
        if backend is not None:
            headers["X-Backend"] = backend
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", data=raw, headers=headers)
            return response.status, await response.json()

    status, payload = asyncio.run(go())
    calls = [c.kwargs for c in service.handle_proxy.await_args_list]
    return status, payload, calls


@pytest.mark.parametrize("token,backend,pin", [
    (HONCHO, None, None), (KDA, None, "Gemma"), (KDA, "Gemma", "Gemma"), (ADMIN, "Gemma", "Gemma"),
])
def test_standalone_flag_on_alias_forwards_gemma(flag_on, token, backend, pin):
    status, payload, calls = _standalone_call(token=token, raw=_body("LLM.Chat"), backend=backend)
    assert status == 200, payload
    assert len(calls) == 1
    assert json.loads(calls[0]["body"])["model"] == "gemma"
    assert calls[0]["headers"].get("X-Backend") == pin


@pytest.mark.parametrize("token,expected", [(HONCHO, 401), (KDA, 403), (ADMIN, 403)])
def test_standalone_flag_on_alias_member_pin_refused(flag_on, token, expected):
    status, _, calls = _standalone_call(token=token, raw=_body("llm.chat"), backend="Qwen-Dual")
    assert status == expected and calls == []


@pytest.mark.parametrize("token,expected", [(HONCHO, 401), (KDA, 403)])
@pytest.mark.parametrize("model", ["llm.chat", "llm.chatx", "gemma2", 7])
def test_standalone_scoped_refusals(tokens, monkeypatch, token, expected, model):
    if model != "llm.chat":
        monkeypatch.setenv(LLM_CHAT_ALIAS_ENV, "1")
    status, _, calls = _standalone_call(token=token, raw=_body(model))
    assert status == expected and calls == []


@pytest.mark.parametrize("flag,token,raw", [
    (None, ADMIN, b'{ "model" : "llm.chat",  "messages": [] }'),
    (None, ADMIN, b'{ "model" : "gemma",  "messages": [] }'),
    ("1", ADMIN, b'{ "model" : "gemma",  "messages": [] }'),
    (None, HONCHO, b'{ "model" : "Gemma",  "messages": [] }'),
    ("1", HONCHO, b'{ "model" : "Gemma",  "messages": [] }'),
])
def test_standalone_non_alias_and_flag_off_bodies_are_byte_identical(tokens, monkeypatch, flag, token, raw):
    if flag:
        monkeypatch.setenv(LLM_CHAT_ALIAS_ENV, flag)
    status, payload, calls = _standalone_call(token=token, raw=raw, backend="Gemma")
    assert status == 200, payload
    assert calls[0]["body"] == raw
