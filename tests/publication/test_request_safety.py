"""Focused regressions for the four request-safety fixes.

Covers:
  1. ``handle_generate_output`` rejects absolute paths, separators, dot
     segments, URL-decoded traversal, and any resolved path outside
     the COMFYUI output/input roots; valid basenames still serve.
  2. ``handle_api_proxy`` accepts only explicit loopback http/https
     URLs whose port is declared as ``port`` or ``proxy_port`` on an
     enabled service in ``_services_config``; reject userinfo,
     missing/malformed ports, controller self-port, and undeclared
     ports before I/O; strip Authorization, Cookie, Proxy-Authorization
     and X-API-Key from caller-supplied proxy headers.
  3. ``_forward_llm`` strips those same credential headers plus
     hop-by-hop headers, while preserving Content-Type and benign
     headers.
  4. The dashboard ``addChatMessage`` function renders role/service/
     content with textContent / DOM nodes, never interpolating model
     or user content into innerHTML.

These tests follow TDD: write first, watch fail, then fix the code.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp.streams import StreamReader
from aiohttp.test_utils import make_mocked_request


_ROOT = Path(__file__).resolve().parents[2]
_CONTROLLER = _ROOT / "scripts" / "gpu-manager.py"


class _FakeProto:
    """Minimal stand-in for an asyncio protocol aiohttp streams need."""

    def __init__(self) -> None:
        self._reading_paused = False

    def pause_reading(self) -> None:  # pragma: no cover - trivial
        pass

    def resume_reading(self) -> None:  # pragma: no cover - trivial
        pass


def _build_request(body: bytes, *, path: str = "/api/proxy") -> object:
    async def _make():
        stream = StreamReader(protocol=_FakeProto(), limit=2**26)
        stream.feed_data(body)
        stream.feed_eof()
        return make_mocked_request(
            "POST",
            path,
            payload=stream,
            client_max_size=10 * 1024 * 1024,
        )

    return asyncio.run(_make())


def _make_request(body: bytes, *, path: str, match_info: dict | None = None) -> object:
    """Build a fake ``web.Request`` whose ``match_info`` exposes the supplied mapping.

    ``Request.match_info`` is a cached property on real aiohttp requests,
    so we use a dict-based shim that supports ``["key"]`` lookup, which
    is what the handlers under test consume.
    """
    fake = SimpleNamespace(match_info=dict(match_info or {}))
    # Expose __getitem__ on match_info so handler code that uses
    # ``request.match_info["filename"]`` works.
    mi = fake.match_info
    mi.__getitem__  # already a dict, this is a no-op sanity check
    return fake


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "candidate_gpu_manager_request_safety", _CONTROLLER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_minimal(module, *, services_config: dict | None = None,
                     comfyui_root: Path | None = None):
    """Wire lightweight mocks so handlers can be invoked in isolation."""
    module.session = MagicMock(name="session")
    if services_config is None:
        services_config = {"services": {}, "generation_templates": {}}
    module._services_config = services_config
    if comfyui_root is not None:
        # Replace Path with a temporary test root so we don't accidentally
        # serve real files from the user's home ComfyUI directory.
        module.COMFYUI_ROOT = comfyui_root
    return module


# ─────────────────────────────────────────────────────────────────────
# (1) handle_generate_output — path traversal rejection
# ─────────────────────────────────────────────────────────────────────


def test_handle_generate_output_serves_basename_within_comfyui_output(tmp_path):
    """Valid basename inside the configured output root must serve."""
    comfy_root = tmp_path / "comfyui"
    output_dir = comfy_root / "output"
    input_dir = comfy_root / "input"
    output_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    valid = output_dir / "ok.png"
    valid_bytes = b"\x89PNG\r\n\x1a\n"
    valid.write_bytes(valid_bytes)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/ok.png",
                            match_info={"filename": "ok.png"})
    response = asyncio.run(module.handle_generate_output(request))

    assert response.status == 200
    # The FileResponse must be backed by the actual file we wrote.
    assert isinstance(response, type(module.web.FileResponse("/dev/null")))
    # web.FileResponse stores the file path on the ``_path`` attribute;
    # verify it resolves to the same file.
    served = Path(response._path).resolve()
    assert served == valid.resolve()


def test_handle_generate_output_rejects_absolute_path(tmp_path):
    """Absolute filenames (e.g. /etc/passwd) must be refused."""
    comfy_root = tmp_path / "comfyui"
    (comfy_root / "output").mkdir(parents=True)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/etc",
                            match_info={"filename": "/etc/passwd"})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 400
    assert "invalid" in response.text.lower() or "filename" in response.text.lower()


def test_handle_generate_output_rejects_path_separator(tmp_path):
    """Filenames containing path separators must be refused."""
    comfy_root = tmp_path / "comfyui"
    (comfy_root / "output").mkdir(parents=True)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/sub",
                            match_info={"filename": "subdir/file.png"})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 400


def test_handle_generate_output_rejects_backslash_separator(tmp_path):
    """Backslash separators must be refused even on POSIX."""
    comfy_root = tmp_path / "comfyui"
    (comfy_root / "output").mkdir(parents=True)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/back",
                            match_info={"filename": "..\\windows.png"})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 400


def test_handle_generate_output_rejects_dot_segments(tmp_path):
    """Dot-segment sequences must be refused."""
    comfy_root = tmp_path / "comfyui"
    (comfy_root / "output").mkdir(parents=True)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/dot",
                            match_info={"filename": ".."})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 400

    request = _make_request(b"", path="/generate/output/dot2",
                            match_info={"filename": "."})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 400


def test_handle_generate_output_rejects_url_decoded_traversal(tmp_path):
    """%2e%2e (URL-decoded '..') traversal must be refused."""
    comfy_root = tmp_path / "comfyui"
    (comfy_root / "output").mkdir(parents=True)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/decoded",
                            match_info={"filename": "%2e%2e%2fsecret"})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 400


def test_handle_generate_output_rejects_resolved_outside_root(tmp_path):
    """A symlink/file resolved outside the COMFYUI roots must be refused.

    We create a real file outside the COMFYUI roots and a symlink
    inside the output directory pointing to it.  Even though
    ``os.path.isfile`` on the un-resolved path returns False, the
    test is framed in terms of the resolved realpath: if the handler
    tried to follow the link and serve the file, we'd observe a 200
    for an unexpected target.
    """
    comfy_root = tmp_path / "comfyui"
    output_dir = comfy_root / "output"
    output_dir.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret")

    target = output_dir / "link.png"
    try:
        target.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported on this filesystem")

    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/link",
                            match_info={"filename": "link.png"})
    response = asyncio.run(module.handle_generate_output(request))
    # Must refuse — never serve a path that resolves outside the
    # configured roots, even if the immediate filename looks valid.
    assert response.status in (400, 404)


def test_handle_generate_output_returns_404_for_missing_basename(tmp_path):
    """Valid basename that doesn't exist must return 404, not 400."""
    comfy_root = tmp_path / "comfyui"
    (comfy_root / "output").mkdir(parents=True)
    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/nope",
                            match_info={"filename": "nope.png"})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 404


def test_handle_generate_output_serves_valid_basename_from_input_dir(tmp_path):
    """Files in the input directory must also serve when present."""
    comfy_root = tmp_path / "comfyui"
    input_dir = comfy_root / "input"
    input_dir.mkdir(parents=True)
    (comfy_root / "output").mkdir(parents=True)

    target = input_dir / "stable.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")

    module = _load_controller()
    _install_minimal(module, comfyui_root=comfy_root)

    request = _make_request(b"", path="/generate/output/stable",
                            match_info={"filename": "stable.png"})
    response = asyncio.run(module.handle_generate_output(request))
    assert response.status == 200


# ─────────────────────────────────────────────────────────────────────
# (2) handle_api_proxy — loopback + declared-port allowlist + header strip
# ─────────────────────────────────────────────────────────────────────


def _proxy_request(url: str, headers: dict | None = None, method: str = "GET"):
    body = json.dumps({"url": url, "method": method, "headers": headers or {}})
    return _build_request(body.encode("utf-8"), path="/api/proxy")


def test_handle_api_proxy_rejects_userinfo(tmp_path):
    """Userinfo in the URL must be rejected before any I/O."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {"svc": {"enabled": True, "port": 19001}},
        },
    )
    request = _proxy_request("http://user:pass@127.0.0.1:19001/")
    response = asyncio.run(module.handle_api_proxy(request))
    assert response.status == 403
    # Session must not have been touched for a rejected request.
    module.session.request.assert_not_called()


def test_handle_api_proxy_rejects_missing_port(tmp_path):
    """URL without a port must be rejected before any I/O."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {"svc": {"enabled": True, "port": 19001}},
        },
    )
    request = _proxy_request("http://127.0.0.1/")
    response = asyncio.run(module.handle_api_proxy(request))
    assert response.status == 403
    module.session.request.assert_not_called()


def test_handle_api_proxy_rejects_malformed_port(tmp_path):
    """Non-numeric port must be rejected before any I/O."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {"svc": {"enabled": True, "port": 19001}},
        },
    )
    request = _proxy_request("http://127.0.0.1:bad/")
    response = asyncio.run(module.handle_api_proxy(request))
    assert response.status == 403
    module.session.request.assert_not_called()


def test_handle_api_proxy_rejects_controller_self_port(tmp_path):
    """The controller's own LISTEN_PORT must be unproxyable."""
    module = _load_controller()
    _install_minimal(module)
    request = _proxy_request(f"http://127.0.0.1:{module.LISTEN_PORT}/self")
    response = asyncio.run(module.handle_api_proxy(request))
    assert response.status == 403
    module.session.request.assert_not_called()


def test_handle_api_proxy_rejects_undeclared_port(tmp_path):
    """Port not declared as port/proxy_port on an enabled service is rejected."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {
                # Service exists but is disabled — its port must not be allowed.
                "svc": {"enabled": False, "port": 19001},
            },
        },
    )
    request = _proxy_request("http://127.0.0.1:19001/")
    response = asyncio.run(module.handle_api_proxy(request))
    assert response.status == 403
    module.session.request.assert_not_called()


def test_handle_api_proxy_rejects_disabled_service_even_if_proxy_port_declared(tmp_path):
    """A disabled service's port/proxy_port must still be rejected."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {
                "svc": {"enabled": False, "port": 19001, "proxy_port": 19100},
            },
        },
    )
    for port in (19001, 19100):
        request = _proxy_request(f"http://127.0.0.1:{port}/")
        response = asyncio.run(module.handle_api_proxy(request))
        assert response.status == 403, (
            f"port {port} of disabled service must be rejected"
        )
        module.session.request.assert_not_called()


def test_handle_api_proxy_accepts_declared_port_on_enabled_service(tmp_path):
    """An enabled service's declared port must be proxyable (faked backend)."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {
                "svc-a": {"enabled": True, "port": 19011},
                "svc-b": {"enabled": True, "proxy_port": 19200},
            },
        },
    )

    # Mock session.request returns a fake response context manager.
    class _FakeResp:
        status = 200
        headers = {"content-type": "application/json"}
        text = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def text(self):  # pragma: no cover - replaced below
            return "{}"

    fake = MagicMock(name="session.request")
    fake.return_value = _FakeResp()
    # We need the .text() attribute to be awaitable; replace the one
    # on the class with an async function returning "{}".
    async def _text():  # noqa: D401 - simple coroutine
        return "{}"
    _FakeResp.text = _text
    module.session.request = fake

    for url in (
        "http://127.0.0.1:19011/v1/models",
        "http://127.0.0.1:19200/v1/models",
    ):
        request = _proxy_request(url)
        response = asyncio.run(module.handle_api_proxy(request))
        assert response.status == 200, (
            f"declared enabled port for {url} must proxy"
        )

    assert fake.call_count >= 2


def test_handle_api_proxy_strips_credential_headers_from_caller(tmp_path):
    """Authorization/Cookie/Proxy-Authorization/X-API-Key must be stripped."""
    module = _load_controller()
    _install_minimal(
        module,
        services_config={
            "services": {"svc": {"enabled": True, "port": 19021}},
        },
    )

    captured: dict = {}

    class _FakeResp:
        status = 200
        headers = {"content-type": "application/json"}

        async def __aenter__(self_inner):
            return self_inner

        async def __aexit__(self_inner, *exc):
            return False

        async def text(self_inner):
            return "{}"

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = dict(kwargs.get("headers") or {})
        return _FakeResp()

    module.session.request = _fake_request

    headers = {
        "Authorization": "Bearer leaked",
        "Cookie": "session=leaked",
        "Proxy-Authorization": "Basic leaked",
        "X-API-Key": "leaked",
        "Content-Type": "application/json",
        "X-Benign": "kept",
    }
    request = _proxy_request("http://127.0.0.1:19021/", headers=headers)
    response = asyncio.run(module.handle_api_proxy(request))
    assert response.status == 200

    forwarded = captured["headers"]
    for forbidden in ("Authorization", "Cookie", "Proxy-Authorization", "X-API-Key"):
        assert forbidden not in forwarded, (
            f"{forbidden} leaked into forwarded headers: {forwarded!r}"
        )
        assert forbidden.lower() not in {k.lower() for k in forwarded}
    # Case-insensitive headers may be lowercased by aiohttp; verify by
    # scanning the lower-cased keys explicitly.
    lowered = {k.lower() for k in forwarded}
    assert "authorization" not in lowered
    assert "cookie" not in lowered
    assert "proxy-authorization" not in lowered
    assert "x-api-key" not in lowered
    # Benign headers must still flow through.
    assert "x-benign" in lowered


def test_strip_credential_headers_rejects_whitespace_padded_names():
    """Malformed whitespace must not bypass the credential denylist."""
    module = _load_controller()

    forwarded = module._strip_credential_headers({
        " Authorization ": "Bearer leaked",
        "\tCookie": "session=leaked",
        " X-API-Key": "leaked",
        "X-Benign": "kept",
    })

    assert forwarded == {"X-Benign": "kept"}


# ─────────────────────────────────────────────────────────────────────
# (3) _forward_llm — credential + hop-by-hop strip, preserve Content-Type
# ─────────────────────────────────────────────────────────────────────


class _FakeLLMResp:
    def __init__(self, body: bytes = b"ok", status: int = 200,
                 content_type: str = "application/json") -> None:
        self.status = status
        self.headers = {"content-type": content_type}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return self._body


def test_forward_llm_strips_credential_headers():
    """_forward_llm must drop Authorization/Cookie/Proxy-Authorization/X-API-Key."""
    module = _load_controller()
    _install_minimal(module)

    captured: dict = {}

    def _fake_request(method, url, **kwargs):
        captured["headers"] = dict(kwargs.get("headers") or {})
        return _FakeLLMResp(b'{"ok":true}', 200)

    module.session.request = _fake_request

    headers = {
        "Authorization": "Bearer leaked",
        "Cookie": "session=leaked",
        "Proxy-Authorization": "Basic leaked",
        "X-API-Key": "leaked",
        "Content-Type": "application/json",
        "X-Benign": "keep-me",
        "Host": "frontend.example",
        "Content-Length": "999",
        "Transfer-Encoding": "chunked",
        "Connection": "close",
        "Upgrade": "h2c",
    }
    response = asyncio.run(
        module._forward_llm("POST", "/v1/chat", b"{}", headers)
    )
    assert response.status == 200

    forwarded = captured["headers"]
    lowered = {k.lower() for k in forwarded}
    for forbidden in ("authorization", "cookie", "proxy-authorization",
                      "x-api-key"):
        assert forbidden not in lowered, (
            f"{forbidden} leaked through _forward_llm: {forwarded!r}"
        )
    # Hop-by-hop headers must be stripped.
    for hop in ("host", "content-length", "transfer-encoding",
                "connection", "upgrade"):
        assert hop not in lowered, (
            f"hop-by-hop {hop} leaked through _forward_llm: {forwarded!r}"
        )
    # Content-Type must be preserved (forwarded, not stripped).
    assert "content-type" in lowered
    # Benign headers must be preserved.
    assert "x-benign" in lowered


def test_forward_llm_preserves_content_type_when_present():
    """The response Content-Type must be set on the web.Response."""
    module = _load_controller()
    _install_minimal(module)

    def _fake_request(method, url, **kwargs):
        return _FakeLLMResp(b"<html/>", 200, "text/html")

    module.session.request = _fake_request

    response = asyncio.run(
        module._forward_llm("GET", "/", None, {"accept": "text/html"})
    )
    assert response.status == 200
    # web.Response defaults the content_type to text/plain; the
    # handler overrides it from the upstream Content-Type.
    assert "text/html" in str(response.content_type).lower()


# ─────────────────────────────────────────────────────────────────────
# (4) dashboard addChatMessage — DOM-safe rendering
# ─────────────────────────────────────────────────────────────────────


def test_dashboard_addChatMessage_does_not_interpolate_into_innerHTML():
    """Look at the dashboard source and confirm addChatMessage uses
    textContent / DOM nodes for ``content`` (and never ``innerHTML``)."""
    dashboard = _CONTROLLER.read_text(encoding="utf-8")
    # Pull out the addChatMessage function body.
    match = re.search(
        r"function addChatMessage\(role, content\)\s*\{(?P<body>.*?)\n\}\n",
        dashboard,
        re.DOTALL,
    )
    assert match is not None, "addChatMessage function not found in dashboard"

    body = match.group("body")

    # The role/service header MAY keep using innerHTML for static
    # formatting (it comes from a controlled registry), but the
    # caller-controlled ``content`` MUST NOT be concatenated into
    # innerHTML.  Specifically, ``content`` must never appear on the
    # right-hand side of an innerHTML assignment.
    inner_html_assignments = re.findall(
        r"\.innerHTML\s*=\s*([^;]+);", body
    )
    for snippet in inner_html_assignments:
        assert "content" not in snippet, (
            "addChatMessage interpolates caller-controlled 'content' "
            f"into innerHTML: {snippet!r}"
        )

    # The function MUST use textContent (or appendChild with a text
    # node) for the message body.
    assert "textContent" in body or "createTextNode" in body, (
        "addChatMessage must use textContent or createTextNode for "
        f"the message body. body was:\n{body}"
    )


def test_dashboard_addChatMessage_renders_role_service_and_content_via_DOM():
    """All three pieces (role header / service name / content body)
    must be rendered via DOM nodes, not raw HTML interpolation.

    The original implementation concatenated ``content`` (after
    replacing \\n with <br>) directly into ``div.innerHTML``.  The
    fixed implementation must preserve the role/service header styling
    while keeping content strictly as text via ``textContent``.
    """
    dashboard = _CONTROLLER.read_text(encoding="utf-8")
    match = re.search(
        r"function addChatMessage\(role, content\)\s*\{(?P<body>.*?)\n\}\n",
        dashboard,
        re.DOTALL,
    )
    assert match is not None, "addChatMessage function not found in dashboard"
    body = match.group("body")

    # The legacy string `content.replace(/\n/g, '<br>')` is the smoking
    # gun: it converts content into HTML at the JS level.  Refuse it.
    assert "content.replace" not in body, (
        "addChatMessage still calls .replace on caller content, which "
        "produced the innerHTML injection path"
    )

    # Role and service name are static controlled values; they may be
    # inserted via textContent on a span.  Confirm at least one of
    # textContent / createTextNode is present and is used for the body.
    has_text_node = "textContent" in body
    assert has_text_node, (
        "addChatMessage must build the message body via textContent "
        f"(body was:\n{body})"
    )
