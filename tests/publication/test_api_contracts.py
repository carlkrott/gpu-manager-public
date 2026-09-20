from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import api_contracts
from api_contracts import (
    MAX_JSON_BODY_BYTES,
    ContractError,
    bounded_int,
    bounded_string,
    bounded_string_list,
    error_envelope,
    is_loopback_url,
    is_safe_identifier,
    parse_json_object,
)


def test_parse_json_object_accepts_bounded_utf8_object():
    payload = parse_json_object('{"name": "café"}')
    assert payload == {"name": "café"}


def test_parse_json_object_rejects_duplicate_keys_at_every_nesting_level():
    for body in (
        '{"name": "first", "name": "second"}',
        '{"outer": {"name": "first", "name": "second"}}',
        '{"outer": [{"name": "first", "name": "second"}]}',
    ):
        with pytest.raises(ContractError, match="valid JSON required"):
            parse_json_object(body)


def test_parse_json_object_rejects_oversize_invalid_and_non_object_bodies():
    with pytest.raises(ContractError, match="request body is too large"):
        parse_json_object(b"{}" + b" " * MAX_JSON_BODY_BYTES)
    with pytest.raises(ContractError, match="valid JSON required"):
        parse_json_object(b"{")
    with pytest.raises(ContractError, match="JSON object required"):
        parse_json_object(json.dumps(["not", "an", "object"]))


def test_safe_identifier_rejects_paths_control_chars_and_wrong_types():
    assert is_safe_identifier("model.v1-2")
    for value in ("", "../model", "model/name", "model name", "model\x00", True, 3):
        assert not is_safe_identifier(value)


def test_loopback_url_requires_http_or_https_literal_loopback_without_credentials():
    accepted = (
        "http://127.0.0.1:8000/health",
        "https://localhost/api",
        "http://[::1]:9/",
    )
    rejected = (
        "http://127.0.0.2:8000/",
        "http://example.test/",
        "http://127.0.0.1.evil.test/",
        "http://user:pass@127.0.0.1/",
        "file:///tmp/service",
        "//127.0.0.1:8000/",
        "http://127.0.0.1:bad/",
    )
    assert all(is_loopback_url(value) for value in accepted)
    assert not any(is_loopback_url(value) for value in rejected)


def test_bounded_string_and_list_reject_booleans_and_boundaries():
    assert bounded_string(" ready ", field="state", max_length=8) == "ready"
    assert bounded_string_list(["a", " b "], field="tags", max_items=2) == ["a", "b"]
    assert bounded_int(2, field="attempt", minimum=1, maximum=3) == 2
    for value in (True, 1, "", "x" * 9):
        with pytest.raises(ContractError):
            bounded_string(value, field="state", max_length=8)
    with pytest.raises(ContractError):
        bounded_int(True, field="attempt", minimum=1, maximum=3)
    with pytest.raises(ContractError):
        bounded_string_list(["a", "b", "c"], field="tags", max_items=2)
    with pytest.raises(ContractError):
        bounded_string_list(["a", 1], field="tags")


def test_error_envelope_is_stable_and_redacts_sensitive_values():
    envelope = error_envelope(
        "invalid_request",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature token",
        status=422,
        details={"password": "top-secret", "field": "name"},
    )
    assert envelope == {
        "error": {
            "code": "invalid_request",
            "message": "Authorization: Bearer <redacted> token",
            "details": {"password": "<redacted>", "field": "name"},
        },
        "status": 422,
    }
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature" not in json.dumps(envelope)
    assert "top-secret" not in json.dumps(envelope)
    assert "<redacted>" in envelope["error"]["message"]


def test_error_envelope_rejects_non_mapping_details():
    # A list passed as details must surface as a clear boundary failure,
    # not silently leak through as a redacted list payload.
    for bad in (["a", "b"], "not-a-mapping", ("tuple", "of", "values"), 42):
        with pytest.raises(ContractError):
            error_envelope(
                "invalid_request",
                "details were the wrong shape",
                details=bad,  # type: ignore[arg-type]
            )


def test_redact_handles_bytes_keys_and_bounds_key_length():
    # Bytes keys decode safely rather than rendering as ``b'...'``.
    # Int keys are skipped to avoid leaking arbitrary repr noise.
    # Oversize string keys are truncated so they cannot amplify output.
    long_key = "x" * (api_contracts.MAX_REDACT_KEY_LENGTH + 50)
    payload = {
        b"Authorization": "Bearer raw-bytes-token-value",
        "session_id": "sess-raw-value",
        42: "ignored-int-key",
        long_key: "still-here",
        "field": "name",
    }
    safe = api_contracts._redact(payload)
    assert safe["Authorization"] == "<redacted>"
    assert safe["session_id"] == "<redacted>"
    assert 42 not in safe
    truncated_key = long_key[: api_contracts.MAX_REDACT_KEY_LENGTH]
    assert safe[truncated_key] == "still-here"
    assert truncated_key in safe
    assert len(safe) == 4  # bytes, session_id, long_key, field (int skipped)
    assert "raw-bytes-token-value" not in json.dumps(safe)
    assert "sess-raw-value" not in json.dumps(safe)


def test_redact_includes_session_cookie_csrf_xsrf_jwt_as_sensitive():
    payload = {
        "session": "sess-abc",
        "cookie": "k=v",
        "csrf_token": "csrf-abc",
        "xsrf_header": "xsrf-abc",
        "jwt": "jwt-abc",
        "benign": "keep",
    }
    safe = api_contracts._redact(payload)
    assert safe == {
        "session": "<redacted>",
        "cookie": "<redacted>",
        "csrf_token": "<redacted>",
        "xsrf_header": "<redacted>",
        "jwt": "<redacted>",
        "benign": "keep",
    }


def test_error_envelope_redacts_clear_bearer_token_to_redacted():
    # Use a distinctive, easily-spotted JWT-like token and assert the
    # redacted placeholder replaces it explicitly.
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
    envelope = error_envelope(
        "invalid_request",
        f"Authorization: Bearer {token}",
        status=401,
    )
    rendered = json.dumps(envelope)
    assert token not in rendered
    assert "<redacted>" in envelope["error"]["message"]
    assert envelope["error"]["message"].startswith("Authorization: Bearer <redacted>")


def test_error_envelope_replaces_non_finite_details_and_is_strict_json():
    envelope = error_envelope(
        "invalid_request",
        "bad input",
        details={
            "nan": math.nan,
            "nested": {"positive": math.inf, "negative": -math.inf},
            "items": [math.nan, {"value": math.inf}],
        },
    )

    rendered = json.dumps(envelope, allow_nan=False)
    assert "NaN" not in rendered
    assert "Infinity" not in rendered
    assert envelope["error"]["details"] == {
        "nan": "<redacted>",
        "nested": {"positive": "<redacted>", "negative": "<redacted>"},
        "items": ["<redacted>", {"value": "<redacted>"}],
    }


# ─────────────────────────────────────────────────────────────────────
# Wired behaviour: handle_submit_job + handle_submit_generation must
# route request reading through parse_json_object and return stable
# error envelopes BEFORE any queue side effect.
# ─────────────────────────────────────────────────────────────────────


_ROOT = Path(__file__).resolve().parents[2]
_CONTROLLER = _ROOT / "scripts" / "gpu-manager.py"


class _FakeProto:
    """Minimal stand-in for an asyncio protocol that aiohttp streams need."""

    def __init__(self) -> None:
        self._reading_paused = False

    def pause_reading(self) -> None:  # pragma: no cover - trivial
        pass

    def resume_reading(self) -> None:  # pragma: no cover - trivial
        pass


def _build_request(body: bytes, *, path: str = "/v1/submit") -> object:
    """Build an aiohttp Request whose raw body is the supplied bytes."""
    from aiohttp.streams import StreamReader
    from aiohttp.test_utils import make_mocked_request

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


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "candidate_gpu_manager_submit_contracts", _CONTROLLER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_submit_mocks(module) -> dict[str, MagicMock]:
    """Wire lightweight mocks so queue side effects can be observed.

    Returns the mocks so tests can assert they were not invoked for
    malformed / oversized input.  Valid requests intentionally fail
    later (no real Redis / routing state) — those assertions live in
    dedicated tests.
    """
    capacity_tracker = MagicMock(name="capacity_tracker")
    job_tracker = MagicMock(name="job_tracker")
    queue_engine = MagicMock(name="queue_engine")
    routing_engine = MagicMock(name="routing_engine")
    routing_engine.find_service_for_group.return_value = None
    routing_engine.get_group_members.return_value = []
    async_gossiper = MagicMock(name="async_gossiper")
    async_gossiper.find_remote_service = AsyncMock_shim(None)

    module.capacity_tracker = capacity_tracker
    module.job_tracker = job_tracker
    module.queue_engine = queue_engine
    module.routing_engine = routing_engine
    module.async_gossiper = async_gossiper
    module.session = MagicMock(name="session")
    module._services_config = {"services": {}, "generation_templates": {}}

    return {
        "capacity_tracker": capacity_tracker,
        "job_tracker": job_tracker,
        "queue_engine": queue_engine,
        "routing_engine": routing_engine,
        "async_gossiper": async_gossiper,
    }


class AsyncMock_shim:
    """Minimal async mock for the few coroutines the handler calls."""

    def __init__(self, return_value=None):
        self.return_value = return_value
        self.calls: list = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


def _envelope(payload: dict) -> tuple[int, dict]:
    """Decode a handler response into ``(status, body)``."""
    status = payload.status
    body = json.loads(payload.text)
    return status, body


def _assert_safe_envelope(status: int, body: dict, *, expected_code: str,
                          expected_status: int) -> None:
    """Stable envelope shape: top-level ``error.code`` + ``status`` fields."""
    assert status == expected_status, (status, body)
    assert set(body) == {"error", "status"}, body
    assert body["status"] == expected_status, body
    assert set(body["error"]) == {"code", "message"}, body
    assert body["error"]["code"] == expected_code, body
    assert isinstance(body["error"]["message"], str) and body["error"]["message"], body


def test_handle_submit_job_rejects_duplicate_keys_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b'{"name": "first", "name": "second"}')

    status, body = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_job_rejects_nan_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b'{"x": NaN}')

    status, body = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    # NaN must never be allowed to leak into the envelope payload.
    assert "NaN" not in json.dumps(body)
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_job_rejects_infinity_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b'{"x": Infinity}')

    status, body = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    assert "Infinity" not in json.dumps(body)
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_job_rejects_non_object_body_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b'["not", "an", "object"]')

    status, body = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, body, expected_code="object_required", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_job_rejects_malformed_json_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b"{unterminated")

    status, body = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_job_rejects_malformed_utf8_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    # 0xC3 0x28 is an invalid UTF-8 sequence.
    request = _build_request(b'{"x": "\xc3\x28"}')

    status, body = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_job_rejects_oversize_body_with_413_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    # Just above the public 1 MiB limit; well-formed JSON so size is the
    # only reason to refuse.
    body = b'{"x":"' + b"A" * (api_contracts.MAX_JSON_BODY_BYTES + 16) + b'"}'
    request = _build_request(body)

    status, payload = _envelope(asyncio.run(module.handle_submit_job(request)))

    _assert_safe_envelope(
        status, payload, expected_code="body_too_large", expected_status=413
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_rejects_duplicate_keys_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(
        b'{"type":"image","name":"first","name":"second"}',
        path="/v1/submit/generation",
    )

    status, body = _envelope(
        asyncio.run(module.handle_submit_generation(request))
    )

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_rejects_nan_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(
        b'{"type":"image","x": NaN}',
        path="/v1/submit/generation",
    )

    status, body = _envelope(
        asyncio.run(module.handle_submit_generation(request))
    )

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    assert "NaN" not in json.dumps(body)
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_rejects_non_object_body_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b'["array","body"]', path="/v1/submit/generation")

    status, body = _envelope(
        asyncio.run(module.handle_submit_generation(request))
    )

    _assert_safe_envelope(
        status, body, expected_code="object_required", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_rejects_malformed_json_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(b"{not-json", path="/v1/submit/generation")

    status, body = _envelope(
        asyncio.run(module.handle_submit_generation(request))
    )

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_rejects_malformed_utf8_with_stable_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    request = _build_request(
        b'{"type":"image","x": "\xc3\x28"}',
        path="/v1/submit/generation",
    )

    status, body = _envelope(
        asyncio.run(module.handle_submit_generation(request))
    )

    _assert_safe_envelope(
        status, body, expected_code="invalid_json", expected_status=400
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_rejects_oversize_body_with_413_envelope():
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    body = b'{"type":"image","x":"' + b"A" * (
        api_contracts.MAX_JSON_BODY_BYTES + 16
    ) + b'"}'
    request = _build_request(body, path="/v1/submit/generation")

    status, payload = _envelope(
        asyncio.run(module.handle_submit_generation(request))
    )

    _assert_safe_envelope(
        status, payload, expected_code="body_too_large", expected_status=413
    )
    for mock in mocks.values():
        mock.assert_not_called()


def test_handle_submit_generation_data_override_skips_json_reading():
    """data_override callers (MCP) must bypass request-body parsing entirely."""
    module = _load_controller()
    mocks = _install_submit_mocks(module)
    # A real Request that would explode if .read() or .json() were called.
    exploding_request = SimpleNamespace(
        read=lambda: (_ for _ in ()).throw(
            AssertionError("request.read must not run when data_override is set")
        ),
        json=lambda: (_ for _ in ()).throw(
            AssertionError("request.json must not run when data_override is set")
        ),
        headers={},
    )
    # Empty override — handler should still proceed through the rest of
    # the validation pipeline without touching the request body.  We only
    # assert that the request body was NOT read here.
    asyncio.run(
        module.handle_submit_generation(
            exploding_request, data_override={}
        )
    )
    # Sanity: nothing mutated shared queue state during the parse step.
    mocks["capacity_tracker"].assert_not_called()
