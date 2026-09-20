from __future__ import annotations

import json
import math

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
