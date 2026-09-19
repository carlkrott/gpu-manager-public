from __future__ import annotations

import json

import pytest

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
        "Authorization: Bearer top-secret token",
        status=422,
        details={"password": "secret", "field": "name"},
    )
    assert envelope == {
        "error": {
            "code": "invalid_request",
            "message": "Authorization: Bearer <redacted> token",
            "details": {"password": "<redacted>", "field": "name"},
        },
        "status": 422,
    }
    assert "top-secret" not in json.dumps(envelope)
    assert "secret" not in json.dumps(envelope)
