"""Small, dependency-free validators for public HTTP boundaries.

This module only handles transport-shaped values.  Registry and workflow
semantics remain in their respective contract modules so the HTTP layer does
not grow a second schema implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlsplit


MAX_JSON_BODY_BYTES = 1_048_576
MAX_IDENTIFIER_LENGTH = 128
MAX_STRING_LENGTH = 256
MAX_LIST_ITEMS = 100
MAX_ERROR_MESSAGE_LENGTH = 256
MAX_REDACT_KEY_LENGTH = 64

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUTH_VALUE_RE = re.compile(r"(?i)(\bAuthorization\s*:\s*)(Bearer\s+)?[^\s,;&]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+\S+")
_SECRET_VALUE_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|password|secret|token|credential)\s*[=:]\s*)([^\s,;&]+)"
)
_SECRET_KEYS = frozenset(
    {
        "access_key",
        "api_key",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "csrf",
        "csrf_token",
        "jwt",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session",
        "token",
        "xsrf",
        "xsrf_token",
    }
)


class ContractError(ValueError):
    """A safe, client-facing boundary validation failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _body_bytes(body: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(body, str):
        try:
            return body.encode("utf-8")
        except UnicodeError as exc:  # pragma: no cover - str encoding is total
            raise ContractError("invalid_json", "valid JSON required") from exc
    if isinstance(body, (bytes, bytearray, memoryview)):
        return bytes(body)
    raise ContractError("invalid_json", "valid JSON required")


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def parse_json_object(
    body: bytes | bytearray | memoryview | str,
    *,
    max_bytes: int = MAX_JSON_BODY_BYTES,
) -> dict[str, Any]:
    """Parse a UTF-8 JSON object without accepting oversized or non-finite input."""
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
        or max_bytes > MAX_JSON_BODY_BYTES
    ):
        raise ValueError("max_bytes must be a positive bounded integer")
    raw = _body_bytes(body)
    if len(raw) > max_bytes:
        raise ContractError("body_too_large", "request body is too large")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError("invalid_json", "valid JSON required") from exc
    if not isinstance(value, dict):
        raise ContractError("object_required", "JSON object required")
    return value


def is_safe_identifier(value: Any, *, max_length: int = MAX_IDENTIFIER_LENGTH) -> bool:
    """Return whether *value* is a bounded inert registry-style identifier."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        return False
    if len(value) > max_length or max_length > MAX_IDENTIFIER_LENGTH:
        return False
    return bool(_IDENTIFIER_RE.fullmatch(value))


def is_loopback_url(value: Any) -> bool:
    """Accept only HTTP(S) URLs addressed to an explicit loopback host.

    Userinfo, malformed ports, control characters, and non-HTTP schemes are
    rejected.  Hostnames are deliberately limited to ``localhost`` and the
    canonical IPv4/IPv6 loopback literals rather than trusting DNS.
    """
    if not isinstance(value, str) or not value or any(ord(char) < 0x20 for char in value):
        return False
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"}:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.hostname is None or parsed.hostname.lower() not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            return False
        # Accessing port validates both non-numeric and out-of-range ports.
        _ = parsed.port
    except ValueError:
        return False
    try:
        if parsed.hostname.lower() == "localhost":
            return True
        return ipaddress.ip_address(parsed.hostname).is_loopback and parsed.hostname in {
            "127.0.0.1",
            "::1",
        }
    except ValueError:
        return False


def bounded_string(
    value: Any,
    *,
    field: str = "value",
    max_length: int = MAX_STRING_LENGTH,
) -> str:
    """Return a trimmed non-empty string within the public boundary limit."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise ContractError("invalid_string", f"{field} must be a string")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or not 1 <= max_length <= MAX_STRING_LENGTH
    ):
        raise ValueError("max_length must be a positive bounded integer")
    result = value.strip()
    if not result:
        raise ContractError("invalid_string", f"{field} must not be empty")
    if len(result) > max_length:
        raise ContractError("string_too_long", f"{field} is too long")
    return result


def bounded_int(
    value: Any,
    *,
    field: str = "value",
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an integer range value while explicitly rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("invalid_integer", f"{field} must be an integer")
    out_of_range = (
        (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    )
    if out_of_range:
        raise ContractError("integer_out_of_range", f"{field} is out of range")
    return value


def bounded_string_list(
    value: Any,
    *,
    field: str = "value",
    max_items: int = MAX_LIST_ITEMS,
    item_max_length: int = MAX_STRING_LENGTH,
) -> list[str]:
    """Return a bounded list of bounded, trimmed, non-empty strings."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, list):
        raise ContractError("invalid_list", f"{field} must be a list")
    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or not 0 <= max_items <= MAX_LIST_ITEMS
    ):
        raise ValueError("max_items must be a non-negative bounded integer")
    if (
        isinstance(item_max_length, bool)
        or not isinstance(item_max_length, int)
        or not 1 <= item_max_length <= MAX_STRING_LENGTH
    ):
        raise ValueError("item_max_length must be a positive bounded integer")
    if len(value) > max_items:
        raise ContractError("list_too_long", f"{field} has too many items")
    return [
        bounded_string(item, field=f"{field}[{index}]", max_length=item_max_length)
        for index, item in enumerate(value)
    ]


def _redact_text(value: str) -> str:
    value = _AUTH_VALUE_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2) or ''}<redacted>", value
    )
    value = _BEARER_RE.sub("Bearer <redacted>", value)
    return _SECRET_VALUE_RE.sub(r"\1<redacted>", value)


def _safe_key_text(key: Any) -> str | None:
    """Normalise a mapping key to a bounded UTF-8 string, skipping unsafe ones."""
    if isinstance(key, str):
        text = key
    elif isinstance(key, (bytes, bytearray, memoryview)):
        try:
            text = bytes(key).decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        # Non-string/bytes keys (ints, tuples, objects) are skipped so the
        # caller cannot leak arbitrary repr noise through the envelope.
        return None
    if not text:
        return None
    return text[:MAX_REDACT_KEY_LENGTH]


def _redact(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "<redacted>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in list(value.items())[:MAX_LIST_ITEMS]:
            key_text = _safe_key_text(key)
            if key_text is None:
                continue
            if any(needle in key_text.lower() for needle in _SECRET_KEYS):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = _redact(child, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [_redact(child, depth=depth + 1) for child in value[:MAX_LIST_ITEMS]]
    if isinstance(value, tuple):
        return [_redact(child, depth=depth + 1) for child in value[:MAX_LIST_ITEMS]]
    if isinstance(value, str):
        return _redact_text(value)[:MAX_ERROR_MESSAGE_LENGTH]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "<redacted>"


def error_envelope(
    code: str,
    message: str,
    *,
    status: int = 400,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable public error shape without leaking secret material."""
    if details is not None and not isinstance(details, Mapping):
        raise ContractError(
            "invalid_details",
            "details must be a mapping of string keys to JSON values",
        )
    safe_code = code if is_safe_identifier(code) else "internal_error"
    safe_status = (
        status
        if isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
        else 500
    )
    safe_message = _redact_text(str(message))[:MAX_ERROR_MESSAGE_LENGTH]
    error: dict[str, Any] = {"code": safe_code, "message": safe_message}
    if details is not None:
        error["details"] = _redact(details)
    return {"error": error, "status": safe_status}


__all__ = [
    "ContractError",
    "MAX_IDENTIFIER_LENGTH",
    "MAX_JSON_BODY_BYTES",
    "MAX_LIST_ITEMS",
    "MAX_STRING_LENGTH",
    "bounded_int",
    "bounded_string",
    "bounded_string_list",
    "error_envelope",
    "is_loopback_url",
    "is_safe_identifier",
    "parse_json_object",
]
