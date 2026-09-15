#!/usr/bin/env python3
"""Sanitize a JSON registry for an export or container fixture.

The command has no default input path and writes to stdout unless ``--output``
is explicitly supplied.  It is intentionally conservative: secret-looking
keys are replaced, local/private endpoints and filesystem roots are redacted,
and public research URLs remain intact.
"""
from __future__ import annotations

import argparse
import copy
import ipaddress
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_SECRET_KEY = re.compile(
    r"(?:pass(?:word)?|token|secret|api[_-]?key|credential|private[_-]?key|auth|"
    r"certificate|cert|tls[_-]?cert|x509|ssh[_-]?key)",
    re.IGNORECASE,
)
_PATH_KEY = re.compile(r"(?:^|[_-])(path|root|mount|directory|dir|file)(?:$|[_-])", re.IGNORECASE)
_ENDPOINT_KEY = re.compile(r"(?:url|endpoint|host|callback|base[_-]?uri)", re.IGNORECASE)
_PRIVATE_PATH = re.compile(
    r"^(?:/(?:home|mnt|opt|tmp|var|run|root|Users)/|"
    r"[A-Za-z]:[\\/]|~[\\/]|[\\/]Users[\\/])"
)
_PUBLIC_ROUTE_PATH = re.compile(r"^/[A-Za-z0-9/_.~:-]{1,2048}$")
_URL_SECRET_PART = re.compile(
    r"(?:^|[&#;])(?:pass(?:word)?|token|secret|api[_-]?key|credential|auth|"
    r"private[_-]?key|certificate|cert|tls[_-]?cert|x509|ssh[_-]?key)(?:=|$)",
    re.IGNORECASE,
)
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")


def _is_private_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower().strip("[]")
    if host in {"localhost", "host.docker.internal"} or host.endswith(".internal"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address in _SHARED_ADDRESS_SPACE
    )


def _sanitize_string(key: str, value: str) -> str:
    # Route paths are public protocol metadata only when they are pure paths.
    if key in {"health_path", "request_path"}:
        if _PUBLIC_ROUTE_PATH.fullmatch(value) and not _PRIVATE_PATH.match(value):
            return value
        return "<redacted-route-path>"
    if _PATH_KEY.search(key):
        if (
            _PRIVATE_PATH.match(value)
            or value.startswith(("/", "~", "\\"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or ".." in Path(value).parts
        ):
            return "<private-path>"
    elif _PRIVATE_PATH.match(value):
        return "<private-path>"
    if _ENDPOINT_KEY.search(key) and _is_private_host(value):
        return "<private-host>"

    parts = urlsplit(value)
    if parts.scheme and parts.netloc:
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        try:
            if parts.port is not None:
                netloc = f"{netloc}:{parts.port}"
        except ValueError:
            return "<redacted-credential-url>"
        query = "" if _URL_SECRET_PART.search(parts.query) else parts.query
        fragment = "" if _URL_SECRET_PART.search(parts.fragment) else parts.fragment
        if parts.username is not None or parts.password is not None:
            value = urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
            parts = urlsplit(value)
        elif query != parts.query or fragment != parts.fragment:
            value = urlunsplit((parts.scheme, parts.netloc, parts.path, query, fragment))
            parts = urlsplit(value)
    if _ENDPOINT_KEY.search(key) and parts.scheme and parts.netloc:
        host = parts.hostname
        if _is_private_host(host):
            return urlunsplit((parts.scheme, "<private-host>", parts.path, "", ""))
    return value


def sanitize_registry(value: Any, *, _key: str = "") -> Any:
    """Return a JSON-compatible sanitized copy without mutating ``value``."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY.search(key_text):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = sanitize_registry(item, _key=key_text)
        return result
    if isinstance(value, list):
        return [sanitize_registry(item, _key=_key) for item in value]
    if isinstance(value, str):
        return _sanitize_string(_key, value)
    return copy.deepcopy(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON file to sanitize, or '-' for stdin")
    parser.add_argument("-o", "--output", type=Path, help="optional output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw = __import__("sys").stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    data = json.loads(raw)
    rendered = json.dumps(sanitize_registry(data), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["sanitize_registry", "main"]
