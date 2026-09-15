from __future__ import annotations

import json
from pathlib import Path

import pytest

from check_public_payload import check_public_payload
from declarative_workflow_registration import (
    DraftWorkflowCompilationError,
    compile_service_processing,
)
from pipeline_provider_runtime import EndpointResolutionError, _resolve_service_endpoint
from sanitize_registry import sanitize_registry


ROOT = Path(__file__).parents[2]


def test_sanitize_private_hosts_and_paths_without_mutating_input():
    payload = {
        "shared": {"host": "100.64.0.5"},
        "rfc1918": {"host": "192.168.10.2"},
        "link_local": {"endpoint": "http://169.254.169.254/latest"},
        "ula": {"callback_url": "http://[fd00::1]/hook"},
        "paths": {
            "path": "/home/operator/private.json",
            "mount": "/mnt/private",
            "directory": "/opt/private",
            "file": "C:\\Users\\operator\\secret.json",
            "relative": "scripts/example.json",
        },
    }
    sanitized = sanitize_registry(payload)
    assert payload["shared"]["host"] == "100.64.0.5"
    assert sanitized["shared"]["host"] == "<private-host>"
    assert sanitized["rfc1918"]["host"] == "<private-host>"
    assert sanitized["link_local"]["endpoint"].startswith("http://<private-host>")
    assert sanitized["ula"]["callback_url"].startswith("http://<private-host>")
    assert sanitized["paths"]["path"] == "<private-path>"
    assert sanitized["paths"]["mount"] == "<private-path>"
    assert sanitized["paths"]["directory"] == "<private-path>"
    assert sanitized["paths"]["file"] == "<private-path>"
    assert sanitized["paths"]["relative"] == "scripts/example.json"


def test_sanitize_credential_urls_and_secret_payload_keys():
    payload = {
        "endpoint_url": "https://user:password@example.com/api?api_key=synthetic#token=synthetic",
        "public_url": "https://example.com/research.pdf?topic=audio",
        "certificate": "-----BEGIN CERTIFICATE-----\nsynthetic\n-----END CERTIFICATE-----",
        "cert": "synthetic-certificate",
        "private_key": "-----BEGIN PRIVATE KEY-----\nsynthetic\n-----END PRIVATE KEY-----",
        "nested": [{"tls_cert": "synthetic-cert"}],
    }
    sanitized = sanitize_registry(payload)
    assert "user:password@" not in sanitized["endpoint_url"]
    assert "api_key" not in sanitized["endpoint_url"]
    assert "token" not in sanitized["endpoint_url"]
    assert sanitized["public_url"] == payload["public_url"]
    assert sanitized["certificate"] == "<redacted>"
    assert sanitized["cert"] == "<redacted>"
    assert sanitized["private_key"] == "<redacted>"
    assert sanitized["nested"][0]["tls_cert"] == "<redacted>"


def test_check_public_payload_rejects_synthetic_leaks_and_binary(tmp_path):
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"endpoint": "http://100.64.0.5/?token=synthetic"}),
        encoding="utf-8",
    )
    report = check_public_payload(tmp_path)
    assert not report["ok"]
    assert {item["rule"] for item in report["violations"]} >= {
        "credential_url_forbidden",
        "sanitizer_would_change_payload",
    }

    payload.write_text("{}", encoding="utf-8")
    binary = tmp_path / "payload.bin"
    binary.write_bytes(b"PK\x03\x04synthetic")
    report = check_public_payload(tmp_path)
    assert not report["ok"]
    assert any(item["rule"] == "binary_or_archive" for item in report["violations"])


def test_check_public_payload_rejects_absolute_paths_and_private_filenames(tmp_path):
    path = tmp_path / "payload.txt"
    path.write_text("source=/home/operator/private", encoding="utf-8")
    report = check_public_payload(tmp_path)
    assert any(item["rule"] == "absolute_provenance_forbidden" for item in report["violations"])

    path.unlink()
    private_report = tmp_path / "gitleaks-summary.json"
    private_report.write_text("{}", encoding="utf-8")
    report = check_public_payload(tmp_path)
    assert any(item["rule"] == "private_report_filename" for item in report["violations"])


def test_exact_candidate_payload_is_clean():
    report = check_public_payload(ROOT)
    assert report["ok"], report["violations"]
    assert report["file_count"] > 0


def _service(**values):
    result = {"enabled": True, "port": 8123}
    result.update(values)
    return {"services": {"synthetic": result}}


def test_endpoint_resolution_allows_loopback_and_explicit_public_only():
    assert _resolve_service_endpoint(_service(), "synthetic")[0] == "http://127.0.0.1:8123"
    with pytest.raises(EndpointResolutionError):
        _resolve_service_endpoint(_service(host="evil.example"), "synthetic")
    assert _resolve_service_endpoint(
        _service(endpoint="https://api.example.com:443", endpoint_public=True), "synthetic"
    )[0] == "https://api.example.com:443"


@pytest.mark.parametrize(
    "host",
    ["10.0.0.1", "192.168.1.5", "169.254.169.254", "100.64.0.5", "fd00::1"],
)
def test_endpoint_resolution_rejects_private_and_shared_hosts(host):
    with pytest.raises(EndpointResolutionError):
        _resolve_service_endpoint(_service(host=host), "synthetic")


def test_endpoint_resolution_rejects_credentials_query_and_fragment():
    for endpoint in (
        "http://user:pass@example.com:8123",
        "http://127.0.0.1:8123/?token=synthetic",
        "http://127.0.0.1:8123/#secret",
    ):
        with pytest.raises(EndpointResolutionError):
            _resolve_service_endpoint(_service(endpoint=endpoint), "synthetic")


def test_declarative_registration_rejects_nested_execution_escape_fields():
    service = {
        "name": "synthetic-service",
        "processing": [
            {
                "stage": "stage-a",
                "operation": "missing-operation",
                "command": "not-allowed",
                "metadata": {"credentials": "not-allowed"},
            }
        ],
    }
    with pytest.raises(DraftWorkflowCompilationError, match="not allowed"):
        compile_service_processing(service, {"operations": {}})
