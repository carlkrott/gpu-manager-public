from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess

import pytest


_SENSITIVE_ENV_PREFIXES = (
    "AWS_", "AZURE_", "GOOGLE_", "HF_", "HUGGINGFACE_", "OPENAI_",
    "GITHUB_", "GPU_MANAGER_", "COMFYUI_", "ACESTEP_", "MINIMAX_",
)
_SENSITIVE_ENV_NAMES = {
    "CREDENTIALS_DIRECTORY", "REDIS_URL", "REDIS_PASSWORD", "VALKEY_PASSWORD",
    "COMBINED_GEMMA_REDIS_URL", "SSH_AUTH_SOCK",
}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for name in tuple(os.environ):
        if name in _SENSITIVE_ENV_NAMES or name.startswith(_SENSITIVE_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def deny_external_network(monkeypatch: pytest.MonkeyPatch):
    original_connect = socket.socket.connect

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise PermissionError("publication tests may connect only to loopback")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def deny_process_launch(monkeypatch: pytest.MonkeyPatch):
    def denied(*_args, **_kwargs):
        raise PermissionError("publication tests may not launch processes")

    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "check_call", denied)
    monkeypatch.setattr(subprocess, "check_output", denied)
