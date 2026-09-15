from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess

import pytest


def test_environment_isolated_from_operator_home_and_runtime_variables(tmp_path: Path):
    assert Path.home() == tmp_path / "home"
    assert os.environ["XDG_CONFIG_HOME"] == str(tmp_path / "config")
    assert not any(name.startswith("GPU_MANAGER_") for name in os.environ)
    assert "CREDENTIALS_DIRECTORY" not in os.environ


def test_network_guard_rejects_non_loopback(deny_external_network):
    with socket.socket() as sock:
        with pytest.raises(PermissionError, match="loopback"):
            sock.connect(("203.0.113.10", 80))


def test_process_guard_rejects_launch(deny_process_launch):
    with pytest.raises(PermissionError, match="launch processes"):
        subprocess.run(["true"], check=True)
