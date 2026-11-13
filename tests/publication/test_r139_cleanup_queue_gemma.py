"""Real scratch-Redis proof for the one-shot queue:Gemma cleanup.

The owned child uses a private temporary persistence directory and Unix socket;
no host service or production credential is used.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import redis

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import r139_cleanup_queue_gemma as cleanup

pytestmark = pytest.mark.skipif(shutil.which("redis-server") is None,
                                reason="redis-server not installed")


class ScratchRedis:
    def __init__(self, path):
        self.path = path
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            ["redis-server", "--port", "0", "--unixsocket", str(self.path / "redis.sock"),
             "--unixsocketperm", "700", "--dir", str(self.path), "--appendonly", "yes",
             "--appendfilename", "appendonly.aof", "--save", ""],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        client = redis.Redis(unix_socket_path=str(self.path / "redis.sock"),
                             decode_responses=True, socket_connect_timeout=0.2)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("owned scratch redis-server exited during startup")
            try:
                client.ping()
                client.close()
                return
            except redis.RedisError:
                time.sleep(0.05)
        self.stop()
        raise TimeoutError("scratch redis-server did not become ready")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

    def client(self):
        return redis.Redis(unix_socket_path=str(self.path / "redis.sock"),
                           decode_responses=True)


@pytest.fixture
def scratch_redis(tmp_path):
    server = ScratchRedis(tmp_path)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_default_dry_run_does_not_mutate_target_or_control_data(scratch_redis):
    client = scratch_redis.client()
    client.xadd(cleanup.TARGET_KEY, {"value": "sentinel"})
    client.set("control:sentinel", "keep")
    client.close()

    assert cleanup.main(["--socket", str(scratch_redis.path / "redis.sock")]) == 0
    client = scratch_redis.client()
    assert client.xlen(cleanup.TARGET_KEY) == 1
    assert client.get("control:sentinel") == "keep"
    client.close()


def test_execute_deletes_only_target_persists_aof_and_rdb_across_restart(scratch_redis):
    client = scratch_redis.client()
    client.xadd(cleanup.TARGET_KEY, {"value": "remove-me"})
    client.set("control:sentinel", "survives-restart")
    client.close()

    assert cleanup.run(socket_path=str(scratch_redis.path / "redis.sock"), execute=True) == 0
    scratch_redis.stop()
    scratch_redis.start()

    client = scratch_redis.client()
    assert client.exists(cleanup.TARGET_KEY) == 0
    assert client.get("control:sentinel") == "survives-restart"
    client.close()


def test_wrong_type_is_refused_without_mutating_any_key(scratch_redis):
    client = scratch_redis.client()
    client.set(cleanup.TARGET_KEY, "wrong-type sentinel")
    client.set("control:sentinel", "keep")
    client.close()

    with pytest.raises(ValueError, match="unexpected Redis type string"):
        cleanup.run(socket_path=str(scratch_redis.path / "redis.sock"), execute=True)

    client = scratch_redis.client()
    assert client.get(cleanup.TARGET_KEY) == "wrong-type sentinel"
    assert client.get("control:sentinel") == "keep"
    client.close()
