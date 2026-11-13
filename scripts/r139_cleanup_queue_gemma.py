#!/usr/bin/env python3
"""One-shot, explicitly targeted cleanup of the literal Redis queue:Gemma.

No mutation occurs unless --execute is supplied. A Unix socket path is always
required, and the shared Redis credential resolver reads protected runtime
credentials without exposing them in argv or output.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, cast

import redis
from queue_engine import redis_connection_kwargs

TARGET_KEY = "queue:Gemma"
DEFAULT_WAIT_SECONDS = 60.0


def _client(socket_path: str) -> redis.Redis:
    options = redis_connection_kwargs()
    options.pop("host", None)
    options.pop("port", None)
    options["unix_socket_path"] = socket_path
    options["decode_responses"] = True
    return redis.Redis(**cast(Any, options))


def _wait_for(client: redis.Redis, *, field: str, in_progress: str,
              timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = client.info("persistence")
        if int(state.get(in_progress, 0)) == 0:
            if state.get(field) != "ok":
                raise RuntimeError(f"Redis persistence failed: {field}")
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"timed out waiting for Redis {in_progress}")


def run(*, socket_path: str, execute: bool, timeout_seconds: float = DEFAULT_WAIT_SECONDS) -> int:
    if not socket_path:
        raise ValueError("an explicit Redis Unix socket target is required")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    client = _client(socket_path)
    try:
        client.ping()
        key_type = client.type(TARGET_KEY)
        if key_type not in ("none", "stream"):
            raise ValueError(f"refusing {TARGET_KEY}: unexpected Redis type {key_type}")
        before_exists = int(client.exists(TARGET_KEY))
        if not execute:
            print(json.dumps({"mode": "dry-run", "target": TARGET_KEY,
                              "type": key_type, "exists": bool(before_exists)}, sort_keys=True))
            return 0

        removed = int(client.delete(TARGET_KEY))
        client.bgrewriteaof()
        _wait_for(client, field="aof_last_bgrewrite_status",
                  in_progress="aof_rewrite_in_progress", timeout_seconds=timeout_seconds)
        client.bgsave()
        _wait_for(client, field="rdb_last_bgsave_status",
                  in_progress="rdb_bgsave_in_progress", timeout_seconds=timeout_seconds)
        remaining = int(client.exists(TARGET_KEY))
        if remaining != 0:
            raise RuntimeError(f"{TARGET_KEY} still exists after cleanup")
        print(json.dumps({"mode": "execute", "target": TARGET_KEY,
                          "type_before": key_type, "deleted": removed,
                          "exists_after": remaining,
                          "aof_rewrite": "ok", "rdb_save": "ok"}, sort_keys=True))
        return 0
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True,
                        help="explicit Redis Unix socket path (no network/default target)")
    parser.add_argument("--execute", action="store_true", help="delete queue:Gemma and persist (default: inspect only)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS,
                        help="maximum seconds to wait for each persistence operation")
    args = parser.parse_args(argv)
    try:
        return run(socket_path=args.socket, execute=args.execute,
                   timeout_seconds=args.timeout)
    except (OSError, redis.RedisError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(1, f"cleanup refused/failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
