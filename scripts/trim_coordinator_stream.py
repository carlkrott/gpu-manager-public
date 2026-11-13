#!/usr/bin/env python3
"""Safely trim the two coordinator-related Redis streams in bounded steps.

The default invocation is read-only. Live execution requires --execute.
Coordinator-stream floor defaults to the existing 10,000-entry retention
policy; queue:Gemma defaults to 500 entries.
"""
from __future__ import annotations

import argparse
import json
import time

import redis
from queue_engine import redis_connection_kwargs

ALLOWED_KEYS = {"pipeline_coordinator:stream", "queue:Gemma"}
DEFAULT_FLOORS = {"pipeline_coordinator:stream": 10_000, "queue:Gemma": 500}

# Redis executes this script atomically, so no group can acquire an older
# pending entry between the pending check and the trim. Exact MINID plus a
# threshold selected from at most chunk+1 stream IDs bounds each deletion.
TRIM_CHUNK_LUA = r"""
local key, floor, chunk = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2])
if redis.call('TYPE', key).ok ~= 'stream' then
  return redis.error_reply('target_not_stream')
end
local length = redis.call('XLEN', key)
local count = math.min(chunk, math.max(0, length - floor))
if count == 0 then return {0, length} end
local rows = redis.call('XRANGE', key, '-', '+', 'COUNT', count + 1)
if #rows <= count then return {0, length} end
local threshold = rows[count + 1][1]
local function less_id(a, b)
  local am, as = string.match(a, '^(%d+)%-(%d+)$')
  local bm, bs = string.match(b, '^(%d+)%-(%d+)$')
  am, as, bm, bs = tonumber(am), tonumber(as), tonumber(bm), tonumber(bs)
  return am < bm or (am == bm and as < bs)
end
local groups = redis.call('XINFO', 'GROUPS', key)
for _, group in ipairs(groups) do
  local name = nil
  for i = 1, #group, 2 do
    if group[i] == 'name' then name = group[i + 1]; break end
  end
  if name then
    local pending = redis.call('XPENDING', key, name, '-', '+', 1)
    if #pending > 0 and less_id(pending[1][1], threshold) then
      threshold = pending[1][1]
    end
  end
end
local removed = redis.call('XTRIM', key, 'MINID', '=', threshold)
return {removed, redis.call('XLEN', key)}
"""


def _pending_min(client, key: str) -> str | None:
    ids = []
    for group in client.xinfo_groups(key):
        summary = client.xpending(key, group["name"])
        pending_id = summary.get("min")
        if pending_id:
            ids.append(str(pending_id))
    return min(ids, key=lambda value: tuple(int(part) for part in value.split("-"))) if ids else None


def run(key: str, *, execute: bool, floor: int, chunk: int,
        pause: float, max_seconds: float) -> int:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"unsupported stream key: {key}")
    if floor < 0 or chunk < 1 or pause < 0 or max_seconds <= 0:
        raise ValueError("floor, chunk, pause, or time budget is invalid")
    client = redis.Redis(**redis_connection_kwargs())
    before_len = int(client.xlen(key))
    before_bytes = int(client.memory_usage(key) or 0)
    pending_min = _pending_min(client, key)
    plan = {"key": key, "mode": "execute" if execute else "dry-run",
            "before_xlen": before_len, "before_bytes": before_bytes,
            "floor": floor, "chunk": chunk, "pending_min_id": pending_min,
            "estimated_removal": max(0, before_len - floor)}
    print(json.dumps(plan, sort_keys=True))
    if not execute:
        return 0

    started = time.monotonic()
    step = 0
    while time.monotonic() - started < max_seconds:
        step_before_len = int(client.xlen(key))
        step_before_bytes = int(client.memory_usage(key) or 0)
        removed, after_len = client.eval(TRIM_CHUNK_LUA, 1, key, floor, chunk)
        removed, after_len = int(removed), int(after_len)
        after_bytes = int(client.memory_usage(key) or 0)
        step += 1
        print(json.dumps({"step": step, "before_xlen": step_before_len,
                          "before_bytes": step_before_bytes, "removed": removed,
                          "after_xlen": after_len, "after_bytes": after_bytes},
                         sort_keys=True))
        if removed == 0 or after_len >= step_before_len:
            break
        remaining = max_seconds - (time.monotonic() - started)
        if pause and remaining > 0:
            time.sleep(min(pause, remaining))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, choices=sorted(ALLOWED_KEYS))
    parser.add_argument("--execute", action="store_true", help="mutate Redis (default is dry-run)")
    parser.add_argument("--floor", type=int, help="keep this many newest entries")
    parser.add_argument("--chunk", type=int, default=50_000, help="maximum entries removed per step")
    parser.add_argument("--pause", type=float, default=0.25, help="seconds between steps")
    parser.add_argument("--max-seconds", type=float, default=3600,
                        help="maximum execution duration")
    args = parser.parse_args(argv)
    floor = args.floor if args.floor is not None else DEFAULT_FLOORS[args.key]
    try:
        return run(args.key, execute=args.execute, floor=floor, chunk=args.chunk,
                   pause=args.pause, max_seconds=args.max_seconds)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
