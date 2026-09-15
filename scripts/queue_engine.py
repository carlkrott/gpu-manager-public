#!/usr/bin/env python3
"""
queue_engine.py — Core Redis queue, job tracking, slot management,
routing, and capacity reporting for the GPU Manager system.

Standalone module. No dependency on gpu-manager.py.

Requires: redis (valkey-compatible) library.
Redis connection: GPU_MANAGER_REDIS_HOST:GPU_MANAGER_REDIS_PORT,
decode_responses=True.
"""

import json
import hashlib
import logging
import math
import os
import socket
import time
import uuid
from datetime import datetime, timezone

import redis
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from gpu_manager_contracts import (
    apply_priority_contract,
    demand_counts,
    service_is_publicly_eligible,
)

logger = logging.getLogger("queue-engine")

MACHINE_ID = socket.gethostname()

# Keep queue calls bounded when Valkey is unavailable.  These are deliberately
# conservative defaults and remain overrideable for a controlled staging
# environment; callers still surface an outage instead of treating it as an
# empty queue.
REDIS_CONNECT_TIMEOUT_S = float(os.environ.get("GPU_MANAGER_REDIS_CONNECT_TIMEOUT_S", "2"))
REDIS_SOCKET_TIMEOUT_S = float(os.environ.get("GPU_MANAGER_REDIS_SOCKET_TIMEOUT_S", "5"))
REDIS_HOST = os.environ.get("GPU_MANAGER_REDIS_HOST", "localhost").strip() or "localhost"
REDIS_PORT = int(os.environ.get("GPU_MANAGER_REDIS_PORT", "6379"))


def _redis_password() -> str | None:
    """Read the Redis password from a systemd-provided credential file."""
    path = os.environ.get("GPU_MANAGER_REDIS_PASSWORD_FILE")
    if not path:
        credentials_dir = os.environ.get("CREDENTIALS_DIRECTORY")
        if credentials_dir:
            path = os.path.join(credentials_dir, "redis-password")
    if not path:
        return None
    try:
        value = open(path, encoding="utf-8").read().strip()
    except OSError:
        return None
    return value or None


def redis_password() -> str | None:
    """Return the systemd-injected Redis credential for sibling components."""
    return _redis_password()


def redis_connection_kwargs() -> dict[str, object]:
    """Return the shared bounded Redis connection policy."""
    return {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "password": _redis_password(),
        "decode_responses": True,
        "socket_connect_timeout": REDIS_CONNECT_TIMEOUT_S,
        "socket_timeout": REDIS_SOCKET_TIMEOUT_S,
        "health_check_interval": 30,
    }

# ---------------------------------------------------------------------------
# Plan 05 — dead-letter retention policy
# ---------------------------------------------------------------------------
# Single source of truth for the dead-letter cap and policy. Public surfaces
# MUST reference this constant rather than hard-coding 100. The cap is
# EXACT (XTRIM MAXLEN 0 approximate=False semantics) — entries beyond the
# cap are dropped deterministically; the dropped count is reported by
# ``get_dead_letters`` so callers can correlate loss vs. retention.
#
# Policy (explicit, no TTL / no archive rotation):
#   * Dead-letter entries live in a per-service Redis stream
#     ``dead-letter:{service_name}`` until explicitly purged.
#   * TTL is NOT applied — entries are durable audit evidence.
#   * Archive rotation is NOT applied — the stream is the archive.
#   * Overflow drops the OLDEST entries (XTRIM with approximate=False
#     forces exact trim semantics).
#   * Purge is the only eviction mechanism, and it requires explicit
#     authorization at the gpu-manager public boundary.
EXACT_DEAD_LETTER_CAP = 100  # exact, not approximate

# Audit log key — durable record of every authorized purge. A Redis
# stream so we can replay the timeline. Never trimmed by retention.
DEAD_LETTER_AUDIT_STREAM = "dead-letter:audit"
DEAD_LETTER_METRICS_HASH = "dead-letter:metrics"

# Atomically transfer one failed queue entry into its capped DLQ and account
# for every entry removed by the exact trim. KEYS: source stream, DLQ stream,
# metrics hash. ARGV: entry id, consumer group, cap, service, payload pairs.
_DEAD_LETTER_TRANSFER_LUA = r"""
-- dead_letter_transfer_v1
local source = KEYS[1]
local dlq = KEYS[2]
local metrics = KEYS[3]
local entry_id = ARGV[1]
local group = ARGV[2]
local cap = tonumber(ARGV[3])
local service = ARGV[4]
local source_type = redis.call('TYPE', source).ok
local dlq_type = redis.call('TYPE', dlq).ok
local metrics_type = redis.call('TYPE', metrics).ok
if source_type ~= 'stream' then return redis.error_reply('source_not_stream') end
if dlq_type ~= 'none' and dlq_type ~= 'stream' then
  return redis.error_reply('dlq_not_stream')
end
if metrics_type ~= 'none' and metrics_type ~= 'hash' then
  return redis.error_reply('metrics_not_hash')
end
local dlq_id = redis.call('XADD', dlq, '*', unpack(ARGV, 5))
local removed = redis.call('XTRIM', dlq, 'MAXLEN', '=', cap)
if removed > 0 then
  redis.call('HINCRBY', metrics, service, removed)
end
local dropped_total = tonumber(redis.call('HGET', metrics, service) or '0')
redis.pcall('XACK', source, group, entry_id)
redis.call('XDEL', source, entry_id)
return {dlq_id, tostring(removed), tostring(dropped_total)}
"""

# Audit-before-delete in one Redis script. Redis executes scripts atomically;
# if XADD fails, XTRIM is never reached, so an unaudited purge cannot occur.
_DEAD_LETTER_PURGE_LUA = r"""
-- dead_letter_purge_v1
local dlq = KEYS[1]
local audit_stream = KEYS[2]
local dlq_type = redis.call('TYPE', dlq).ok
local audit_type = redis.call('TYPE', audit_stream).ok
if dlq_type ~= 'none' and dlq_type ~= 'stream' then
  return redis.error_reply('dlq_not_stream')
end
if audit_type ~= 'none' and audit_type ~= 'stream' then
  return redis.error_reply('audit_not_stream')
end
local count = redis.call('XLEN', dlq)
local audit_id = redis.call(
  'XADD', audit_stream, '*',
  'actor', ARGV[1],
  'reason', ARGV[2],
  'timestamp', ARGV[3],
  'count', tostring(count),
  'service', ARGV[4],
  'cap', ARGV[5]
)
redis.call('DEL', dlq)
return {tostring(count), audit_id}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def dead_letter_policy() -> dict:
    """Return the explicit dead-letter retention policy.

    Stable, public-facing metadata so callers (gpu-manager, tests,
    operators) can introspect the contract rather than hard-coding it.
    """
    return {
        "cap": EXACT_DEAD_LETTER_CAP,
        "cap_semantics": "exact",
        "ttl": "none",
        "archive_rotation": "none",
        "overflow_action": "drop_oldest",
        "purge_authorization": "required_at_public_boundary",
        "audit_stream": DEAD_LETTER_AUDIT_STREAM,
    }


def _dead_letter_stream(service_name: str) -> str:
    """Return the per-service dead-letter stream key."""
    return f"dead-letter:{service_name}"


def _flatten_stream_payload(payload: dict) -> list:
    """Flatten a stream field mapping for Redis EVAL/XADD arguments."""
    flat: list = []
    for key, value in payload.items():
        flat.extend((key, value if value is not None else ""))
    return flat


def _redis_conn() -> redis.Redis:
    """Return a default Redis connection. Callers should catch exceptions."""
    return redis.Redis(**redis_connection_kwargs())


def _is_missing_stream_error(error: BaseException) -> bool:
    """Return whether a Redis response means the stream is not present yet."""
    return "no such key" in str(error).lower()


def _safe_int(value: object, *, default: int = 0, minimum: int | None = None) -> int:
    """Coerce legacy/null numeric metadata without crashing routing."""
    if isinstance(value, bool):
        result = default
    else:
        try:
            result = int(value) if value is not None else default
        except (TypeError, ValueError):
            result = default
    return max(minimum, result) if minimum is not None else result


def _safe_float(value: object, *, default: float = 0.0, minimum: float | None = None) -> float:
    """Coerce optional speed/priority metadata to a finite number."""
    if isinstance(value, bool):
        result = default
    else:
        try:
            result = float(value) if value is not None else default
        except (TypeError, ValueError):
            result = default
    if not math.isfinite(result):
        result = default
    return max(minimum, result) if minimum is not None else result


# New admissions may use three server-owned Redis Streams while the legacy
# stream remains readable for drain/replay.  The handle returned to callers
# carries the lane because Redis Stream IDs are only unique within one stream.
PRIORITY_LANES: tuple[str, ...] = ("interactive", "normal", "background")
_PRIORITY_LANE_RANK = {lane: index for index, lane in enumerate(PRIORITY_LANES)}
_ENTRY_HANDLE_PREFIX = "lane:"


def _priority_lane(contracted: dict[str, object]) -> str:
    """Return the canonical lane already derived by the priority contract."""

    lane = str(contracted.get("priority_class") or "normal").strip().lower()
    if lane not in _PRIORITY_LANE_RANK:
        # ``apply_priority_contract`` should make this unreachable.  Keep the
        # queue boundary fail-closed rather than creating an arbitrary stream.
        raise ValueError(f"unknown priority lane: {lane!r}")
    return lane


def _lane_stream_name(service_name: str, routing_group: str | None, lane: str) -> str:
    authority = routing_group or service_name
    if not authority:
        raise ValueError("queue lane requires service_name or routing_group")
    if lane not in _PRIORITY_LANE_RANK:
        raise ValueError(f"unknown priority lane: {lane!r}")
    return f"queue:{authority}:lane:{lane}"


def _legacy_stream_name(service_name: str, routing_group: str | None) -> str:
    authority = routing_group or service_name
    if not authority:
        raise ValueError("queue stream requires service_name or routing_group")
    return f"queue:{authority}"


def _entry_handle(lane: str, entry_id: str) -> str:
    return f"{_ENTRY_HANDLE_PREFIX}{lane}:{entry_id}"


def _parse_entry_handle(
    entry_id: str,
    service_name: str,
    routing_group: str | None,
) -> tuple[str, str, str | None]:
    """Resolve an opaque queue handle to ``(stream, raw_id, lane)``.

    Legacy entries retain their plain Redis ID.  New lane entries are
    represented as ``lane:<class>:<redis-id>`` so ACK/NACK/reaper paths can
    target the exact stream without a second mapping key or authority.
    """

    if isinstance(entry_id, bytes):
        text = entry_id.decode("utf-8", errors="replace")
    else:
        text = str(entry_id or "")
    if text.startswith(_ENTRY_HANDLE_PREFIX):
        parts = text.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid priority queue entry handle: {text!r}")
        _, lane, raw_id = parts
        if lane not in _PRIORITY_LANE_RANK or not raw_id:
            raise ValueError(f"invalid priority queue entry handle: {text!r}")
        return _lane_stream_name(service_name, routing_group, lane), raw_id, lane
    return _legacy_stream_name(service_name, routing_group), text, None


def _priority_age_score(
    lane: str | None,
    entry_id: str,
    *,
    now: float,
    aging_interval_s: float,
    aging_cap: int,
) -> tuple[int, float, int]:
    """Score a ready entry without preempting an already claimed job."""

    lane_name = lane if lane in _PRIORITY_LANE_RANK else "normal"
    base = len(PRIORITY_LANES) - 1 - _PRIORITY_LANE_RANK[lane_name]
    try:
        timestamp_ms = int(str(entry_id).split("-", 1)[0])
        age = max(0.0, now - (timestamp_ms / 1000.0))
    except (TypeError, ValueError, IndexError):
        age = 0.0
    interval = max(1.0, float(aging_interval_s))
    age_boost = min(max(0, int(aging_cap)), int(age // interval))
    return base + age_boost, age, -_safe_int(
        str(entry_id).split("-", 1)[0] if "-" in str(entry_id) else 0
    )


def _observe_stream_depth(r, stream: str, group: str) -> dict[str, object]:
    """Observe one stream's unclaimed depth without inferring from XLEN."""

    try:
        try:
            info = r.xinfo_stream(stream)
        except redis.ResponseError as exc:
            missing = _is_missing_stream_error(exc)
            return {
                "stream": stream,
                "depth": 0,
                "known": missing,
                "reason": "stream_missing" if missing else "redis_response_error",
                "pending": 0,
                **({} if missing else {
                    "error_type": type(exc).__name__,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }),
            }
        length = max(0, int(info.get("length", 0) or 0))
        if length == 0:
            return {"stream": stream, "depth": 0, "known": True, "reason": "empty", "pending": 0}
        try:
            groups = r.xinfo_groups(stream)
            for item in groups:
                if item.get("name") != group:
                    continue
                lag = item.get("lag")
                pending = max(0, int(item.get("pending", 0) or 0))
                if lag is not None:
                    return {
                        "stream": stream,
                        "depth": min(max(0, int(lag)), length),
                        "known": True,
                        "reason": "consumer_group_lag",
                        "pending": pending,
                    }
                return {
                    "stream": stream,
                    "depth": 0,
                    "known": False,
                    "reason": "consumer_group_lag_unavailable",
                    "pending": pending,
                }
        except redis.ResponseError as exc:
            return {
                "stream": stream,
                "depth": 0,
                "known": False,
                "reason": "redis_response_error",
                "pending": 0,
                "error_type": type(exc).__name__,
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }
        return {"stream": stream, "depth": length, "known": True, "reason": "no_consumer_group", "pending": 0}
    except redis.RedisError as exc:
        return {
            "stream": stream,
            "depth": 0,
            "known": False,
            "reason": "redis_error",
            "pending": 0,
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


async def _async_redis_conn():
    """Return a default async Redis connection pool. Callers should catch exceptions."""
    pool = aioredis.ConnectionPool(
        **redis_connection_kwargs(), max_connections=30,
    )
    return aioredis.Redis(connection_pool=pool)


# ===================================================================
# 1. RedisQueue  —  Stream-based FIFO job queue
# ===================================================================

class RedisQueue:
    """
    Uses Redis Streams as per-service job queues.

    Key layout
    ----------
    stream   : ``queue:{service_name}``
    group    : ``workers``
    consumer : any caller-supplied name
    """

    GROUP = "workers"

    def __init__(
        self,
        r: redis.Redis | None = None,
        *,
        externally_owned_routing_groups: set[str] | frozenset[str] | None = None,
        priority_lanes_enabled: bool = False,
        priority_aging_interval_s: float = 60.0,
        priority_aging_cap: int = 3,
    ):
        self.r = r or _redis_conn()
        self.externally_owned_routing_groups = frozenset(
            externally_owned_routing_groups or ()
        )
        self.priority_lanes_enabled = bool(priority_lanes_enabled)
        self.priority_aging_interval_s = max(1.0, float(priority_aging_interval_s))
        self.priority_aging_cap = max(0, int(priority_aging_cap))

    # --- internal helpers ------------------------------------------------

    def _stream(self, service_name: str, routing_group: str | None = None) -> str:
        # Phase 5.2.80 R1 guard: refuse to silently create "queue:" (empty
        # stream key). Caller MUST supply either a routing_group OR a
        # service_name; if both are empty, raise so the bug surfaces in
        # tests rather than silently accumulating orphans. Defense in depth
        # alongside the gpu-manager proxy-direct routing_group root-cause fix.
        if routing_group:
            return f"queue:{routing_group}"
        if service_name:
            return f"queue:{service_name}"
        raise ValueError(
            "queue_engine._stream requires service_name or routing_group"
        )

    def _stream_specs(
        self, service_name: str, routing_group: str | None = None
    ) -> list[tuple[str, str | None]]:
        """Return new priority lanes followed by the legacy drain stream."""

        legacy = self._stream(service_name, routing_group=routing_group)
        if not getattr(self, "priority_lanes_enabled", False):
            return [(legacy, None)]
        return [
            *(
                (_lane_stream_name(service_name, routing_group, lane), lane)
                for lane in PRIORITY_LANES
            ),
            (legacy, None),
        ]

    def resolve_entry_stream(
        self, service_name: str, entry_id: str, routing_group: str | None = None
    ) -> tuple[str, str, str | None]:
        """Resolve a worker/reaper handle to its exact stream and raw ID."""

        return _parse_entry_handle(entry_id, service_name, routing_group)

    def _ensure_group(self, service_name: str, routing_group: str | None = None):
        """Create consumer group if it doesn't exist (idempotent)."""
        stream = self._stream(service_name, routing_group=routing_group)
        try:
            self.r.xgroup_create(stream, self.GROUP, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # --- public API ------------------------------------------------------

    def enqueue(self, service_name: str, job_data: dict,
                  routing_group: str | None = None, *,
                  tracked_job: dict | None = None) -> str | None:
        """
        Add a job to the stream.  Returns the stream entry-id (used as
        the handle for ack / nack), or None on error.
        """
        try:
            self._ensure_group(service_name, routing_group=routing_group)
            convention = job_data.get("priority_convention")
            if convention is None:
                convention = (
                    "generation_higher_is_urgent"
                    if job_data.get("routing_group_type") == "generation"
                    else "legacy_lower_is_urgent"
                )
            contracted = apply_priority_contract(
                job_data, convention=convention
            )
            lane = _priority_lane(contracted)
            stream = self._stream(service_name, routing_group=routing_group)
            handle_lane: str | None = None
            if getattr(self, "priority_lanes_enabled", False):
                stream = _lane_stream_name(service_name, routing_group, lane)
                # Keep the original admission age explicit for diagnostics and
                # future migrations.  The selector uses the Redis ID today so
                # it remains compatible with old payloads and does not trust a
                # caller-supplied timestamp for ordering.
                contracted.setdefault("enqueued_at", _now_iso())
                contracted["priority_lane"] = lane
                handle_lane = lane
            payload = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in contracted.items()
            }
            if tracked_job is not None:
                # Internal deterministic children commit identity + stream
                # entry together. Retrying an ambiguous Redis response returns
                # the original handle, never a second generation request.
                record = {str(k): str(v) for k, v in tracked_job.items() if v is not None}
                key = "job:" + str(record["job_id"])
                handle = self.r.eval("""
                    local oldtype = redis.call('TYPE', KEYS[1]).ok
                    local streamtype = redis.call('TYPE', KEYS[2]).ok
                    if oldtype ~= 'none' and oldtype ~= 'hash' then
                        return redis.error_reply('invalid tracked job type')
                    end
                    if streamtype ~= 'none' and streamtype ~= 'stream' then
                        return redis.error_reply('invalid tracked queue type')
                    end
                    if oldtype == 'hash' then
                        if redis.call('HGET', KEYS[1], 'request_sha256') ~= ARGV[3] then
                            return redis.error_reply('deterministic child identity conflict')
                        end
                        local prior = redis.call('HGET', KEYS[1], 'entry_id')
                        if not prior then return redis.error_reply('child outcome unknown: missing queue receipt') end
                        return prior
                    end
                    local entry = redis.call('XADD', KEYS[2], '*', unpack(cjson.decode(ARGV[1])))
                    local handle = ARGV[4] .. entry
                    redis.call('HSET', KEYS[1], unpack(cjson.decode(ARGV[2])))
                    redis.call('HSET', KEYS[1], 'entry_id', handle)
                    return handle
                    """, 2, key, stream,
                    json.dumps([item for pair in payload.items() for item in pair]),
                    json.dumps([item for pair in record.items() for item in pair]),
                    record["request_sha256"], _entry_handle(handle_lane, "") if handle_lane else "")
                return handle
            entry_id = self.r.xadd(stream, payload)
            handle = _entry_handle(handle_lane, entry_id) if handle_lane else entry_id
            logger.info("enqueue %s -> %s", service_name, handle)
            return handle
        except redis.RedisError:
            logger.exception("enqueue failed for %s", service_name)
            return None

    def dequeue(self, service_name: str, consumer_name: str,
                count: int = 1,
                routing_group: str | None = None) -> list[dict]:
        """
        Claim up to *count* pending jobs.  Returns a list of dicts
        with an extra ``_entry_id`` key for later ack/nack.
        """
        try:
            specs = self._stream_specs(service_name, routing_group=routing_group)
            for stream, _lane in specs:
                try:
                    self.r.xgroup_create(stream, self.GROUP, id="0", mkstream=True)
                except redis.ResponseError as exc:
                    if "BUSYGROUP" not in str(exc):
                        raise

            # New lane entries are selected by bounded priority/age before
            # claim.  This keeps the Redis PEL/reaper contract unchanged and
            # never preempts an already accepted GPU stage.
            if getattr(self, "priority_lanes_enabled", False):
                candidates: list[tuple[tuple[int, float, int], str, str | None]] = []
                now = time.time()
                for stream, lane in specs:
                    try:
                        groups = self.r.xinfo_groups(stream)
                        last_id = "0-0"
                        for group in groups:
                            if group.get("name") == self.GROUP:
                                last_id = group.get("last-delivered-id") or "0-0"
                                break
                        rows = self.r.xread({stream: last_id}, count=1)
                    except (AttributeError, TypeError, redis.RedisError):
                        rows = []
                    if rows:
                        for _stream_key, entries in rows:
                            if entries:
                                raw_id = str(entries[0][0])
                                candidates.append(
                                    (
                                        _priority_age_score(
                                            lane,
                                            raw_id,
                                            now=now,
                                            aging_interval_s=self.priority_aging_interval_s,
                                            aging_cap=self.priority_aging_cap,
                                        ),
                                        stream,
                                        lane,
                                    )
                                )
                if candidates:
                    _, selected_stream, selected_lane = max(candidates, key=lambda item: item[0])
                    claimed = self.r.xreadgroup(
                        groupname=self.GROUP,
                        consumername=consumer_name,
                        streams={selected_stream: ">"},
                        count=count,
                    )
                    return self._decode_claimed_entries(claimed, selected_lane)

            # Empty-at-check/race path and legacy-only mode.  A combined read
            # is retained as the blocking fallback so workers do not busy-loop.
            streams = self.r.xreadgroup(
                groupname=self.GROUP,
                consumername=consumer_name,
                streams={stream: ">" for stream, _lane in specs},
                count=count,
            )
            return self._decode_claimed_entries(streams, None)
        except redis.RedisError:
            logger.exception("dequeue failed for %s", service_name)
            return []

    @staticmethod
    def _decode_claimed_entries(streams, selected_lane: str | None = None) -> list[dict]:
        results: list[dict] = []
        for stream_key, entries in streams or []:
            stream_text = str(stream_key)
            lane = selected_lane
            if lane is None and ":lane:" in stream_text:
                lane = stream_text.rsplit(":lane:", 1)[-1]
            for entry_id, data in entries:
                data["_entry_id"] = _entry_handle(lane, entry_id) if lane in _PRIORITY_LANE_RANK else entry_id
                results.append(data)
        return results

    # Lua script for atomic queued-cancel. All-or-nothing:
    #   1. Verify the exact entry_id is still in the stream (XRANGE).
    #   2. If absent -> {"no_entry"} (nothing happened).
    #   3. If the tracker hash is already in a terminal state ->
    #      {"already_terminal", current_status} (nothing happened, audit
    #      fields preserved).
    #   4. Else XACK + XDEL the entry AND write the terminal hash with
    #      cancellation audit fields in one atomic step ->
    #      {"removed_and_terminalized"}.
    #
    # A single EVAL is the only way to make these guarantees without
    # leaving a window where the entry is gone but the tracker is still
    # "queued" (or vice-versa) — the handler uses the outcome to gate
    # the slot/capacity release so the releases happen EXACTLY ONCE per
    # cancel request, not more, not less.
    _CANCEL_QUEUED_LUA = """
    local stream_key = KEYS[1]
    local tracker_key = KEYS[2]
    local entry_id = ARGV[1]
    local expected_job_id = ARGV[2]
    local group_name = ARGV[3]
    local completed_at = ARGV[4]
    local error_msg = ARGV[5]
    local audit_source = ARGV[6]
    local cancelled_by = ARGV[7]
    local terminal_ttl = tonumber(ARGV[8])

    -- Step 1: verify the entry still exists. XRANGE is exact-match when
    -- min == max == entry_id.
    local entries = redis.call('XRANGE', stream_key, entry_id, entry_id)
    if (not entries) or (#entries == 0) then
        return {'no_entry'}
    end

    -- Step 1b: bind the exact stream entry to the requested job. A stale
    -- tracker entry_id must never allow cancellation of another job's
    -- stream record.
    local fields = entries[1][2]
    local stream_job_id = nil
    for i = 1, #fields, 2 do
        if fields[i] == 'job_id' then
            stream_job_id = fields[i + 1]
            break
        end
    end
    if stream_job_id ~= expected_job_id then
        return {'entry_job_mismatch', tostring(stream_job_id or '')}
    end

    -- Step 2: refuse if the tracker hash is already terminal — never
    -- overwrite a real completion/failure/cancellation with cancel.
    local current_status = redis.call('HGET', tracker_key, 'status')
    if current_status and (current_status == 'completed'
        or current_status == 'failed'
        or current_status == 'cancelled') then
        return {'already_terminal', tostring(current_status)}
    end

    -- The queued-cancel owner must win a strict queued -> cancelled
    -- transition. If a worker has already claimed the job, leave both
    -- the PEL entry and tracker untouched so the normal cooperative
    -- in-flight cancellation path owns the result. A missing tracker
    -- hash is allowed: its TTL may have expired while the stream entry
    -- remained queued, and the terminal write below recreates the audit
    -- record instead of leaving an orphaned stream entry.
    if current_status and current_status ~= 'queued' then
        return {'not_queued', tostring(current_status)}
    end

    -- Step 3: XACK + XDEL the stream entry.
    redis.call('XACK', stream_key, group_name, entry_id)
    redis.call('XDEL', stream_key, entry_id)

    -- Step 4: write the durable terminal hash with audit fields and
    -- apply the short terminal TTL.
    redis.call('HSET', tracker_key,
        'status', 'cancelled',
        'completed_at', completed_at,
        'error', error_msg,
        'cancelled_by', cancelled_by,
        'cancellation_audit_source', audit_source,
        'cancellation_at', completed_at)
    redis.call('EXPIRE', tracker_key, terminal_ttl)

    return {'removed_and_terminalized'}
    """

    def cancel_queued_atomically(
        self,
        service_name: str,
        routing_group: str | None,
        entry_id: str,
        consumer: str,
        *,
        job_id: str,
        error: str = "user_cancel",
        audit_source: str = "cancel-api",
        cancelled_by: str = "cancel-api",
        terminal_ttl: int = 30 * 86400,
    ) -> dict:
        """Atomically remove a queued stream entry AND record durable
        terminal cancellation on the JobTracker hash.

        Backwards compatible: this method is additive. ``ack()`` and
        ``JobTracker.update_status()`` retain their existing contracts
        and continue to be callable in isolation; the queued-cancel
        HTTP handler is the only intended caller of this primitive.

        Returns a dict with at minimum an ``outcome`` key. Outcomes:

          * ``"removed_and_terminalized"`` — the entry existed and the
            terminal hash was written. Caller MUST perform slot /
            capacity / admission release exactly once on this branch.

          * ``"no_entry"`` — the entry_id was not in the stream. Caller
            MUST NOT release (either someone else already removed it,
            or the wrong entry_id was supplied; in both cases the
            durable state is consistent and a release here would
            double-count).

          * ``"entry_job_mismatch"`` — the exact stream entry exists but
            its durable ``job_id`` payload does not match the requested
            job. Caller MUST NOT release; the primitive leaves both
            records untouched so the tracker/stream mismatch can be
            investigated safely.

          * ``"already_terminal"`` — the tracker hash is already in a
            terminal state (completed/failed/cancelled). The cancellation
            audit fields from this call are NOT applied (we never
            overwrite a real terminal record). Caller MUST NOT release
            (a prior cancel / completion already handled it).

          * ``"error"`` — Redis EVAL faulted. Nothing happened. Caller
            MUST NOT release (handler should return 503 and let the
            caller retry; if the entry was already gone, the retry
            will land on ``no_entry``).

        The handler uses ``outcome`` to gate the release step so that:

          * Split-brain on crash is impossible: either the stream
            entry AND the tracker terminal write both happened, or
            neither did.
          * Duplicate DELETEs do not double-release slot/capacity: only
            the call that returns ``"removed_and_terminalized"``
            performs the release; the rest fall into the
            ``"no_entry"`` / ``"already_terminal"`` branch and skip.
        """
        if not entry_id or not job_id:
            return {
                "outcome": "error",
                "error": "missing required argument (entry_id / job_id)",
            }
        try:
            stream_key, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )
        except ValueError as exc:
            return {"outcome": "error", "error": str(exc)}
        tracker_key = f"{JobTracker.HASH_PREFIX}{job_id}"
        completed_at = _now_iso()
        try:
            result = self.r.eval(
                self._CANCEL_QUEUED_LUA,
                2,
                stream_key,
                tracker_key,
                raw_entry_id,
                str(job_id),
                self.GROUP,
                completed_at,
                str(error or ""),
                str(audit_source or ""),
                str(cancelled_by or ""),
                str(int(terminal_ttl)),
            )
        except redis.RedisError as exc:
            logger.exception(
                "cancel_queued_atomically EVAL failed %s/%s", job_id, entry_id
            )
            return {"outcome": "error", "error": str(exc)}

        # Decode Lua return: {'removed_and_terminalized'} or
        # {'no_entry'} or {'already_terminal', current_status}.
        try:
            outcome_raw = result[0] if isinstance(result, (list, tuple)) else result
        except (IndexError, TypeError):
            outcome_raw = result
        if isinstance(outcome_raw, bytes):
            outcome_raw = outcome_raw.decode("utf-8", errors="replace")
        outcome = str(outcome_raw or "error")

        if outcome == "removed_and_terminalized":
            logger.info(
                "cancel_queued_atomically %s/%s removed+terminal",
                job_id,
                entry_id,
            )
            return {
                "outcome": "removed_and_terminalized",
                "stream": stream_key,
                "entry_id": entry_id,
                "previous_status": "queued",
            }

        if outcome == "no_entry":
            logger.info(
                "cancel_queued_atomically %s/%s no_entry",
                job_id,
                entry_id,
            )
            return {
                "outcome": "no_entry",
                "stream": stream_key,
                "entry_id": entry_id,
            }

        if outcome == "entry_job_mismatch":
            stream_job_id = ""
            try:
                if len(result) > 1:
                    value = result[1]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    stream_job_id = str(value or "")
            except Exception:
                stream_job_id = ""
            logger.error(
                "cancel_queued_atomically %s/%s entry job mismatch stream_job_id=%s",
                job_id,
                entry_id,
                stream_job_id,
            )
            return {
                "outcome": "entry_job_mismatch",
                "stream": stream_key,
                "entry_id": entry_id,
                "stream_job_id": stream_job_id,
            }

        if outcome == "not_queued":
            current = ""
            try:
                if len(result) > 1:
                    cur = result[1]
                    if isinstance(cur, bytes):
                        cur = cur.decode("utf-8", errors="replace")
                    current = str(cur or "")
            except Exception:
                current = ""
            return {
                "outcome": "not_queued",
                "stream": stream_key,
                "entry_id": entry_id,
                "current_status": current,
            }

        if outcome == "already_terminal":
            current = ""
            try:
                if len(result) > 1:
                    cur = result[1]
                    if isinstance(cur, bytes):
                        cur = cur.decode("utf-8", errors="replace")
                    current = str(cur or "")
            except Exception:
                current = ""
            logger.info(
                "cancel_queued_atomically %s/%s already_terminal current=%s",
                job_id,
                entry_id,
                current,
            )
            return {
                "outcome": "already_terminal",
                "stream": stream_key,
                "entry_id": entry_id,
                "current_status": current,
            }

        # Unknown Lua return: defensive — treat as error so the caller
        # never releases on a malformed outcome.
        logger.error(
            "cancel_queued_atomically %s/%s unexpected outcome %r",
            job_id,
            entry_id,
            outcome,
        )
        return {
            "outcome": "error",
            "error": f"unexpected primitive outcome: {outcome!r}",
        }

    def ack(self, service_name: str, consumer_name: str, entry_id: str,
            routing_group: str | None = None) -> bool:
        """Mark a job as successfully processed and trim the stream."""
        try:
            stream, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )
            # XACK only clears PEL, XDEL clears XLEN, and transaction keeps them atomic.
            pipe = self.r.pipeline(transaction=True)
            pipe.xack(stream, self.GROUP, raw_entry_id)
            pipe.xdel(stream, raw_entry_id)
            # Keep exact trim to mirror async ack R3b. approximate=True can leave small
            # audit-trail streams untrimmed; XDEL already removes the processed entry,
            # so exact trim normally has little work. If profiling ever shows XTRIM cost,
            # throttle frequency rather than switching back to approximate trim.
            pipe.xtrim(stream, maxlen=500, approximate=False)
            pipe.execute()
            logger.debug("ack %s %s", service_name, entry_id)
            return True
        except redis.RedisError:
            logger.exception("ack failed %s %s", service_name, entry_id)
            return False

    def nack(self, service_name: str, entry_id: str, max_retries: int = 3,
             routing_group: str | None = None) -> bool:
        """
        Re-queue a failed / stuck job.  After ``max_retries`` nacks,
        moves the entry to a dead-letter stream instead of re-queuing.
        """
        try:
            stream, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )
            # Read the original payload before ack-ing
            entries = self.r.xrange(stream, min=raw_entry_id, max=raw_entry_id, count=1)
            if not entries:
                pipe = self.r.pipeline(transaction=True)
                pipe.xack(stream, self.GROUP, raw_entry_id)
                pipe.execute()
                logger.debug("nack: entry %s already trimmed from %s", entry_id, stream)
                return True
            _, original_data = entries[0]

            # Check nack count
            raw_nack_count = original_data.get("_nack_count", original_data.get(b"_nack_count", 0))
            if isinstance(raw_nack_count, bytes): raw_nack_count = raw_nack_count.decode()
            nack_count = int(raw_nack_count or 0) + 1

            if nack_count >= max_retries:
                dlq_stream = _dead_letter_stream(service_name)
                payload = {
                    **original_data,
                    "_nack_count": str(nack_count),
                    "_dead_letter_cap": str(EXACT_DEAD_LETTER_CAP),
                    "_dead_letter_ts": _now_iso(),
                }
                result = self.r.eval(
                    _DEAD_LETTER_TRANSFER_LUA,
                    3,
                    stream,
                    dlq_stream,
                    DEAD_LETTER_METRICS_HASH,
                    raw_entry_id,
                    self.GROUP,
                    EXACT_DEAD_LETTER_CAP,
                    service_name,
                    *_flatten_stream_payload(payload),
                )
                dropped = int(result[1])
                dropped_total = int(result[2])
                logger.warning(
                    "nack: moved %s to dead-letter after %d attempts on %s "
                    "(cap=%d dropped=%d dropped_total=%d)",
                    entry_id, nack_count, service_name,
                    EXACT_DEAD_LETTER_CAP, dropped, dropped_total,
                )
                return True

            # Re-add with incremented nack count
            pipe = self.r.pipeline(transaction=True)
            pipe.xack(stream, self.GROUP, raw_entry_id)
            pipe.xdel(stream, raw_entry_id)
            payload = {**original_data, "_nack_count": str(nack_count)}
            pipe.xadd(stream, payload)
            # Exact trim prevents small audit-trail streams from lingering.
            # XDEL removes processed entry so exact trim normally has little work.
            pipe.xtrim(stream, maxlen=500, approximate=False)
            results = pipe.execute()
            new_id = results[2]
            logger.info("nack (re-queued %d/%d) %s %s → %s",
                        nack_count, max_retries, service_name, entry_id, new_id)
            return True
        except redis.RedisError:
            logger.exception("nack failed %s %s", service_name, entry_id)
            return False

    def get_dead_letters(self, service_name: str, count: int = 50) -> dict:
        """Return dead-lettered entries for a service.

        Return shape (Plan 05 — outage distinguishable from empty):
            {
                "status":      "available" | "outage",
                "service":     service_name,
                "cap":         EXACT_DEAD_LETTER_CAP,
                "entries":     [ {"entry_id": ..., ...}, ... ],
                "count":       int,        # entries actually returned
                "total":       int,        # XLEN at read time
                "dropped":     int,        # deterministic overflow metric
                                        # (entries that fell off the cap)
                "ttl":         "none",
                "archive_rotation": "none",
                "policy":      dead_letter_policy(),
            }

        On RedisError we DO NOT silently coerce to ``{"entries": []}``.
        We surface ``status == "outage"`` so operators can distinguish
        an empty DLQ (legitimately zero dead-letters) from an outage
        (we don't know). Backward-compatible callers that indexed
        ``["entries"]`` keep working because we always emit the key.
        """
        dlq_stream = _dead_letter_stream(service_name)
        policy = dead_letter_policy()
        empty = {
            "status": "available",
            "service": service_name,
            "cap": EXACT_DEAD_LETTER_CAP,
            "entries": [],
            "count": 0,
            "total": 0,
            "dropped": 0,
            "ttl": "none",
            "archive_rotation": "none",
            "policy": policy,
        }
        try:
            raw = self.r.xrange(dlq_stream, count=count)
            total = int(self.r.xlen(dlq_stream))
            dropped = int(
                self.r.hget(DEAD_LETTER_METRICS_HASH, service_name) or 0
            )
        except redis.RedisError:
            logger.exception("get_dead_letters outage %s", service_name)
            # Outage is distinguishable from empty: same shape, status flag.
            return {
                **empty,
                "status": "outage",
                "entries": [],
                "count": 0,
                "total": 0,
            }
        entries = [{"entry_id": e[0], **e[1]} for e in raw]
        # Durable cumulative count maintained atomically by the transfer Lua.
        return {
            "status": "available",
            "service": service_name,
            "cap": EXACT_DEAD_LETTER_CAP,
            "entries": entries,
            "count": len(entries),
            "total": total,
            "dropped": dropped,
            "ttl": "none",
            "archive_rotation": "none",
            "policy": policy,
        }

    def purge_dead_letters(
        self,
        service_name: str,
        actor: str,
        reason: str,
    ) -> dict:
        """Atomically audit and purge one service dead-letter stream.

        Actor and reason are mandatory. The Redis Lua script appends the audit
        record before clearing the stream in one atomic execution; there is no
        unaudited production or test bypass. Redis errors leave the stream
        untouched when the audit append cannot be performed.
        """
        actor_s = (actor or "").strip()
        reason_s = (reason or "").strip()
        if not actor_s:
            return {"status": "unauthorized", "service": service_name,
                    "purged": 0, "audit": None, "error": "actor_required"}
        if not reason_s:
            return {"status": "unauthorized", "service": service_name,
                    "purged": 0, "audit": None, "error": "reason_required"}
        timestamp = _now_iso()
        audit_record = {
            "actor": actor_s,
            "reason": reason_s,
            "timestamp": timestamp,
            "count": 0,
            "service": service_name,
            "cap": EXACT_DEAD_LETTER_CAP,
        }
        try:
            result = self.r.eval(
                _DEAD_LETTER_PURGE_LUA,
                2,
                _dead_letter_stream(service_name),
                DEAD_LETTER_AUDIT_STREAM,
                actor_s,
                reason_s,
                timestamp,
                service_name,
                EXACT_DEAD_LETTER_CAP,
            )
            purged = int(result[0])
            audit_id = result[1]
        except redis.RedisError:
            logger.exception(
                "purge_dead_letters failed atomically %s actor=%s",
                service_name, actor_s,
            )
            return {"status": "error", "service": service_name,
                    "purged": 0, "audit": None, "error": "redis_error"}
        audit_record["count"] = purged
        audit_record["audit_id"] = audit_id
        return {"status": "ok", "service": service_name,
                "purged": purged, "audit": audit_record}

    def get_all_dead_letter_services(self) -> dict:
        """List services that have dead-lettered entries.

        Return shape (Plan 05 — outage distinguishable from empty):
            {
                "status":      "available" | "outage",
                "services":    [svc_name, ...],   # XLEN > 0
                "scanned":     int,                # total DLQ keys discovered
                "ttl":         "none",
                "archive_rotation": "none",
                "policy":      dead_letter_policy(),
            }

        Outage: any RedisError during SCAN / XLEN. Callers that previously
        used ``return []`` semantics should check ``status == "outage"``.
        """
        policy = dead_letter_policy()
        try:
            services: list[str] = []
            scanned = 0
            for key in self.r.scan_iter("dead-letter:*"):
                # The audit stream is NOT a per-service DLQ; skip it.
                if key in (DEAD_LETTER_AUDIT_STREAM, DEAD_LETTER_METRICS_HASH):
                    continue
                scanned += 1
                try:
                    length = int(self.r.xlen(key))
                except redis.RedisError:
                    # Treat a single per-key outage as outage overall.
                    raise
                if length > 0:
                    # Service name is everything after the prefix.
                    svc = key.split(":", 1)[1]
                    services.append(svc)
            return {
                "status": "available",
                "services": services,
                "scanned": scanned,
                "ttl": "none",
                "archive_rotation": "none",
                "policy": policy,
            }
        except redis.RedisError:
            logger.exception("get_all_dead_letter_services outage")
            return {
                "status": "outage",
                "services": [],
                "scanned": 0,
                "ttl": "none",
                "archive_rotation": "none",
                "policy": policy,
            }

    def get_queue_depth_observation(
        self, service_name: str, routing_group: str | None = None
    ) -> dict[str, object]:
        """Return proven unclaimed depth without guessing from stream history."""

        streams = self._stream_specs(service_name, routing_group=routing_group)
        legacy_stream = self._stream(service_name, routing_group=routing_group)
        stream = streams[0][0]
        authority = routing_group or service_name
        if authority in self.externally_owned_routing_groups:
            return {
                # The delegated broker still owns the legacy logical stream;
                # do not report an unowned priority lane as its authority.
                "stream": legacy_stream,
                "depth": 0,
                "known": False,
                "reason": "external_queue_authority",
                "pending": 0,
            }
        try:
            observations = [
                _observe_stream_depth(self.r, stream_name, self.GROUP)
                for stream_name, _lane in streams
            ]
            known = all(item.get("known") is True for item in observations)
            result: dict[str, object] = {
                "stream": stream,
                "streams": [item["stream"] for item in observations],
                "depth": sum(int(item.get("depth", 0) or 0) for item in observations),
                "known": known,
                "reason": (
                    "priority_lanes"
                    if getattr(self, "priority_lanes_enabled", False)
                    else observations[0].get("reason", "unknown")
                ),
                "pending": sum(int(item.get("pending", 0) or 0) for item in observations),
            }
            errors = [item for item in observations if item.get("error")]
            if errors:
                result["error_type"] = errors[0].get("error_type")
                result["error"] = errors[0].get("error")
            return result
        except redis.RedisError as exc:
            logger.exception("get_queue_depth failed %s", service_name)
            return {
                "stream": stream,
                "depth": 0,
                "known": False,
                "reason": "redis_error",
                "pending": 0,
                "error_type": type(exc).__name__,
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }

    def get_queue_depth(self, service_name: str,
                         routing_group: str | None = None,
                         *, strict: bool = False) -> int:
        """Return proven unclaimed depth through the legacy integer API."""

        observation = self.get_queue_depth_observation(
            service_name, routing_group=routing_group
        )
        if strict and observation.get("known") is not True:
            if observation.get("reason") != "external_queue_authority":
                reason = observation.get("reason")
                detail = observation.get("error")
                suffix = f": {detail}" if detail else ""
                # Preserve the legacy strict API's Redis exception classes
                # while keeping the observation itself safe to serialize.
                error_type = observation.get("error_type")
                redis_error_type = getattr(redis, str(error_type), None)
                if (
                    isinstance(redis_error_type, type)
                    and issubclass(redis_error_type, redis.RedisError)
                ):
                    raise redis_error_type(detail or str(reason))
                raise RuntimeError(f"queue depth unavailable: {reason}{suffix}")
        return int(observation.get("depth", 0) or 0)

    def get_oldest_wait_seconds(
        self, service_name: str, routing_group: str | None = None
    ) -> float | None:
        """Return the age of the oldest untrimmed stream entry.

        Redis stream IDs carry the enqueue millisecond, so this adds no
        payload convention and does not scan the stream.  ``None`` means the
        stream is empty or its timestamp cannot be trusted.
        """
        if (routing_group or service_name) in self.externally_owned_routing_groups:
            return None
        try:
            waits: list[float] = []
            now = time.time()
            for stream, _lane in self._stream_specs(service_name, routing_group=routing_group):
                entries = self.r.xrange(stream, count=1)
                if not entries:
                    continue
                timestamp_ms = int(str(entries[0][0]).split("-", 1)[0])
                waits.append(max(0.0, now - (timestamp_ms / 1000.0)))
            return max(waits) if waits else None
        except (redis.RedisError, TypeError, ValueError, IndexError):
            logger.exception("get_oldest_wait_seconds failed %s", service_name)
            return None

    def get_pending(self, service_name: str, routing_group: str | None = None,
                    *, strict: bool = False) -> list[dict]:
        """
        Return in-flight jobs with idle time (ms) via XPENDING.
        """
        try:
            result: list[dict] = []
            for stream, lane in self._stream_specs(service_name, routing_group=routing_group):
                # Ensure consumer group exists before querying pending
                try:
                    self.r.xgroup_create(stream, self.GROUP, id="0", mkstream=True)
                except Exception as exc:
                    if strict and "BUSYGROUP" not in str(exc):
                        raise
                    # Group already exists.
                pending = self.r.xpending_range(
                    stream, self.GROUP, min="-", max="+", count=100
                )
                for p in pending:
                    raw_id = p.get("message_id")
                    result.append({
                        "entry_id": _entry_handle(lane, raw_id) if lane else raw_id,
                        "stream": stream,
                        "consumer": p.get("consumer_name"),
                        "idle_ms": p.get("time_since_delivered", 0),
                        "deliveries": p.get("times_delivered", 1),
                    })
            return result
        except redis.RedisError:
            logger.exception("get_pending failed %s", service_name)
            if strict:
                raise
            return []

    def get_stale_pel_candidates(self, service_name: str, routing_group: str | None = None, min_idle_ms: int = 60000) -> list[dict]:
        """Dry-run helper: identify bounded stale PEL candidates without XAUTOCLAIM."""
        pending = self.get_pending(service_name, routing_group=routing_group)
        return [p for p in pending if p.get("idle_ms", 0) > min_idle_ms]


    def purge(self, service_name: str,
              routing_group: str | None = None) -> int:
        """Remove all entries from the selected queue stream(s)."""
        try:
            removed = 0
            for stream, _lane in self._stream_specs(
                service_name, routing_group=routing_group
            ):
                try:
                    info = self.r.xinfo_stream(stream)
                except redis.ResponseError as exc:
                    if _is_missing_stream_error(exc):
                        continue
                    raise
                length = int(info.get("length", 0) or 0)
                if length > 0:
                    self.r.xtrim(stream, maxlen=0, approximate=False)
                    removed += length
            logger.info("purge %s: removed %d entries", service_name, removed)
            return removed
        except redis.RedisError:
            logger.exception("purge failed %s", service_name)
            return 0

    def cleanup(self, service_name: str, max_entries: int = 10000,
                routing_group: str | None = None) -> int:
        """Trim old completed entries across selected queue stream(s)."""
        try:
            trimmed_total = 0
            for stream, _lane in self._stream_specs(
                service_name, routing_group=routing_group
            ):
                try:
                    before = int(self.r.xinfo_stream(stream).get("length", 0) or 0)
                except redis.ResponseError as exc:
                    if _is_missing_stream_error(exc):
                        continue
                    raise
                self.r.xtrim(stream, maxlen=max_entries, approximate=True)
                after = int(self.r.xinfo_stream(stream).get("length", 0) or 0)
                trimmed_total += max(0, before - after)
            if trimmed_total:
                logger.info(
                    "cleanup %s: trimmed %d entries", service_name, trimmed_total
                )
            return trimmed_total
        except redis.RedisError:
            logger.exception("cleanup failed %s", service_name)
            return 0


# ===================================================================
# 2. JobTracker  —  Redis-hash job state with TTL
# ===================================================================

class JobTracker:
    """
    Stores per-job state in Redis hashes.  Job IDs cycle from
    J-0001 through J-9999.
    """

    HASH_PREFIX = "job:"
    COUNTER_KEY = "job_counter"
    IDEMPOTENCY_PREFIX = "job-idempotency:"

    def __init__(self, r: redis.Redis | None = None):
        self.r = r or _redis_conn()

    # --- ID generation ---------------------------------------------------

    def next_job_id(self) -> str | None:
        """Atomically increment and return the next J-NNNN id."""
        try:
            val = self.r.incr(self.COUNTER_KEY)
            if val > 9999:
                # Reset to 1 atomically
                # WATCH + MULTI/EXEC for atomic reset
                with self.r.pipeline() as pipe:
                    pipe.watch(self.COUNTER_KEY)
                    current = int(pipe.get(self.COUNTER_KEY) or 0)
                    if current > 9999:
                        pipe.multi()
                        pipe.set(self.COUNTER_KEY, 1)
                        pipe.execute()
                        val = 1
                    else:
                        pipe.unwatch()
                        val = current
            return f"J-{val:04d}"
        except redis.RedisError:
            logger.exception("next_job_id failed")
            return None

    # --- hash helpers ----------------------------------------------------

    def _key(self, job_id: str) -> str:
        return f"{self.HASH_PREFIX}{job_id}"

    def preserve_unfinished_receipts(self, limit: int = 10000) -> int:
        """Migrate old TTLs without expiring, deleting or replaying any job."""
        preserved = 0
        for index, key in enumerate(self.r.scan_iter(match=self.HASH_PREFIX + '*', count=200)):
            if index >= limit:
                logger.warning("unfinished receipt retention scan reached %s keys", limit)
                break
            preserved += int(self.r.eval("""
                if redis.call('TYPE', KEYS[1]).ok ~= 'hash' then return 0 end
                local status = redis.call('HGET', KEYS[1], 'status')
                if not status or status == 'completed' or status == 'failed' or status == 'cancelled' then return 0 end
                local changed = redis.call('PERSIST', KEYS[1])
                local digest = redis.call('HGET', KEYS[1], 'idempotency_key_sha256')
                if digest then redis.call('PERSIST', ARGV[1] .. digest) end
                return changed
            """, 1, key, self.IDEMPOTENCY_PREFIX))
        return preserved

    def _idempotency_key(self, key: str) -> str:
        """Return a bounded Redis key without persisting caller input."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{self.IDEMPOTENCY_PREFIX}{digest}"

    def claim_idempotency(
        self,
        idempotency_key: str,
        request_sha256: str,
        job_id: str,
        *,
        service_name: str = "",
        routing_group: str = "",
        ttl_seconds: int = 86400,
    ) -> dict[str, str]:
        """Claim or replay one explicit admission key.

        The marker is deliberately separate from the job hash: it can be
        claimed before the queue write and lets a retry return the original
        job instead of enqueueing a second copy.  Only the SHA-256 of the
        caller key is used as the Redis key; the raw key and request body are
        never logged or persisted here.  A malformed or unavailable marker is
        reported as ``unavailable`` so callers fail closed rather than
        guessing that a duplicate is safe.
        """
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")
        if len(idempotency_key) > 256:
            raise ValueError("idempotency_key exceeds 256 characters")
        if not isinstance(request_sha256, str) or len(request_sha256) != 64:
            raise ValueError("request_sha256 must be a SHA-256 hex digest")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        ttl = int(ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        marker_key = self._idempotency_key(idempotency_key)
        marker = {
            "job_id": job_id,
            "request_sha256": request_sha256,
            "service_name": str(service_name or ""),
            "routing_group": str(routing_group or ""),
        }
        encoded = json.dumps(
            marker, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        try:
            # A queued or outcome-unknown admission can outlive a day. Its
            # uniqueness fence must not expire before the owning job finishes.
            if self.r.set(marker_key, encoded, nx=True):
                return {"status": "claimed", **marker}
            existing = self.r.get(marker_key)
        except redis.RedisError:
            logger.exception("claim_idempotency failed")
            return {"status": "unavailable", "error": "idempotency store unavailable"}
        if existing is None:
            # The marker disappeared between SET NX and GET.  Do not silently
            # create a second job; the caller can retry the admission.
            return {"status": "unavailable", "error": "idempotency marker disappeared"}
        if isinstance(existing, bytes):
            existing = existing.decode("utf-8", errors="replace")
        try:
            prior = json.loads(str(existing))
        except (TypeError, json.JSONDecodeError):
            return {"status": "unavailable", "error": "idempotency marker invalid"}
        if not isinstance(prior, dict):
            return {"status": "unavailable", "error": "idempotency marker invalid"}
        prior_hash = prior.get("request_sha256")
        prior_job = prior.get("job_id")
        if prior_hash != request_sha256:
            return {
                "status": "conflict",
                "error": "idempotency_key_reused_for_different_request",
            }
        if not isinstance(prior_job, str) or not prior_job:
            return {"status": "unavailable", "error": "idempotency marker missing job"}
        return {
            "status": "replay",
            "job_id": prior_job,
            "request_sha256": str(prior_hash),
            "service_name": str(prior.get("service_name") or ""),
            "routing_group": str(prior.get("routing_group") or ""),
        }

    # --- CRUD ------------------------------------------------------------

    def create_job(self, job_id: str, service_name: str, source: str,
                   prompt: str, max_tokens: int,
                   routing_group: str = "",
                   priority: int | None = None,
                   priority_class: str | None = None,
                   broker_priority: int | None = None,
                   parent_job_id: str = "",
                   priority_convention: str | None = None,
                   priority_source: str | None = None,
                   idempotency_key_sha256: str | None = None,
                   request_sha256: str | None = None,
                   service_revision: int | str | None = None,
                   service_revision_fingerprint: str | None = None,
                   workflow_revision: int | str | None = None,
                   workflow_fingerprint: str | None = None) -> dict | None:
        """Create a job hash. Returns the stored dict or None."""
        now = _now_iso()
        job = {
            "job_id": job_id,
            "service_name": service_name,
            "routing_group": routing_group,
            "status": "queued",
            "source": source,
            "submitted_at": now,
            "started_at": "",
            "completed_at": "",
            "assigned_slot": "",
            "prompt_preview": (prompt or "")[:200],
            "max_tokens": max_tokens,
            "result": "",
            "error": "",
            "position": "",
        }
        if priority is not None:
            job["priority"] = str(priority)
        if priority_class:
            job["priority_class"] = str(priority_class)
        if broker_priority is not None:
            job["broker_priority"] = str(broker_priority)
        if priority_convention:
            job["priority_convention"] = str(priority_convention)
        if priority_source:
            job["priority_source"] = str(priority_source)
        if idempotency_key_sha256:
            job["idempotency_key_sha256"] = str(idempotency_key_sha256)
        if request_sha256:
            job["request_sha256"] = str(request_sha256)
        if parent_job_id:
            job["parent_job_id"] = str(parent_job_id)
        for field, value in (
            ("service_revision", service_revision),
            ("service_revision_fingerprint", service_revision_fingerprint),
            ("workflow_revision", workflow_revision),
            ("workflow_fingerprint", workflow_fingerprint),
        ):
            if value not in (None, ""):
                job[field] = str(value)
        try:
            self.r.hset(self._key(job_id), mapping=job)
            # Nonterminal identities must survive long queues and outages.
            # Retention starts only once a proved terminal result is stored.
            self.r.persist(self._key(job_id))
            logger.info("create_job %s -> %s", job_id, service_name)
            return job
        except redis.RedisError:
            logger.exception("create_job failed %s", job_id)
            return None

    def update_status(self, job_id: str, status: str, **kwargs) -> bool:
        """Update status + arbitrary fields. Adjusts TTL on completion.

        Never expire a queued, running or unknown task identity. Recovery
        reconciles these records; a wall-clock TTL cannot prove completion.
        """
        try:
            key = self._key(job_id)
            updates: dict = {"status": status}
            if status in ("completed", "failed", "cancelled"):
                updates["completed_at"] = _now_iso()
            else:
                if status == "in_flight":
                    updates["started_at"] = _now_iso()
                    # A retry/re-delivery must not retain terminal metadata
                    # from the previous attempt. Otherwise an in-flight job
                    # can appear completed/failed at the same time.
                    updates["completed_at"] = ""
                    updates["error"] = ""
                    updates["result"] = ""
            updates.update({k: str(v) for k, v in kwargs.items() if v is not None})
            pipe = self.r.pipeline(transaction=True)
            pipe.hset(key, mapping=updates)
            if status in ("completed", "failed", "cancelled"):
                pipe.expire(key, 30 * 86400)
            else:
                pipe.persist(key)
            pipe.execute()
            marker_digest = self.r.hget(key, "idempotency_key_sha256")
            if marker_digest:
                # Re-read status inside Lua: a late heartbeat/completion must
                # not attach a TTL after an operator has resumed this job.
                self.r.eval("""
                    local raw = redis.call('GET', KEYS[2])
                    if not raw then return 0 end
                    local ok, marker = pcall(cjson.decode, raw)
                    if not ok or marker.job_id ~= ARGV[1] then return 0 end
                    local status = redis.call('HGET', KEYS[1], 'status')
                    if status == 'completed' or status == 'failed' or status == 'cancelled' then
                        return redis.call('EXPIRE', KEYS[2], 2592000)
                    end
                    return redis.call('PERSIST', KEYS[2])
                """, 2, key, self.IDEMPOTENCY_PREFIX + marker_digest, job_id)
            logger.debug("update_status %s -> %s", job_id, status)
            return True
        except redis.RedisError:
            logger.exception("update_status failed %s", job_id)
            return False

    def update_fields(self, job_id: str, **fields) -> bool:
        """Persist non-lifecycle fields without rewriting the job status.

        Admission handlers learn the Redis stream entry ID only after XADD.
        A worker can claim that entry before the handler records the ID; using
        ``update_status(..., "queued")`` at that point would incorrectly roll
        an already ``in_flight`` job back to queued.  This narrow field-only
        seam preserves the worker's lifecycle transition while still making
        the entry correlation durable.
        """
        try:
            updates = {
                str(name): str(value)
                for name, value in fields.items()
                if value is not None
            }
            if not updates:
                return True
            key = self._key(job_id)
            if not self.r.exists(key):
                return False
            self.r.hset(key, mapping=updates)
            return True
        except redis.RedisError:
            logger.exception("update_fields failed %s", job_id)
            return False

    def update_heartbeat(self, job_id: str, **fields) -> bool:
        """Persist liveness fields without performing a lifecycle transition.

        In particular this must not rewrite ``started_at`` or clear terminal
        metadata, which ``update_status(..., "in_flight")`` intentionally does
        when a job first enters execution or is redelivered.
        """
        try:
            key = self._key(job_id)
            updates = {
                str(name): str(value)
                for name, value in fields.items()
                if value is not None
            }
            if not updates:
                return True
            # Heartbeats must not silently put an expiry back on accepted
            # work, or resurrect an already expired terminal receipt.
            args = [item for pair in updates.items() for item in pair]
            changed = self.r.eval("""
                if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
                redis.call('HSET', KEYS[1], unpack(ARGV))
                local status = redis.call('HGET', KEYS[1], 'status')
                if status == 'completed' or status == 'failed' or status == 'cancelled' then
                    redis.call('EXPIRE', KEYS[1], 2592000)
                else
                    redis.call('PERSIST', KEYS[1])
                end
                return 1
            """, 1, key, *args)
            if not changed:
                return False
            logger.debug("update_heartbeat %s", job_id)
            return True
        except redis.RedisError:
            logger.exception("update_heartbeat failed %s", job_id)
            return False

    def mark_cancellation_intent_if_in_flight(self, job_id: str) -> dict:
        """Atomically mark cooperative cancel intent without rewriting status.

        WATCH prevents a completion transition between the status read and the
        intent write. The method never writes ``status`` itself, so it cannot
        roll a terminal record back to ``in_flight``.
        """
        key = self._key(job_id)
        for _attempt in range(3):
            try:
                with self.r.pipeline() as pipe:
                    pipe.watch(key)
                    status = pipe.hget(key, "status")
                    if not status:
                        pipe.unwatch()
                        return {"updated": False, "status": "missing"}
                    if isinstance(status, bytes):
                        status = status.decode("utf-8", errors="replace")
                    else:
                        status = str(status)
                    if status != "in_flight":
                        pipe.unwatch()
                        return {"updated": False, "status": status}
                    pipe.multi()
                    pipe.hset(key, mapping={"cancellation_intent": "1"})
                    pipe.execute()
                    return {"updated": True, "status": "in_flight"}
            except redis.WatchError:
                continue
            except redis.RedisError as exc:
                logger.exception("mark cancellation intent failed %s", job_id)
                return {"updated": False, "status": "error", "error": str(exc)}
        return {
            "updated": False,
            "status": "conflict",
            "error": "job status changed repeatedly during cancellation CAS",
        }

    def get_job(self, job_id: str) -> dict | None:
        """Return full job dict or None."""
        try:
            data = self.r.hgetall(self._key(job_id))
            return data if data else None
        except redis.RedisError:
            logger.exception("get_job failed %s", job_id)
            return None

    def get_jobs_by_status(self, service_name: str, status: str) -> list[dict]:
        """Scan for jobs of a given service + status."""
        try:
            results: list[dict] = []
            cursor = 0
            while True:
                cursor, keys = self.r.scan(
                    cursor, match=f"{self.HASH_PREFIX}*", count=200
                )
                for key in keys:
                    data = self.r.hgetall(key)
                    if (data.get("service_name") == service_name
                            and data.get("status") == status):
                        results.append(data)
                if cursor == 0:
                    break
            return results
        except redis.RedisError:
            logger.exception("get_jobs_by_status failed %s/%s", service_name, status)
            return []

    def get_jobs_by_source(self, source: str) -> list[dict]:
        """Scan for jobs from a given source."""
        try:
            results: list[dict] = []
            cursor = 0
            while True:
                cursor, keys = self.r.scan(
                    cursor, match=f"{self.HASH_PREFIX}*", count=200
                )
                for key in keys:
                    data = self.r.hgetall(key)
                    if data.get("source") == source:
                        results.append(data)
                if cursor == 0:
                    break
            return results
        except redis.RedisError:
            logger.exception("get_jobs_by_source failed %s", source)
            return []

    def demand_snapshot(
        self,
        service_name: str | None = None,
        routing_group: str | None = None,
    ) -> dict:
        """Return queued/claimed/loading/running demand from durable hashes.

        This is a diagnostic and lifecycle guard, not a dispatch hot path.
        It intentionally counts claimed and in-flight jobs so a worker claim
        cannot make an otherwise active service look idle.  Redis errors are
        surfaced as ``status=unavailable`` instead of being reported as zero.
        """
        records: list[dict] = []
        try:
            cursor = 0
            while True:
                cursor, keys = self.r.scan(
                    cursor, match=f"{self.HASH_PREFIX}*", count=200
                )
                for key in keys:
                    record = self.r.hgetall(key)
                    if not record:
                        continue
                    if service_name is not None and record.get("service_name") != service_name:
                        continue
                    if routing_group is not None and record.get("routing_group") != routing_group:
                        continue
                    records.append(record)
                if cursor == 0:
                    break
            result = demand_counts(records)
            result.update({
                "status": "available",
                "service": service_name,
                "routing_group": routing_group,
            })
            return result
        except redis.RedisError as exc:
            logger.exception("demand_snapshot failed %s", service_name)
            return {
                "status": "unavailable",
                "service": service_name,
                "routing_group": routing_group,
                "error": str(exc),
                "active_demand": None,
            }

    def delete_job(self, job_id: str) -> bool:
        """Delete a job hash entirely."""
        try:
            self.r.delete(self._key(job_id))
            logger.info("delete_job %s", job_id)
            return True
        except redis.RedisError:
            logger.exception("delete_job failed %s", job_id)
            return False


# ===================================================================
# 3. Guarded terminalization for impossible-to-resume jobs
# ===================================================================

def terminalize_if_impossible(
    tracker: JobTracker,
    job_id: str,
    *,
    stream_has_pel: bool,
    stream_has_entry: bool,
    service_running: bool,
    force: bool = False,
) -> dict:
    """Cancel an in-flight job only when every continuation signal is absent."""
    job = tracker.get_job(job_id)
    if not job:
        return {"action": "noop", "reason": "job_not_found"}
    if job.get("status") != "in_flight":
        return {"action": "noop", "reason": f"status={job.get('status')!r}"}
    if not force:
        return {"action": "noop", "reason": "force flag not set"}
    if service_running:
        return {"action": "noop", "reason": "service_running"}
    if stream_has_pel:
        return {"action": "noop", "reason": "stream_has_pel"}
    if stream_has_entry:
        return {"action": "noop", "reason": "stream_has_entry"}

    updated = tracker.update_status(
        job_id,
        "cancelled",
        error=(
            "terminalize_if_impossible: no PEL, no stream entry, "
            "service stopped"
        ),
    )
    if updated is not True:
        logger.error(
            "terminalize_if_impossible failed to persist cancellation for %s",
            job_id,
        )
        return {
            "action": "error",
            "status": "in_flight",
            "reason": "status_update_failed",
        }
    return {
        "action": "terminalized",
        "status": "cancelled",
        "reason": "no_pel_no_entry_service_stopped",
    }


# ===================================================================
# 4. SlotManager  —  Parallel worker slot tracking
# ===================================================================

class SlotManager:
    """
    Tracks per-service parallel worker slots in Redis hashes.

    Key layout: ``slots:{machine_id}:{service_name}``
    Field: slot index (0..max_slots-1) → JSON {job_id, started_at, status}
    """

    def __init__(self, r: redis.Redis | None = None,
                 service_slots: dict[str, int] | None = None):
        """
        service_slots: {service_name: max_parallel_slots}
        """
        self.r = r or _redis_conn()
        self.service_slots: dict[str, int] = service_slots or {}

    # Lua script for atomic slot acquisition: find first free slot and mark busy
    _ACQUIRE_LUA = """
    local key = KEYS[1]
    local job_id = ARGV[1]
    local now = ARGV[2]
    local all_slots = redis.call('HGETALL', key)
    for i = 1, #all_slots, 2 do
        local slot_id = all_slots[i]
        local raw = all_slots[i+1]
        local data = cjson.decode(raw)
        if data.status == 'free' then
            data.status = 'busy'
            data.job_id = job_id
            data.started_at = now
            redis.call('HSET', key, slot_id, cjson.encode(data))
            return tonumber(slot_id)
        end
    end
    return -1
    """

    def _key(self, service_name: str) -> str:
        return f"slots:{MACHINE_ID}:{service_name}"

    def _init_slots(self, service_name: str):
        """Reconcile persisted slots with the configured parallelism."""
        key = self._key(service_name)
        try:
            max_s = self.service_slots.get(service_name, 0)
            existing = self.r.hgetall(key) if self.r.exists(key) else {}
            pipe = self.r.pipeline()
            for i in range(max_s):
                if str(i) not in existing:
                    pipe.hset(key, str(i), json.dumps({
                        "job_id": "",
                        "started_at": "",
                        "status": "free",
                    }))
            for slot_id, raw in existing.items():
                if int(slot_id) >= max_s and json.loads(raw).get("status") == "free":
                    pipe.hdel(key, slot_id)
            pipe.execute()
        except redis.RedisError:
            logger.exception("_init_slots failed %s", service_name)

    # --- public API ------------------------------------------------------

    def acquire_slot(self, service_name: str, job_id: str = "") -> int | None:
        """
        Find a free slot, mark it busy, return the slot index.
        Returns None if all slots are occupied.
        Uses atomic Lua script to prevent TOCTOU race.
        """
        try:
            self._init_slots(service_name)
            key = self._key(service_name)
            slot = self.r.eval(self._ACQUIRE_LUA, 1, key, job_id, _now_iso())
            if slot < 0:
                return None
            logger.debug("acquire_slot %s -> slot %s for %s",
                         service_name, slot, job_id)
            return int(slot)
        except redis.RedisError:
            logger.exception("acquire_slot failed %s", service_name)
            return None

    def release_slot(self, service_name: str, slot_id: int) -> bool:
        """Mark a slot as free."""
        try:
            key = self._key(service_name)
            raw = self.r.hget(key, str(slot_id))
            if raw is None:
                return False
            data = json.loads(raw)
            data["status"] = "free"
            data["job_id"] = ""
            data["started_at"] = ""
            self.r.hset(key, str(slot_id), json.dumps(data))
            logger.debug("release_slot %s slot %s", service_name, slot_id)
            return True
        except redis.RedisError:
            logger.exception("release_slot failed %s/%s", service_name, slot_id)
            return False

    def get_slots(self, service_name: str) -> dict:
        """Return all slots with their current state."""
        try:
            self._init_slots(service_name)
            raw = self.r.hgetall(self._key(service_name))
            return {k: json.loads(v) for k, v in raw.items()}
        except redis.RedisError:
            logger.exception("get_slots failed %s", service_name)
            return {}

    def get_available_slots(self, service_name: str) -> int:
        """Count of free slots for a service."""
        try:
            self._init_slots(service_name)
            raw = self.r.hgetall(self._key(service_name))
            count = 0
            for v in raw.values():
                if json.loads(v).get("status") == "free":
                    count += 1
            return count
        except redis.RedisError:
            logger.exception("get_available_slots failed %s", service_name)
            return 0

    def get_slot_for_job(self, service_name: str, job_id: str) -> int | None:
        """Find which slot is running a given job."""
        try:
            raw = self.r.hgetall(self._key(service_name))
            for slot_id, v in raw.items():
                data = json.loads(v)
                if data.get("job_id") == job_id:
                    return int(slot_id)
            return None
        except redis.RedisError:
            logger.exception("get_slot_for_job failed %s/%s", service_name, job_id)
            return None

    def reap_stale_slots(self, service_name: str, max_age_seconds: float) -> list[dict]:
        """
        Force-release any slot that has been 'busy' longer than max_age_seconds.
        Returns list of reaped slot info [{slot_id, job_id, started_at, age_seconds}].
        """
        reaped = []
        try:
            self._init_slots(service_name)
            key = self._key(service_name)
            all_slots = self.r.hgetall(key)
            now = datetime.now(timezone.utc)

            for slot_id_str, raw in all_slots.items():
                data = json.loads(raw)
                if data.get("status") != "busy":
                    continue
                started = data.get("started_at", "")
                if not started:
                    # No timestamp but busy — orphan, reap immediately
                    reaped.append({
                        "slot_id": int(slot_id_str),
                        "job_id": data.get("job_id", ""),
                        "started_at": started,
                        "age_seconds": float("inf"),
                    })
                    self.release_slot(service_name, int(slot_id_str))
                    continue

                try:
                    started_dt = datetime.fromisoformat(started)
                    age = (now - started_dt).total_seconds()
                except (ValueError, TypeError):
                    age = float("inf")

                if age > max_age_seconds:
                    logger.warning(
                        "Reaping stale slot %s/%s: job=%s age=%.0fs > max=%.0fs",
                        service_name, slot_id_str, data.get("job_id", ""), age, max_age_seconds,
                    )
                    reaped.append({
                        "slot_id": int(slot_id_str),
                        "job_id": data.get("job_id", ""),
                        "started_at": started,
                        "age_seconds": age,
                    })
                    self.release_slot(service_name, int(slot_id_str))
        except redis.RedisError:
            logger.exception("reap_stale_slots failed for %s", service_name)
        return reaped


    def flush_all(self):
        """Delete all slot hashes for this machine."""
        flushed = 0
        try:
            for key in self.r.scan_iter(f"slots:{MACHINE_ID}:*"):
                self.r.delete(key)
                flushed += 1
                logger.info("flushed slot key: %s", key)
        except redis.RedisError:
            logger.exception("flush_all failed")
        return flushed


# ===================================================================
# 4. RoutingEngine  —  Dynamic inter-service routing
# ===================================================================

# ===================================================================
# 4.5. CapacityTracker — Instant in-memory capacity tracking
# ===================================================================

class CapacityTracker:
    """
    Instant capacity tracking for routing decisions.

    Tracks active (in-flight) and pending (queued) jobs per service
    using in-memory counters. Updates instantly on submit/start/complete.
    """

    def __init__(self, redis_conn=None, health_check_fn=None):
        self._active = {}    # service → count of in-flight jobs
        self._pending = {}   # service → count of submitted but not started
        self._r = redis_conn
        self._health_check_fn = health_check_fn  # callback: service_name → bool
        self._restore_from_redis()

    def _restore_from_redis(self):
        """Restore counters from Redis on startup, validating against service health."""
        if not self._r:
            return
        try:
            for key in self._r.scan_iter("capacity:track:*"):
                service = key.split(":")[-1]
                data = self._r.hgetall(key)
                redis_active = int(data.get("active", 0))
                redis_pending = int(data.get("pending", 0))

                # Validate: if service is DOWN, reset counters to zero
                # (crash left stale state)
                is_healthy = True
                if self._health_check_fn:
                    try:
                        is_healthy = self._health_check_fn(service)
                    except Exception:
                        is_healthy = True  # optimistic if check fails

                if not is_healthy and (redis_active > 0 or redis_pending > 0):
                    logger.warning("CapacityTracker: %s is DOWN but Redis says "
                                  "active=%d pending=%d — resetting to 0",
                                  service, redis_active, redis_pending)
                    self._active[service] = 0
                    self._pending[service] = 0
                    self._persist_to_redis(service)
                else:
                    self._active[service] = redis_active
                    self._pending[service] = redis_pending
        except Exception:
            logger.exception("_restore_from_redis failed")

    def _persist_to_redis(self, service):
        """Persist counters to Redis for durability."""
        if not self._r:
            return
        try:
            key = f"capacity:track:{service}"
            self._r.hset(key, mapping={
                "active": self._active.get(service, 0),
                "pending": self._pending.get(service, 0),
            })
            self._r.expire(key, 86400)
        except Exception:
            pass

    def on_submit(self, service, routing_group=None):
        """Called when a job is submitted to a service queue.

        For shared-queue services (routing_group set), the pending counter
        is tracked at the routing group level, not per-service, because
        any worker from any service in the group can claim the job.
        """
        key = routing_group if routing_group else service
        self._pending[key] = self._pending.get(key, 0) + 1
        self._persist_to_redis(key)

    def on_start(self, service, routing_group=None):
        """Called when a job starts processing.

        For shared queues, decrements the group-level pending counter.
        """
        pending_key = routing_group if routing_group else service
        self._pending[pending_key] = max(0, self._pending.get(pending_key, 0) - 1)
        self._active[service] = self._active.get(service, 0) + 1
        self._persist_to_redis(service)
        if pending_key != service:
            self._persist_to_redis(pending_key)

    def on_complete(self, service):
        """Called when a job finishes processing."""
        self._active[service] = max(0, self._active.get(service, 0) - 1)
        self._persist_to_redis(service)

    def on_cancel(self, service, routing_group=None):
        """Called when a queued job is cancelled."""
        pending_key = routing_group if routing_group else service
        self._pending[pending_key] = max(0, self._pending.get(pending_key, 0) - 1)
        self._persist_to_redis(pending_key)

    def get_capacity(self, service, slots_total, routing_group=None):
        """Get available capacity for a service.

        For shared-queue groups, pending is tracked at the group level,
        so we check the group's pending count instead of per-service.
        """
        active = self._active.get(service, 0)
        # Use group-level pending for shared queues
        pending_key = routing_group if routing_group else service
        pending = self._pending.get(pending_key, 0)
        # If never tracked, assume full capacity available
        if service not in self._active and pending_key not in self._pending:
            return slots_total
        return max(0, slots_total - active - pending)

    def get_load(self, service):
        """Get current load (active + pending) for a service."""
        return self._active.get(service, 0) + self._pending.get(service, 0)

    def get_stats(self, service):
        """Get detailed stats for a service."""
        return {
            "active": self._active.get(service, 0),
            "pending": self._pending.get(service, 0),
            "total": self._active.get(service, 0) + self._pending.get(service, 0),
        }

    def get_all_stats(self):
        """Get stats for all tracked services."""
        all_services = set(list(self._active.keys()) + list(self._pending.keys()))
        return {svc: self.get_stats(svc) for svc in all_services}

class RoutingEngine:
    """
    Routes requests within a routing group to the best available
    local service based on free slots and speed.

    service_config example:
    {
        "llm-primary": {
            "routing_group": "llm-group",
            "parallel": 16,
            "speed": 100,
        },
        ...
    }
    """

    def __init__(self, r: redis.Redis | None = None,
                 service_config: dict | None = None,
                 slot_manager: SlotManager | None = None,
                 capacity_tracker: CapacityTracker | None = None,
                 priority_lanes_enabled: bool = False):
        self.r = r or _redis_conn()
        self.service_config: dict = service_config or {}
        self.slot_mgr = slot_manager or SlotManager(r=self.r)
        self.capacity_tracker = capacity_tracker
        # Keep the lane selector opt-in until the live owner has been migrated
        # and observed in staging.  Routing depth must nevertheless understand
        # the same streams when the queue is configured for lanes.
        self.priority_lanes_enabled = bool(priority_lanes_enabled)
        # Tie-break only: capacity, lane and speed remain the policy.  This
        # cursor prevents stable config order from starving equal candidates.
        self._fair_cursors: dict[str, int] = {}

    def _choose_fair(self, routing_group: str, candidates: list[dict], key) -> dict:
        """Choose the next candidate among the best equal-scored members."""
        if not candidates:
            raise ValueError("candidates must not be empty")
        best_score = key(candidates[0])
        tied = [candidate for candidate in candidates if key(candidate) == best_score]
        if len(tied) == 1:
            return tied[0]
        cursors = getattr(self, "_fair_cursors", None)
        if cursors is None:
            cursors = self._fair_cursors = {}
        cursor = cursors.get(routing_group, 0) % len(tied)
        chosen = tied[cursor]
        cursors[routing_group] = (cursor + 1) % len(tied)
        return chosen

    # --- public API ------------------------------------------------------

    def find_service_for_group(self, routing_group: str,
                               exclude: set[str] | None = None,
                               *,
                               queue_depth: int = 0,
                               prefer_priority: bool = False) -> str | None:
        """
        Lane-fill routing within a routing_group.

        - prefer_priority=True   -> Tier 1: pick first lane (asc) with capacity > 0
        - prefer_priority=False  -> Tier 2: pick the lane with the most free slots
                                   (slot-weighted spread)
        - all full               -> Tier 3: return None  (shared queue absorbs job)

        Backward compatible: existing callers passing only routing_group and
        exclude still work — they hit Tier 2 (spread), which is the
        queue-loaded behaviour-intent.
        """
        exclude = exclude or set()
        members = self.get_group_members(routing_group)

        # 0. Filter excluded
        candidates = [m for m in members if m["service"] not in exclude]
        if not candidates:
            return None

        # Helper: read live capacity for a member
        def _cap(m: dict) -> int:
            if self.capacity_tracker:
                return self.capacity_tracker.get_capacity(
                    m["service"], m.get("slots_total", 0),
                    routing_group=routing_group,
                )
            return m.get("slots_free", 0)

        # Annotate with live capacity + lane
        for m in candidates:
            m["capacity"] = _safe_int(_cap(m), minimum=0)
            m["lane"] = _safe_int(m.get("lane"), minimum=0)

        available = [m for m in candidates if m["capacity"] > 0]

        # --- Tier 1: priority / lane-fill -----------------------------------
        if prefer_priority and available:
            # Lane-eligible first (lane >= 1), then lane asc, then speed desc
            lane_pool = [m for m in available if m["lane"] >= 1]
            if not lane_pool:
                # No lane assigned on this group — fall through to spread
                pass
            else:
                lane_pool.sort(
                    key=lambda m: (m["lane"], -m.get("speed", 0), -m.get("slots_total", 0))
                )
                chosen = self._choose_fair(
                    routing_group,
                    lane_pool,
                    key=lambda m: (
                        m["lane"],
                        -m.get("speed", 0),
                        -m.get("slots_total", 0),
                    ),
                )
                return chosen["service"]

        # --- Tier 2: spread (slot-weighted) ---------------------------------
        if available:
            # Prefer lane-assigned services; if none, use the whole pool
            pool = [m for m in available if m["lane"] >= 1] or available
            # Score: free capacity, tiebreak by lane asc, then speed desc
            pool.sort(
                key=lambda m: (
                    -m["capacity"],                # most headroom first
                    m["lane"] if m["lane"] >= 1 else 10**9,  # lane asc
                    -m.get("speed", 0),            # faster is better
                )
            )
            return self._choose_fair(
                routing_group,
                pool,
                key=lambda m: (
                    -m["capacity"],
                    m["lane"] if m["lane"] >= 1 else 10**9,
                    -m.get("speed", 0),
                ),
            )["service"]

        # --- Tier 3: all full -> shared queue -------------------------------
        return None

    def _get_queue_depth(self, service_name: str, routing_group: str | None = None) -> int:
        """Get queue depth for a service from Redis streams."""
        try:
            streams = [
                _legacy_stream_name(service_name, routing_group)
            ]
            if getattr(self, "priority_lanes_enabled", False):
                streams = [
                    *(
                        _lane_stream_name(service_name, routing_group, lane)
                        for lane in PRIORITY_LANES
                    ),
                    *streams,
                ]
            total = 0
            for stream in streams:
                try:
                    info = self.r.xinfo_stream(stream)
                except redis.ResponseError as exc:
                    if _is_missing_stream_error(exc):
                        continue
                    raise
                length = int(info.get("length", 0) or 0)
                if not length:
                    continue
                depth = length
                try:
                    groups = self.r.xinfo_groups(stream)
                    for group in groups:
                        if group.get("name") != RedisQueue.GROUP:
                            continue
                        lag = group.get("lag")
                        if lag is not None:
                            depth = min(max(0, int(lag)), length)
                        else:
                            delivered = int(group.get("entries_read", 0) or 0)
                            depth = max(0, length - delivered)
                        break
                except redis.ResponseError:
                    pass
                total += depth
            return total
        except Exception:
            return 0

    def get_group_members(self, routing_group: str) -> list[dict]:
        """All services in a routing group with current capacity info."""
        results: list[dict] = []
        for svc_name, cfg in self.service_config.items():
            if cfg.get("routing_group") != routing_group:
                continue
            # A routing group is a schedulable queue only for the two engine
            # families understood by this selector.  Provider/control-plane
            # records may still appear in the registry for health and GUI
            # discovery, but treating them as queue members creates phantom
            # capacity and can route work to a service with no GPU slot.
            declared_group_type = cfg.get("routing_group_type")
            if declared_group_type is not None and declared_group_type not in {
                "llm", "generation"
            }:
                continue
            declared_type = cfg.get("type")
            if declared_type in {
                "cloud", "provider", "orchestrator", "control_plane",
            }:
                continue
            # Disabled/retired services are registry inventory, not routing
            # candidates.  Filtering here keeps every caller (LLM, generation
            # and remote fallback) from selecting a backend that the operator
            # has explicitly taken out of service.
            if not service_is_publicly_eligible(cfg):
                continue
            slots_total = _safe_int(cfg.get("parallel", 1), default=1, minimum=1)
            slots_free = min(
                slots_total,
                _safe_int(self.slot_mgr.get_available_slots(svc_name), minimum=0),
            )
            # Include queue depth for accurate load balancing
            queue_depth = self._get_queue_depth(svc_name, routing_group)
            results.append({
                "service": svc_name,
                "routing_group": routing_group,
                "slots_total": slots_total,
                "slots_free": slots_free,
                "slots_active": slots_total - slots_free,
                "queue_depth": queue_depth,
                "total_load": (slots_total - slots_free) + queue_depth,
                "speed": _safe_float(cfg.get("speed"), minimum=0.0),
                "priority": _safe_float(cfg.get("priority"), minimum=0.0),
                "lane": _safe_int(cfg.get("lane"), minimum=0),   # NEW (lane-based routing)
            })
        return results

    def get_group_status(self, routing_group: str) -> dict:
        """Aggregated status for a routing group."""
        members = self.get_group_members(routing_group)
        if not members:
            return {"routing_group": routing_group, "members": [],
                    "slots_total": 0, "slots_free": 0, "can_accept": False}
        total = sum(m["slots_total"] for m in members)
        free = sum(m["slots_free"] for m in members)
        return {
            "routing_group": routing_group,
            "members": members,
            "slots_total": total,
            "slots_free": free,
            "can_accept": free > 0,
        }


# ===================================================================
# 5. CapacityReporter  —  Publish / consume peer capacity
# ===================================================================

class CapacityReporter:
    """
    Publishes local capacity to Redis for peer GPU Managers to read.
    Peers not seen in 30 s are considered stale.

    Key: ``capacity:{machine_id}``
    """

    PEER_TTL = 30          # seconds before a peer is stale
    PUBLISH_INTERVAL = 5   # seconds between publishes

    def __init__(self, r: redis.Redis | None = None,
                 slot_manager: SlotManager | None = None,
                 queue: RedisQueue | None = None,
                 service_config: dict | None = None):
        self.r = r or _redis_conn()
        self.slot_mgr = slot_manager or SlotManager(r=self.r)
        self.queue = queue or RedisQueue(r=self.r)
        self.service_config: dict = service_config or {}

    def _key(self, machine_id: str | None = None) -> str:
        return f"capacity:{machine_id or MACHINE_ID}"

    # --- public API ------------------------------------------------------

    def publish(self) -> bool:
        """Write current machine's capacity to Redis."""
        try:
            data = self.get_local_capacity()
            now = _now_iso()
            data["published_at"] = now
            data["last_seen"] = now
            self.r.set(self._key(), json.dumps(data), ex=self.PEER_TTL + 10)
            logger.debug("publish capacity for %s", MACHINE_ID)
            return True
        except redis.RedisError:
            logger.exception("publish capacity failed")
            return False

    def get_local_capacity(self) -> dict:
        """Build a capacity dict for every configured service."""
        capacity: dict = {"machine_id": MACHINE_ID, "services": {}}
        now = time.time()
        for svc_name, cfg in self.service_config.items():
            slots_total = cfg.get("parallel", 1)
            slots_free = self.slot_mgr.get_available_slots(svc_name)
            queue_depth = self.queue.get_queue_depth(svc_name)
            capacity["services"][svc_name] = {
                "slots_total": slots_total,
                "slots_active": slots_total - slots_free,
                "queue_depth": queue_depth,
                "can_accept": slots_free > 0,
                # Required capacity contract fields
                "capacity_source": "slot_manager",
                "identity_echo": svc_name,
                "observed_at": now,
                "freshness": "current",
            }
        return capacity

    def get_peer_capacity(self, machine_id: str) -> dict | None:
        """Read a specific peer's capacity."""
        try:
            raw = self.r.get(self._key(machine_id))
            if raw is None:
                return None
            data = json.loads(raw)
            # Check staleness
            published = data.get("published_at", "")
            if published:
                pub_t = datetime.fromisoformat(published)
                age = (datetime.now(timezone.utc) - pub_t).total_seconds()
                if age > self.PEER_TTL:
                    return None
            return data
        except (redis.RedisError, json.JSONDecodeError, ValueError):
            logger.exception("get_peer_capacity failed %s", machine_id)
            return None

    def get_all_peers(self) -> dict:
        """Return {machine_id: capacity_dict} for all known peers."""
        peers: dict = {}
        try:
            cursor = 0
            while True:
                cursor, keys = self.r.scan(
                    cursor, match="capacity:*", count=100
                )
                for key in keys:
                    # key = "capacity:{machine_id}"
                    mid = key.split(":", 1)[1] if ":" in key else key
                    raw = self.r.get(key)
                    if raw:
                        try:
                            data = json.loads(raw)
                            published = data.get("published_at", "")
                            if published:
                                pub_t = datetime.fromisoformat(published)
                                age = (datetime.now(timezone.utc) - pub_t).total_seconds()
                                if age <= self.PEER_TTL:
                                    peers[mid] = data
                        except (json.JSONDecodeError, ValueError):
                            pass
                if cursor == 0:
                    break
        except redis.RedisError:
            logger.exception("get_all_peers failed")
        return peers

    def mark_stale(self, machine_id: str) -> bool:
        """Delete a peer's capacity key to mark it stale."""
        try:
            self.r.delete(self._key(machine_id))
            logger.info("mark_stale %s", machine_id)
            return True
        except redis.RedisError:
            logger.exception("mark_stale failed %s", machine_id)
            return False


# ===================================================================
# 6. FleetGossiper  —  Cross-machine capacity gossip & remote routing
# ===================================================================

class FleetGossiper:
    """
    Push-model cross-machine capacity gossip.

    Each machine pushes its local capacity to ALL peer Redis instances,
    so any machine can read peer capacity from its own local Redis without
    needing direct connections at read time.

    Key layout (on each Redis instance):
        ``capacity:{machine_id}``  —  JSON with ``last_seen`` timestamp
    """

    STALE_SECONDS = 30       # peer capacity older than this is stale
    CAPACITY_TTL = 40        # Redis key TTL (seconds) — must be > STALE_SECONDS

    def __init__(self, local_redis, machine_id: str, peers: dict,
                 password: str, local_capacity_fn):
        """
        Parameters
        ----------
        local_redis : redis.Redis
            Connection to the local Redis instance.
        machine_id : str
            This machine's hostname (e.g. ``'host'``).
        peers : dict
            ``{machine_id: ip_address}`` for every peer machine.
            Does NOT include self.
        password : str
            Shared Redis AUTH password for all instances.
        local_capacity_fn : callable
            ``() -> dict`` returning the current capacity snapshot
            (same format as ``CapacityReporter.get_local_capacity()``).
        """
        self.r = local_redis
        self.machine_id = machine_id
        self.peers: dict[str, str] = peers          # {mid: ip}
        self.password = password
        self.local_capacity_fn = local_capacity_fn

        # Cache peer Redis connections: {ip: redis.Redis}
        self._peer_conns: dict[str, redis.Redis] = {}

    # --- internal helpers ------------------------------------------------

    def _get_peer_conn(self, ip: str) -> redis.Redis:
        """Return a cached Redis connection to a peer (create on first use)."""
        if ip not in self._peer_conns:
            self._peer_conns[ip] = redis.Redis(
                host=ip, port=REDIS_PORT, password=self.password,
                socket_timeout=2, socket_connect_timeout=2,
                decode_responses=True,
            )
        return self._peer_conns[ip]

    def _capacity_key(self, machine_id: str | None = None) -> str:
        return f"capacity:{machine_id or self.machine_id}"

    @staticmethod
    def _is_stale(data: dict, max_age: float | None = None) -> bool:
        """Return True if the capacity dict is stale or missing a timestamp."""
        max_age = max_age or FleetGossiper.STALE_SECONDS
        last_seen = data.get("last_seen", "")
        if not last_seen:
            return True
        try:
            ts = datetime.fromisoformat(last_seen)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return age > max_age
        except (ValueError, TypeError):
            return True

    # --- public API ------------------------------------------------------

    def publish_to_peers(self) -> dict[str, bool]:
        """
        Push local capacity to ALL peer Redis instances AND to local Redis.

        Returns
        -------
        dict
            ``{machine_id: bool}`` — True if write succeeded for that peer.
        """
        capacity = self.local_capacity_fn()
        capacity["last_seen"] = _now_iso()
        payload = json.dumps(capacity)
        key = self._capacity_key()
        results: dict[str, bool] = {}

        # Always write to local Redis first
        try:
            self.r.set(key, payload, ex=self.CAPACITY_TTL)
            results[self.machine_id] = True
        except redis.RedisError:
            logger.exception("publish_to_peers: local write failed")
            results[self.machine_id] = False

        # Push to every peer
        for peer_mid, peer_ip in self.peers.items():
            try:
                conn = self._get_peer_conn(peer_ip)
                conn.set(key, payload, ex=self.CAPACITY_TTL)
                results[peer_mid] = True
                logger.debug("publish_to_peers: pushed to %s (%s)", peer_mid, peer_ip)
            except (redis.RedisError, OSError, ConnectionError, TimeoutError):
                logger.warning("publish_to_peers: peer %s (%s) unreachable, skipping",
                               peer_mid, peer_ip)
                results[peer_mid] = False

        return results

    def read_peer_capacity(self, peer_machine_id: str) -> dict | None:
        """
        Read a specific peer's capacity from LOCAL Redis.

        Returns the capacity dict, or None if missing / stale.
        """
        try:
            key = self._capacity_key(peer_machine_id)
            raw = self.r.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            if self._is_stale(data):
                return None
            return data
        except (redis.RedisError, json.JSONDecodeError):
            logger.exception("read_peer_capacity failed for %s", peer_machine_id)
            return None

    def read_all_peers(self) -> dict[str, dict | None]:
        """
        Return ``{machine_id: capacity_dict_or_None}`` for all known peers.

        Peers with stale / missing data map to None.
        """
        result: dict[str, dict | None] = {}
        for peer_mid in self.peers:
            result[peer_mid] = self.read_peer_capacity(peer_mid)
        return result

    def find_remote_service(self, routing_group: str,
                            exclude_services: set[str] | None = None) -> tuple | None:
        """
        Find a peer that has a service in *routing_group* with free slots.

        Returns
        -------
        ``(machine_id, service_name, capacity_info)`` or None

        Does NOT check health of remote services — only published capacity.
        """
        exclude_services = exclude_services or set()
        candidates: list[tuple[str, str, dict, int]] = []  # (mid, svc, info, free)

        for peer_mid in self.peers:
            cap = self.read_peer_capacity(peer_mid)
            if cap is None:
                continue
            services = cap.get("services", {})
            for svc_name, svc_info in services.items():
                if svc_name in exclude_services:
                    continue
                if not svc_info.get("can_accept", False):
                    continue
                # The service must belong to the target routing group.
                # Peer capacity dicts may include routing_group per service,
                # or we fall back to matching by name convention.
                svc_rg = svc_info.get("routing_group", "")
                if svc_rg and svc_rg != routing_group:
                    continue
                slots_free = svc_info.get("slots_free",
                                          svc_info.get("slots_total", 0)
                                          - svc_info.get("slots_active", 0))
                if slots_free <= 0:
                    continue
                candidates.append((peer_mid, svc_name, svc_info, slots_free))

        if not candidates:
            return None

        # Sort by most free slots first
        candidates.sort(key=lambda c: c[3], reverse=True)
        mid, svc, info, _ = candidates[0]
        return (mid, svc, info)


# ===================================================================
# 8. QueueMetrics — Time-windowed performance tracking
# ===================================================================

class QueueMetrics:
    """Collect queue performance metrics with time-windowed aggregation.

    Uses Redis Sorted Sets for sliding-window metrics:
    - Key: metrics:{service}:completed  (score=timestamp, member=event_id:duration_ms)
    - Key: metrics:{service}:failed     (score=timestamp, member=event_id:error_type)
    - Key: metrics:{service}:latency    (score=timestamp, member=event_id:latency_ms)
    - TTL: 48h on all metrics keys
    """

    WINDOW_5M = 300
    WINDOW_1H = 3600
    WINDOW_24H = 86400
    KEY_TTL = 86400 * 2  # 48h

    def __init__(self, r):
        self.r = r  # synchronous Redis — callers must use run_in_executor in async context

    def _key(self, service: str, metric: str) -> str:
        return f"metrics:{service}:{metric}"

    def record_completion(self, service: str, duration_ms: int):
        """Record a completed job with its duration."""
        key = self._key(service, "completed")
        pipe = self.r.pipeline()
        pipe.zadd(key, {f"{uuid.uuid4().hex}:{int(duration_ms)}": time.time()})
        pipe.expire(key, self.KEY_TTL)
        pipe.execute()

    def record_failure(self, service: str, error_type: str = "unknown"):
        """Record a failed job."""
        key = self._key(service, "failed")
        pipe = self.r.pipeline()
        pipe.zadd(key, {f"{uuid.uuid4().hex}:{error_type}": time.time()})
        pipe.expire(key, self.KEY_TTL)
        pipe.execute()

    def record_latency(self, service: str, latency_ms: int):
        """Record request latency (time from submit to first response)."""
        key = self._key(service, "latency")
        pipe = self.r.pipeline()
        pipe.zadd(key, {f"{uuid.uuid4().hex}:{int(latency_ms)}": time.time()})
        pipe.expire(key, self.KEY_TTL)
        pipe.execute()

    def get_stats(self, service: str, window: int = WINDOW_1H) -> dict:
        """Get stats for a service within a time window."""
        def numeric_value(member) -> int:
            text = member.decode("utf-8") if isinstance(member, bytes) else str(member)
            # New records use a unique prefix so equal measurements do not
            # replace one another. Legacy values remain readable until expiry.
            return int(text.rsplit(":", 1)[-1])

        now = time.time()
        cutoff = now - window

        # Get completed count
        completed_key = self._key(service, "completed")
        completed = self.r.zrangebyscore(completed_key, cutoff, now)
        total_completed = len(completed)
        avg_duration = (
            sum(numeric_value(member) for member in completed) // total_completed
            if total_completed
            else None
        )

        # Get failed count
        failed_key = self._key(service, "failed")
        failed = self.r.zrangebyscore(failed_key, cutoff, now)
        total_failed = len(failed)

        # Throughput: completed per minute
        throughput = total_completed / (window / 60) if window > 0 else 0

        # Success rate
        total = total_completed + total_failed
        success_rate = (total_completed / total * 100) if total else None

        # Latency percentiles
        latency_key = self._key(service, "latency")
        latencies = self.r.zrangebyscore(latency_key, cutoff, now)
        latencies_int = sorted(numeric_value(member) for member in latencies)
        p50 = latencies_int[len(latencies_int) // 2] if latencies_int else None
        p90 = latencies_int[int(len(latencies_int) * 0.9)] if latencies_int else None
        p99 = latencies_int[int(len(latencies_int) * 0.99)] if latencies_int else None

        return {
            "total_completed": total_completed,
            "total_failed": total_failed,
            "avg_duration_ms": avg_duration,
            "success_rate": round(success_rate, 1) if success_rate is not None else None,
            "sample_count": total,
            "sample_state": "observed" if total else "no_samples",
            "throughput_per_min": round(throughput, 2),
            "latency_p50_ms": p50,
            "latency_p90_ms": p90,
            "latency_p99_ms": p99,
            "window_seconds": window,
        }

    def get_all_stats(self, services: list[str], window: int = WINDOW_1H) -> dict:
        """Get stats for all services."""
        return {svc: self.get_stats(svc, window) for svc in services}

    def cleanup_old_entries(self, max_age_seconds: int = WINDOW_24H * 2):
        """Remove entries older than max_age (called periodically)."""
        cutoff = time.time() - max_age_seconds
        for pattern in ["metrics:*:completed", "metrics:*:failed", "metrics:*:latency"]:
            for key in self.r.scan_iter(pattern):
                self.r.zremrangebyscore(key, "-inf", cutoff)


# ===================================================================
# Async Redis Wrappers — non-blocking versions for the event loop
# ===================================================================

class AsyncRedisQueue:
    """
    Async version of RedisQueue.  Uses redis.asyncio for non-blocking I/O.
    Same key layout: ``queue:{service_name}``, group ``workers``.
    """

    GROUP = "workers"

    def __init__(
        self,
        r=None,
        *,
        priority_lanes_enabled: bool = False,
        priority_aging_interval_s: float = 60.0,
        priority_aging_cap: int = 3,
    ):
        """
        Parameters
        ----------
        r : aioredis.Redis | None
            An async Redis connection.  If None, one is created on first use
            (lazy via ``_get_conn``).
        """
        self._r = r  # may be None — lazily created
        self._owns_conn = r is None  # True if we created the connection
        self.priority_lanes_enabled = bool(priority_lanes_enabled)
        self.priority_aging_interval_s = max(1.0, float(priority_aging_interval_s))
        self.priority_aging_cap = max(0, int(priority_aging_cap))

    async def _get_conn(self):
        """Return the async Redis connection, creating it lazily if needed."""
        if self._r is None:
            self._r = await _async_redis_conn()
        return self._r

    async def close(self):
        """Close the async connection if we own it."""
        if self._r and self._owns_conn:
            try:
                await self._r.aclose()
            except Exception:
                pass
            self._r = None

    def _stream(self, service_name: str, routing_group: str | None = None) -> str:
        # Phase 5.2.80 R1 guard: refuse to silently create "queue:" (empty
        # stream key). Caller MUST supply either a routing_group OR a
        # service_name; if both are empty, raise so the bug surfaces in
        # tests rather than silently accumulating orphans. Defense in depth
        # alongside the gpu-manager proxy-direct routing_group root-cause fix.
        if routing_group:
            return f"queue:{routing_group}"
        if service_name:
            return f"queue:{service_name}"
        raise ValueError(
            "queue_engine._stream requires service_name or routing_group"
        )

    def _stream_specs(
        self, service_name: str, routing_group: str | None = None
    ) -> list[tuple[str, str | None]]:
        legacy = self._stream(service_name, routing_group=routing_group)
        if not getattr(self, "priority_lanes_enabled", False):
            return [(legacy, None)]
        return [
            *(
                (_lane_stream_name(service_name, routing_group, lane), lane)
                for lane in PRIORITY_LANES
            ),
            (legacy, None),
        ]

    def resolve_entry_stream(
        self, service_name: str, entry_id: str, routing_group: str | None = None
    ) -> tuple[str, str, str | None]:
        return _parse_entry_handle(entry_id, service_name, routing_group)

    async def _ensure_group(self, service_name: str,
                            routing_group: str | None = None):
        """Create consumer group if it doesn't exist (idempotent)."""
        r = await self._get_conn()
        stream = self._stream(service_name, routing_group=routing_group)
        try:
            await r.xgroup_create(stream, self.GROUP, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue(self, service_name: str, job_data: dict,
                        routing_group: str | None = None) -> str | None:
        """Add a job to the stream (async)."""
        try:
            await self._ensure_group(service_name, routing_group=routing_group)
            r = await self._get_conn()
            convention = job_data.get("priority_convention")
            if convention is None:
                convention = (
                    "generation_higher_is_urgent"
                    if job_data.get("routing_group_type") == "generation"
                    else "legacy_lower_is_urgent"
                )
            contracted = apply_priority_contract(
                job_data, convention=convention
            )
            lane = _priority_lane(contracted)
            stream = self._stream(service_name, routing_group=routing_group)
            handle_lane: str | None = None
            if getattr(self, "priority_lanes_enabled", False):
                stream = _lane_stream_name(service_name, routing_group, lane)
                contracted.setdefault("enqueued_at", _now_iso())
                contracted["priority_lane"] = lane
                handle_lane = lane
            payload = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in contracted.items()
            }
            entry_id = await r.xadd(stream, payload)
            handle = _entry_handle(handle_lane, entry_id) if handle_lane else entry_id
            logger.info("async enqueue %s -> %s", service_name, handle)
            return handle
        except Exception:
            logger.exception("async enqueue failed for %s", service_name)
            return None

    async def dequeue(self, service_name: str, consumer_name: str,
                      count: int = 1, block_ms: int = 0,
                      routing_group: str | None = None) -> list[dict]:
        """Claim up to *count* pending jobs (async).

        block_ms > 0: block up to N milliseconds waiting for a job (XREADGROUP BLOCK).
        block_ms = 0: return immediately if no jobs available.

        Phase 5.2.3 Path A: removed XAUTOCLAIM. See
        plans/PATH_A_VS_B_TRADEOFF.md
        for the full tradeoff. XAUTOCLAIM (Phase 5.2.1 Fix B) was solving
        the dead-consumer-PEL problem that the reaper (Phase 5.2.1 Fix A)
        already solves at idle > 300s. XAUTOCLAIM introduced a regression
        where stale entries (idle > 60s, before the reaper fires) were
        claimed by the worker and the worker hung on a broken payload.
        The reaper-only path is simpler and observably correct.
        """
        try:
            r = await self._get_conn()
            specs = self._stream_specs(service_name, routing_group=routing_group)
            for stream, _lane in specs:
                try:
                    await r.xgroup_create(stream, self.GROUP, id="0", mkstream=True)
                except Exception as exc:
                    if "BUSYGROUP" not in str(exc):
                        raise

            if getattr(self, "priority_lanes_enabled", False):
                candidates: list[tuple[tuple[int, float, int], str, str | None]] = []
                now = time.time()
                for stream, lane in specs:
                    try:
                        groups = await r.xinfo_groups(stream)
                        last_id = "0-0"
                        for group in groups:
                            if group.get("name") == self.GROUP:
                                last_id = group.get("last-delivered-id") or "0-0"
                                break
                        rows = await r.xread({stream: last_id}, count=1)
                    except (AttributeError, TypeError):
                        # Preserve the existing XREADGROUP path for minimal
                        # Redis/fakeredis doubles that do not implement the
                        # optional ready-entry peek.
                        rows = []
                    except redis.RedisError:
                        # A real Redis outage must remain visible to the outer
                        # handler rather than being misreported as an empty
                        # queue.
                        raise
                    if rows:
                        for _stream_key, entries in rows:
                            if entries:
                                raw_id = str(entries[0][0])
                                candidates.append(
                                    (
                                        _priority_age_score(
                                            lane,
                                            raw_id,
                                            now=now,
                                            aging_interval_s=self.priority_aging_interval_s,
                                            aging_cap=self.priority_aging_cap,
                                        ),
                                        stream,
                                        lane,
                                    )
                                )
                if candidates:
                    _, selected_stream, selected_lane = max(candidates, key=lambda item: item[0])
                    claimed = await r.xreadgroup(
                        groupname=self.GROUP,
                        consumername=consumer_name,
                        streams={selected_stream: ">"},
                        count=count,
                    )
                    return self._decode_claimed_entries(claimed, selected_lane)

            kwargs = dict(
                groupname=self.GROUP,
                consumername=consumer_name,
                streams={stream: ">" for stream, _lane in specs},
                count=count,
            )
            if block_ms and block_ms > 0:
                kwargs["block"] = block_ms
            streams = await r.xreadgroup(**kwargs)
            return self._decode_claimed_entries(streams)
        except Exception:
            logger.exception("async dequeue failed for %s", service_name)
            return []  # noqa: E501

    @staticmethod
    def _decode_claimed_entries(streams, selected_lane: str | None = None) -> list[dict]:
        results: list[dict] = []
        for stream_key, entries in streams or []:
            stream_text = str(stream_key)
            lane = selected_lane
            if lane is None and ":lane:" in stream_text:
                lane = stream_text.rsplit(":lane:", 1)[-1]
            for entry_id, data in entries:
                data["_entry_id"] = _entry_handle(lane, entry_id) if lane in _PRIORITY_LANE_RANK else entry_id
                results.append(data)
        return results

    async def ack(self, service_name: str, consumer_name: str, entry_id: str,
                   routing_group: str | None = None) -> bool:
        """Mark a job as successfully processed (async)."""
        try:
            r = await self._get_conn()
            stream, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )
            # Phase 5.2.80 R3b: XACK only removes from the consumer-group
            # PEL; XLEN still counts the entry until explicit XDEL or
            # XTRIM-0. approximate=True only trims when XLEN >= maxlen, so
            # at small XLEN the audit-trail-of-4 stuck forever.
            #
            # The three independent awaits (XACK, XDEL, XTRIM) are
            # non-atomic. If the worker crashes or the XDEL raises between
            # XACK and XDEL, the entry is acked from the PEL but stays in
            # the stream forever (until XTRIM happens to fire on a stream
            # that has accumulated >= maxlen entries).
            #
            # Fix: use a Redis pipeline with MULTI/EXEC so XACK+XDEL+XTRIM
            # commit atomically. The pipeline pattern is already used in
            # this file at _init_slots below/in this file. Drop the inner
            # try/except around XTRIM (transaction atomicity means partial
            # failure should roll back, not swallow).
            pipe = r.pipeline(transaction=True)
            pipe.xack(stream, self.GROUP, raw_entry_id)
            pipe.xdel(stream, raw_entry_id)
            pipe.xtrim(stream, maxlen=500, approximate=False)
            await pipe.execute()
            logger.debug("async ack %s %s", service_name, entry_id)
            return True
        except Exception:
            logger.exception("async ack failed %s %s", service_name, entry_id)
            return False

    async def renew_pending(
        self,
        service_name: str,
        consumer_name: str,
        entry_id: str,
        routing_group: str | None = None,
    ) -> bool:
        """Reset one owned PEL entry's idle clock without changing its payload.

        Long-running internal workflows use this as a transport heartbeat. If
        the controller dies, renewal stops and the normal pending-entry reaper
        can safely requeue the parent after its declared lease.
        """
        try:
            r = await self._get_conn()
            stream, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )
            claimed = await r.xclaim(
                stream,
                self.GROUP,
                consumer_name,
                min_idle_time=0,
                message_ids=[raw_entry_id],
                justid=True,
            )
            return bool(claimed)
        except Exception:
            logger.exception("async renew_pending failed %s %s", service_name, entry_id)
            return False

    async def nack(self, service_name: str, entry_id: str, max_retries: int = 3,
                    routing_group: str | None = None) -> bool:
        """Re-queue a failed / stuck job (async)."""
        try:
            r = await self._get_conn()
            stream, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )

            entries = await r.xrange(stream, min=raw_entry_id, max=raw_entry_id, count=1)
            if not entries:
                pipe = r.pipeline(transaction=True)
                pipe.xack(stream, self.GROUP, raw_entry_id)
                await pipe.execute()
                logger.debug("nack: entry %s already trimmed from %s", entry_id, stream)
                return True  # PEL cleaned, that's what matters
            _, original_data = entries[0]

            raw_nack_count = original_data.get("_nack_count", original_data.get(b"_nack_count", 0))
            if isinstance(raw_nack_count, bytes): raw_nack_count = raw_nack_count.decode()
            nack_count = int(raw_nack_count or 0) + 1

            if nack_count >= max_retries:
                dlq_stream = _dead_letter_stream(service_name)
                payload = {
                    **original_data,
                    "_nack_count": str(nack_count),
                    "_dead_letter_cap": str(EXACT_DEAD_LETTER_CAP),
                    "_dead_letter_ts": _now_iso(),
                }
                result = await r.eval(
                    _DEAD_LETTER_TRANSFER_LUA,
                    3,
                    stream,
                    dlq_stream,
                    DEAD_LETTER_METRICS_HASH,
                    raw_entry_id,
                    self.GROUP,
                    EXACT_DEAD_LETTER_CAP,
                    service_name,
                    *_flatten_stream_payload(payload),
                )
                dropped = int(result[1])
                dropped_total = int(result[2])
                logger.warning(
                    "async nack: moved %s to dead-letter after %d attempts on %s "
                    "(cap=%d dropped=%d dropped_total=%d)",
                    entry_id, nack_count, service_name,
                    EXACT_DEAD_LETTER_CAP, dropped, dropped_total,
                )
                return True

            pipe = r.pipeline(transaction=True)
            pipe.xack(stream, self.GROUP, raw_entry_id)
            pipe.xdel(stream, raw_entry_id)
            payload = {**original_data, "_nack_count": str(nack_count)}
            pipe.xadd(stream, payload)
            # Exact trim prevents small audit-trail streams from lingering.
            # XDEL removes processed entry so exact trim normally has little work.
            pipe.xtrim(stream, maxlen=500, approximate=False)
            results = await pipe.execute()
            new_id = results[2]
            logger.info("async nack (re-queued %d/%d) %s %s -> %s",
                        nack_count, max_retries, service_name, entry_id, new_id)
            return True
        except Exception:
            logger.exception("async nack failed %s %s", service_name, entry_id)
            return False

    async def yield_pending(self, service_name: str, entry_id: str,
                            routing_group: str | None = None) -> bool:
        """
        Re-queue a job without failing it (e.g. paused / backoff).
        ACKs current entry and XADDs original payload WITHOUT incrementing
        _nack_count or dead-lettering.
        """
        try:
            r = await self._get_conn()
            stream, raw_entry_id, _lane = self.resolve_entry_stream(
                service_name, entry_id, routing_group=routing_group
            )

            entries = await r.xrange(stream, min=raw_entry_id, max=raw_entry_id, count=1)
            if not entries:
                pipe = r.pipeline(transaction=True)
                pipe.xack(stream, self.GROUP, raw_entry_id)
                await pipe.execute()
                logger.debug("yield_pending: entry %s already trimmed from %s", entry_id, stream)
                return True
            _, original_data = entries[0]

            raw_yield_count = original_data.get("_yield_count", original_data.get(b"_yield_count", 0))
            if isinstance(raw_yield_count, bytes): raw_yield_count = raw_yield_count.decode()
            yield_count = int(raw_yield_count or 0) + 1

            pipe = r.pipeline(transaction=True)
            pipe.xack(stream, self.GROUP, raw_entry_id)
            pipe.xdel(stream, raw_entry_id)

            payload = {**original_data, "_yield_count": str(yield_count)}
            pipe.xadd(stream, payload)
            pipe.xtrim(stream, maxlen=500, approximate=False)
            results = await pipe.execute()
            new_id = results[2]
            logger.info("async yield_pending (yield #%d) %s %s -> %s",
                        yield_count, service_name, entry_id, new_id)
            return True
        except Exception:
            logger.exception("async yield_pending failed %s %s", service_name, entry_id)
            return False

    # --- Plan 05 dead-letter mirrors (async surface, semantically aligned) -

    async def get_dead_letters(self, service_name: str, count: int = 50) -> dict:
        """Async mirror of ``RedisQueue.get_dead_letters``.

        Same return shape — outage distinguishable from empty.
        """
        dlq_stream = _dead_letter_stream(service_name)
        policy = dead_letter_policy()
        empty = {
            "status": "available",
            "service": service_name,
            "cap": EXACT_DEAD_LETTER_CAP,
            "entries": [],
            "count": 0,
            "total": 0,
            "dropped": 0,
            "ttl": "none",
            "archive_rotation": "none",
            "policy": policy,
        }
        try:
            r = await self._get_conn()
            raw = await r.xrange(dlq_stream, count=count)
            total = int(await r.xlen(dlq_stream))
            dropped = int(
                await r.hget(DEAD_LETTER_METRICS_HASH, service_name) or 0
            )
        except Exception:
            logger.exception("async get_dead_letters outage %s", service_name)
            return {**empty, "status": "outage"}
        entries = [{"entry_id": e[0], **e[1]} for e in raw]
        return {
            "status": "available",
            "service": service_name,
            "cap": EXACT_DEAD_LETTER_CAP,
            "entries": entries,
            "count": len(entries),
            "total": total,
            "dropped": dropped,
            "ttl": "none",
            "archive_rotation": "none",
            "policy": policy,
        }

    async def purge_dead_letters(
        self,
        service_name: str,
        actor: str,
        reason: str,
    ) -> dict:
        """Async mirror of the atomic audit-before-purge contract."""
        actor_s = (actor or "").strip()
        reason_s = (reason or "").strip()
        if not actor_s:
            return {"status": "unauthorized", "service": service_name,
                    "purged": 0, "audit": None, "error": "actor_required"}
        if not reason_s:
            return {"status": "unauthorized", "service": service_name,
                    "purged": 0, "audit": None, "error": "reason_required"}
        timestamp = _now_iso()
        audit_record = {
            "actor": actor_s, "reason": reason_s, "timestamp": timestamp,
            "count": 0, "service": service_name,
            "cap": EXACT_DEAD_LETTER_CAP,
        }
        try:
            r = await self._get_conn()
            result = await r.eval(
                _DEAD_LETTER_PURGE_LUA,
                2,
                _dead_letter_stream(service_name),
                DEAD_LETTER_AUDIT_STREAM,
                actor_s, reason_s, timestamp, service_name,
                EXACT_DEAD_LETTER_CAP,
            )
            purged = int(result[0])
            audit_id = result[1]
        except Exception:
            logger.exception(
                "async purge_dead_letters failed atomically %s actor=%s",
                service_name, actor_s,
            )
            return {"status": "error", "service": service_name,
                    "purged": 0, "audit": None, "error": "redis_error"}
        audit_record["count"] = purged
        audit_record["audit_id"] = audit_id
        return {"status": "ok", "service": service_name,
                "purged": purged, "audit": audit_record}

    async def get_all_dead_letter_services(self) -> dict:
        """Async mirror of ``RedisQueue.get_all_dead_letter_services``.

        Outage distinguishable from empty.
        """
        policy = dead_letter_policy()
        try:
            r = await self._get_conn()
            services: list[str] = []
            scanned = 0
            async for key in r.scan_iter("dead-letter:*"):
                if key in (DEAD_LETTER_AUDIT_STREAM, DEAD_LETTER_METRICS_HASH):
                    continue
                scanned += 1
                length = int(await r.xlen(key))
                if length > 0:
                    svc = key.split(":", 1)[1]
                    services.append(svc)
            return {
                "status": "available",
                "services": services,
                "scanned": scanned,
                "ttl": "none",
                "archive_rotation": "none",
                "policy": policy,
            }
        except Exception:
            logger.exception("async get_all_dead_letter_services outage")
            return {
                "status": "outage",
                "services": [],
                "scanned": 0,
                "ttl": "none",
                "archive_rotation": "none",
                "policy": policy,
            }

    async def get_queue_depth(self, service_name: str,
                               routing_group: str | None = None) -> int:
        """Number of entries still waiting to be consumed (async).

        When the opt-in lane reader is active, depth is the sum of the three
        lane streams and the legacy stream retained for drain/replay.  Pending
        entries are deliberately not counted as ready work, matching the
        synchronous observation contract and avoiding false capacity.
        """
        try:
            r = await self._get_conn()
            total = 0
            for stream, _lane in self._stream_specs(
                service_name, routing_group=routing_group
            ):
                try:
                    info = await r.xinfo_stream(stream)
                except redis.ResponseError as exc:
                    if _is_missing_stream_error(exc):
                        continue
                    raise
                length = int(info.get("length", 0) or 0)
                if length == 0:
                    continue
                depth = length
                try:
                    groups = await r.xinfo_groups(stream)
                    for group in groups:
                        if group.get("name") != self.GROUP:
                            continue
                        lag = group.get("lag")
                        if lag is not None:
                            depth = min(max(0, int(lag)), length)
                        else:
                            delivered = int(group.get("entries_read", 0) or 0)
                            depth = max(0, length - delivered)
                        break
                except redis.ResponseError:
                    # A stream can exist briefly before its worker group is
                    # created; XLEN is the safe conservative fallback.
                    pass
                total += depth
            return total
        except Exception:
            logger.exception("async get_queue_depth failed %s", service_name)
            return 0

    async def get_pending(
        self,
        service_name: str,
        routing_group: str | None = None,
        *,
        strict: bool = False,
    ) -> list[dict]:
        """Return in-flight jobs with idle time (async).

        Phase 5.2.1 Fix A: routing_group targets the per-rg stream
        (``queue:{routing_group}``) instead of ``queue:{service_name}``.
        None falls back to the legacy stream name.

        ``strict=True`` preserves Redis/Valkey errors for recovery callers
        that must distinguish an empty PEL from an unavailable broker.  The
        compatibility default retains the historical empty-list fallback for
        non-critical observers.
        """
        try:
            r = await self._get_conn()
            result: list[dict] = []
            for stream, lane in self._stream_specs(
                service_name, routing_group=routing_group
            ):
                try:
                    await r.xgroup_create(
                        stream, self.GROUP, id="0", mkstream=True
                    )
                except Exception:
                    pass
                pending = await r.xpending_range(
                    stream, self.GROUP, min="-", max="+", count=100
                )
                for item in pending:
                    raw_id = item.get("message_id")
                    result.append({
                        "entry_id": _entry_handle(lane, raw_id) if lane else raw_id,
                        "stream": stream,
                        "consumer": item.get("consumer_name"),
                        "idle_ms": item.get("time_since_delivered", 0),
                        "deliveries": item.get("times_delivered", 1),
                    })
            return result
        except redis.RedisError:
            logger.exception("async get_pending failed %s", service_name)
            if strict:
                raise
            return []

    async def get_stale_pel_candidates(self, service_name: str, routing_group: str | None = None, min_idle_ms: int = 60000) -> list[dict]:
        """Dry-run helper: identify bounded stale PEL candidates without XAUTOCLAIM."""
        pending = await self.get_pending(service_name, routing_group=routing_group)
        return [p for p in pending if p.get("idle_ms", 0) > min_idle_ms]

    async def purge(self, service_name: str,
                    routing_group: str | None = None) -> int:
        """Remove all entries from the selected queue stream(s) (async)."""
        try:
            r = await self._get_conn()
            removed = 0
            for stream, _lane in self._stream_specs(
                service_name, routing_group=routing_group
            ):
                try:
                    info = await r.xinfo_stream(stream)
                except redis.ResponseError as exc:
                    if _is_missing_stream_error(exc):
                        continue
                    raise
                length = int(info.get("length", 0) or 0)
                if length > 0:
                    await r.xtrim(stream, maxlen=0, approximate=False)
                    removed += length
            logger.info("async purge %s: removed %d entries", service_name, removed)
            return removed
        except Exception:
            logger.exception("async purge failed %s", service_name)
            return 0

    async def cleanup(self, service_name: str, max_entries: int = 10000,
                      routing_group: str | None = None) -> int:
        """Trim old completed entries across selected queue stream(s) (async)."""
        try:
            r = await self._get_conn()
            trimmed_total = 0
            for stream, _lane in self._stream_specs(
                service_name, routing_group=routing_group
            ):
                try:
                    info_before = await r.xinfo_stream(stream)
                except redis.ResponseError as exc:
                    if _is_missing_stream_error(exc):
                        continue
                    raise
                before = int(info_before.get("length", 0) or 0)
                await r.xtrim(stream, maxlen=max_entries, approximate=True)
                info_after = await r.xinfo_stream(stream)
                after = int(info_after.get("length", 0) or 0)
                trimmed_total += max(0, before - after)
            if trimmed_total:
                logger.info(
                    "async cleanup %s: trimmed %d entries",
                    service_name, trimmed_total,
                )
            return trimmed_total
        except Exception:
            logger.exception("async cleanup failed %s", service_name)
            return 0


class AsyncSlotManager:
    """
    Async version of SlotManager.  Same key layout: ``slots:{machine_id}:{service_name}``.
    """

    def __init__(self, r=None, service_slots: dict[str, int] | None = None):
        self._r = r
        self._owns_conn = r is None
        self.service_slots: dict[str, int] = service_slots or {}

    # Lua script for atomic slot acquisition (same as sync version)
    _ACQUIRE_LUA = """
    local key = KEYS[1]
    local job_id = ARGV[1]
    local now = ARGV[2]
    local all_slots = redis.call('HGETALL', key)
    for i = 1, #all_slots, 2 do
        local slot_id = all_slots[i]
        local raw = all_slots[i+1]
        local data = cjson.decode(raw)
        if data.status == 'free' then
            data.status = 'busy'
            data.job_id = job_id
            data.started_at = now
            redis.call('HSET', key, slot_id, cjson.encode(data))
            return tonumber(slot_id)
        end
    end
    return -1
    """

    async def _get_conn(self):
        if self._r is None:
            self._r = await _async_redis_conn()
        return self._r

    async def close(self):
        if self._r and self._owns_conn:
            try:
                await self._r.aclose()
            except Exception:
                pass
            self._r = None

    def _key(self, service_name: str) -> str:
        return f"slots:{MACHINE_ID}:{service_name}"

    async def _init_slots(self, service_name: str):
        """Pre-populate slot fields if empty."""
        r = await self._get_conn()
        key = self._key(service_name)
        try:
            if await r.exists(key):
                return
            max_s = self.service_slots.get(service_name, 0)
            pipe = r.pipeline()
            for i in range(max_s):
                pipe.hset(key, str(i), json.dumps({
                    "job_id": "",
                    "started_at": "",
                    "status": "free",
                }))
            await pipe.execute()
        except Exception:
            logger.exception("async _init_slots failed %s", service_name)

    async def acquire_slot(self, service_name: str, job_id: str = "") -> int | None:
        """Find a free slot, mark it busy, return the slot index (async).
        Uses atomic Lua script to prevent TOCTOU race.
        """
        try:
            await self._init_slots(service_name)
            r = await self._get_conn()
            key = self._key(service_name)
            slot = await r.eval(self._ACQUIRE_LUA, 1, key, job_id, _now_iso())
            if slot < 0:
                return None
            logger.debug("async acquire_slot %s -> slot %s for %s",
                         service_name, slot, job_id)
            return int(slot)
        except Exception:
            logger.exception("async acquire_slot failed %s", service_name)
            return None

    async def release_slot(self, service_name: str, slot_id: int) -> bool:
        """Mark a slot as free (async)."""
        try:
            r = await self._get_conn()
            key = self._key(service_name)
            raw = await r.hget(key, str(slot_id))
            if raw is None:
                return False
            data = json.loads(raw)
            data["status"] = "free"
            data["job_id"] = ""
            data["started_at"] = ""
            await r.hset(key, str(slot_id), json.dumps(data))
            logger.debug("async release_slot %s slot %s", service_name, slot_id)
            return True
        except Exception:
            logger.exception("async release_slot failed %s/%s", service_name, slot_id)
            return False

    async def get_slots(self, service_name: str) -> dict:
        """Return all slots with their current state (async)."""
        try:
            await self._init_slots(service_name)
            r = await self._get_conn()
            raw = await r.hgetall(self._key(service_name))
            return {k: json.loads(v) for k, v in raw.items()}
        except Exception:
            logger.exception("async get_slots failed %s", service_name)
            return {}

    async def get_available_slots(self, service_name: str) -> int:
        """Count of free slots for a service (async)."""
        try:
            await self._init_slots(service_name)
            r = await self._get_conn()
            raw = await r.hgetall(self._key(service_name))
            count = 0
            for v in raw.values():
                if json.loads(v).get("status") == "free":
                    count += 1
            return count
        except Exception:
            logger.exception("async get_available_slots failed %s", service_name)
            return 0

    async def reap_stale_slots(self, service_name: str, max_age_seconds: float) -> list[dict]:
        """
        Force-release any slot that has been 'busy' longer than max_age_seconds (async).
        Returns list of reaped slot info [{slot_id, job_id, started_at, age_seconds}].
        """
        reaped = []
        try:
            await self._init_slots(service_name)
            r = await self._get_conn()
            key = self._key(service_name)
            all_slots = await r.hgetall(key)
            now = datetime.now(timezone.utc)

            for slot_id_str, raw in all_slots.items():
                data = json.loads(raw)
                if data.get("status") != "busy":
                    continue
                started = data.get("started_at", "")
                if not started:
                    reaped.append({
                        "slot_id": int(slot_id_str),
                        "job_id": data.get("job_id", ""),
                        "started_at": started,
                        "age_seconds": float("inf"),
                    })
                    await self.release_slot(service_name, int(slot_id_str))
                    continue

                try:
                    started_dt = datetime.fromisoformat(started)
                    age = (now - started_dt).total_seconds()
                except (ValueError, TypeError):
                    age = float("inf")

                if age > max_age_seconds:
                    logger.warning(
                        "Async reaping stale slot %s/%s: job=%s age=%.0fs > max=%.0fs",
                        service_name, slot_id_str, data.get("job_id", ""), age, max_age_seconds,
                    )
                    reaped.append({
                        "slot_id": int(slot_id_str),
                        "job_id": data.get("job_id", ""),
                        "started_at": started,
                        "age_seconds": age,
                    })
                    await self.release_slot(service_name, int(slot_id_str))
        except Exception:
            logger.exception("async reap_stale_slots failed for %s", service_name)
        return reaped


    async def flush_all(self):
        """Delete all slot hashes for this machine (async)."""
        flushed = 0
        try:
            r = await self._get_conn()
            cursor = 0
            while True:
                cursor, keys = await r.scan(cursor, match=f"slots:{MACHINE_ID}:*")
                for key in keys:
                    await r.delete(key)
                    flushed += 1
                    logger.info("async flushed slot key: %s", key)
                if cursor == 0:
                    break
        except Exception:
            logger.exception("async flush_all failed")
        return flushed


class AsyncCapacityReporter:
    """
    Async version of CapacityReporter.
    Key: ``capacity:{machine_id}``
    """

    PEER_TTL = 30
    PUBLISH_INTERVAL = 5

    def __init__(self, r=None, slot_manager=None, queue=None, service_config: dict | None = None):
        self._r = r
        self._owns_conn = r is None
        self._slot_mgr = slot_manager  # AsyncSlotManager
        self._queue = queue            # AsyncRedisQueue
        self.service_config: dict = service_config or {}

    async def _get_conn(self):
        if self._r is None:
            self._r = await _async_redis_conn()
        return self._r

    async def close(self):
        if self._r and self._owns_conn:
            try:
                await self._r.aclose()
            except Exception:
                pass
            self._r = None

    def _key(self, machine_id: str | None = None) -> str:
        return f"capacity:{machine_id or MACHINE_ID}"

    async def publish(self) -> bool:
        """Write current machine's capacity to Redis (async)."""
        try:
            r = await self._get_conn()
            data = await self.get_local_capacity()
            now = _now_iso()
            data["published_at"] = now
            data["last_seen"] = now
            await r.set(self._key(), json.dumps(data), ex=self.PEER_TTL + 10)
            logger.debug("async publish capacity for %s", MACHINE_ID)
            return True
        except Exception:
            logger.exception("async publish capacity failed")
            return False

    async def get_local_capacity(self) -> dict:
        """Build a capacity dict for every configured service (async)."""
        capacity: dict = {"machine_id": MACHINE_ID, "services": {}}
        now = time.time()
        for svc_name, cfg in self.service_config.items():
            slots_total = cfg.get("parallel", 1)
            slots_free = await self._slot_mgr.get_available_slots(svc_name) if self._slot_mgr else 0
            queue_depth = await self._queue.get_queue_depth(svc_name) if self._queue else 0
            capacity["services"][svc_name] = {
                "slots_total": slots_total,
                "slots_active": slots_total - slots_free,
                "queue_depth": queue_depth,
                "can_accept": slots_free > 0,
                # Required capacity contract fields
                "capacity_source": "slot_manager",
                "identity_echo": svc_name,
                "observed_at": now,
                "freshness": "current",
            }
        return capacity

    async def get_peer_capacity(self, machine_id: str) -> dict | None:
        """Read a specific peer's capacity (async)."""
        try:
            r = await self._get_conn()
            raw = await r.get(self._key(machine_id))
            if raw is None:
                return None
            data = json.loads(raw)
            published = data.get("published_at", "")
            if published:
                pub_t = datetime.fromisoformat(published)
                age = (datetime.now(timezone.utc) - pub_t).total_seconds()
                if age > self.PEER_TTL:
                    return None
            return data
        except Exception:
            logger.exception("async get_peer_capacity failed %s", machine_id)
            return None

    async def get_all_peers(self) -> dict:
        """Return {machine_id: capacity_dict} for all known peers (async)."""
        peers: dict = {}
        try:
            r = await self._get_conn()
            cursor = 0
            while True:
                cursor, keys = await r.scan(cursor, match="capacity:*", count=100)
                for key in keys:
                    mid = key.split(":", 1)[1] if ":" in key else key
                    raw = await r.get(key)
                    if raw:
                        try:
                            data = json.loads(raw)
                            published = data.get("published_at", "")
                            if published:
                                pub_t = datetime.fromisoformat(published)
                                age = (datetime.now(timezone.utc) - pub_t).total_seconds()
                                if age <= self.PEER_TTL:
                                    peers[mid] = data
                        except (json.JSONDecodeError, ValueError):
                            pass
                if cursor == 0:
                    break
        except Exception:
            logger.exception("async get_all_peers failed")
        return peers


class AsyncFleetGossiper:
    """
    Async version of FleetGossiper.
    Push-model cross-machine capacity gossip.
    """

    STALE_SECONDS = 30
    CAPACITY_TTL = 40

    def __init__(self, local_redis, machine_id: str, peers: dict,
                 password: str, local_capacity_fn):
        """
        Parameters
        ----------
        local_redis : aioredis.Redis
            Async connection to the local Redis instance.
        machine_id : str
            This machine's hostname.
        peers : dict
            ``{machine_id: ip_address}`` for every peer machine.
        password : str
            Shared Redis AUTH password for all instances.
        local_capacity_fn : async callable
            ``() -> dict`` returning the current capacity snapshot.
        """
        self.r = local_redis
        self.machine_id = machine_id
        self.peers: dict[str, str] = peers
        self.password = password
        self.local_capacity_fn = local_capacity_fn  # async callable
        self._peer_conns: dict[str, aioredis.Redis] = {}

    async def close(self):
        """Close all cached peer connections."""
        for conn in self._peer_conns.values():
            try:
                await conn.aclose()
            except Exception:
                pass
        self._peer_conns.clear()

    async def _get_peer_conn(self, ip: str):
        """Return a cached async Redis connection to a peer."""
        if ip not in self._peer_conns:
            self._peer_conns[ip] = aioredis.Redis(
                host=ip, port=REDIS_PORT, password=self.password,
                socket_timeout=2, socket_connect_timeout=2,
                decode_responses=True,
            )
        return self._peer_conns[ip]

    def _capacity_key(self, machine_id: str | None = None) -> str:
        return f"capacity:{machine_id or self.machine_id}"

    @staticmethod
    def _is_stale(data: dict, max_age: float | None = None) -> bool:
        max_age = max_age or AsyncFleetGossiper.STALE_SECONDS
        last_seen = data.get("last_seen", "")
        if not last_seen:
            return True
        try:
            ts = datetime.fromisoformat(last_seen)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return age > max_age
        except (ValueError, TypeError):
            return True

    async def publish_to_peers(self) -> dict[str, bool]:
        """Push local capacity to ALL peer Redis instances (async)."""
        capacity = await self.local_capacity_fn()
        capacity["last_seen"] = _now_iso()
        payload = json.dumps(capacity)
        key = self._capacity_key()
        results: dict[str, bool] = {}

        # Always write to local Redis first
        try:
            await self.r.set(key, payload, ex=self.CAPACITY_TTL)
            results[self.machine_id] = True
        except Exception:
            logger.exception("async publish_to_peers: local write failed")
            results[self.machine_id] = False

        # Push to every peer
        for peer_mid, peer_ip in self.peers.items():
            try:
                conn = await self._get_peer_conn(peer_ip)
                await conn.set(key, payload, ex=self.CAPACITY_TTL)
                results[peer_mid] = True
                logger.debug("async publish_to_peers: pushed to %s (%s)", peer_mid, peer_ip)
            except Exception:
                logger.warning("async publish_to_peers: peer %s (%s) unreachable, skipping",
                               peer_mid, peer_ip)
                results[peer_mid] = False

        return results

    async def read_peer_capacity(self, peer_machine_id: str) -> dict | None:
        """Read a specific peer's capacity from LOCAL Redis (async)."""
        try:
            key = self._capacity_key(peer_machine_id)
            raw = await self.r.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            if self._is_stale(data):
                return None
            return data
        except Exception:
            logger.exception("async read_peer_capacity failed for %s", peer_machine_id)
            return None

    async def read_all_peers(self) -> dict[str, dict | None]:
        """Return {machine_id: capacity_dict_or_None} for all known peers (async)."""
        result: dict[str, dict | None] = {}
        for peer_mid in self.peers:
            result[peer_mid] = await self.read_peer_capacity(peer_mid)
        return result

    async def find_remote_service(self, routing_group: str,
                                   exclude_services: set[str] | None = None) -> tuple | None:
        """Find a peer with a service in routing_group with free slots (async)."""
        exclude_services = exclude_services or set()
        candidates: list[tuple[str, str, dict, int]] = []

        for peer_mid in self.peers:
            cap = await self.read_peer_capacity(peer_mid)
            if cap is None:
                continue
            services = cap.get("services", {})
            for svc_name, svc_info in services.items():
                if svc_name in exclude_services:
                    continue
                if not svc_info.get("can_accept", False):
                    continue
                svc_rg = svc_info.get("routing_group", "")
                if svc_rg and svc_rg != routing_group:
                    continue
                slots_free = svc_info.get("slots_free",
                                          svc_info.get("slots_total", 0)
                                          - svc_info.get("slots_active", 0))
                if slots_free <= 0:
                    continue
                candidates.append((peer_mid, svc_name, svc_info, slots_free))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[3], reverse=True)
        mid, svc, info, _ = candidates[0]
        return (mid, svc, info)


# ===================================================================
# Self-test
# ===================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    r = _redis_conn()

    # --- Config (mirrors the real services) ------------------------------
    TEST_SERVICES = {
        "llm-primary": {"routing_group": "llm-group", "parallel": 16, "speed": 100},
        "llm-secondary":     {"routing_group": "llm-group", "parallel": 12, "speed": 80},
        "llm-cpu":   {"routing_group": "llm-group", "parallel": 4,  "speed": 60},
        "llama-qwen36":   {"routing_group": "qwen36",     "parallel": 1,  "speed": 50},
    }

    # --- Cleanup any prior test data ------------------------------------
    for svc in TEST_SERVICES:
        r.delete(f"queue:{svc}")
        r.delete(f"slots:{MACHINE_ID}:{svc}")
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="job:J-*", count=200)
        if keys:
            r.delete(*keys)
        if cursor == 0:
            break

    separator = "=" * 60

    # -------------------------------------------------------------------
    #  1.  RedisQueue + JobTracker
    # -------------------------------------------------------------------
    print(f"\n{separator}")
    print("  1. RedisQueue + JobTracker")
    print(separator)

    q = RedisQueue(r=r)
    tracker = JobTracker(r=r)

    job_ids = []
    for i in range(5):
        jid = tracker.next_job_id()
        if jid is None:
            print("  ERROR: failed to generate job_id")
            sys.exit(1)
        job = tracker.create_job(
            job_id=jid,
            service_name="llm-primary",
            source="test",
            prompt=f"Test prompt #{i+1}: generate a short summary.",
            max_tokens=256,
            routing_group="llm-group",
        )
        entry_id = q.enqueue("llm-primary", {"job_id": jid, "prompt": f"test {i}"})
        job_ids.append((jid, entry_id))
        print(f"  submitted {jid}  entry={entry_id}")

    depth = q.get_queue_depth("llm-primary")
    print(f"\n  queue depth = {depth}")

    # Dequeue 2 jobs
    claimed = q.dequeue("llm-primary", "worker-1", count=2)
    print(f"\n  dequeued {len(claimed)} jobs:")
    for c in claimed:
        print(f"    {c.get('job_id')}  entry={c.get('_entry_id')}")
        tracker.update_status(c["job_id"], "in_flight")

    # Ack the first claimed job
    first_entry = None
    first_jid = None
    if claimed:
        first_entry = claimed[0]["_entry_id"]
        first_jid = claimed[0]["job_id"]
        q.ack("llm-primary", "worker-1", first_entry)
        tracker.update_status(first_jid, "completed", result="ok")
        print(f"\n  acked {first_jid}")

    # Check pending
    pending = q.get_pending("llm-primary")
    print(f"\n  pending in-flight: {len(pending)}")
    for p in pending:
        print(f"    {p['entry_id']}  idle={p['idle_ms']}ms  deliveries={p['deliveries']}")

    depth_after = q.get_queue_depth("llm-primary")
    print(f"\n  queue depth after dequeue+ack = {depth_after}")

    # Verify job retrieval
    if first_jid:
        j = tracker.get_job(first_jid)
        if j:
            print(f"\n  get_job({first_jid}): status={j['status']}  completed_at={j['completed_at']}")

    # -------------------------------------------------------------------
    #  2.  SlotManager
    # -------------------------------------------------------------------
    print(f"\n{separator}")
    print("  2. SlotManager")
    print(separator)

    sm = SlotManager(r=r, service_slots={s: c["parallel"] for s, c in TEST_SERVICES.items()})

    avail = sm.get_available_slots("llm-primary")
    total = TEST_SERVICES["llm-primary"]["parallel"]
    print(f"  llm-primary: {avail}/{total} slots free")

    # Acquire 3 slots
    acquired = []
    for idx in range(3):
        s = sm.acquire_slot("llm-primary", job_id=f"test-{idx}")
        acquired.append(s)
        print(f"  acquired slot {s}")

    avail_after = sm.get_available_slots("llm-primary")
    print(f"  after 3 acquires: {avail_after}/{total} slots free")

    # Release slot 1
    sm.release_slot("llm-primary", acquired[0])
    print(f"  released slot {acquired[0]}")

    avail_final = sm.get_available_slots("llm-primary")
    print(f"  after release: {avail_final}/{total} slots free")

    # Get slot for job
    slot_of_job = sm.get_slot_for_job("llm-primary", "test-1")
    print(f"  slot for test-1 = {slot_of_job}")

    # -------------------------------------------------------------------
    #  3.  RoutingEngine
    # -------------------------------------------------------------------
    print(f"\n{separator}")
    print("  3. RoutingEngine")
    print(separator)

    re = RoutingEngine(r=r, service_config=TEST_SERVICES, slot_manager=sm)

    members = re.get_group_members("llm-group")
    print("  group members:")
    for m in members:
        print(f"    {m['service']}: {m['slots_free']}/{m['slots_total']} free  speed={m['speed']}")

    best = re.find_service_for_group("llm-group")
    print(f"\n  best service for llm-group: {best}")

    status = re.get_group_status("llm-group")
    print(f"  group status: total={status['slots_total']}  free={status['slots_free']}  can_accept={status['can_accept']}")

    best_qwen = re.find_service_for_group("qwen36")
    print(f"  best service for qwen36: {best_qwen}")

    # -------------------------------------------------------------------
    #  4.  CapacityReporter
    # -------------------------------------------------------------------
    print(f"\n{separator}")
    print("  4. CapacityReporter")
    print(separator)

    cr = CapacityReporter(
        r=r,
        slot_manager=sm,
        queue=q,
        service_config=TEST_SERVICES,
    )

    cap = cr.get_local_capacity()
    print(f"  local capacity:")
    for svc, info in cap["services"].items():
        print(f"    {svc}: {info['slots_active']}/{info['slots_total']} active  queue={info['queue_depth']}  accept={info['can_accept']}")

    cr.publish()
    print(f"  published capacity to Redis")

    peers = cr.get_all_peers()
    print(f"  known peers: {list(peers.keys())}")

    # -------------------------------------------------------------------
    #  5.  FleetGossiper  (mock-peer test using local Redis)
    # -------------------------------------------------------------------
    print(f"\n{separator}")
    print("  5. FleetGossiper")
    print(separator)

    # Simulate a peer publishing its capacity into our local Redis.
    # In production each peer pushes to all other Redis instances;
    # here we just inject the peer data directly.
    MOCK_PEER_ID = "macbook"
    mock_peer_cap = {
        "machine_id": MOCK_PEER_ID,
        "last_seen": _now_iso(),
        "services": {
            "lfm25-macbook": {
                "routing_group": "lfm25",
                "slots_total": 1,
                "slots_active": 0,
                "slots_free": 1,
                "queue_depth": 0,
                "can_accept": True,
            },
        },
    }
    r.set(f"capacity:{MOCK_PEER_ID}", json.dumps(mock_peer_cap), ex=40)
    print(f"  injected mock peer capacity for {MOCK_PEER_ID}")

    # Create a FleetGossiper with no real peers (we mock via local Redis)
    gossiper = FleetGossiper(
        local_redis=r,
        machine_id=MACHINE_ID,
        peers={MOCK_PEER_ID: "127.0.0.1"},  # point to local for testing
        password=_redis_password(),
        local_capacity_fn=cr.get_local_capacity,
    )

    # Test publish_to_peers (will write to local + try 127.0.0.1)
    pub_results = gossiper.publish_to_peers()
    print(f"  publish_to_peers results: {pub_results}")

    # Test read_peer_capacity — read the mock peer we injected
    peer_cap = gossiper.read_peer_capacity(MOCK_PEER_ID)
    assert peer_cap is not None, "Expected to read mock peer capacity"
    assert peer_cap["machine_id"] == MOCK_PEER_ID
    print(f"  read_peer_capacity({MOCK_PEER_ID}): OK  services={list(peer_cap['services'].keys())}")

    # Test read_all_peers
    all_peers = gossiper.read_all_peers()
    print(f"  read_all_peers: {list(all_peers.keys())}")
    assert MOCK_PEER_ID in all_peers
    assert all_peers[MOCK_PEER_ID] is not None

    # Test find_remote_service
    remote = gossiper.find_remote_service("lfm25")
    assert remote is not None, "Expected to find remote lfm25 service"
    rmid, rsvc, rinfo = remote
    print(f"  find_remote_service(lfm25): machine={rmid}  service={rsvc}  can_accept={rinfo.get('can_accept')}")
    assert rmid == MOCK_PEER_ID
    assert rsvc == "lfm25-macbook"

    # Test staleness: inject a stale peer
    stale_peer = {
        "machine_id": "stale-box",
        "last_seen": "2020-01-01T00:00:00+00:00",  # old timestamp
        "services": {"old-svc": {"slots_total": 1, "slots_active": 0, "can_accept": True}},
    }
    r.set("capacity:stale-box", json.dumps(stale_peer), ex=40)
    gossiper.peers["stale-box"] = "127.0.0.1"
    stale_cap = gossiper.read_peer_capacity("stale-box")
    assert stale_cap is None, "Expected stale peer to return None"
    print(f"  staleness check: stale-box correctly returned None")

    # Test find_remote_service with no matching group
    no_match = gossiper.find_remote_service("nonexistent-group")
    assert no_match is None
    print(f"  find_remote_service(nonexistent-group): None  (correct)")

    # Clean up mock peer keys
    r.delete(f"capacity:{MOCK_PEER_ID}")
    r.delete("capacity:stale-box")
    del gossiper.peers["stale-box"]
    print(f"  cleaned up mock peer keys")

    # -------------------------------------------------------------------
    #  6.  Cleanup
    # -------------------------------------------------------------------
    print(f"\n{separator}")
    print("  6. Cleanup")
    print(separator)

    q.purge("llm-primary")
    for svc in TEST_SERVICES:
        r.delete(f"slots:{MACHINE_ID}:{svc}")
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="job:J-*", count=200)
        if keys:
            r.delete(*keys)
        if cursor == 0:
            break
    r.delete(cr._key())
    r.delete(gossiper._capacity_key())
    print("  cleaned up all test keys")

    print(f"\n{separator}")
    print("  ALL TESTS PASSED")
    print(separator)


# ===================================================================
# Example GPU Manager Integration
# ===================================================================
# The following shows how GPU Manager (gpu-manager.py) would initialize
# the FleetGossiper for cross-machine capacity gossip.
#
# from queue_engine import (
#     RedisQueue, JobTracker, SlotManager, RoutingEngine,
#     CapacityReporter, FleetGossiper,
# )
#
# # --- Local infrastructure (existing GPU Manager init) ----------------
# r = redis.Redis(host="localhost", port=6379, password=_redis_password(),
#                 decode_responses=True)
#
# service_config = {
#     "llm-primary": {"routing_group": "llm-group", "parallel": 16, "speed": 100},
#     "llama-qwen36":   {"routing_group": "qwen36",     "parallel": 1,  "speed": 50},
# }
#
# slot_mgr = SlotManager(r=r, service_slots={s: c["parallel"] for s, c in service_config.items()})
# queue    = RedisQueue(r=r)
# tracker  = JobTracker(r=r)
# capacity_reporter = CapacityReporter(
#     r=r, slot_manager=slot_mgr, queue=queue, service_config=service_config,
# )
#
# # --- Fleet gossip (new) ----------------------------------------------
# gossiper = FleetGossiper(
#     local_redis=r,
#     machine_id="host",
#     peers={
#         "remote-a": "tailnet-or-dns-address-a",
#         "remote-b": "tailnet-or-dns-address-b",
#     },
#     password=_redis_password(),
#     local_capacity_fn=capacity_reporter.get_local_capacity,
# )
#
# # In the main event loop (every 5 seconds):
# #   capacity_reporter.publish()        # local capacity
# #   gossiper.publish_to_peers()        # push to all peers
#
# # When local routing finds no available service, check remote peers:
# #   remote = gossiper.find_remote_service(routing_group="lfm25")
# #   if remote:
# #       machine_id, service_name, info = remote
# #       # Route job to remote machine (e.g. via HTTP proxy or direct enqueue)
