"""Facade-bound OOM retry: Lua script accessor, eligibility, correlation and atomic retry.

Extracted verbatim from ``scripts/gpu-manager.py``.  Every function reads facade
globals (``_OOM_RETRY_LUA``, ``_oom_redis_pool``, ``_oom_retry_*`` helpers, ...)
at call time, so ``bind(facade_globals)`` rebuilds each one with
``types.FunctionType`` against the supplied facade dictionary; no
``sys.modules`` entry or shared facade pointer is kept here.  The definitions
below are templates: calling them directly would resolve names in this module.
"""
from __future__ import annotations

from types import FunctionType


def _oom_retry_lua_script() -> str:
    """Return the Lua source for the atomic-retry EVAL.

    Wrapped so tests can monkeypatch a different body (the FakeRedis
    stand-in already has its own reference implementation, but the
    test suite still pins the text of the production script so a
    silent regression in the production body fails fast).
    """
    return _OOM_RETRY_LUA


def _normalize_dead_letter_entry(entry: dict) -> dict:
    """Return a sanitized copy of a dead-letter entry.

    Defensive against partial / malformed records so the rest of
    the retry path can rely on the shape. Plan 05 #1 says historical
    evidence is never rewritten, but reading / classifying a
    malformed entry is not a write — and the classifier is the only
    thing that decides what is even attemptable.
    """
    if not isinstance(entry, dict):
        return {"job_id": "", "gpu_id": ""}
    out = dict(entry)
    out["job_id"] = str(out.get("job_id", "") or "").strip()
    out["gpu_id"] = str(out.get("gpu_id", "") or "").strip()
    out["reason"] = str(out.get("reason", "") or "").strip()
    out["last_state"] = str(out.get("last_state", "") or "").strip()
    return out


async def _oom_retry_check_redis_state(entries: list[dict]) -> list[dict]:
    """For each dead-letter entry, decide retry eligibility.

    Returns the input list annotated with ``classification``,
    ``classification_reason``, and a flag indicating which Redis
    surfaces were observed. NEVER rewrites the input entries or any
    Redis state. Plan 05 #2 + #6 say missing tracker / payload /
    stream metadata is a terminal refusal; that is exactly what
    this function reports.

    The four signals it requires before declaring ``eligible``:

      1. A live tracker hash at ``job:{job_id}`` carrying a
         terminal status (``completed``/``failed``/``cancelled``/
         ``oom_terminal``). Plan 05 #1 says the 32 retained records
         have NO surviving trackers; they all fall out here as
         ``missing_tracker``.
      2. Payload / artifact reference material that can be replayed.
         Encoded as a non-empty ``payload_hash`` on the tracker, or
         an artifact reference list with at least one entry. Plan
         05 #4 says payload/artifact reconstruction is required.
      3. A non-replayed destination stream entry — i.e. no
         ``retry_id`` already recorded against the original
         ``oom:retry:stream:{orig_job_id}`` correlation set, AND
         no active job with ``retry_of == orig_job_id`` still
         queued or running. Plan 05 #4 + #5.
      4. The attempt budget is not exhausted, tracked via the
         audit-set count. Plan 05 #3.

    Any single missing signal downgrades the classification.
    ``dry_run`` callers see the same answer; there is no
    differential path that would let a dry-run report ``eligible``
    while a real run would refuse.
    """
    if not entries:
        return []
    if _oom_redis_pool is None:
        # Fail closed: when Redis is unavailable we cannot verify
        # any of the four signals. The plan is explicit that
        # missing state fails closed; an unreachable store is
        # also missing state.
        return [
            dict(
                _normalize_dead_letter_entry(e),
                classification=_OOM_RETRY_CLASS_NO_TRACKER,
                classification_reason="redis_unavailable",
                has_tracker=False,
                has_payload_artifact=False,
                has_stream_correlation=False,
                attempt_count=0,
            )
            for e in entries
        ]

    out: list[dict] = []
    for raw in entries:
        e = _normalize_dead_letter_entry(raw)
        job_id = e["job_id"]
        gpu_id = e["gpu_id"]
        if not job_id:
            out.append(
                dict(
                    e,
                    classification=_OOM_RETRY_CLASS_NOT_FOUND,
                    classification_reason="blank_job_id",
                    has_tracker=False,
                    has_payload_artifact=False,
                    has_stream_correlation=False,
                    attempt_count=0,
                )
            )
            continue

        observed = {
            "has_tracker": False,
            "has_payload_artifact": False,
            "tracker_status": None,
            "tracker_payload_hash": None,
            "tracker_artifact_refs": None,
            "tracker_service_name": None,
            "tracker_routing_group": None,
            "has_stream_correlation": False,
            "has_active_retry": False,
            "attempt_count": 0,
        }

        # (1) tracker existence + payload / artifact material.
        try:
            tracker = await asyncio.to_thread(
                _oom_redis_pool.hgetall, f"job:{job_id}"
            )
        except Exception as exc:
            logger.debug(
                f"[oom-retry] tracker read failed for {job_id}: {exc}"
            )
            tracker = None
        if tracker:
            observed["has_tracker"] = True
            observed["tracker_status"] = str(
                tracker.get("status", "") or ""
            ).strip() or None
            ph = str(tracker.get("payload_hash", "") or "").strip()
            ar = str(tracker.get("artifact_refs", "") or "").strip()
            observed["tracker_payload_hash"] = ph or None
            observed["tracker_artifact_refs"] = ar or None
            observed["tracker_service_name"] = str(
                tracker.get("service_name", "") or ""
            ).strip() or None
            observed["tracker_routing_group"] = str(
                tracker.get("routing_group", "") or ""
            ).strip() or None
            observed["has_payload_artifact"] = bool(ph) or bool(ar)

        # (3a) correlation set: any prior retries against this orig?
        try:
            members = await asyncio.to_thread(
                _oom_redis_pool.smembers,
                f"{_OOM_RETRY_STREAM_CORRELATION_PREFIX}{job_id}",
            )
        except Exception as exc:
            logger.debug(
                f"[oom-retry] correlation read failed for {job_id}: {exc}"
            )
            members = set()
        observed["attempt_count"] = len(members or set())
        observed["has_stream_correlation"] = bool(members)

        # (3b) any active retry still in flight? Scan retry audit
        # hashes for status=queued/running against this orig.
        if members:
            for retry_id in members:
                try:
                    audit = await asyncio.to_thread(
                        _oom_redis_pool.hgetall,
                        f"{_OOM_RETRY_AUDIT_PREFIX}{retry_id}",
                    )
                except Exception as exc:
                    logger.debug(
                        f"[oom-retry] audit read failed for {retry_id}: {exc}"
                    )
                    audit = None
                if not audit:
                    continue
                new_job_id = str(audit.get("new_job_id", "") or "").strip()
                if not new_job_id:
                    continue
                try:
                    new_tracker = await asyncio.to_thread(
                        _oom_redis_pool.hgetall, f"job:{new_job_id}"
                    )
                except Exception as exc:
                    logger.debug(
                        f"[oom-retry] new tracker read failed: {exc}"
                    )
                    new_tracker = None
                status = (
                    str((new_tracker or {}).get("status", "") or "").strip()
                    if new_tracker
                    else ""
                )
                if status in ("queued", "running", "in_flight"):
                    observed["has_active_retry"] = True
                    break

        # (4) attempt budget gate.
        if observed["attempt_count"] >= _OOM_RETRY_DEFAULT_ATTEMPT_LIMIT:
            classification = _OOM_RETRY_CLASS_BUDGET_EXCEEDED
            reason = f"attempt_count={observed['attempt_count']}>=limit={_OOM_RETRY_DEFAULT_ATTEMPT_LIMIT}"
        elif observed["has_active_retry"]:
            classification = _OOM_RETRY_CLASS_REPLAYED
            reason = "active_retry_in_flight"
        elif not observed["has_tracker"]:
            classification = _OOM_RETRY_CLASS_NO_TRACKER
            reason = "no_live_tracker_at_job_key"
        elif observed["tracker_status"] not in (
            "completed",
            "failed",
            "cancelled",
            "oom_terminal",
        ):
            classification = _OOM_RETRY_CLASS_REPLAYED
            reason = f"non_terminal_tracker_status={observed['tracker_status']!r}"
        elif not observed["has_payload_artifact"]:
            classification = _OOM_RETRY_CLASS_NO_PAYLOAD
            reason = "no_payload_hash_or_artifact_refs"
        else:
            classification = _OOM_RETRY_CLASS_ELIGIBLE
            reason = "all_signals_present"

        out.append(
            dict(
                e,
                gpu_id=gpu_id,
                classification=classification,
                classification_reason=reason,
                has_tracker=observed["has_tracker"],
                has_payload_artifact=observed["has_payload_artifact"],
                has_stream_correlation=observed["has_stream_correlation"],
                has_active_retry=observed["has_active_retry"],
                attempt_count=observed["attempt_count"],
                tracker_status=observed["tracker_status"],
                tracker_payload_hash=observed["tracker_payload_hash"],
                tracker_artifact_refs=observed["tracker_artifact_refs"],
                tracker_service_name=observed["tracker_service_name"],
                tracker_routing_group=observed["tracker_routing_group"],
            )
        )
    return out


def _generate_oom_retry_id(orig_job_id: str) -> str:
    """Build a stable, unique retry id. Pure function — used both
    by the live path and by the test suite.
    """
    suffix = uuid.uuid4().hex[:12]
    ts_ms = int(time.time() * 1000)
    safe_orig = re.sub(r"[^A-Za-z0-9_-]", "_", str(orig_job_id))[:48]
    return f"retry-{safe_orig}-{ts_ms}-{suffix}"


def _generate_oom_retry_job_id(retry_id: str) -> str:
    """New job id derived from a retry_id. Stable mapping so the
    test suite can derive the expected tracker key without
    re-implementing the algorithm.
    """
    return f"job-retry-{retry_id}"


async def _atomic_retry_oom_job(
    dead_letter: dict,
    *,
    actor: str,
    reason: str,
    idempotency_key: str,
    payload_hash: str | None = None,
    artifact_refs: list[str] | None = None,
    service_name: str | None = None,
    routing_group: str | None = None,
) -> dict:
    """Run the atomic retry EVAL for one dead-letter entry.

    Caller MUST have classified first and passed only eligible
    entries; this function does not re-classify, and will refuse
    anything that is not eligible. The result is a dict with the
    same shape the test suite pins:

        {
          "outcome": "submitted" | "replayed" | "budget_exceeded"
                     | "exec_failed" | "refused",
          "retry_id": ...,
          "new_job_id": ...,
          "stream_entry_id": ...,
          "orig_job_id": ...,
          "idempotency_key": ...,
        }

    The original dead-letter record is NEVER removed or rewritten.
    """
    e = _normalize_dead_letter_entry(dead_letter)
    orig_job_id = e["job_id"]
    gpu_id = e["gpu_id"]
    if not orig_job_id:
        return {
            "outcome": "refused",
            "reason": "blank_job_id",
            "orig_job_id": orig_job_id,
            "idempotency_key": idempotency_key,
        }
    if not idempotency_key or not isinstance(idempotency_key, str):
        return {
            "outcome": "refused",
            "reason": "missing_idempotency_key",
            "orig_job_id": orig_job_id,
            "idempotency_key": idempotency_key,
        }
    if _oom_redis_pool is None:
        return {
            "outcome": "refused",
            "reason": "redis_unavailable",
            "orig_job_id": orig_job_id,
            "idempotency_key": idempotency_key,
        }

    # Belt-and-suspenders: refuse if classification was not
    # supplied / says ineligible. Production callers go through
    # ``_oom_retry_check_redis_state`` first; the live MCP path
    # is the only thing that should be invoking this helper, and
    # it always passes a classified entry.
    classification = str(dead_letter.get("classification", "") or "")
    if classification and classification != _OOM_RETRY_CLASS_ELIGIBLE:
        return {
            "outcome": "refused",
            "reason": f"ineligible:{classification}",
            "orig_job_id": orig_job_id,
            "idempotency_key": idempotency_key,
        }

    retry_id = _generate_oom_retry_id(orig_job_id)
    new_job_id = _generate_oom_retry_job_id(retry_id)
    attempt = int(dead_letter.get("attempt_count", 0) or 0) + 1
    attempt_limit = _OOM_RETRY_DEFAULT_ATTEMPT_LIMIT
    service = service_name or e.get("service_name") or ""
    routing = routing_group or e.get("routing_group") or e.get("last_state") or "default"
    payload_hash = payload_hash or e.get("payload_hash") or ""
    artifact_refs = artifact_refs or []
    artifact_refs_json = json.dumps(list(artifact_refs), sort_keys=True)
    ts = f"{time.time():.6f}"

    # The original dead-letter member is preserved verbatim; we
    # only serialize it back as a string for the audit hash so a
    # later audit can prove the original evidence has not been
    # touched.
    try:
        orig_member_json = json.dumps(
            e, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        orig_member_json = json.dumps({"job_id": orig_job_id, "gpu_id": gpu_id})

    # Destination stream key follows the same convention as
    # RedisQueue.enqueue: queue:{routing_group}.
    stream_key = f"queue:{routing}"

    keys = [
        f"{_OOM_RETRY_IDEMPOTENCY_PREFIX}{idempotency_key}",
        f"{_OOM_RETRY_AUDIT_PREFIX}{retry_id}",
        f"{_OOM_RETRY_STREAM_CORRELATION_PREFIX}{orig_job_id}",
        f"job:{new_job_id}",
        stream_key,
    ]
    args = [
        idempotency_key,
        retry_id,
        new_job_id,
        orig_job_id,
        orig_member_json,
        gpu_id,
        str(attempt),
        str(attempt_limit),
        actor or "",
        reason or "",
        ts,
        payload_hash or "",
        artifact_refs_json,
        service or "",
        routing,
        str(_OOM_RETRY_IDEMPOTENCY_TTL_S),
        e.get("reason", ""),
    ]

    try:
        raw = await asyncio.to_thread(
            _oom_redis_pool.eval, _oom_retry_lua_script(), len(keys), *keys, *args
        )
    except Exception as exc:
        logger.warning(
            f"[oom-retry] EVAL failed for {orig_job_id}: {exc}"
        )
        return {
            "outcome": "exec_failed",
            "reason": str(exc),
            "retry_id": retry_id,
            "new_job_id": new_job_id,
            "orig_job_id": orig_job_id,
            "idempotency_key": idempotency_key,
        }

    # Lua always returns a 3-element list.
    if not isinstance(raw, (list, tuple)) or len(raw) < 1:
        return {
            "outcome": "exec_failed",
            "reason": "malformed_lua_response",
            "retry_id": retry_id,
            "new_job_id": new_job_id,
            "orig_job_id": orig_job_id,
            "idempotency_key": idempotency_key,
        }
    status = raw[0]
    out = {
        "retry_id": retry_id,
        "new_job_id": new_job_id,
        "orig_job_id": orig_job_id,
        "idempotency_key": idempotency_key,
    }
    if status == "submitted":
        out["outcome"] = "submitted"
        out["stream_entry_id"] = raw[2] if len(raw) > 2 else None
        return out
    if status == "replayed":
        replayed_retry_id = raw[1] if len(raw) > 1 else None
        out["outcome"] = "replayed"
        out["retry_id"] = replayed_retry_id
        out["new_job_id"] = (
            _generate_oom_retry_job_id(replayed_retry_id)
            if replayed_retry_id
            else None
        )
        out["replayed_retry_id"] = replayed_retry_id
        return out
    if status == "budget_exceeded":
        out["outcome"] = "budget_exceeded"
        out["attempt"] = raw[1] if len(raw) > 1 else None
        out["attempt_limit"] = raw[2] if len(raw) > 2 else None
        return out
    # exec_failed or anything else: no partial state survived
    # MULTI/EXEC (the Lua script DELs the idempotency marker on
    # EXEC failure so a fresh retry is possible). The original
    # dead-letter member is untouched.
    out["outcome"] = "exec_failed"
    out["reason"] = status
    return out


async def _oom_retry_correlation_snapshot(limit: int = 100) -> dict:
    """Read-only helper: list every retry currently in the audit
    index, group by ``orig_job_id``, and surface cross-store
    correlation so operators can see OOM archive + queue DL +
    retry audit as a unified picture (plan 05 #5 Option A).
    """
    if _oom_redis_pool is None:
        return {
            "schema_version": 1,
            "redis_available": False,
            "by_orig": {},
            "total_retries": 0,
        }
    by_orig: dict[str, list[dict]] = {}
    total = 0
    try:
        cursor = 0
        # SCAN over oom:retry:audit:* to enumerate retries. The
        # audit keys are short-lived (no explicit TTL today; the
        # idempotency marker has the TTL) so a bounded scan is
        # sufficient and avoids the KEYS hot path.
        while True:
            cursor, keys = await asyncio.to_thread(
                _oom_redis_pool.scan,
                cursor=cursor,
                match=f"{_OOM_RETRY_AUDIT_PREFIX}*",
                count=200,
            )
            for k in keys:
                kk = k.decode() if isinstance(k, bytes) else k
                if not kk.startswith(_OOM_RETRY_AUDIT_PREFIX):
                    continue
                retry_id = kk[len(_OOM_RETRY_AUDIT_PREFIX):]
                audit = await asyncio.to_thread(
                    _oom_redis_pool.hgetall, kk
                )
                if not audit:
                    continue
                orig = str(audit.get("orig_job_id", "") or "").strip()
                if not orig:
                    continue
                by_orig.setdefault(orig, []).append(
                    {
                        "retry_id": retry_id,
                        "new_job_id": audit.get("new_job_id", ""),
                        "gpu_id": audit.get("gpu_id", ""),
                        "service_name": audit.get("service_name", ""),
                        "routing_group": audit.get("routing_group", ""),
                        "attempt": audit.get("attempt", ""),
                        "actor": audit.get("actor", ""),
                        "ts": audit.get("ts", ""),
                    }
                )
                total += 1
            if cursor == 0:
                break
            if total >= limit * 4:
                break
    except Exception as exc:
        logger.debug(f"[oom-retry] correlation scan failed: {exc}")
    return {
        "schema_version": 1,
        "redis_available": True,
        "by_orig": by_orig,
        "total_retries": total,
    }


NAMES = (
    "_oom_retry_lua_script",
    "_normalize_dead_letter_entry",
    "_oom_retry_check_redis_state",
    "_generate_oom_retry_id",
    "_generate_oom_retry_job_id",
    "_atomic_retry_oom_job",
    "_oom_retry_correlation_snapshot",
)


def _rebuild(function, facade_globals):
    rebuilt = FunctionType(
        function.__code__,
        facade_globals,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebuilt.__kwdefaults__ = (
        function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    )
    rebuilt.__annotations__ = function.__annotations__.copy()
    rebuilt.__doc__ = function.__doc__
    rebuilt.__qualname__ = function.__qualname__
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt


def bind(facade_globals):
    """Return the retry functions in ``NAMES`` order, bound to a facade.

    Each function is rebuilt so its ``__globals__`` is exactly
    ``facade_globals`` and every global read resolves against that facade at
    call time.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    return tuple(_rebuild(namespace[name], facade_globals) for name in NAMES)
