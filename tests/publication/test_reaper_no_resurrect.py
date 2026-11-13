"""The pending-entry reaper must dead-letter long-abandoned jobs, not re-queue them."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "scripts" / "gpu-manager.py"
SCRIPTS = SRC.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _helper():
    tree = ast.parse(SRC.read_text())
    wanted = {
        "REAPER_STALE_DEAD_LETTER_FLOOR_MS", "_reaper_nack_max_retries",
        "_pending_orphan_ttl_ms", "_pending_entry_submission_age_ms",
    }
    nodes = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in wanted)
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_reaper_nack_max_retries"]


def test_ordinary_stuck_entry_keeps_retry_budget():
    fn = _helper()
    assert fn(20 * 60_000, 900_000) == 4


def test_three_hour_old_entry_is_dead_lettered():
    fn = _helper()
    assert fn(10_863_587, 900_000) == 1  # 13:52 incident entry


def test_ceiling_scales_with_long_lease():
    fn = _helper()
    assert fn(5_000_000, 2_000_000) == 4
    assert fn(8_500_000, 2_000_000) == 1


def test_orphan_ttl_is_configurable_bounded_and_defaults_to_30_minutes():
    from collections.abc import Mapping

    namespace = {"Mapping": Mapping}
    tree = ast.parse(SRC.read_text())
    wanted = {"_pending_orphan_ttl_ms"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), "exec"), namespace)
    ttl = namespace["_pending_orphan_ttl_ms"]
    assert ttl({}) == 1_800_000
    assert ttl({"admission": {"orphan_ttl_seconds": 900}}) == 900_000
    assert ttl({"admission": {"orphan_ttl_seconds": 1}}) == 60_000
    assert ttl({"admission": {"orphan_ttl_seconds": 10**9}}) == 86_400_000


def test_submission_age_uses_immutable_timestamp_and_legacy_ids_fail_closed():
    namespace = {"time": __import__("time")}
    tree = ast.parse(SRC.read_text())
    nodes = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_pending_entry_submission_age_ms"
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), "exec"), namespace)
    age = namespace["_pending_entry_submission_age_ms"]
    assert age("900000-0", now_ms=1_801_000, submitted_at_ms=1000) == 1_800_000
    assert age("1800000-0", now_ms=1_801_000) == 0
    assert age("bad-id", now_ms=1_801_000, submitted_at_ms="bad") == 0


def test_reaper_passes_computed_budget_to_nack():
    src = SRC.read_text()
    start = src.index("async def _reap_stale_jobs")
    body = src[start:src.index("async def _reap_stale_slots", start)]
    assert "retry_budget = 1 if entry_age_ms >= orphan_ttl_ms else 4" in body
    assert "max_retries=4" not in body
    assert '"stale_orphan_ttl" if retry_budget == 1' in body
    assert "_pending_orphan_ttl_ms(svc_cfg)" in body
    assert "_pending_entry_submission_age_ms" in body


def test_worker_supervisor_reaper_nacks_only_expired_submission_and_keeps_reason():
    import asyncio
    import importlib.util
    import time


    spec = importlib.util.spec_from_file_location("reaper_candidate", SRC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Queue:
        def __init__(self, entry_id, entry_age_ms):
            self.entry_id = entry_id
            self.entry_age_ms = entry_age_ms
            self.nacks = []

        async def get_pending(self, *_args, **_kwargs):
            return [{
                "entry_id": self.entry_id, "idle_ms": 120_000,
                "entry_created_ms": int(time.time() * 1000) - self.entry_age_ms,
                "job_id": "job-1",
            }]

        async def nack(self, *args, **kwargs):
            self.nacks.append((args, kwargs))

    def run_for(entry_age_ms, tracker_result=None, active_slot=False):
        queue = Queue(f"{int(time.time() * 1000) - entry_age_ms}-0", entry_age_ms)
        module._services_config = {
            "services": {
                "comfyui": {
                    "enabled": True, "type": "generation_backend",
                    "admission": {"pending_lease_seconds": 60},
                }
            }
        }
        module.async_queue = queue
        class Tracker:
            def _key(self, job_id):
                return f"job:{job_id}"
            def get_job(self, _job_id):
                if tracker_result is Exception:
                    raise RuntimeError("tracker unavailable")
                return tracker_result or {"status": "in_flight"}
            def update_status(self, *_args, **_kwargs):
                return True
        module.job_tracker = Tracker()
        class Slots:
            def _key(self, service):
                return f"slots:machine:{service}"
            async def get_slots(self, _service):
                return {"slot-0": {
                    "status": "busy" if active_slot else "free",
                    "job_id": "job-1" if active_slot else "",
                }}
        module.async_slot_manager = Slots()
        supervisor = object.__new__(module.WorkerSupervisor)
        supervisor._pools = {}
        asyncio.run(supervisor._reap_stale_jobs())
        return queue.nacks

    fresh_nacks = run_for(10 * 60_000)
    assert len(fresh_nacks) == 1
    assert fresh_nacks[0][1]["max_retries"] == 4
    old_nacks = run_for(31 * 60_000)
    assert len(old_nacks) == 1
    assert old_nacks[0][1]["max_retries"] == 1
    assert old_nacks[0][1]["dead_letter_reason"] == "stale_orphan_ttl"
    assert run_for(31 * 60_000, active_slot=True) == []
    assert run_for(31 * 60_000, tracker_result=Exception) == []


def test_async_nack_retains_typed_stale_orphan_reason_in_dead_letter():
    import asyncio
    import queue_engine

    class _Redis:
        def __init__(self):
            self.eval_args = None

        async def xrange(self, *args, **kwargs):
            return [(b"1-0", {b"job_id": b"orphan", b"payload": b"audit"})]

        async def eval(self, *args):
            self.eval_args = args
            return [1, 0, 0]

    redis = _Redis()
    queue = queue_engine.AsyncRedisQueue(redis)
    assert asyncio.run(queue.nack(
        "svc", "1-0", max_retries=1, dead_letter_reason="stale_orphan_ttl",
        orphan_tracker_key="job:orphan", orphan_slot_key="slots:machine:svc",
        minimum_idle_ms=60_000,
    ))
    assert "stale_orphan_terminalize_v1" in redis.eval_args[0]
    assert "XPENDING" in redis.eval_args[0]
    assert "tracker_changed" in redis.eval_args[0]
    assert "stale_orphan_ttl" in redis.eval_args


class _Tracker:
    def __init__(self, status="in_flight"):
        self.job = {"job_id": "7199ecf2", "status": status}
        self.updates = []

    def get_job(self, job_id):
        return self.job

    def update_status(self, job_id, status, error=None):
        self.updates.append((status, error))
        self.job["status"] = status
        return True


def _terminalize():
    import queue_engine

    return queue_engine.terminalize_if_impossible


def test_dead_lettered_job_with_stopped_backend_ends_typed_cancelled():
    tracker = _Tracker()
    decision = _terminalize()(
        tracker, "7199ecf2",
        stream_has_pel=False, stream_has_entry=False, service_running=False, force=True,
    )
    assert decision["action"] == "terminalized"
    assert tracker.job["status"] == "cancelled"
    assert tracker.updates[0][1]


def test_live_continuation_signals_preserve_the_job():
    for pel, entry, running in ((True, False, False), (False, True, False), (False, False, True)):
        tracker = _Tracker()
        decision = _terminalize()(
            tracker, "7199ecf2",
            stream_has_pel=pel, stream_has_entry=entry, service_running=running, force=True,
        )
        assert decision["action"] == "noop"
        assert tracker.job["status"] == "in_flight"
