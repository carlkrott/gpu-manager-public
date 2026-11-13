"""The queue being worked is finished before priority is consulted again.

Exercises the real scheduler evaluator and the broker repository ordering; only
the external precheck/topology readers are stubbed.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

from gemma_broker.contracts import JobRecord, Priority
from gemma_broker.redis_store import InMemoryJobRepository

SRC = Path(__file__).resolve().parents[2] / "scripts" / "gpu-manager.py"


@pytest.fixture(scope="module")
def gm():
    scripts = str(SRC.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("drain_candidate", SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator(gm, monkeypatch):
    monkeypatch.setattr(gm, "_phase3_scheduling_config", lambda: {
        "scheduler_owns_load": True, "scheduler_dry_run_mode": True,
    })
    monkeypatch.setattr(gm, "queue_engine", object(), raising=False)
    monkeypatch.setattr(gm, "_services_config", {"services": {"x": {}}})
    monkeypatch.setattr(gm, "_scheduler_run_precheck", lambda group: {"head_job_ids": [f"{group}-job"]})
    monkeypatch.setattr(gm, "_resolve_group_target_gpus", lambda group: ["card0", "card1"])
    monkeypatch.setattr(gm, "_target_gpus_hold_warm_bundle", lambda group: False)
    monkeypatch.setattr(gm, "_record_load_history", lambda decision: None)
    monkeypatch.setattr(gm, "_drain_cutoffs", {})
    monkeypatch.setattr(gm, "_last_next_decision", {"next_decision": "demand:comfyui-gen"})
    return gm


def test_group_a_inside_its_cutoff_holds_group_b_then_b_loads(evaluator):
    gm = evaluator
    gm._drain_cutoffs["llm-chat"] = time.time()
    held = gm._scheduler_evaluate_load()
    assert held["action"] == "hold_drain"
    assert held["target_gpus"] == [] and held["dispatch_performed"] is False

    gm._drain_cutoffs["llm-chat"] = time.time() - 61
    released = gm._scheduler_evaluate_load()
    assert released["action"] == "dual_gpu_load"
    assert released["target_gpus"] == ["card0", "card1"]


def test_group_own_cutoff_never_holds_itself(evaluator):
    gm = evaluator
    gm._drain_cutoffs["comfyui-gen"] = time.time()
    assert gm._scheduler_evaluate_load()["action"] == "dual_gpu_load"


def test_no_durable_head_job_never_loads_even_without_a_hold(evaluator, monkeypatch):
    gm = evaluator
    monkeypatch.setattr(gm, "_scheduler_run_precheck", lambda group: {
        "head_job_ids": [], "skipped_reason": "empty",
    })
    decision = gm._scheduler_evaluate_load()
    assert decision["action"] == "pre_load_failed_skip"
    assert decision["head_job_ids"] == [] and decision["target_gpus"] == []


def _job(job_id, priority, submitted_at, sequence):
    return JobRecord.new(
        job_id=job_id, idempotency_key=f"i-{job_id}", request_sha256="0" * 64,
        request_body={"messages": []}, submitted_at=submitted_at,
        enqueue_sequence=sequence, required_capabilities=("chat",),
        base_priority=priority,
    )


def test_priority_applies_after_the_drained_queue_then_aging_prevents_starvation():
    repo = InMemoryJobRepository(aging_seconds=300.0)
    repo.submit(_job("bg-old", Priority.BACKGROUND, 0.0, 1), caller_scope="t")
    repo.submit(_job("hi-new", Priority.INTERACTIVE, 100.0, 2), caller_scope="t")
    # Fresh: the higher-priority job goes first.
    assert repo.peek_next(now=101.0).job_id == "hi-new"
    # After two aging tiers the long-waiting background job reaches the top
    # tier and wins the sequence tie-break: no starvation.
    assert repo.peek_next(now=601.0).job_id == "bg-old"
