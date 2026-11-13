"""Regressions for coordinator readiness retries and dead generation children."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_coordinator import (  # noqa: E402
    InMemoryCoordinatorStore,
    PipelineCoordinator,
    PipelineRunState,
    StageAuthorityResolver,
    ExternalExecutor,
    LeafGenerationClient,
    RedisCoordinatorStore,
)
from gpu_manager_pipeline_runtime import ControllerPipelineRuntime  # noqa: E402


class _RedisPipeline:
    def __init__(self, owner):
        self.owner = owner

    def xadd(self, *_args):
        self.owner.attempts += 1
        return "1001-1"

    def xack(self, *_args):
        return 1

    def xdel(self, *_args):
        return 1

    def execute(self):
        return ["1001-1", 1, 1]


class _Redis:
    def __init__(self):
        self.attempts = 0

    def pipeline(self, **_kwargs):
        return _RedisPipeline(self)

    def xlen(self, _key):
        return 0


def test_loading_backend_does_not_hot_requeue(monkeypatch):
    """Readiness polling waits without charging retries and reuses the child."""
    store = object.__new__(RedisCoordinatorStore)
    redis = _Redis()
    store._r = redis
    store._entry_ids = {"loading-image": "1000-1"}
    store._consecutive_requeues = {}
    store._last_requeue_warning = {}
    store._requeue_backoff = lambda _run: store._consecutive_requeues.__setitem__(
        "loading-image", store._consecutive_requeues.get("loading-image", 0) + 1
    )
    state = PipelineRunState(
        parent_job_id="image-1", run_id="loading-image", pipeline_id="image",
        revision_hash="r1", status="running",
    )

    assert store.requeue_parent(state, ordinary_wait=True)
    assert store._consecutive_requeues.get(state.run_id, 0) == 0
    assert redis.attempts <= 1

    class _ReadyAfterOnePoll(_DyingBackend):
        def __init__(self):
            super().__init__()
            self.submits = 0

        def submit(self, *args, **kwargs):
            self.submits += 1
            return "child-ready-on-second-poll"

        def poll(self, child_job_id):
            self.polls += 1
            status = "in_flight" if self.polls == 1 else "completed"
            return {"job_id": child_job_id, "status": status, "output": {}}

    memory_store = InMemoryCoordinatorStore()
    leaf = _ReadyAfterOnePoll()
    coordinator = PipelineCoordinator(memory_store, _Leaf(), _NoExternal(), leaf)
    pipeline = {
        "pipeline_id": "image", "revision_hash": "r1",
        "stages": [{"id": "generate", "kind": "generate", "service_id": "image",
                    "resource_id": "comfyui", "retry": {"max_attempts": 1}}],
        "loops": [],
    }
    waiting = PipelineRunState(
        parent_job_id="image-ready", run_id="loading-ready", pipeline_id="image",
        revision_hash="r1", status="running", compiled_pipeline=pipeline,
        current_stage_id="generate",
    )
    memory_store.submit_parent(waiting)
    sleep_calls = []
    monkeypatch.setattr(
        "gpu_manager_pipeline_runtime.time.sleep", lambda seconds: sleep_calls.append(seconds)
    )
    runtime = ControllerPipelineRuntime(coordinator=coordinator, store=memory_store)
    first = runtime.process_one()
    assert first is not None and first.status == "running"
    same_child_id = memory_store.load_state(waiting.run_id).child_jobs[0].child_job_id
    second = runtime.process_one()
    assert second is not None and second.status == "completed"
    assert memory_store.load_state(waiting.run_id).child_jobs[0].child_job_id == same_child_id
    assert leaf.submits == 1
    assert leaf.polls == 2
    assert sleep_calls == [1.0]


class _Leaf(StageAuthorityResolver):
    def resolve(self, _stage):
        return "leaf"


class _NoExternal(ExternalExecutor):
    def execute(self, *_args, **_kwargs):
        raise AssertionError("unexpected external stage")

    def poll(self, *_args, **_kwargs):
        raise AssertionError("unexpected external stage")

    def cancel(self, *_args, **_kwargs):
        return True


class _DyingBackend(LeafGenerationClient):
    """Fake accepted child whose backend dies before the next health poll."""
    def __init__(self):
        self.polls = 0
        self.submits = 0

    def submit(self, stage, parent_run_id, parent_state, idempotency_key, trusted_config):
        self.submits += 1
        return "child-oom-1"

    def poll(self, child_job_id):
        self.polls += 1
        # Model the tracker after WorkerSupervisor reconciles confirmed unit death.
        return {
            "job_id": child_job_id,
            "status": "failed",
            "error": "Backend comfyui (comfyui.service) confirmed failed: systemd result=oom-kill",
        }

    def cancel(self, _child_job_id):
        return True


def test_backend_death_terminalizes_by_next_health_cycle():
    """An accepted image child killed by its backend cannot remain in_flight."""
    store = InMemoryCoordinatorStore()
    leaf = _DyingBackend()
    coordinator = PipelineCoordinator(store, _Leaf(), _NoExternal(), leaf)
    pipeline = {
        "pipeline_id": "image", "revision_hash": "r1",
        "stages": [{"id": "generate", "kind": "generate", "service_id": "image",
                    "resource_id": "comfyui", "retry": {"max_attempts": 1}}],
        "loops": [],
    }
    runtime = ControllerPipelineRuntime(coordinator=coordinator, store=store)
    assert runtime.submit_parent(
        compiled_pipeline=pipeline,
        original_request={},
        revision="r1",
        parent_job_id="image-2",
        run_id="oom-child",
    )
    ended = runtime.process_one()
    assert ended is not None and ended.status == "failed"
    child_id = store.load_state("oom-child").child_jobs[0].child_job_id
    observed = store.load_state("oom-child")
    assert observed.status == "failed"
    assert "oom-kill" in (observed.terminal_reason or "")
    assert leaf.polls >= 1
    assert child_id
    # Terminal parent ACK releases the claim; a later tick cannot replay submission.
    assert runtime.process_one() is None
    assert leaf.submits == 1
