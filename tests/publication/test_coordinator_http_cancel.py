"""Parent cancellation reaches coordinator and converges terminal child state."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from test_idle_restore_http import lifecycle, manager  # noqa: E402,F401


class _Tracker:
    def __init__(self, job):
        self.job = job

    def get_job(self, _job_id):
        return self.job


class _Coordinator:
    def __init__(self):
        self.calls = []

    def cancel(self, run_id):
        self.calls.append(run_id)
        return True


@pytest.mark.asyncio
async def test_parent_delete_requests_coordinator_cancellation(lifecycle):
    m = lifecycle.m
    coordinator = _Coordinator()
    job_id = "parent-job"
    m.job_tracker = _Tracker({
        "job_id": job_id, "status": "in_flight",
        "workflow_execution_authority": "coordinator",
        "coordinator_run_id": "run-parent",
    })
    m.queue_engine = object()
    app = {m.WORKFLOW_COORDINATOR_RUNTIME_KEY: coordinator}
    request = SimpleNamespace(
        app=app, match_info={"job_id": job_id},
        get=lambda *_args: False,
    )

    response = await m.handle_cancel_job(request)

    assert response.status == 200
    assert json.loads(response.text) == {
        "job_id": job_id, "status": "cancellation_requested"
    }
    assert coordinator.calls == ["run-parent"]


def test_cancelled_child_status_is_terminal_not_unknown(lifecycle):
    from pipeline_coordinator import (
        ChildJobRef, ExternalExecutor, InMemoryCoordinatorStore,
        LeafGenerationClient, PipelineCoordinator, PipelineRunState,
        StageAuthorityResolver,
    )

    child = ChildJobRef("child-1", "generate", "now", "idem", status="cancelled")
    assert child.status in PipelineCoordinator.TERMINAL_CHILD_STATUSES

    class Authority(StageAuthorityResolver):
        def resolve(self, stage): return "leaf"
    class External(ExternalExecutor):
        def execute(self, *args): raise AssertionError
        def cancel(self, *args): return True
    class Leaf(LeafGenerationClient):
        def submit(self, *args): return "child-1"
        def poll(self, child_job_id): return {"status": "cancelled"}
        def cancel(self, child_job_id): raise AssertionError("already stopped")

    store = InMemoryCoordinatorStore()
    coordinator = PipelineCoordinator(store, Authority(), External(), Leaf())
    state = PipelineRunState(
        parent_job_id="parent", run_id="run", pipeline_id="p",
        revision_hash="r1", status="running", cancellation_intent=True,
        child_jobs=[ChildJobRef("child-1", "generate", "now", "idem", status="queued")],
    )
    store.save_state(state)
    result = coordinator._cascade_cancellation(state)
    assert result.status == "cancelled"
    assert result.child_jobs[0].status == "cancelled"
    assert result.child_jobs[0].cancelled is True
    assert result.cancelled_children == ["child-1"]


def test_repeated_parent_cancellation_is_idempotent(lifecycle):
    from pipeline_coordinator import InMemoryCoordinatorStore, PipelineCoordinator
    from pipeline_coordinator import ExternalExecutor, LeafGenerationClient, StageAuthorityResolver
    from pipeline_coordinator import PipelineRunState

    class Authority(StageAuthorityResolver):
        def resolve(self, stage): return "leaf"
    class External(ExternalExecutor):
        def execute(self, *args): raise AssertionError
        def cancel(self, *args): return True
    class Leaf(LeafGenerationClient):
        def submit(self, *args): return "child"
        def poll(self, child_job_id): return {"status": "cancelled"}
        def cancel(self, child_job_id): return True

    store = InMemoryCoordinatorStore()
    coordinator = PipelineCoordinator(store, Authority(), External(), Leaf())
    store.save_state(PipelineRunState(
        parent_job_id="parent-job", run_id="run-id", pipeline_id="p",
        revision_hash="r1", status="running",
        compiled_pipeline={"pipeline_id": "p", "revision_hash": "r1", "stages": [], "loops": []},
    ))
    assert coordinator.request_cancellation("run-id") is True
    queue_size = len(store._queue)
    assert coordinator.request_cancellation("run-id") is True
    assert len(store._queue) == queue_size


def test_orphaned_terminal_parent_is_acked_not_requeued(lifecycle):
    from gpu_manager_pipeline_runtime import ControllerPipelineRuntime
    from pipeline_coordinator import (
        ChildJobRef, ExternalExecutor, InMemoryCoordinatorStore,
        LeafGenerationClient, PipelineCoordinator, PipelineRunState,
        StageAuthorityResolver, TerminalCode,
    )

    class Authority(StageAuthorityResolver):
        def resolve(self, stage): return "leaf"
    class External(ExternalExecutor):
        def execute(self, *args): raise AssertionError
        def cancel(self, *args): return True
    class Leaf(LeafGenerationClient):
        def submit(self, *args): raise AssertionError
        def poll(self, child_job_id): raise AssertionError
        def cancel(self, child_job_id): raise AssertionError

    store = InMemoryCoordinatorStore()
    coordinator = PipelineCoordinator(store, Authority(), External(), Leaf())
    state = PipelineRunState(
        parent_job_id="parent", run_id="terminal-run", pipeline_id="p",
        revision_hash="r1", status="cancelled", terminal_code=TerminalCode.COORDINATOR_CANCELLED,
        cancellation_intent=True, cancelled=True,
        child_jobs=[ChildJobRef("child-1", "generate", "now", "idem", status="cancelled", cancelled=True)],
    )
    store.save_state(state)
    store.enqueue_parent(state)
    runtime = ControllerPipelineRuntime(coordinator=coordinator, store=store, consumer_name="test")

    recovered = runtime.process_one()

    assert recovered.status == "cancelled"
    assert store.dequeue_parent("test") is None
