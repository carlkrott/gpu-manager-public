"""A missing workflow asset ends the parent typed instead of retrying forever.

Job fb3af112 (2026-10-06): the ``qwen-correct`` stage raised an untyped
``Could not load workflow ...`` 500 because the release did not ship the graph,
and the coordinator re-ticked it every ~4 s without end. These tests run the
production leaf admission (``_admit_internal_generation_leaf``) under the real
``PipelineCoordinator``/``ControllerPipelineRuntime`` inside the production
``create_app`` TestServer harness. Only external boundaries are injected.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_manager_pipeline_runtime import (  # noqa: E402
    ControllerPipelineRuntime,
    GPUManagerLeafGenerationClient,
)
from pipeline_coordinator import (  # noqa: E402
    MAX_TICK_FAILURES,
    ExternalExecutor,
    InMemoryCoordinatorStore,
    PipelineCoordinator,
    StageAuthorityResolver,
    TerminalCode,
    TerminalLeafError,
)

from test_idle_restore_http import (  # noqa: E402,F401  (fixtures + helpers)
    AUTH,
    _age_generation,
    _baselines_then_generation,
    lifecycle,
    manager,
)

MISSING = "workflows/qwen-image-edit-2511-not-shipped.json"
SHIPPED = ROOT / "workflows" / "qwen-image-edit-2511-q5-dualgpu.json"


def _configure(env, workflow_template=MISSING):
    cfg = env.cfg
    cfg["pipeline_providers"] = {
        "gm-leaf": {"adapter": "gpu_manager_generation", "endpoint_ref": "comfyui"},
    }
    cfg["generation_templates"] = {
        "image-edit-test": {
            "backend": "comfyui",
            "workflow_backend": "comfyui",
            "routing_group": "comfyui-gen",
            "bundle": "comfyui",
            "endpoint": "http://127.0.0.1:8188",
            "request_path": "/prompt",
            "workflow_template": workflow_template,
            "timeout": 60,
        }
    }
    m = env.m
    # Qualification gates are exercised elsewhere; this suite is about the asset.
    env.mp.setattr(m, "_service_is_publicly_qualified", lambda _x: True)
    env.mp.setattr(m, "_validate_template_qualification", lambda *_a, **_k: None)
    env.mp.setattr(m, "_validate_q5_template_identity", lambda *_a, **_k: None)


class _Leaf(StageAuthorityResolver):
    def resolve(self, stage):
        return "leaf"


class _NoExternal(ExternalExecutor):
    def execute(self, stage, state, idempotency_key):
        raise AssertionError("no external stage in this pipeline")

    def poll(self, stage, state, idempotency_key, provider_handle):
        raise AssertionError("no external stage in this pipeline")

    def cancel(self, run_id, stage_id):
        return True


def _pipeline():
    return {
        "pipeline_id": "krea-image-qc-test",
        "revision_hash": "r1",
        "stages": [{
            "id": "qwen-correct", "kind": "generate", "service_id": "gm-leaf",
            "resource_id": "image-edit-test", "retry": {"max_attempts": 5},
        }],
        "loops": [],
    }


class _PendingEntryStore(InMemoryCoordinatorStore):
    """Redis-stream semantics: an unACKed claim is redelivered on every reclaim
    until ``ack_parent`` (the in-memory store hands an orphan out only once)."""

    def reclaim_orphaned_parents(self, consumer_name):
        return [
            self._states[rid] for rid, owner in self._claimed.items()
            if owner == consumer_name and rid in self._states and rid not in self._acked
        ]


class _Harness:
    def __init__(self, env, submit_override=None):
        m = env.m
        self.loop = asyncio.get_running_loop()
        self.store = _PendingEntryStore()
        self.submits = 0

        def submit_leaf(service_id, resource_id, job_id, params):
            self.submits += 1
            if submit_override is not None:
                return submit_override(service_id, resource_id, job_id, params)
            future = asyncio.run_coroutine_threadsafe(
                m._admit_internal_generation_leaf(service_id, resource_id, job_id, params),
                self.loop,
            )
            return future.result(timeout=30)

        leaf = GPUManagerLeafGenerationClient(
            submit_leaf=submit_leaf,
            poll_leaf=m._poll_internal_generation_leaf,
            cancel_leaf=m._cancel_internal_generation_leaf,
        )
        coordinator = PipelineCoordinator(self.store, _Leaf(), _NoExternal(), leaf)
        self.runtime = ControllerPipelineRuntime(
            coordinator=coordinator, store=self.store, consumer_name="test")
        assert self.runtime.submit_parent(
            compiled_pipeline=_pipeline(), original_request={"prompt": "a cat"},
            revision="r1", parent_job_id="parent-1", run_id="run-1")

    async def tick(self):
        return await asyncio.to_thread(self.runtime.process_one)

    def state(self):
        return self.runtime.get("run-1")


@pytest.mark.asyncio
async def test_builder_reports_typed_reason_for_missing_and_corrupt_asset(lifecycle, tmp_path):
    m = lifecycle.m
    tmpl = {"workflow_backend": "comfyui", "backend": "comfyui", "routing_group": "comfyui-gen",
            "endpoint": "http://127.0.0.1:8188", "request_path": "/prompt",
            "workflow_template": MISSING}
    entry, error = await m._build_comfyui_workflow_entry(
        service_name="comfyui", tmpl=tmpl, params={}, job_id="b1", priority=50,
        callback_url=None, metadata={})
    assert entry is None
    assert error["reason"] == "workflow_asset_missing"
    assert error["status"] == 500
    assert "Could not load workflow" in error["error"]

    bad = tmp_path / "corrupt.json"
    bad.write_text("{not json")
    entry, error = await m._build_comfyui_workflow_entry(
        service_name="comfyui", tmpl={**tmpl, "workflow_template": str(bad)},
        params={}, job_id="b2", priority=50,
        callback_url=None, metadata={})
    assert entry is None and error["reason"] == "workflow_asset_invalid"


@pytest.mark.asyncio
async def test_leaf_admission_raises_typed_terminal_error(lifecycle):
    _configure(lifecycle)
    m = lifecycle.m
    with pytest.raises(TerminalLeafError) as caught:
        await m._admit_internal_generation_leaf("gm-leaf", "image-edit-test", "gm-leaf-1", {
            "parent_job_id": "parent-1", "id": "qwen-correct",
            "context": {"request_params": {"prompt": "a cat"}, "effective_prompt": "a cat"},
        })
    assert caught.value.code == "workflow_asset_missing"
    assert lifecycle.m.job_tracker.get_job("gm-leaf-1") is None  # nothing was admitted


def test_shipped_qwen_workflow_is_a_loadable_graph():
    graph = json.loads(SHIPPED.read_text())
    assert graph["4"]["inputs"] and graph["13"]["inputs"] and graph["15"]["inputs"]
    assert all("class_type" in node for node in graph.values()
               if isinstance(node, dict) and not str(node).startswith("_meta"))


@pytest.mark.asyncio
async def test_shipped_asset_resolves_through_the_real_builder(lifecycle):
    _configure(lifecycle, "workflows/qwen-image-edit-2511-q5-dualgpu.json")
    m = lifecycle.m
    tmpl = dict(lifecycle.cfg["generation_templates"]["image-edit-test"])
    entry, error = await m._build_comfyui_workflow_entry(
        service_name="comfyui", tmpl=tmpl, params={}, job_id="ok-1", priority=50,
        callback_url=None, metadata={})
    assert error is None and entry is not None


@pytest.mark.asyncio
async def test_missing_asset_terminalizes_once_without_retries(lifecycle):
    _configure(lifecycle)
    h = _Harness(lifecycle)
    first = await h.tick()
    assert first.status == "failed"
    assert first.terminal_code is TerminalCode.WORKFLOW_ASSET_MISSING
    assert "workflow_asset_missing" in first.terminal_reason
    # One admission attempt only, despite a stage retry budget of 5.
    assert h.submits == 1
    assert len(h.state().stage_attempts["qwen-correct"]) == 1
    # The claim was ACKed: nothing is left to re-tick or reclaim.
    assert await h.tick() is None
    assert h.submits == 1


@pytest.mark.asyncio
async def test_untyped_failure_is_capped_and_ends_coordinator_error(lifecycle):
    def boom(*_a):
        raise RuntimeError("HTTP 500 from admission")

    h = _Harness(lifecycle, submit_override=boom)
    for _ in range(MAX_TICK_FAILURES - 1):
        with pytest.raises(RuntimeError):
            await h.tick()
        assert h.state().status == "running"
    ended = await h.tick()  # the capping tick terminalizes instead of raising
    assert ended.status == "failed"
    assert ended.terminal_code is TerminalCode.COORDINATOR_ERROR
    assert "tick_retry_cap_exceeded" in ended.terminal_reason
    assert h.submits == MAX_TICK_FAILURES
    assert await h.tick() is None
    assert h.submits == MAX_TICK_FAILURES


@pytest.mark.asyncio
async def test_a_returning_tick_resets_the_failure_counter(lifecycle):
    calls = {"n": 0}

    def flaky(_service, _resource, job_id, _params):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient")
        return {"job_id": job_id}

    h = _Harness(lifecycle, submit_override=flaky)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await h.tick()
    assert h.state().tick_failures == 2
    state = await h.tick()  # admission works now; the tick returns (child not tracked yet)
    assert state is not None and state.status == "running"
    assert h.state().tick_failures == 0


@pytest.mark.asyncio
async def test_cancellation_is_honoured_while_retrying(lifecycle):
    def boom(*_a):
        raise RuntimeError("HTTP 500 from admission")

    h = _Harness(lifecycle, submit_override=boom)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await h.tick()
    assert h.submits == 2
    assert h.runtime.cancel("run-1") is True
    ended = await h.tick()
    assert ended.status == "cancelled"
    assert ended.terminal_code is TerminalCode.COORDINATOR_CANCELLED
    assert h.submits == 2  # no further admission attempt after the cancel
    assert await h.tick() is None


@pytest.mark.asyncio
async def test_typed_failure_then_release_and_both_baselines_restored(lifecycle):
    """The failed parent holds no work: the comfyui bundle is released after the
    idle TTL and BOTH idle baselines come back through the real restore."""
    m = lifecycle.m
    _configure(lifecycle)
    await _baselines_then_generation(lifecycle)  # comfyui loaded, card0 Gemma displaced
    h = _Harness(lifecycle)
    ended = await h.tick()
    assert ended.terminal_code is TerminalCode.WORKFLOW_ASSET_MISSING
    _age_generation(lifecycle)
    await m.scheduler._auto_evict_idle_bundles()
    assert "comfyui.service" not in lifecycle.active
    assert {"gemma-rx.service", "aux-mi50.service"} <= lifecycle.active
    report = m.scheduler._last_idle_restore_report
    assert report["partial"] is False and sorted(report["results"]) == ["card0", "card1"]
    health = await (await lifecycle.client.get("/health", headers=AUTH)).json()
    assert "llm_evicted_with_llm_loaded_state" not in health["contradictions"]
    queue = await lifecycle.client.get("/v1/queue", headers=AUTH)
    assert queue.status == 200


@pytest.mark.asyncio
async def test_supervisor_terminalizes_only_owned_accepted_child_on_failed_unit(lifecycle):
    """Confirmed unit death is visible to the accepted child before pool teardown."""
    m = lifecycle.m
    tracked = {
        "job_id": "gm-oom-child", "service_name": "comfyui", "status": "in_flight",
        "source": "pipeline_coordinator_child", "parent_job_id": "parent-oom",
        "entry_id": "1710000000000-0", "comfyui_prompt_id": "prompt-oom",
    }

    class Tracker:
        def get_job(self, job_id):
            return tracked if job_id == tracked["job_id"] else None

        def update_status(self, job_id, status, **fields):
            assert job_id == tracked["job_id"]
            tracked.update(status=status, **fields)
            return True

    class Slots:
        async def get_slots(self, service):
            assert service == "comfyui"
            return {"slot-0": {"status": "busy", "job_id": tracked["job_id"]}}

    lifecycle.cfg["services"] = {
        "comfyui": {"enabled": True, "type": "generation_backend", "port": 8188,
                    "systemd_unit": "comfyui.service"},
    }
    lifecycle.mp.setattr(m, "_services_config", lifecycle.cfg, raising=False)
    lifecycle.mp.setattr(m, "job_tracker", Tracker(), raising=False)
    lifecycle.mp.setattr(m, "async_slot_manager", Slots(), raising=False)
    sup = object.__new__(m.WorkerSupervisor)
    sup._pools = {}
    sup._pool_tasks = {}
    async def failed_unit(_cfg):
        return {"confirmed_failed": True, "result": "oom-kill", "active_state": "failed"}
    sup._backend_failure_facts = failed_unit
    async def unhealthy(*_a, **_k):
        return False
    sup._check_health = unhealthy
    await sup._validate_pools()

    assert tracked["status"] == "failed"
    assert tracked["error_type"] == "backend_died"
    assert "oom-kill" in tracked["error"]
    assert "comfyui.service" in tracked["error"]


@pytest.mark.parametrize("result", ["signal", "exit-code", "core-dump"])
@pytest.mark.asyncio
async def test_supervisor_terminalizes_restart_window_child(lifecycle, result):
    """Real systemd facts flow through pool validation to the owned child."""
    m = lifecycle.m
    tracked = {
        "job_id": "gm-restart-child", "service_name": "comfyui", "status": "in_flight",
        "source": "pipeline_coordinator_child", "parent_job_id": "parent-restart",
        "entry_id": "1710000000000-1", "comfyui_prompt_id": "prompt-restart",
    }

    class Tracker:
        def get_job(self, job_id):
            return tracked if job_id == tracked["job_id"] else None

        def update_status(self, job_id, status, **fields):
            assert job_id == tracked["job_id"]
            tracked.update(status=status, **fields)
            return True

    class Slots:
        async def get_slots(self, service):
            assert service == "comfyui"
            return {"slot-0": {"status": "busy", "job_id": tracked["job_id"]}}

    class Proc:
        returncode = 0

        async def communicate(self):
            return (
                f"ActiveState=activating\nSubState=auto-restart\nResult={result}\n"
                "ExecMainCode=exited\nExecMainStatus=1\n"
            ).encode(), b""

    async def create(*_args, **_kwargs):
        return Proc()

    lifecycle.mp.setattr(m.asyncio, "create_subprocess_exec", create)
    lifecycle.cfg["services"] = {
        "comfyui": {"enabled": True, "type": "generation_backend", "port": 8188,
                    "systemd_unit": "comfyui.service"},
    }
    lifecycle.mp.setattr(m, "_services_config", lifecycle.cfg, raising=False)
    lifecycle.mp.setattr(m, "job_tracker", Tracker(), raising=False)
    lifecycle.mp.setattr(m, "async_slot_manager", Slots(), raising=False)
    sup = object.__new__(m.WorkerSupervisor)
    sup._pools = {}
    sup._pool_tasks = {}

    async def unhealthy(*_a, **_k):
        return False

    sup._check_health = unhealthy
    await sup._validate_pools()

    assert tracked["status"] == "failed"
    assert tracked["error_type"] == "backend_died"
    assert result in tracked["error"]
    assert "comfyui.service" in tracked["error"]


@pytest.mark.parametrize("result", ["signal", "exit-code", "core-dump"])
def test_backend_auto_restart_death_is_confirmed(lifecycle, result):
    m = lifecycle.m

    class Proc:
        returncode = 0

        async def communicate(self):
            return (
                f"ActiveState=activating\nSubState=auto-restart\nResult={result}\n"
                "ExecMainCode=exited\nExecMainStatus=1\n"
            ).encode(), b""

    async def create(*_args, **_kwargs):
        return Proc()

    lifecycle.mp.setattr(m.asyncio, "create_subprocess_exec", create)
    sup = object.__new__(m.WorkerSupervisor)
    facts = asyncio.run(sup._backend_failure_facts({"systemd_unit": "comfyui.service"}))
    assert facts["confirmed_failed"] is True
    assert facts["sub_state"] == "auto-restart"
    assert facts["result"] == result


@pytest.mark.parametrize("active,sub,result", [
    ("activating", "start", "signal"),
    ("activating", "auto-restart", "success"),
    ("active", "running", "success"),
])
def test_backend_non_death_states_are_not_confirmed(lifecycle, active, sub, result):
    m = lifecycle.m

    class Proc:
        returncode = 0

        async def communicate(self):
            return (
                f"ActiveState={active}\nSubState={sub}\nResult={result}\n"
                "ExecMainCode=exited\nExecMainStatus=0\n"
            ).encode(), b""

    async def create(*_args, **_kwargs):
        return Proc()

    lifecycle.mp.setattr(m.asyncio, "create_subprocess_exec", create)
    sup = object.__new__(m.WorkerSupervisor)
    facts = asyncio.run(sup._backend_failure_facts({"systemd_unit": "comfyui.service"}))
    assert facts["confirmed_failed"] is False
