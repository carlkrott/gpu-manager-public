"""
gpu_manager_pipeline_runtime.py — Phase 4A dependency-injected coordinator-to-GPUManager adapter layer.

Provides two components wired to the existing pipeline_coordinator.py protocols:

1. GPUManagerLeafGenerationClient
   - Satisfies pipeline_coordinator.LeafGenerationClient exactly.
   - Injects three synchronous callables: submit_leaf, poll_leaf, cancel_leaf.
   - submit(): deterministic job_id generation, deep-copied trusted params,
     recursive stripping of bypass keys (service_pipeline_id, worker_pipeline,
     pipeline, internal_leaf, skip_pipeline, force_qc_fail), validated response.
   - poll(): normalises GPUManager tracker statuses (queued/in_flight/running →
     in_flight; completed/failed/cancelled pass-through), extracts bounded
     artifact/QC fields, rejects raw binary/base64.
   - cancel(): single delegation, truthful bool.

2. ControllerPipelineRuntime
   - Wraps PipelineCoordinator + CoordinatorStore with a synchronous API
     suitable for asyncio.to_thread() from gpu-manager.py.
   - submit_parent(): enqueues via durable store queue (not in-memory parallel).
   - process_one(): returns None on no-message (not error); ACKs only after
     persisted terminal/progress state; no busy-loop assumptions.
   - get(): returns state or None.
   - cancel(): delegates to coordinator.request_cancellation().
   - reclaim_orphaned(): delegates to store.reclaim_orphaned_parents().

No live Redis, no live GPUManager HTTP calls in tests.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from pipeline_coordinator import (
    CoordinatorStore,
    LeafGenerationClient,
    PipelineCoordinator,
    PipelineRunState,
    QCVerdict,
    _deep_sanitize,
    _now_iso,
)

# ── Exceptions ─────────────────────────────────────────────────────────────────

class GPUMgrSubmitError(Exception):
    """Raised when GPUManager submit_leaf returns an unexpected/mismatched job_id."""

    pass


class GPUMgrPollError(Exception):
    """Raised when poll_leaf raises (connection/Redis error)."""

    pass


class GPUMgrCancelError(Exception):
    """Raised when cancel_leaf raises."""

    pass


# ── GPUManagerLeafGenerationClient ─────────────────────────────────────────────

# Keys that are stripped (never forwarded to the GPUManager) at any nesting level.
# These correspond to pipeline-internal routing fields that could bypass
# safety if propagated to the generation backend.
_LEAF_REJECT_KEYS: frozenset[str] = frozenset({
    "service_pipeline_id",
    "worker_pipeline",
    "pipeline",
    "internal_leaf",
    "skip_pipeline",
    "force_qc_fail",
})


def _reject_recursive(value: Any, path: str = "$") -> None:
    """
    Raise ValueError if value contains any _LEAF_REJECT_KEYS at any nesting level.
    Applied to trusted_config before it is passed to submit_leaf.
    """
    if isinstance(value, dict):
        for k in value:
            if k in _LEAF_REJECT_KEYS:
                raise ValueError(
                    f"{path}.{k}: forbidden key '{k}' found in trusted_config "
                    "(must not be forwarded to GPUManager)"
                )
            _reject_recursive(value[k], f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _reject_recursive(item, f"{path}[{i}]")


def _bounded_leaf_output(value: Any, path: str = "$") -> Any:
    """Keep only bounded, non-binary child outputs for parent bindings."""

    if isinstance(value, bytes):
        raise ValueError(f"{path}: binary child output is not persistable")
    if isinstance(value, str):
        return value[:10_000]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if (
                lowered in {"base64", "blob", "bytes", "authorization", "secret"}
                or lowered.endswith(("_b64", "_base64", "_blob", "_bytes"))
            ):
                continue
            output[key_text] = _bounded_leaf_output(item, f"{path}.{key_text}")
        return output
    if isinstance(value, (list, tuple)):
        if len(value) > 1_000:
            raise ValueError(f"{path}: child output list exceeds 1000 items")
        return [_bounded_leaf_output(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


class GPUManagerLeafGenerationClient(LeafGenerationClient):
    """
    Trusted adapter implementing LeafGenerationClient using injected GPUManager
    synchronous callables.

    Parameters
    ----------
    submit_leaf : callable
        Signature: (service_id: str, resource_id: str, job_id: str, params: dict) -> dict
        Returns {"job_id": str, "service_id": str, "resource_id": str}.
        The returned job_id is validated against the expected deterministic value.
    poll_leaf : callable
        Signature: (job_id: str) -> dict
        Returns a GPUManager tracker status dict.  Recognised status values:
        "queued", "in_flight", "running", "completed", "failed", "cancelled".
        May contain "artifact", "qc_verdict" fields.
    cancel_leaf : callable
        Signature: (job_id: str) -> bool
        Returns True if cancelled, False if the job was already terminal/absent.
    """

    def __init__(
        self,
        *,
        submit_leaf: Callable[[str, str, str, dict], dict],
        poll_leaf: Callable[[str], dict],
        cancel_leaf: Callable[[str], bool],
    ) -> None:
        self._submit_leaf = submit_leaf
        self._poll_leaf = poll_leaf
        self._cancel_leaf = cancel_leaf

    def submit(
        self,
        stage: dict,
        parent_run_id: str,
        parent_state: PipelineRunState,
        idempotency_key: str,
        trusted_config: dict,
    ) -> str:
        """
        Submit a GPU generation child job via the injected submit_leaf callable.

        Returns the assigned child_job_id string.

        Raises
        ------
        GPUMgrSubmitError
            If the GPUManager returns a job_id that does not match the expected
            deterministic value (indicating the service may have remapped the job
            to a different pipeline).
        ValueError
            If ``trusted_config`` contains any _LEAF_REJECT_KEYS at any nesting
            level (service_pipeline_id, worker_pipeline, pipeline, internal_leaf,
            skip_pipeline, force_qc_fail).
        """
        # ── 1. Validate trusted_config recursively ──────────────────────────────
        _reject_recursive(trusted_config)

        # ── 2. Extract service_id and resource_id from trusted_config ────────────
        # The coordinator's _build_trusted_config resolves these from the stage.
        service_id = trusted_config.get("service_id", "")
        resource_id = trusted_config.get("resource_id", "")
        # Declarative service-pipeline stages may omit the optional explicit
        # leaf identity.  The coordinator has already resolved the provider
        # through the reviewed adapter registry, so using that provider as the
        # service fallback is deterministic and does not expose an endpoint or
        # executable command to the request. Resource/template IDs remain
        # server-compiled stage params.
        if not service_id:
            service_id = trusted_config.get("provider", "")
        if not resource_id:
            params = trusted_config.get("params")
            if isinstance(params, dict):
                resource_id = (
                    params.get("resource_id")
                    or params.get("template")
                    or params.get("qwen_template")
                    or ""
                )

        # ── 3. Deterministic job_id ────────────────────────────────────────────
        # job_id is derived from idempotency_key so the coordinator can re-submit
        # the same leaf job after a controller restart and receive the same ID
        # from the GPUManager (enabling safe polling of a pre-existing child).
        expected_job_id = f"gm-{idempotency_key[:12]}"

        # ── 4. Deep-copy params so mutations don't affect caller ────────────────
        params = copy.deepcopy(trusted_config)
        # Server-created execution identity is attached only after caller-derived
        # config has passed recursive bypass-key validation.
        params["parent_run_id"] = parent_run_id
        params["parent_job_id"] = parent_state.parent_job_id
        params["idempotency_key"] = idempotency_key

        # ── 5. Call GPUManager submit ──────────────────────────────────────────
        result = self._submit_leaf(service_id, resource_id, expected_job_id, params)

        # ── 6. Validate returned job_id ────────────────────────────────────────
        returned_job_id = result.get("job_id", "") if isinstance(result, dict) else ""
        if returned_job_id != expected_job_id:
            raise GPUMgrSubmitError(
                f"GPUManager returned mismatched job_id: {returned_job_id!r} "
                f"(expected {expected_job_id!r}); "
                f"service may have routed to an unintended pipeline"
            )

        return returned_job_id

    def poll(self, child_job_id: str) -> dict:
        """
        Poll a GPU generation child job via the injected poll_leaf callable.

        Normalises tracker statuses:
          queued / in_flight / running  →  in_flight
          completed / failed / cancelled →  pass-through

        Extracts bounded artifact and QC fields only; discards raw binary/base64.

        Returns
        -------
        dict
            {"status": str, "artifact": dict|None, "error": str|None,
             "qc_verdict": QCVerdict|None}

        Raises
        ------
        GPUMgrPollError
            If poll_leaf raises (connection/Redis error).
        """
        try:
            raw = self._poll_leaf(child_job_id)
        except Exception as exc:
            raise GPUMgrPollError(f"poll_leaf failed for {child_job_id!r}: {exc}") from exc

        if not isinstance(raw, dict):
            raw = {}

        # ── Normalise status ───────────────────────────────────────────────────
        raw_status = str(raw.get("status", "")).lower()
        if raw_status in ("queued", "in_flight", "running"):
            status = "in_flight"
        elif raw_status in ("completed", "failed", "cancelled"):
            status = raw_status
        else:
            status = raw_status if raw_status else "unknown"

        # ── Extract bounded artifact ────────────────────────────────────────────
        artifact: dict | None = None
        raw_artifact = raw.get("artifact")
        if isinstance(raw_artifact, dict):
            artifact = {
                "path_or_url": str(raw_artifact.get("path_or_url", "")),
                "sha256": str(raw_artifact.get("sha256", "")),
                "producer_stage_id": str(raw_artifact.get("producer_stage_id", "")),
                "produced_at": str(raw_artifact.get("produced_at", "")),
            }

        output: dict[str, Any] = {}
        raw_output = raw.get("output")
        if isinstance(raw_output, dict):
            output = _bounded_leaf_output(raw_output)
        if artifact is not None:
            output.setdefault("artifact", artifact)

        # ── Extract bounded QC verdict ─────────────────────────────────────────
        qc_verdict: QCVerdict | None = None
        raw_qc = raw.get("qc_verdict")
        if isinstance(raw_qc, dict):
            qc_pass = raw_qc.get("passed")
            if qc_pass is not None:
                qc_verdict = QCVerdict(
                    stage_id=str(raw_qc.get("stage_id", "")),
                    passed=bool(qc_pass),
                    report=str(raw_qc.get("report", "")),
                    raw_data=str(raw_qc["raw_data"]) if raw_qc.get("raw_data") is not None else None,
                    scored_at=str(raw_qc.get("scored_at", _now_iso())),
                )

        # ── Extract error (bounded string) ─────────────────────────────────────
        error: str | None = None
        raw_error = raw.get("error")
        if raw_error is not None:
            error = str(raw_error)

        return {
            "status": status,
            "artifact": artifact,
            "output": output,
            "error": error,
            "qc_verdict": qc_verdict,
        }

    def cancel(self, child_job_id: str) -> bool:
        """
        Cancel a GPU generation child job via the injected cancel_leaf callable.

        Delegates exactly once. Returns the result of cancel_leaf (True = cancelled,
        False = job already terminal/absent). Propagates any exception raised by
        cancel_leaf as GPUMgrCancelError.

        Returns
        -------
        bool
            True if the cancel signal was accepted; False if the job was already
            terminal or not found (truthful no-op).
        """
        try:
            return self._cancel_leaf(child_job_id)
        except Exception as exc:
            raise GPUMgrCancelError(
                f"cancel_leaf failed for {child_job_id!r}: {exc}"
            ) from exc


# ── ControllerPipelineRuntime ─────────────────────────────────────────────────

_ControllerPipelineRuntimeCtx = None  # Placeholder; runtime type only


class ControllerPipelineRuntime:
    """
    Synchronous wrapper around PipelineCoordinator + CoordinatorStore suitable
    for calling via asyncio.to_thread() from gpu-manager.py.

    This class does NOT implement the CoordinatorStore protocol — it holds a
    CoordinatorStore reference for direct queue operations and wraps the
    PipelineCoordinator for submit/cancel/get.  It is backend/queue agnostic
    through the CoordinatorStore protocol.

    Parameters
    ----------
    coordinator : PipelineCoordinator
        The pipeline coordinator to delegate to.
    store : CoordinatorStore
        The durable store providing the parent run queue (e.g. RedisCoordinatorStore
        or InMemoryCoordinatorStore).
    consumer_name : str
        Consumer identifier used when dequeuing from the store queue.
    """

    def __init__(
        self,
        *,
        coordinator: PipelineCoordinator,
        store: CoordinatorStore,
        consumer_name: str = "controller",
    ) -> None:
        self._coordinator = coordinator
        self._store = store
        self._consumer_name = consumer_name

    # ── Public synchronous API ─────────────────────────────────────────────────

    def submit_parent(
        self,
        *,
        compiled_pipeline: dict,
        original_request: dict,
        revision: str,
        parent_job_id: str,
        run_id: str,
        idempotency_key: str | None = None,
    ) -> bool:
        """
        Submit a new parent pipeline run and enqueue it via the durable store queue.

        This method uses the store's enqueue_parent directly (not an in-memory
        parallel queue) to ensure the parent is visible to process_one() calls
        even after a controller restart.

        Idempotency is enforced by run_id: submitting a second parent with the
        same run_id that is already in the store is a no-op (returns False).

        Parameters
        ----------
        compiled_pipeline : dict
            Frozen compile output from service_pipeline.compile_generation_pipeline.
        original_request : dict
            Bounded request parameters.
        revision : str
            Revision hash from compile output.
        parent_job_id : str
            GPUManager top-level job ID.
        run_id : str
            Unique pipeline run ID.  Used as idempotency key — duplicates are rejected.
        idempotency_key : str | None
            Ignored; run_id is the idempotency key.

        Returns
        -------
        bool
            True if the parent was enqueued; False if a parent with this run_id
            already exists in the store.
        """
        # Fail-closed: do not submit twice for the same run_id
        if self._store.load_state(run_id) is not None:
            return False

        ok = self._coordinator.submit(
            parent_job_id=parent_job_id,
            run_id=run_id,
            compiled_pipeline=compiled_pipeline,
            request_params=original_request,
        )
        return ok

    def process_one(self) -> PipelineRunState | None:
        """
        Claim and process one pending parent run from the durable store queue.

        This method:
          1. Calls store.dequeue_parent() — returns None on empty queue (no error).
          2. On None, attempts store.reclaim_orphaned_parents() for crash recovery.
          3. On a claimed state, calls coordinator.advance() exactly once.
          4. advance() persists its resulting state before returning.
          5. Terminal result (completed/failed/cancelled) is ACKed.
          6. Nonterminal result is re-enqueued via coordinator for the next bounded tick.
          7. If advance() raises, the claimed message is left unACKed for crash recovery.
          8. Cancellation-woken parent (cancellation_intent=True) is terminalized and ACKed.

        This method does NOT busy-loop: it returns None immediately when the
        queue is empty and no orphaned entries exist for this consumer.

        Returns
        -------
        PipelineRunState | None
            The processed (or resumed) state, or None if no parent was available.
        """
        # Step 1: dequeue one parent (single dequeue owner)
        state = self._store.dequeue_parent(self._consumer_name)
        if not state:
            # Step 2: attempt orphan recovery
            orphaned = self._store.reclaim_orphaned_parents(self._consumer_name)
            state = orphaned[0] if orphaned else None
        if not state:
            return None

        # Step 3: reload fresh state from store (process_one always reloads)
        claimed_run_id = state.run_id
        state = self._store.load_state(claimed_run_id)
        if not state:
            raise RuntimeError(
                f"claimed parent state disappeared for run_id={claimed_run_id!r}; "
                "stream entry remains unACKed for fail-closed recovery"
            )

        # Step 4: call advance() exactly once — this persists progress/terminal state
        try:
            state = self._coordinator.advance(state)
        except Exception:
            # advance() raised: leave the claimed message unACKed for crash recovery.
            # Do not re-enqueue, do not ACK, do not swallow the error.
            raise

        # Step 5: ACK terminal results; re-enqueue nonterminal via requeue_parent
        if state.status in ("completed", "failed", "cancelled"):
            # Fail-closed: if ACK fails for a terminal state, raise so the
            # persisted state remains inspectable and the stream message stays
            # recoverable via reclaim_orphaned_parents.
            ok = self._store.ack_parent(state.run_id)
            if not ok:
                raise RuntimeError(
                    f"ack_parent failed for run_id={state.run_id!r}; "
                    f"terminal state is persisted but stream ACK failed; "
                    f"process must not continue silently"
                )
        else:
            # Nonterminal — use requeue_parent to atomically ACK the claimed
            # message and enqueue the replacement. This prevents PEL accumulation.
            # Fail-closed: if the atomic replacement fails, raise so the
            # unACKed message can be recovered via reclaim_orphaned.
            ok = self._store.requeue_parent(state)
            if not ok:
                raise RuntimeError(
                    f"requeue_parent failed for run_id={state.run_id!r}; "
                    f"state is still claimed and unrecoverable without orphan reclaim"
                )

        return state

    def get(self, run_id: str) -> PipelineRunState | None:
        """
        Return the current PipelineRunState for run_id, or None if not found.
        """
        return self._store.load_state(run_id)

    def get_events(self, run_id: str) -> list[dict]:
        """Return bounded durable coordinator events for a parent run."""
        return self._store.get_events(run_id)

    def cancel(self, run_id: str) -> bool:
        """
        Request cancellation of a running parent pipeline.

        Delegates to coordinator.request_cancellation() which sets the
        cancellation_intent flag; the cascade happens on the next advance().

        Returns True if the cancellation was requested; False if the run
        is unknown or already terminal.
        """
        return self._coordinator.request_cancellation(run_id)

    def reclaim_orphaned(self) -> list[PipelineRunState]:
        """
        Reclaim all parents claimed by this runtime's consumer_name that
        were not ACKed before a crash.

        Delegates to store.reclaim_orphaned_parents().
        """
        return self._store.reclaim_orphaned_parents(self._consumer_name)
