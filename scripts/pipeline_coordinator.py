"""
pipeline_coordinator.py — Durable parent-pipeline coordinator and trusted leaf interface.

Provider/queue-agnostic. Owns the declarative DAG and finite QC/correction loop.
Never touches gpu-manager.py, service_pipeline.py, provider runtimes, or config.

Phase 2 contract
────────────────
• Immutable compiled pipeline input (from service_pipeline.compile_generation_pipeline).
• Durable PipelineRunState: plan fields, bounded inputs/outputs, child IDs, attempts,
  correction count/max, QC evidence, artifact refs+sha256+producer, cancellation,
  terminal code, restoration evidence. Never persists binary/base64 blobs.
• Injected CoordinatorStore protocol + hermetic InMemory store + Redis adapter.
  Redis adapter: dedicated namespaced parent stream + hashes/events, claim/ack/reclaim APIs.
• Injected stage authority resolver + external executor protocol.
• Trusted LeafGenerationClient: submit/poll/cancel. Leaf input is validated compiled
  stage + trusted resolved config. Rejects endpoint/unit/command/routing/bypass fields.
• Stage idempotency key = SHA256(parent_run_id, stage_id, attempt).
  Completed stage delivery is never rerun. Failed unsafe resume terminalizes stable code.
• Coordinator owns declarative DAG + bounded QC/correction loop: QC pass → skip;
  fail → bounded correction attempts + re-run declared on_correction stages;
  exhausted budget → stable terminal code.
• Cascade cancellation to active leaf child / external execution before terminal parent.
• All failures normalized to stable codes; evidence persisted.

Design seams
────────────
[pipeline_coordinator] ←─ [CoordinatorStore protocol]
[pipeline_coordinator] ←─ [StageAuthorityResolver protocol]
[pipeline_coordinator] ←─ [ExternalExecutor protocol]
[pipeline_coordinator] ←─ [LeafGenerationClient protocol]
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import redis

# Cache Redis exception types lazily so they work even when redis is mocked
# in test contexts (conftest replaces sys.modules["redis"]).
def _get_redis_error_types():
    try:
        return (redis.RedisError, redis.exceptions.RedisError)
    except AttributeError:
        return (Exception,)  # Fallback — should never happen in production

_REDIS_ERROR_TYPES: tuple = _get_redis_error_types()

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# ── Stable terminal codes ──────────────────────────────────────────────────────

class StoreError(Exception):
    """
    Raised by CoordinatorStore operations when a Redis error occurs and
    fail-closed behavior is required.

    This distinguishes "not found" (returns None) from "error" (raises
    StoreError), ensuring the consumer loop can differentiate between
    "nothing to do" and "something is wrong with Redis".
    """
    pass


class TerminalCode(str, Enum):
    # Graceful completions
    SUCCESS        = "success"
    # QC exhausted corrections
    QC_EXHAUSTED  = "qc_exhausted"
    # Provider / stage failures
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR   = "provider_error"
    LEAF_ERROR       = "leaf_error"
    # Coordinator-level failures
    COORDINATOR_ERROR   = "coordinator_error"
    COORDINATOR_CANCELLED = "coordinator_cancelled"
    UNSAFE_RESUME       = "unsafe_resume"
    # No terminal artifact produced
    NO_ARTIFACT         = "no_artifact"


# ── Stage state machine ───────────────────────────────────────────────────────

class StageStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    SKIPPED    = "skipped"


# ── Artifact reference ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class ArtifactRef:
    path_or_url: str          # Artifact path or URL (not binary data)
    sha256: str               # Hex-encoded SHA-256 of the artifact content
    producer_stage_id: str     # Stage that produced this artifact
    produced_at: str          # ISO-8601 timestamp

    def to_dict(self) -> dict:
        return {
            "path_or_url": self.path_or_url,
            "sha256": self.sha256,
            "producer_stage_id": self.producer_stage_id,
            "produced_at": self.produced_at,
        }


# ── QC verdict ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class QCVerdict:
    stage_id: str
    passed: bool
    report: str                # Bounded text summary, not raw model output
    raw_data: str | None       # Optional reference to fuller evidence (path/URL)
    scored_at: str             # ISO-8601 timestamp

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "passed": self.passed,
            "report": self.report,
            "raw_data": self.raw_data,
            "scored_at": self.scored_at,
        }


# ── Child job record ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class ChildJobRef:
    child_job_id: str
    stage_id: str
    submitted_at: str
    idempotency_key: str = ""
    status: str = "queued"
    cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "child_job_id": self.child_job_id,
            "stage_id": self.stage_id,
            "submitted_at": self.submitted_at,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "cancelled": self.cancelled,
        }


# ── Stage attempt record ───────────────────────────────────────────────────────

@dataclass(slots=True)
class StageAttempt:
    attempt: int               # 1-indexed
    status: StageStatus
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)  # Bounded stage outputs
    # QC-verdict-aware: a "completed" QC stage with verdict.passed=False is re-runnable
    qc_verdict: QCVerdict | None = None
    idempotency_key: str = ""
    # External providers may return an opaque, durable handle before their
    # result is ready.  Keep it on the attempt so a controller restart can
    # poll the same provider operation instead of replaying the submission.
    provider_handle: str | None = None

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "output": self.output,
            "qc_verdict": self.qc_verdict.to_dict() if self.qc_verdict else None,
            "idempotency_key": self.idempotency_key,
            "provider_handle": self.provider_handle,
        }


# ── Restoration evidence ────────────────────────────────────────────────────────

@dataclass(slots=True)
class RestorationEvidence:
    resumed_from: str           # run_id we resumed from
    stage_id: str | None       # stage we were at when resumed
    resumed_at: str            # ISO-8601
    resume_count: int
    restoration_artifact_refs: list[str] = field(default_factory=list)  # paths/URLs only

    def to_dict(self) -> dict:
        return {
            "resumed_from": self.resumed_from,
            "stage_id": self.stage_id,
            "resumed_at": self.resumed_at,
            "resume_count": self.resume_count,
            "restoration_artifact_refs": list(self.restoration_artifact_refs),
        }


# ── Bounded persistence sanitizer ─────────────────────────────────────────────

_MAX_STR_LEN = 10_000
_MAX_COLLECTION_ITEMS = 1_000

_SANITIZE_REJECT_KEYS = frozenset({
    "base64", "blob", "bytes", "password", "api_key",
    "authorization", "secret", "private_key",
})
_SANITIZE_REJECT_SUFFIXES = ("_base64", "_blob", "_bytes")
_SANITIZE_ALLOW_AS_PATH = frozenset({"raw_data"})  # QCVerdict.raw_data — allowed as path/URL ref


def _deep_sanitize(value: Any, path: str = "$", allow_path: bool = False) -> Any:
    """
    Recursive sanitizer for request_params, compiled_pipeline, stage outputs,
    QC reports and events.  Rejects bytes; replaces oversized strings/
    collections with stable placeholder errors; preserves artifact references.

    ``allow_path`` permits the ``raw_data`` field to pass through as a
    bounded path/URL string (used for QCVerdict.raw_data).
    """
    if isinstance(value, bytes):
        raise ValueError(f"{path}: bytes value rejected for persistence safety")

    if isinstance(value, str):
        if len(value) > _MAX_STR_LEN:
            return (
                f"{path}: string value exceeds {_MAX_STR_LEN} chars "
                f"(truncated for persistence safety)"
            )
        return value

    if isinstance(value, dict):
        for reject_key in _SANITIZE_REJECT_KEYS:
            if reject_key in value:
                raise ValueError(
                    f"{path}.{reject_key}: rejected key '{reject_key}' "
                    f"for persistence safety"
                )
        for k in list(value.keys()):
            if any(k.endswith(s) for s in _SANITIZE_REJECT_SUFFIXES):
                # Numeric byte counts (for example ``size_bytes`` in an
                # integrity report) are metadata, not an inline binary
                # payload.  Keep the raw-payload fence for every other value.
                if (
                    k.endswith("_bytes")
                    and isinstance(value[k], int)
                    and not isinstance(value[k], bool)
                    and 0 <= value[k] <= (2**63 - 1)
                ):
                    continue
                raise ValueError(
                    f"{path}.{k}: key suffix rejected for persistence safety"
                )
        return {
            k: (
                _deep_sanitize(value[k], f"{path}.{k}", allow_path=True)
                if k in _SANITIZE_ALLOW_AS_PATH
                else _deep_sanitize(value[k], f"{path}.{k}", allow_path=False)
            )
            for k in value
        }

    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError(
                f"{path}: collection exceeds {_MAX_COLLECTION_ITEMS} items "
                f"for persistence safety"
            )
        return [_deep_sanitize(item, f"{path}[{i}]", allow_path=allow_path) for i, item in enumerate(value)]

    return value  # Primitives pass through unchanged


# ── PipelineRunState ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class PipelineRunState:
    # Identity
    parent_job_id: str         # GPUManager job ID (top-level job)
    run_id: str                # Unique pipeline run ID (controller-scoped)
    pipeline_id: str           # Declarative pipeline/service ID
    revision_hash: str         # From compile_generation_pipeline output
    status: str                # "running" | "completed" | "failed" | "cancelled"

    # Cursor
    current_stage_id: str | None = None
    cursor_note: str | None = None  # e.g. "correction iteration 2"

    # Bounded inputs (no binary, no base64 blobs)
    request_params: dict[str, Any] = field(default_factory=dict)
    compiled_pipeline: dict[str, Any] = field(default_factory=dict)  # Frozen compile output

    # Child job references
    child_jobs: list[ChildJobRef] = field(default_factory=list)

    # Per-stage attempt tracking
    stage_attempts: dict[str, list[StageAttempt]] = field(default_factory=dict)
    # stage_id → list of QC verdicts (last one is current)
    stage_qc_verdicts: dict[str, list[QCVerdict]] = field(default_factory=dict)

    # Correction loop
    correction_count: int = 0
    max_corrections: int = 0
    # Loops that have been re-entered (correction → on_correction stages)
    loop_iterations: dict[str, int] = field(default_factory=dict)

    # Pending stage IDs for correction loop continuation.
    # When QC fails, this is populated with the on_correction stage IDs to run
    # before re-entering the QC stage. Empty means normal linear/correction flow.
    pending_stage_ids: list[str] = field(default_factory=list)

    # Terminal artifact
    artifact: ArtifactRef | None = None
    # Preserve the first generated artifact and the most recent QC-accepted
    # artifact separately from the currently attempted edit.  A correction
    # may produce a newer image that later fails QC; terminal delivery must
    # never silently select that rejected attempt.
    initial_artifact: ArtifactRef | None = None
    last_good_artifact: ArtifactRef | None = None

    # Cancellation
    cancelled: bool = False
    cancellation_intent: bool = False
    cancelled_children: list[str] = field(default_factory=list)  # child_job_ids

    # Terminal
    terminal_code: TerminalCode | None = None
    terminal_reason: str | None = None
    terminal_at: str | None = None

    # Timestamps
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    # Restoration evidence
    last_resumed_at: str | None = None
    resume_count: int = 0
    restoration_evidence: list[RestorationEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "parent_job_id": self.parent_job_id,
            "run_id": self.run_id,
            "pipeline_id": self.pipeline_id,
            "revision_hash": self.revision_hash,
            "status": self.status,
            "current_stage_id": self.current_stage_id,
            "cursor_note": self.cursor_note,
            "request_params": _deep_sanitize(self.request_params),
            "compiled_pipeline": _deep_sanitize(self.compiled_pipeline),
            "child_jobs": [c.to_dict() for c in self.child_jobs],
            "stage_attempts": {
                sid: [a.to_dict() for a in attempts]
                for sid, attempts in self.stage_attempts.items()
            },
            "stage_qc_verdicts": {
                sid: [v.to_dict() for v in verdicts]
                for sid, verdicts in self.stage_qc_verdicts.items()
            },
            "correction_count": self.correction_count,
            "max_corrections": self.max_corrections,
            "loop_iterations": dict(self.loop_iterations),
            "pending_stage_ids": list(self.pending_stage_ids),
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "initial_artifact": (
                self.initial_artifact.to_dict() if self.initial_artifact else None
            ),
            "last_good_artifact": (
                self.last_good_artifact.to_dict() if self.last_good_artifact else None
            ),
            "cancelled": self.cancelled,
            "cancellation_intent": self.cancellation_intent,
            "cancelled_children": list(self.cancelled_children),
            "terminal_code": self.terminal_code.value if self.terminal_code else None,
            "terminal_reason": self.terminal_reason,
            "terminal_at": self.terminal_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_resumed_at": self.last_resumed_at,
            "resume_count": self.resume_count,
            "restoration_evidence": [e.to_dict() for e in self.restoration_evidence],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineRunState:
        state = cls(
            parent_job_id=d["parent_job_id"],
            run_id=d["run_id"],
            pipeline_id=d["pipeline_id"],
            revision_hash=d["revision_hash"],
            status=d["status"],
            current_stage_id=d.get("current_stage_id"),
            cursor_note=d.get("cursor_note"),
            request_params=d.get("request_params", {}),
            compiled_pipeline=d.get("compiled_pipeline", {}),
            correction_count=d.get("correction_count", 0),
            max_corrections=d.get("max_corrections", 0),
            cancelled=d.get("cancelled", False),
            cancellation_intent=d.get("cancellation_intent", False),
            cancelled_children=d.get("cancelled_children", []),
            terminal_code=TerminalCode(d["terminal_code"])
                if d.get("terminal_code") else None,
            terminal_reason=d.get("terminal_reason"),
            terminal_at=d.get("terminal_at"),
            created_at=d.get("created_at", _now_iso()),
            updated_at=d.get("updated_at", _now_iso()),
            last_resumed_at=d.get("last_resumed_at"),
            resume_count=d.get("resume_count", 0),
        )

        for cjd in d.get("child_jobs", []):
            state.child_jobs.append(ChildJobRef(
                child_job_id=cjd["child_job_id"],
                stage_id=cjd["stage_id"],
                submitted_at=cjd["submitted_at"],
                idempotency_key=cjd.get("idempotency_key", ""),
                status=cjd.get("status", "queued"),
                cancelled=cjd.get("cancelled", False),
            ))

        for sid, attempts in d.get("stage_attempts", {}).items():
            state.stage_attempts[sid] = [
                StageAttempt(
                    attempt=a["attempt"],
                    status=StageStatus(a["status"]),
                    started_at=a["started_at"],
                    completed_at=a.get("completed_at"),
                    error=a.get("error"),
                    output=a.get("output", {}),
                    qc_verdict=(
                        QCVerdict(
                            stage_id=a["qc_verdict"]["stage_id"],
                            passed=a["qc_verdict"]["passed"],
                            report=a["qc_verdict"]["report"],
                            raw_data=a["qc_verdict"].get("raw_data"),
                            scored_at=a["qc_verdict"]["scored_at"],
                        )
                        if a.get("qc_verdict") else None
                    ),
                    idempotency_key=a.get("idempotency_key", ""),
                    provider_handle=(
                        None
                        if a.get("provider_handle") is None
                        else str(a.get("provider_handle"))
                    ),
                )
                for a in attempts
            ]

        for sid, verdicts in d.get("stage_qc_verdicts", {}).items():
            state.stage_qc_verdicts[sid] = [
                QCVerdict(
                    stage_id=v["stage_id"],
                    passed=v["passed"],
                    report=v["report"],
                    raw_data=v.get("raw_data"),
                    scored_at=v["scored_at"],
                )
                for v in verdicts
            ]

        for lid, count in d.get("loop_iterations", {}).items():
            state.loop_iterations[lid] = count

        state.pending_stage_ids = list(d.get("pending_stage_ids", []))

        def _artifact(value: Any) -> ArtifactRef | None:
            if not isinstance(value, dict):
                return None
            required = ("path_or_url", "sha256", "producer_stage_id", "produced_at")
            if any(key not in value for key in required):
                return None
            return ArtifactRef(
                path_or_url=value["path_or_url"],
                sha256=value["sha256"],
                producer_stage_id=value["producer_stage_id"],
                produced_at=value["produced_at"],
            )

        state.artifact = _artifact(d.get("artifact"))
        state.initial_artifact = _artifact(d.get("initial_artifact"))
        state.last_good_artifact = _artifact(d.get("last_good_artifact"))

        for rev in d.get("restoration_evidence", []):
            state.restoration_evidence.append(RestorationEvidence(
                resumed_from=rev["resumed_from"],
                stage_id=rev.get("stage_id"),
                resumed_at=rev["resumed_at"],
                resume_count=rev.get("resume_count", 0),
                restoration_artifact_refs=rev.get("restoration_artifact_refs", []),
            ))

        return state


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_idempotency_key(parent_run_id: str, stage_id: str, attempt: int) -> str:
    """SHA256(parent_run_id, stage_id, attempt) → hex string."""
    raw = f"{parent_run_id}:{stage_id}:{attempt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _provider_handle(value: Any) -> str:
    """Validate one bounded opaque provider operation identity."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("provider_handle must be a non-empty string")
    handle = value.strip()
    if len(handle) > 1024:
        raise ValueError("provider_handle exceeds 1024 characters")
    return handle


# ── Protocols ─────────────────────────────────────────────────────────────────

class CoordinatorStore(ABC):
    """
    Persisted coordinator state backed by Redis (or in-memory for hermetic tests).

    Key layout
    ──────────
    Coordinator-owned Redis keys:
      parent-stream : pipeline_coordinator:stream         (Redis stream, XADD/XREAD)
      state:{run_id}  : pipeline_coordinator:state:{run_id}  (Redis hash)
      event:{run_id}  : pipeline_coordinator:event:{run_id}  (Redis stream, append-only log)
    """

    @abstractmethod
    def save_state(self, state: PipelineRunState) -> bool: ...

    def submit_parent(self, state: PipelineRunState) -> bool:
        """Persist a new parent and enqueue it, failing closed on any write.

        Stores with transactional support should override this method so the
        state hash, submission event and parent-stream entry share one atomic
        boundary.  The default keeps compatibility with small test/local
        stores while still refusing to report success after a failed write.
        """

        if not self.save_state(state):
            return False
        self.append_event(state.run_id, "submitted", {"parent_job_id": state.parent_job_id})
        return self.enqueue_parent(state)

    @abstractmethod
    def load_state(self, run_id: str) -> PipelineRunState | None: ...

    @abstractmethod
    def enqueue_parent(self, state: PipelineRunState) -> bool: ...

    @abstractmethod
    def dequeue_parent(self, consumer_name: str) -> PipelineRunState | None: ...

    @abstractmethod
    def claim_parent(self, run_id: str, consumer_name: str) -> bool: ...

    @abstractmethod
    def ack_parent(self, run_id: str) -> bool: ...

    @abstractmethod
    def requeue_parent(self, state: PipelineRunState) -> bool:
        """
        Durably replace the currently-claimed message for run_id with a fresh
        entry carrying the updated state, atomically ACK-ing the old message.

        This is NOT ordinary enqueue — it is used for nonterminal requeue in the
        runtime loop and for cancellation wake. It must be crash-safe: if the
        process crashes between persisting state and ACK, the old message remains
        recoverable via reclaim_orphaned_parents.

        Implementation contract (Redis):
          - If _entry_ids has a claimed entry ID for this run_id:
              Use a Redis transaction/pipeline to atomically XADD the new message
              and XACK the old claimed ID. After transaction success: **clear**
              _entry_ids[run_id]. The new XADD entry is unread/unclaimed until
              XREADGROUP establishes the next claim. Only XREADGROUP may set a
              new _entry_ids mapping.
              Return False if the transaction fails; the old entry_id mapping is
              preserved for crash recovery.
          - If _entry_ids has no entry for this run_id (no current claim, e.g.
              formerly-ACKed or cancelled parent being re-woken):
              Safely call enqueue_parent to create a fresh entry. Return True
              if the enqueue succeeds, False otherwise.

        Implementation contract (InMemory):
          - Remove any existing _claimed[run_id] and _acked[run_id] markers.
          - Deduplicate: if run_id already has an entry in the queue, remove it.
          - Append the fresh state as a new queue entry.
          - Return True.

        Note: only XREADGROUP establishes a claim. After requeue_parent
        succeeds, _entry_ids has no mapping for that run_id until the new
        entry is read via XREADGROUP.

        Parameters
        ----------
        state : PipelineRunState
            The updated nonterminal state to carry in the replacement message.

        Returns
        -------
        bool
            True if the replacement succeeded; False if it failed (in which case
            the caller must treat the state as still-claimed and fail-closed).
        """

    @abstractmethod
    def reclaim_orphaned_parents(self, consumer_name: str) -> list[PipelineRunState]: ...

    @abstractmethod
    def append_event(self, run_id: str, event_type: str, payload: dict) -> bool: ...

    @abstractmethod
    def get_events(self, run_id: str) -> list[dict]: ...

    @abstractmethod
    def clear_cancellation(self, run_id: str) -> None: ...


class StageAuthorityResolver(ABC):
    """
    Resolves a stage kind to an execution authority.

    • "external" → run through ExternalExecutor (e.g. HTTP provider)
    • "leaf"     → run through LeafGenerationClient (trusted GPUManager child job)
    """

    @abstractmethod
    def resolve(self, stage: dict) -> str: ...
    # Returns "external" or "leaf"


class ExternalExecutor(ABC):
    """
    Executes a non-leaf stage (e.g. prompt_enhance, QC observation) and
    returns a normalized result dict.
    """

    @abstractmethod
    def execute(
        self,
        stage: dict,
        state: PipelineRunState,
        idempotency_key: str,
    ) -> dict:
        """
        Execute stage and return result dict:
          {status: "completed"|"failed"|"queued"|"in_flight"|"pending",
           output: dict, error: str|None, qc_verdict: QCVerdict|None,
           provider_handle: str|None}

        The idempotency_key (SHA256 of parent_run_id:stage_id:attempt) must be
        passed so duplicate delivery can be detected and rejected safely.
        A non-terminal status is valid only when ``provider_handle`` is a
        durable opaque provider operation identity that ``poll`` can use.
        """
        ...

    def poll(
        self,
        stage: dict,
        state: PipelineRunState,
        idempotency_key: str,
        provider_handle: str,
    ) -> dict:
        """Poll a previously accepted external operation after a restart.

        Implementations that only support bounded submit-and-poll keep this
        default. The explicit unsafe result prevents a controller restart from
        silently submitting the same external request twice.
        """
        return {
            "status": "failed",
            "error": (
                "unsafe_external_resume: no provider poll implementation"
            ),
        }

    @abstractmethod
    def cancel(self, run_id: str, stage_id: str) -> bool:
        """
        Cancel an active external execution for run_id/stage_id.
        Returns True if cancelled.  On an unsafe_resume (detected duplicate
        delivery of already-completed work), raises RuntimeError.
        """
        ...


class LeafGenerationClient(ABC):
    """
    Trusted in-process interface to submit a GPU generation child job.
    No public recursive pipeline selection; trusted template resolution.
    """

    @abstractmethod
    def submit(
        self,
        stage: dict,
        parent_run_id: str,
        parent_state: PipelineRunState,
        idempotency_key: str,
        trusted_config: dict,
    ) -> str:
        """
        Submit a child generation job. Returns child_job_id.

        The idempotency_key (SHA256 of parent_run_id:stage_id:attempt) must be
        passed so duplicate delivery can be detected and rejected safely.

        The trusted_config is derived from the compiled stage by the coordinator
        and is the only configuration the leaf may use.  The caller-controlled
        stage dict fields (endpoint, unit, command, routing_group, bypass,
        skip_pipeline, internal_leaf, etc.) are stripped before submission.
        """
        ...

    @abstractmethod
    def poll(self, child_job_id: str) -> dict:
        """
        Poll child job status. Returns dict:
          {status: "queued"|"in_flight"|"completed"|"failed",
           artifact: ArtifactRef|None, error: str|None,
           qc_verdict: QCVerdict|None}
        """
        ...

    @abstractmethod
    def cancel(self, child_job_id: str) -> bool:
        """Cancel an active child job. Returns True if cancelled."""
        ...


# ── InMemory store (hermetic tests) ──────────────────────────────────────────

class InMemoryCoordinatorStore(CoordinatorStore):
    """Fully in-memory store for hermetic tests. Not thread-safe."""

    def __init__(self):
        self._states: dict[str, PipelineRunState] = {}
        self._queue: list[PipelineRunState] = []
        self._claimed: dict[str, str] = {}  # run_id → consumer_name
        self._acked: set[str] = set()
        self._events: dict[str, list[dict]] = {}

    def save_state(self, state: PipelineRunState) -> bool:
        state.updated_at = _now_iso()
        self._states[state.run_id] = state
        return True

    def load_state(self, run_id: str) -> PipelineRunState | None:
        return self._states.get(run_id)

    def enqueue_parent(self, state: PipelineRunState) -> bool:
        # Deduplicate: remove any existing queued entry for this run_id
        self._queue = [s for s in self._queue if s.run_id != state.run_id]
        self._queue.append(state)
        return True

    def dequeue_parent(self, consumer_name: str) -> PipelineRunState | None:
        for i, s in enumerate(self._queue):
            if s.run_id not in self._claimed and s.run_id not in self._acked:
                self._queue.pop(i)
                self._claimed[s.run_id] = consumer_name
                return s
        return None

    def claim_parent(self, run_id: str, consumer_name: str) -> bool:
        if self._states.get(run_id) is None:
            return False
        self._claimed[run_id] = consumer_name
        return True

    def ack_parent(self, run_id: str) -> bool:
        self._acked.add(run_id)
        self._claimed.pop(run_id, None)
        return True

    def reclaim_orphaned_parents(self, consumer_name: str) -> list[PipelineRunState]:
        orphaned_rids = [
            rid for rid, cn in self._claimed.items()
            if cn == consumer_name and rid not in self._acked
        ]
        results = []
        for rid in orphaned_rids:
            state = self._states.get(rid)
            if state:
                results.append(state)
            self._claimed.pop(rid, None)
        return results

    def append_event(self, run_id: str, event_type: str, payload: dict) -> bool:
        if run_id not in self._events:
            self._events[run_id] = []
        self._events[run_id].append({
            "type": event_type,
            "timestamp": _now_iso(),
            "payload": payload,
        })
        return True

    def get_events(self, run_id: str) -> list[dict]:
        return list(self._events.get(run_id, []))

    def clear_cancellation(self, run_id: str) -> None:
        if run_id in self._states:
            self._states[run_id].cancellation_intent = False

    def requeue_parent(self, state: PipelineRunState) -> bool:
        """
        Replace the currently-claimed message for run_id with a fresh entry.

        Removes any existing _claimed[run_id] and _acked[run_id] markers,
        deduplicates queued entries for the same run_id, then appends the
        fresh state as a new queue entry.
        """
        # Clear claim and ACK markers
        self._claimed.pop(state.run_id, None)
        self._acked.discard(state.run_id)

        # Deduplicate: remove any existing entry for this run_id from queue
        self._queue = [s for s in self._queue if s.run_id != state.run_id]

        # Append fresh entry
        self._queue.append(state)
        return True


# ── Redis store adapter ───────────────────────────────────────────────────────

class RedisCoordinatorStore(CoordinatorStore):
    """
    Redis-backed coordinator store using a dedicated parent stream + hashes + events.

    Key layout
    ──────────
    pipeline_coordinator:stream    — parent run entries (XADD)
    pipeline_coordinator:state:{run_id}  — state hash
    pipeline_coordinator:event:{run_id}  — append-only event stream

    No hardcoded host credentials; reads from GPU_MANAGER_REDIS_PASSWORD_FILE
    or CREDENTIALS_DIRECTORY, mirroring queue_engine behaviour.
    """

    HASH_PREFIX = "pipeline_coordinator:state:"
    EVENT_PREFIX = "pipeline_coordinator:event:"
    STREAM_KEY = "pipeline_coordinator:stream"
    GROUP = "coordinator"
    BLOCK_MS = 2000  # Finite block timeout to avoid indefinite blocking
    RECLAIM_MIN_IDLE_MS = 30_000
    # One atomic boundary for a new parent.  The state key is the uniqueness
    # fence: a concurrent admission for the same deterministic run_id returns
    # 0 and cannot append a second parent-stream entry.  Hash/event/stream
    # writes happen in the same Redis/Valkey script, so a successful response
    # always describes a runnable parent.
    _SUBMIT_PARENT_LUA = r"""
local state_key = KEYS[1]
if redis.call('EXISTS', state_key) == 1 then
  return 0
end

local pair_count = tonumber(ARGV[1])
local index = 2
for _ = 1, pair_count do
  redis.call('HSET', state_key, ARGV[index], ARGV[index + 1])
  index = index + 2
end

redis.call('XADD', KEYS[2], '*',
  'type', ARGV[index],
  'timestamp', ARGV[index + 1],
  'payload', ARGV[index + 2])
index = index + 3

redis.call('XADD', KEYS[3], '*',
  'run_id', ARGV[index],
  'parent_job_id', ARGV[index + 1],
  'pipeline_id', ARGV[index + 2])
return 1
"""

    def __init__(self, r=None):
        import redis
        self._r = r or self._make_conn()
        self._ensure_group()
        self._entry_ids: dict[str, str] = {}  # run_id → Redis stream entry ID

    def _make_conn(self):
        import os
        import redis

        def _password() -> str | None:
            path = os.environ.get("GPU_MANAGER_REDIS_PASSWORD_FILE")
            if not path:
                cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
                if cred_dir:
                    path = os.path.join(cred_dir, "redis-password")
            if not path:
                return None
            try:
                return open(path, encoding="utf-8").read().strip() or None
            except OSError:
                return None

        return redis.Redis(host="localhost", port=6379,
                           password=_password(), decode_responses=True)

    def _ensure_group(self):
        try:
            self._r.xgroup_create(self.STREAM_KEY, self.GROUP, id="0", mkstream=True)
        except Exception:  # BUSYGROUP
            pass

    def _state_key(self, run_id: str) -> str:
        return f"{self.HASH_PREFIX}{run_id}"

    def _event_key(self, run_id: str) -> str:
        return f"{self.EVENT_PREFIX}{run_id}"

    def save_state(self, state: PipelineRunState) -> bool:
        state.updated_at = _now_iso()
        data = state.to_dict()
        # Encode every field with one type-preserving codec.  Redis hashes store
        # bytes/strings only; stringifying scalars loses bool/int/null identity.
        flat = {k: json.dumps(v) for k, v in data.items()}
        try:
            self._r.hset(self._state_key(state.run_id), mapping=flat)
            return True
        except Exception:
            return False

    def submit_parent(self, state: PipelineRunState) -> bool:
        """Atomically and idempotently persist a parent and queue entry.

        ``run_id`` is the uniqueness fence.  The script returns ``False`` for
        an existing state, allowing a redelivered source entry to consult the
        durable parent rather than appending another runnable stream entry.
        """

        state.updated_at = _now_iso()
        flat = {key: json.dumps(value) for key, value in state.to_dict().items()}
        event_payload = {
            "type": "submitted",
            "timestamp": _now_iso(),
            "payload": json.dumps(
                {"parent_job_id": state.parent_job_id},
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        try:
            args: list[str] = [str(len(flat))]
            for key, value in flat.items():
                args.extend((str(key), str(value)))
            args.extend(
                (
                    str(event_payload["type"]),
                    str(event_payload["timestamp"]),
                    str(event_payload["payload"]),
                    state.run_id,
                    state.parent_job_id,
                    state.pipeline_id,
                )
            )
            result = self._r.eval(
                self._SUBMIT_PARENT_LUA,
                3,
                self._state_key(state.run_id),
                self._event_key(state.run_id),
                self.STREAM_KEY,
                *args,
            )
            return int(result or 0) == 1
        except Exception:
            # Script errors leave no partial write visible; the caller receives
            # a truthful rejection and may reconcile/retry.
            return False

    def load_state(self, run_id: str) -> PipelineRunState | None:
        try:
            raw = self._r.hgetall(self._state_key(run_id))
            if not raw:
                return None
            decoded = {}
            for key, value in raw.items():
                try:
                    decoded[key] = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    # Backward compatibility for hashes written by the former
                    # str(value) codec.  Other plain strings remain unchanged.
                    if value == "True":
                        decoded[key] = True
                    elif value == "False":
                        decoded[key] = False
                    else:
                        decoded[key] = value
            return PipelineRunState.from_dict(decoded)
        except _REDIS_ERROR_TYPES as e:
            # Redis connection/timeout errors must raise StoreError (fail-closed),
            # not return None which is indistinguishable from "key not found".
            raise StoreError(
                f"load_state: Redis HGETALL failed for {run_id!r}: {e}"
            ) from e
        except Exception:
            # JSON decode errors, type errors — return None (not found or corrupt)
            return None

    def enqueue_parent(self, state: PipelineRunState) -> bool:
        # Store as raw strings (canonical).  The run_id field is read by
        # dequeue_parent / reclaim_orphaned_parents without JSON-decode, so
        # keeping it raw avoids a spurious extra loads() on the read path.
        # Backward compatibility for legacy JSON-quoted entries is handled by
        # _decode_stream_field in the decode path.
        payload = {
            "run_id": state.run_id,
            "parent_job_id": state.parent_job_id,
            "pipeline_id": state.pipeline_id,
        }
        try:
            self._r.xadd(self.STREAM_KEY, payload)
            return True
        except Exception:
            return False

    def dequeue_parent(self, consumer_name: str) -> PipelineRunState | None:
        try:
            streams = self._r.xreadgroup(
                groupname=self.GROUP,
                consumername=consumer_name,
                streams={self.STREAM_KEY: ">"},
                count=1,
                block=self.BLOCK_MS,
            )
            if not streams:
                return None
            for _sk, entries in streams:
                for entry_id, data in entries:
                    # Support both raw string and legacy JSON-quoted run_id
                    raw_run_id = data.get("run_id", "")
                    run_id = self._decode_stream_field(str(raw_run_id)) if raw_run_id else ""
                    if not run_id:
                        raise StoreError(
                            "dequeue_parent: claimed stream entry has no run_id; "
                            "leave it recoverable instead of treating it as idle"
                        )
                    self._entry_ids[run_id] = entry_id
                    state = self.load_state(run_id)
                    if state is None:
                        raise StoreError(
                            f"dequeue_parent: claimed parent {run_id!r} has no "
                            "durable state"
                        )
                    return state
            return None
        except StoreError:
            raise
        except _REDIS_ERROR_TYPES as exc:
            # Redis outages and WRONGTYPE/schema errors are not an empty queue.
            # Returning None here makes the controller look idle and can hide a
            # claimed entry until its lease expires.
            raise StoreError(
                f"dequeue_parent: Redis XREADGROUP failed: {exc}"
            ) from exc
        except Exception as exc:
            raise StoreError(
                f"dequeue_parent: unexpected stream decode failure: {exc}"
            ) from exc

    def claim_parent(self, run_id: str, consumer_name: str) -> bool:
        # Redis stream consumer group tracks claims; just verify state exists
        state = self.load_state(run_id)
        return state is not None

    def ack_parent(self, run_id: str) -> bool:
        try:
            entry_id = self._entry_ids.pop(run_id, None)
            if entry_id:
                self._r.xack(self.STREAM_KEY, self.GROUP, entry_id)
            return True
        except Exception:
            return False

    def _decode_stream_field(self, value: str) -> str:
        """
        Decode a stream field value that may be stored as raw string (canonical)
        or as a JSON-quoted string (legacy enqueue_parent bug).

        Returns the decoded string.  If the value is neither raw nor JSON-quoted,
        returns it unchanged.
        """
        # Canonical: raw string not starting with JSON metacharacters
        if not value.startswith(('"', "'")):
            return value
        # Legacy: JSON-quoted string — tolerate double-quoted values
        try:
            decoded = json.loads(value)
            if isinstance(decoded, str):
                return decoded
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return value

    def reclaim_orphaned_parents(self, consumer_name: str) -> list[PipelineRunState]:
        """Take over idle PEL entries and return one recoverable state per run.

        Entries already owned by ``consumer_name`` are immediately recoverable.
        Entries owned by another consumer are claimed only after the bounded idle
        threshold, preventing a second live consumer from stealing active work.
        Duplicate PEL messages for one run are consolidated to the newest stream
        ID while one claimed message remains available for normal ACK/requeue.
        """
        try:
            pending = self._r.xpending_range(
                self.STREAM_KEY,
                self.GROUP,
                min="-",
                max="+",
                count=100,
            )
            entries_by_run: dict[str, list[str]] = {}

            for item in pending or []:
                if not item:
                    continue
                if isinstance(item, dict):
                    msg_id = item.get("message_id")
                    owner = item.get("consumer")
                    idle_ms = item.get("time_since_delivered", 0)
                elif isinstance(item, (list, tuple)):
                    msg_id = item[0] if len(item) > 0 else None
                    owner = item[1] if len(item) > 1 else None
                    idle_ms = item[2] if len(item) > 2 else 0
                else:
                    msg_id = getattr(item, "message_id", None)
                    owner = getattr(item, "consumer", None)
                    idle_ms = getattr(item, "time_since_delivered", 0)

                if isinstance(msg_id, bytes):
                    msg_id = msg_id.decode()
                if isinstance(owner, bytes):
                    owner = owner.decode()
                if not msg_id:
                    continue

                messages = []
                if owner == consumer_name:
                    messages = self._r.xrange(self.STREAM_KEY, msg_id, msg_id)
                elif int(idle_ms or 0) >= self.RECLAIM_MIN_IDLE_MS:
                    messages = self._r.xclaim(
                        self.STREAM_KEY,
                        self.GROUP,
                        consumer_name,
                        min_idle_time=self.RECLAIM_MIN_IDLE_MS,
                        message_ids=[msg_id],
                    )

                for claimed_id, fields in messages or []:
                    if isinstance(claimed_id, bytes):
                        claimed_id = claimed_id.decode()
                    raw_run_id = fields.get("run_id", "") or ""
                    if isinstance(raw_run_id, bytes):
                        raw_run_id = raw_run_id.decode()
                    run_id = self._decode_stream_field(str(raw_run_id)) if raw_run_id else ""
                    if run_id:
                        entries_by_run.setdefault(run_id, []).append(str(claimed_id))

            def stream_id_key(entry_id: str) -> tuple[int, int]:
                milliseconds, sequence = entry_id.split("-", 1)
                return int(milliseconds), int(sequence)

            results: list[PipelineRunState] = []
            for run_id in sorted(entries_by_run):
                entry_ids = sorted(set(entries_by_run[run_id]), key=stream_id_key)
                retained_id = entry_ids[-1]
                duplicate_ids = entry_ids[:-1]
                if duplicate_ids:
                    acked = self._r.xack(
                        self.STREAM_KEY,
                        self.GROUP,
                        *duplicate_ids,
                    )
                    if acked != len(duplicate_ids):
                        raise StoreError(
                            "reclaim_orphaned_parents: failed to ACK all duplicate "
                            f"PEL entries for {run_id!r}"
                        )

                state = self.load_state(run_id)
                if state is None:
                    raise StoreError(
                        f"reclaim_orphaned_parents: state {run_id!r} found in PEL "
                        "but load_state returned None; data inconsistency detected"
                    )
                self._entry_ids[run_id] = retained_id
                results.append(state)

            return results
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(
                f"reclaim_orphaned_parents: Redis PEL recovery failed: {e}"
            ) from e

    def append_event(self, run_id: str, event_type: str, payload: dict) -> bool:
        try:
            self._r.xadd(self._event_key(run_id), {
                "type": event_type,
                "timestamp": _now_iso(),
                "payload": json.dumps(payload),
            })
            return True
        except Exception:
            return False

    def get_events(self, run_id: str) -> list[dict]:
        try:
            raw = self._r.xrange(self._event_key(run_id))
            events = []
            for _eid, fields in raw:
                event = {
                    "type": fields.get("type", ""),
                    "timestamp": fields.get("timestamp", ""),
                    "payload": json.loads(fields.get("payload", "{}")),
                }
                events.append(event)
            return events
        except Exception:
            return []

    def clear_cancellation(self, run_id: str) -> None:
        state = self.load_state(run_id)
        if state:
            state.cancellation_intent = False
            self.save_state(state)

    def requeue_parent(self, state: PipelineRunState) -> bool:
        """
        Durably replace the currently-claimed message for run_id with a fresh entry.

        If _entry_ids has a claimed entry ID for this run_id:
          - Atomically XADD the new message and XACK the old claimed ID via a
            Redis transaction/pipeline.
          - After transaction success: **clear** _entry_ids[run_id] (the new
            XADD entry is unread/unclaimed until XREADGROUP establishes the next
            claim).  Only XREADGROUP may set a new _entry_ids mapping.
          - Return False if the transaction fails; the old entry_id mapping is
            preserved for crash recovery.
        If _entry_ids has no entry for this run_id (no current claim):
          - Safely enqueue a fresh entry via enqueue_parent.

        Note: mapping is cleared after transaction success; only XREADGROUP
        establishes a new claim.  The new XADD entry must NOT be tracked in
        _entry_ids because it has not yet been read by any consumer.
        """
        claimed_entry_id = self._entry_ids.get(state.run_id)

        payload = {
            "run_id": state.run_id,
            "parent_job_id": state.parent_job_id,
            "pipeline_id": state.pipeline_id,
        }

        if claimed_entry_id is not None:
            # Atomic: XADD new entry + XACK old claimed entry
            try:
                pipe = self._r.pipeline(transaction=True)
                new_entry_id = pipe.xadd(self.STREAM_KEY, payload)
                pipe.xack(self.STREAM_KEY, self.GROUP, claimed_entry_id)
                results = pipe.execute()
                # results[0] = new_entry_id, results[1] = xack_count
                if not results or results[0] is None:
                    return False
                # Transaction succeeded: **clear** the mapping.
                # The new XADD entry is unread until XREADGROUP; tracking it
                # would cause phantom-claim bugs (XACKing an unclaimed entry).
                self._entry_ids.pop(state.run_id, None)
                return True
            except Exception:
                # Transaction failed: preserve old entry_id for recovery
                return False
        else:
            # No current claim: use ordinary enqueue
            return self.enqueue_parent(state)


# ── PipelineCoordinator ───────────────────────────────────────────────────────

class PipelineCoordinator:
    """
    Controller-owned parent pipeline coordinator.

    Owns the declarative DAG execution, QC/correction loop, cancellation cascade,
    and terminalization. Delegates external stages and leaf generation to injected
    protocols.
    """

    def __init__(
        self,
        store: CoordinatorStore,
        authority_resolver: StageAuthorityResolver,
        external_executor: ExternalExecutor,
        leaf_client: LeafGenerationClient,
    ):
        self._store = store
        self._authority = authority_resolver
        self._external = external_executor
        self._leaf = leaf_client

    def submit(
        self,
        parent_job_id: str,
        run_id: str,
        compiled_pipeline: dict,
        request_params: dict,
    ) -> bool:
        """
        Submit a new parent pipeline run. Persists state and enqueues for processing.
        """
        pipeline_id = compiled_pipeline.get("pipeline_id", compiled_pipeline.get("service_id", "?"))
        # ``workflow.v1`` plans carry their source identity as
        # ``workflow_fingerprint``; the older image compiler calls this
        # ``revision_hash``.  Persist whichever reviewed identity is present
        # without inventing a new revision.
        revision_hash = compiled_pipeline.get(
            "revision_hash",
            compiled_pipeline.get("workflow_fingerprint", ""),
        )

        state = PipelineRunState(
            parent_job_id=parent_job_id,
            run_id=run_id,
            pipeline_id=pipeline_id,
            revision_hash=revision_hash,
            status="running",
            request_params=_deep_sanitize(copy.deepcopy(request_params)),
            compiled_pipeline=_deep_sanitize(copy.deepcopy(compiled_pipeline)),
            max_corrections=compiled_pipeline.get("max_loop_count", 0),
        )

        # Select the first dependency-ready stage instead of assuming the
        # declaration order is already topological.  The bridge emits a
        # stable topological order, but this guard also protects older or
        # GUI-authored compiled plans.
        stage_list = compiled_pipeline.get("stages", [])
        stage_map = {
            str(stage.get("id")): stage
            for stage in stage_list
            if isinstance(stage, dict) and stage.get("id")
        }
        state.current_stage_id = self._first_ready_stage_id(state, stage_map)

        # Admission is only successful once the store's state + event + parent
        # queue boundary succeeds. Redis uses one transaction; local stores
        # retain the compatible fail-closed fallback in CoordinatorStore.
        return self._store.submit_parent(state)

    @staticmethod
    def _latest_stage_status(
        state: PipelineRunState, stage_id: str
    ) -> StageStatus | None:
        attempts = state.stage_attempts.get(stage_id, [])
        return attempts[-1].status if attempts else None

    @classmethod
    def _stage_ready(
        cls,
        state: PipelineRunState,
        stage: dict,
        stage_map: dict[str, dict],
    ) -> bool:
        """Return whether a stage's declared dependencies have completed."""

        if stage.get("enabled_by_default", True) is False:
            return False
        stage_id = str(stage.get("id") or "")
        if not stage_id:
            return False
        current_status = cls._latest_stage_status(state, stage_id)
        if current_status in {
            StageStatus.COMPLETED,
            StageStatus.SKIPPED,
        }:
            return False
        for dependency in stage.get("depends_on", []) or []:
            dep_id = str(dependency)
            if dep_id not in stage_map:
                return False
            if stage_map[dep_id].get("enabled_by_default", True) is False:
                # Disabled nodes are semantically skipped at admission; a
                # dependent stage must not deadlock waiting for an attempt
                # record that will never be created.
                continue
            dep_status = cls._latest_stage_status(state, dep_id)
            if dep_status not in {StageStatus.COMPLETED, StageStatus.SKIPPED}:
                return False
        return True

    @classmethod
    def _first_ready_stage_id(
        cls, state: PipelineRunState, stage_map: dict[str, dict]
    ) -> str | None:
        """Choose the first ready non-correction stage deterministically."""

        for stage in stage_map.values():
            if stage.get("kind") == "correct":
                # Correction stages enter only through an explicit loop.
                continue
            if cls._stage_ready(state, stage, stage_map):
                return str(stage["id"])
        return None

    def process_one(self, consumer_name: str) -> PipelineRunState | None:
        """
        Claim and process one pending parent run. Called by the controller consumer loop.

        Raises StoreError if Redis operations fail (fail-closed), or if state
        disappears between reclaim and load (data inconsistency).
        """
        state = self._store.dequeue_parent(consumer_name)
        if not state:
            # XPENDING/XRANGE errors surface as StoreError — must not be silently swallowed.
            # This ensures fail-closed behavior: the consumer loop knows something is wrong.
            orphaned = self._store.reclaim_orphaned_parents(consumer_name)
            state = orphaned[0] if orphaned else None
        if not state:
            return None

        # Reload fresh state while retaining the claimed identity for a
        # fail-closed diagnostic if the durable hash disappeared.
        claimed_run_id = state.run_id
        state = self._store.load_state(claimed_run_id)
        if not state:
            # State was in PEL but load failed — data inconsistency. Raise StoreError
            # so the caller knows something is seriously wrong (not "nothing to do").
            raise StoreError(
                f"process_one: state {claimed_run_id!r} was reclaimed from PEL "
                f"but load_state returned None; data inconsistency"
            )

        # Resume cursor
        if state.current_stage_id is None:
            stage_list = state.compiled_pipeline.get("stages", [])
            if stage_list:
                state.current_stage_id = stage_list[0]["id"]

        state.last_resumed_at = _now_iso()
        state.resume_count += 1
        self._store.save_state(state)
        self._store.append_event(state.run_id, "resumed", {
            "stage_id": state.current_stage_id,
            "resume_count": state.resume_count,
        })

        return state

    def advance(self, state: PipelineRunState) -> PipelineRunState:
        """
        Advance the state machine one step. Returns updated state.

        Called iteratively by the controller consumer loop until terminal.
        """
        state = self._store.load_state(state.run_id) or state

        # Check cancellation cascade
        if state.cancellation_intent and not state.cancelled:
            return self._cascade_cancellation(state)

        # Terminal guard
        if state.status in ("completed", "failed", "cancelled"):
            return state

        stage_list = state.compiled_pipeline.get("stages", [])
        stage_map = {s["id"]: s for s in stage_list}

        # ── pending_stage_ids queue ────────────────────────────────────────────
        # If there are pending correction-loop stages, run the next one.
        # This queue is populated when QC fails and drained as we execute
        # the on_correction stages before re-entering the QC stage.
        current_attempts = state.stage_attempts.get(state.current_stage_id or "", [])
        current_stage_running = bool(
            current_attempts
            and current_attempts[-1].status == StageStatus.RUNNING
        )
        if state.pending_stage_ids and not current_stage_running:
            next_pending = state.pending_stage_ids[0]
            remaining = state.pending_stage_ids[1:]
            # Verify the stage still exists
            if next_pending not in stage_map:
                # Stage missing; skip it
                state.pending_stage_ids = remaining
                self._store.save_state(state)
                return state
            state.current_stage_id = next_pending
            state.pending_stage_ids = remaining
            state.cursor_note = (
                f"correction iteration {state.correction_count}"
                if state.correction_count else None
            )
            self._store.save_state(state)

        # ── Resolve current stage ─────────────────────────────────────────────
        current_id = state.current_stage_id
        if current_id is None or current_id not in stage_map:
            # No more stages → success
            return self._terminalize(state, TerminalCode.SUCCESS, "All stages completed")

        current_stage = stage_map[current_id]
        authority = self._authority.resolve(current_stage)

        # ── Idempotency check ─────────────────────────────────────────────────
        # If this stage already completed with a passing QC verdict (or no
        # verdict), skip it.  QC stages with a failing verdict may be re-entered
        # during correction loops even though their attempt status is COMPLETED.
        attempts = state.stage_attempts.setdefault(current_id, [])

        if attempts and attempts[-1].status == StageStatus.COMPLETED:
            last_attempt = attempts[-1]
            is_qc_with_fail = (
                last_attempt.qc_verdict is not None
                and not last_attempt.qc_verdict.passed
            )
            is_declared_loop_reentry = (
                state.correction_count > 0
                and bool(state.cursor_note)
            )
            # A declared loop may intentionally re-run any stage in its
            # on_correction sequence. Outside that cursor, completed work is
            # never repeated on duplicate delivery.
            if not is_qc_with_fail and not is_declared_loop_reentry:
                return self._advance_to_next_stage(state, stage_map)

        resuming_running_attempt = bool(
            attempts and attempts[-1].status == StageStatus.RUNNING
        )
        if resuming_running_attempt:
            attempt_record = attempts[-1]
            current_attempt_num = attempt_record.attempt
            idem_key = attempt_record.idempotency_key or _stage_idempotency_key(
                state.run_id, current_id, current_attempt_num
            )
            attempt_record.idempotency_key = idem_key
        else:
            current_attempt_num = len(attempts) + 1
            idem_key = _stage_idempotency_key(
                state.run_id, current_id, current_attempt_num
            )
            attempt_record = StageAttempt(
                attempt=current_attempt_num,
                status=StageStatus.RUNNING,
                started_at=_now_iso(),
                idempotency_key=idem_key,
            )
            attempts.append(attempt_record)
            self._store.save_state(state)
            self._store.append_event(state.run_id, "stage_started", {
                "stage_id": current_id,
                "authority": authority,
                "attempt": current_attempt_num,
                "idempotency_key": idem_key,
            })

        # Resolve typed workflow bindings against the frozen request and
        # completed stage outputs immediately before execution.  The compiled
        # plan remains immutable; this copy is the only per-attempt payload
        # passed to an adapter/leaf.
        result: dict | None = None
        try:
            execution_stage = self._materialize_stage_inputs(current_stage, state)
        except (TypeError, ValueError, KeyError) as exc:
            result = {
                "status": "failed",
                "error": f"binding_resolution_failed: {exc}",
            }
            execution_stage = current_stage

        # ── Execute ───────────────────────────────────────────────────────────
        if result is None and authority == "external":
            try:
                if resuming_running_attempt:
                    # A provider handle makes an accepted external operation
                    # resumable without replaying its submission.  Older or
                    # bounded adapters still fail closed when no handle was saved.
                    if attempt_record.provider_handle:
                        result = self._external.poll(
                            execution_stage,
                            state,
                            idempotency_key=idem_key,
                            provider_handle=attempt_record.provider_handle,
                        )
                    else:
                        result = {
                            "status": "failed",
                            "error": (
                                "unsafe_external_resume: prior delivery outcome is "
                                f"unknown for stage {current_id}"
                            ),
                        }
                else:
                    result = self._external.execute(
                        execution_stage, state, idempotency_key=idem_key,
                    )
            except Exception as exc:
                # Reviewed synchronous adapters have no remote acceptance
                # boundary. Persist their exception as the terminal outcome of
                # this attempt so a later consumer tick cannot mistake it for
                # an accepted-but-untracked external operation.
                result = {
                    "status": "failed",
                    "error": f"external_adapter_failed: {str(exc)[:1000]}",
                }
        elif result is None and authority == "leaf":
            result = self._execute_leaf(
                state, execution_stage, current_attempt_num, idem_key,
            )
        elif result is None:
            result = {"status": "failed", "error": f"Unknown authority: {authority}"}

        if result["status"] in ("queued", "in_flight", "pending"):
            if authority == "external":
                try:
                    attempt_record.provider_handle = _provider_handle(
                        result.get("provider_handle")
                    )
                except (TypeError, ValueError) as exc:
                    # A non-terminal provider response without a durable
                    # handle cannot be safely retried after a crash. Treat it
                    # as an explicit unsafe outcome rather than leaving a
                    # permanently RUNNING attempt that would later replay.
                    result = {
                        "status": "failed",
                        "error": f"unsafe_external_resume: {exc}",
                    }
                else:
                    self._store.save_state(state)
                    self._store.append_event(state.run_id, "external_handle_saved", {
                        "stage_id": current_id,
                        "attempt": current_attempt_num,
                    })
                    return state
            else:
                self._store.save_state(state)
                return state

        # ── Record terminal attempt result ────────────────────────────────────
        completed_at = _now_iso()
        attempt_record.status = (
            StageStatus.COMPLETED
            if result["status"] == "completed"
            else StageStatus.FAILED
        )
        attempt_record.completed_at = completed_at
        attempt_record.error = result.get("error")
        attempt_record.output = _deep_sanitize(result.get("output", {}))
        if result.get("provider_handle") is not None:
            try:
                attempt_record.provider_handle = _provider_handle(
                    result.get("provider_handle")
                )
            except (TypeError, ValueError) as exc:
                attempt_record.status = StageStatus.FAILED
                attempt_record.error = f"invalid provider_handle: {exc}"
                result = {
                    **result,
                    "status": "failed",
                    "error": attempt_record.error,
                }

        qc_verdict = result.get("qc_verdict")
        if qc_verdict:
            state.stage_qc_verdicts.setdefault(current_id, []).append(qc_verdict)
            attempt_record.qc_verdict = qc_verdict

        if result.get("artifact"):
            art = result["artifact"]
            artifact = ArtifactRef(
                path_or_url=art["path_or_url"],
                sha256=art["sha256"],
                producer_stage_id=current_id,
                produced_at=completed_at,
            )
            state.artifact = artifact
            if current_stage.get("kind") in {"generate", "generation"}:
                if state.initial_artifact is None:
                    state.initial_artifact = artifact

        self._store.save_state(state)

        if result["status"] == "failed":
            return self._handle_stage_failure(state, current_stage, stage_map, result)

        # Success — check if this is a QC stage with a verdict
        if qc_verdict and current_stage["kind"] == "qc":
            return self._handle_qc_result(state, qc_verdict, stage_map)

        return self._advance_to_next_stage(state, stage_map)

    def _materialize_stage_inputs(
        self,
        stage: dict,
        state: PipelineRunState,
    ) -> dict:
        """Resolve a compiled stage's typed bindings for one attempt.

        Bindings are limited to ``workflow.<input>`` and
        ``stage.<stage>.<output>`` by ``workflow_capabilities``.  Keeping the
        resolver here means every external/leaf adapter receives the same
        inputs and no adapter needs to interpret user-facing graph syntax.
        """

        execution_stage = copy.deepcopy(stage)
        stage_params = execution_stage.get("params")
        if not isinstance(stage_params, dict):
            return execution_stage
        bindings = stage_params.get("bindings")
        if not isinstance(bindings, dict):
            return execution_stage
        input_bindings = bindings.get("inputs")
        if not isinstance(input_bindings, dict):
            return execution_stage

        resolved: dict[str, Any] = {}
        for name, binding in input_bindings.items():
            if not isinstance(binding, dict):
                raise ValueError(f"input binding {name!r} is not an object")
            source = binding.get("source")
            value_found = True
            value: Any = None
            if isinstance(source, str) and source.startswith("workflow."):
                input_name = source.removeprefix("workflow.")
                if input_name not in state.request_params:
                    input_defs = state.compiled_pipeline.get("inputs", {})
                    input_def = (
                        input_defs.get(input_name)
                        if isinstance(input_defs, dict)
                        else None
                    )
                    if isinstance(input_def, dict) and "default" in input_def:
                        value = input_def["default"]
                    else:
                        value_found = False
                else:
                    value = state.request_params[input_name]
            elif isinstance(source, str) and source.startswith("stage."):
                parts = source.split(".", 2)
                if len(parts) != 3:
                    raise ValueError(f"invalid stage binding source {source!r}")
                source_stage, output_name = parts[1], parts[2]
                attempts = state.stage_attempts.get(source_stage, [])
                if not attempts:
                    value_found = False
                else:
                    latest = attempts[-1]
                    value_found = output_name in (latest.output or {})
                    if value_found:
                        value = latest.output[output_name]
            else:
                raise ValueError(f"unsupported binding source {source!r}")

            if not value_found:
                if binding.get("optional") is True:
                    continue
                raise ValueError(
                    f"required input {name!r} has no resolved value from {source!r}"
                )
            resolved[str(name)] = _deep_sanitize(copy.deepcopy(value))

        execution_stage["resolved_inputs"] = resolved
        return execution_stage

    def _execute_leaf(
        self,
        state: PipelineRunState,
        stage: dict,
        attempt: int,
        idem_key: str,
    ) -> dict:
        """
        Execute a leaf (GPU generation) stage through the trusted LeafGenerationClient.
        """
        trusted_config = self._build_trusted_config(stage, state)

        child_ref = next(
            (
                child for child in reversed(state.child_jobs)
                if child.stage_id == stage["id"]
                and child.idempotency_key == idem_key
                and not child.cancelled
            ),
            None,
        )
        # A transient admission/observation failure can mark the stage
        # attempt failed after the leaf has already been accepted. Before
        # creating a new child for the retry, inspect the most recent
        # non-cancelled child for this stage. Reusing a queued, in-flight, or
        # completed child preserves the provider side effect and prevents a
        # late ComfyUI completion from being followed by a duplicate submit.
        if child_ref is None:
            for candidate in reversed(state.child_jobs):
                if candidate.stage_id != stage["id"] or candidate.cancelled:
                    continue
                try:
                    prior = self._leaf.poll(candidate.child_job_id)
                except Exception:
                    continue
                if prior.get("status") in {"queued", "in_flight", "completed"}:
                    child_ref = candidate
                    break
        if child_ref is None:
            child_job_id = self._leaf.submit(
                stage, state.run_id, state, idem_key, trusted_config,
            )
            child_ref = ChildJobRef(
                child_job_id=child_job_id,
                stage_id=stage["id"],
                submitted_at=_now_iso(),
                idempotency_key=idem_key,
                status="queued",
            )
            state.child_jobs.append(child_ref)
            self._store.save_state(state)

        poll_result = self._leaf.poll(child_ref.child_job_id)
        status = poll_result["status"]
        child_ref.status = status
        self._store.save_state(state)

        if status == "completed":
            artifact = poll_result.get("artifact")
            output = poll_result.get("output")
            if not isinstance(output, dict):
                output = {}
            if artifact:
                output = {**output, "artifact": artifact}
            return {
                "status": "completed",
                "output": output,
                "artifact": artifact,
                "qc_verdict": poll_result.get("qc_verdict"),
            }
        if status == "failed":
            return {
                "status": "failed",
                "error": poll_result.get("error", "leaf generation failed"),
            }
        if status in ("queued", "in_flight"):
            return {"status": status}
        return {
            "status": "failed",
            "error": f"Unknown child status: {status}",
        }

    # Caller-controlled field prefixes/suffixes to strip from leaf stage
    _LEAF_STRIP_PREFIXES = (
        "endpoint", "unit", "command", "routing_group",
        "bypass", "skip_pipeline", "internal_leaf",
    )
    _LEAF_STRIP_KEYS = frozenset({
        "endpoint", "unit", "command", "routing_group",
        "bypass", "skip_pipeline", "internal_leaf",
        "routing", "url", "host",
    })

    def _build_trusted_config(
        self,
        stage: dict,
        state: PipelineRunState | None = None,
    ) -> dict:
        """
        Build a trusted config from a compiled stage by recursively stripping all
        caller-controlled fields that a malicious or buggy caller might inject to
        redirect work or bypass safety checks.
        """
        result = {}
        for k, v in stage.items():
            if k in self._LEAF_STRIP_KEYS:
                continue
            if any(k.startswith(p) for p in self._LEAF_STRIP_PREFIXES):
                continue
            result[k] = self._deep_strip(v)
        if state is not None:
            effective_prompt = str(state.request_params.get("prompt") or "")
            stage_defs = {
                str(item.get("id")): item
                for item in state.compiled_pipeline.get("stages", [])
            }
            for stage_id, attempts in state.stage_attempts.items():
                if stage_defs.get(stage_id, {}).get("kind") != "prompt_enhance":
                    continue
                for attempt in reversed(attempts):
                    if attempt.status != StageStatus.COMPLETED:
                        continue
                    output = attempt.output or {}
                    enhanced = output.get("enhanced_prompt") or output.get("prompt")
                    if enhanced:
                        effective_prompt = str(enhanced)
                        break

            latest_qc = None
            for verdicts in state.stage_qc_verdicts.values():
                if verdicts:
                    latest_qc = verdicts[-1].to_dict()

            result["context"] = self._deep_strip({
                "request_params": copy.deepcopy(state.request_params),
                "effective_prompt": effective_prompt,
                "artifact": state.artifact.to_dict() if state.artifact else None,
                "qc_verdict": latest_qc,
            })
        return result

    def _deep_strip(self, value: Any) -> Any:
        """Recursively strip forbidden keys from nested dicts and lists."""
        _strip_suffixes = ("_base64", "_blob", "_bytes", "_data")
        if isinstance(value, dict):
            return {
                k: self._deep_strip(v)
                for k, v in value.items()
                if k not in self._LEAF_STRIP_KEYS
                and not any(k.startswith(p) for p in self._LEAF_STRIP_PREFIXES)
                and not any(k.endswith(s) for s in _strip_suffixes)
            }
        if isinstance(value, (list, tuple)):
            return [self._deep_strip(item) for item in value]
        return value

    def _handle_qc_result(
        self,
        state: PipelineRunState,
        verdict: QCVerdict,
        stage_map: dict[str, dict],
    ) -> PipelineRunState:
        """
        Evaluate QC verdict. Pass → advance. Fail → correction loop or terminal.
        """
        self._store.append_event(state.run_id, "qc_result", {
            "stage_id": verdict.stage_id,
            "passed": verdict.passed,
            "report": verdict.report,
        })

        if verdict.passed:
            if state.artifact is not None:
                state.last_good_artifact = state.artifact
                self._store.save_state(state)
            # QC pass — advance to next stage
            return self._advance_to_next_stage(state, stage_map)

        # QC fail — find the loop whose trigger_stage_id is this QC stage
        current_qc_id = verdict.stage_id
        loops = state.compiled_pipeline.get("loops", [])
        loop_entry = None
        for loop in loops:
            if loop.get("trigger_stage_id") == current_qc_id:
                loop_entry = loop
                break

        correction_stage_id: str
        on_correction: list[str]
        if loop_entry:
            correction_stage_id = loop_entry["correction_stage_id"]
            on_correction = loop_entry.get("on_correction", [])
        else:
            # Fallback: use first correction-kind stage and default continuation
            correct_stages = [
                s for s in state.compiled_pipeline.get("stages", [])
                if s.get("kind") == "correct"
            ]
            if not correct_stages:
                return self._terminalize(
                    state,
                    TerminalCode.QC_EXHAUSTED,
                    "QC failed but no correct stage defined",
                )
            correction_stage_id = correct_stages[0]["id"]
            on_correction = []

        loop_id = loop_entry["id"] if loop_entry else correction_stage_id
        loop_iterations = state.loop_iterations.get(loop_id, 0)
        loop_max = loop_entry["max_iterations"] if loop_entry else state.max_corrections

        if loop_iterations >= loop_max:
            return self._terminalize(
                state,
                TerminalCode.QC_EXHAUSTED,
                f"QC failed after {loop_iterations}/{loop_max} correction iterations",
            )

        # Increment per-loop iteration counter and total correction count
        state.loop_iterations[loop_id] = loop_iterations + 1
        state.correction_count += 1

        # Schedule the exact declarative continuation: correction_stage then on_correction
        state.pending_stage_ids = [correction_stage_id, *on_correction]
        state.current_stage_id = None
        state.cursor_note = f"correction iteration {state.loop_iterations[loop_id]}"
        self._store.save_state(state)
        self._store.append_event(state.run_id, "correction_started", {
            "correction_stage_id": correction_stage_id,
            "loop_id": loop_id,
            "iteration": state.loop_iterations[loop_id],
            "on_correction": on_correction,
        })

        return state

    def _handle_stage_failure(
        self,
        state: PipelineRunState,
        failed_stage: dict,
        stage_map: dict[str, dict],
        result: dict,
    ) -> PipelineRunState:
        """Handle a non-QC stage failure with a bounded retry contract."""
        error_text = str(result.get("error") or "unknown")

        # A resumed external stage has no durable provider handle.  Repeating
        # it could duplicate a paid or generative request, so this outcome is
        # terminal and must remain distinguishable from an ordinary provider
        # failure.
        if "unsafe_external_resume" in error_text:
            return self._terminalize(
                state,
                TerminalCode.UNSAFE_RESUME,
                f"Stage {failed_stage['id']} outcome is unknown: {error_text}",
            )

        retry = failed_stage.get("retry")
        if isinstance(retry, dict):
            max_attempts = retry.get("max_attempts", 1)
        else:
            # Backward-compatible image-pipeline plans use ``retries`` as a
            # retry count rather than a total-attempt count.
            raw_retries = failed_stage.get("retries", 0)
            max_attempts = int(raw_retries) + 1 if isinstance(raw_retries, int) else 1
        try:
            max_attempts = int(max_attempts)
        except (TypeError, ValueError):
            max_attempts = 1
        max_attempts = max(1, min(max_attempts, 11))
        attempts = state.stage_attempts.get(str(failed_stage.get("id")), [])
        current_attempt = attempts[-1].attempt if attempts else 1
        if current_attempt < max_attempts:
            state.current_stage_id = str(failed_stage["id"])
            state.cursor_note = f"retry attempt {current_attempt + 1}/{max_attempts}"
            self._store.save_state(state)
            self._store.append_event(state.run_id, "stage_retry_scheduled", {
                "stage_id": failed_stage["id"],
                "attempt": current_attempt + 1,
                "max_attempts": max_attempts,
                "reconcile_before_retry": bool(
                    retry.get("reconcile_before_retry", False)
                    if isinstance(retry, dict) else False
                ),
            })
            return state

        code: TerminalCode
        if failed_stage["kind"] in {"generate", "generation"}:
            code = TerminalCode.LEAF_ERROR
        elif "timeout" in error_text.lower():
            code = TerminalCode.PROVIDER_TIMEOUT
        else:
            code = TerminalCode.PROVIDER_ERROR

        return self._terminalize(
            state,
            code,
            f"Stage {failed_stage['id']} failed: {error_text}",
        )

    def _advance_to_next_stage(
        self,
        state: PipelineRunState,
        stage_map: dict[str, dict],
    ) -> PipelineRunState:
        """Advance to the next dependency-ready stage.

        Declaration order is used only as a deterministic tie-breaker.  This
        prevents a GUI-authored graph with fan-in or out-of-order declarations
        from running a stage before its inputs are durably complete.
        """
        if state.pending_stage_ids:
            state.current_stage_id = None
            self._store.save_state(state)
            return state

        stage_list = state.compiled_pipeline.get("stages", [])
        previous_stage_id = state.current_stage_id
        next_stage_id = self._first_ready_stage_id(state, stage_map)
        if next_stage_id is None:
            enabled = [
                stage for stage in stage_list
                if stage.get("enabled_by_default", True) is not False
                and stage.get("kind") != "correct"
            ]
            terminal = all(
                self._latest_stage_status(state, str(stage["id"]))
                in {StageStatus.COMPLETED, StageStatus.SKIPPED}
                for stage in enabled
            )
            if terminal:
                return self._terminalize(state, TerminalCode.SUCCESS, "All stages completed")
            # A valid DAG should always expose a ready stage after a success.
            # Treat a missing one as a coordinator error rather than spinning
            # or silently dropping a declared operation.
            return self._terminalize(
                state,
                TerminalCode.COORDINATOR_ERROR,
                "no dependency-ready stage remains; workflow state is inconsistent",
            )

        if next_stage_id == previous_stage_id:
            # This is only expected when a caller invokes advance twice without
            # recording a result.  Leave the state unchanged and let the outer
            # lease/replay path retry the same stage idempotently.
            self._store.save_state(state)
            return state

        if next_stage_id not in stage_map:
            return self._terminalize(state, TerminalCode.SUCCESS, "All stages completed")

        state.current_stage_id = next_stage_id
        state.cursor_note = None
        self._store.save_state(state)
        self._store.append_event(state.run_id, "stage_advanced", {
            "to": next_stage_id,
            "from": previous_stage_id,
        })
        return state

    def _cascade_cancellation(self, state: PipelineRunState) -> PipelineRunState:
        """
        Cascade cancellation to active leaf children and external execution,
        then terminalize the parent only when every accepted child has a
        truthful stop result. A false/failed stop is an unresolved provider
        outcome: do not mark the child cancelled or release its GPU ownership
        under a misleading parent ``cancelled`` status.
        """
        self._store.append_event(state.run_id, "cancellation_cascade", {
            "child_jobs": [c.child_job_id for c in state.child_jobs],
            "current_stage_id": state.current_stage_id,
        })

        unresolved: list[str] = []

        # Cancel external execution only when the current stage is an active
        # external attempt. Calling the external callback for a leaf stage (or
        # for a completed stage whose cursor has not advanced yet) can cancel
        # an unrelated provider operation.
        if state.current_stage_id:
            current_id = state.current_stage_id
            current_attempts = state.stage_attempts.get(current_id, [])
            current_active = bool(
                current_attempts
                and current_attempts[-1].status == StageStatus.RUNNING
            )
            stage_map = {
                str(stage.get("id")): stage
                for stage in state.compiled_pipeline.get("stages", [])
                if isinstance(stage, dict) and stage.get("id")
            }
            current_stage = stage_map.get(current_id)
            authority = None
            if current_active and current_stage is not None:
                try:
                    authority = self._authority.resolve(current_stage)
                except Exception as exc:
                    unresolved.append(f"unknown:{current_id}")
                    self._store.append_event(state.run_id, "cancellation_error", {
                        "stage_id": current_id,
                        "error": f"authority resolution failed: {str(exc)[:10_000]}",
                    })
            if current_active and authority == "external":
                # A missing callback/false result is not proof that a provider
                # stopped, so retain an explicit unresolved outcome.
                try:
                    stopped = bool(
                        self._external.cancel(state.run_id, current_id)
                    )
                except Exception as exc:
                    stopped = False
                    self._store.append_event(state.run_id, "cancellation_error", {
                        "stage_id": current_id,
                        "error": str(exc)[:10_000],
                    })
                if not stopped:
                    unresolved.append(f"external:{current_id}")

        # Cancel only non-terminal leaf children. Completed/failed child
        # receipts are historical evidence and must never be rewritten as
        # cancelled during a later parent cancellation request.
        for child_ref in state.child_jobs:
            if child_ref.cancelled or child_ref.status in {
                "completed", "failed", "cancelled"
            }:
                continue
            try:
                stopped = bool(self._leaf.cancel(child_ref.child_job_id))
                if stopped:
                    child_ref.cancelled = True
                    child_ref.status = "cancelled"
                    state.cancelled_children.append(child_ref.child_job_id)
                else:
                    unresolved.append(f"leaf:{child_ref.child_job_id}")
            except Exception as exc:
                unresolved.append(f"leaf:{child_ref.child_job_id}")
                self._store.append_event(state.run_id, "cancellation_error", {
                    "child_job_id": child_ref.child_job_id,
                    "error": str(exc)[:10_000],
                })

        state.cancelled = True
        state.cancellation_intent = False
        if unresolved:
            # ``UNSAFE_RESUME`` is the existing stable code for an outcome that
            # cannot be proven. It deliberately keeps the parent failed rather
            # than claiming that a provider/GPU child stopped successfully.
            return self._terminalize(
                state,
                TerminalCode.UNSAFE_RESUME,
                "Cancellation requested but provider stop is unresolved: "
                + ", ".join(unresolved[:32]),
            )
        return self._terminalize(
            state,
            TerminalCode.COORDINATOR_CANCELLED,
            "Cancelled by request",
        )

    def _terminalize(
        self,
        state: PipelineRunState,
        code: TerminalCode,
        reason: str,
    ) -> PipelineRunState:
        """Mark parent as terminal and persist."""
        if state.status in ("completed", "failed", "cancelled"):
            return state  # Already terminal

        # A failed correction/provider attempt must not publish its rejected
        # artifact. Prefer the most recent QC-passed artifact, falling back to
        # the original generation when no QC pass was recorded. The terminal
        # status remains failed/review-needed; this only selects the safest
        # artifact reference for downstream inspection or manual review.
        if code != TerminalCode.SUCCESS:
            fallback = state.last_good_artifact or state.initial_artifact
            if fallback is not None and state.artifact != fallback:
                state.artifact = fallback
                reason = f"{reason}; fallback_artifact={fallback.producer_stage_id}"

        # Map terminal code to canonical status
        if code == TerminalCode.SUCCESS:
            state.status = "completed"
        elif code == TerminalCode.COORDINATOR_CANCELLED:
            state.status = "cancelled"
        else:
            state.status = "failed"

        state.terminal_code = code
        state.terminal_reason = reason
        state.terminal_at = _now_iso()

        self._store.save_state(state)
        self._store.append_event(state.run_id, "terminalized", {
            "code": code.value,
            "reason": reason,
            "artifact": state.artifact.to_dict() if state.artifact else None,
        })
        return state

    def request_cancellation(self, run_id: str) -> bool:
        """Persist cancellation intent and wake the parent for its next advance."""
        state = self._store.load_state(run_id)
        if not state:
            return False
        if state.status in ("completed", "failed", "cancelled"):
            return False
        state.cancellation_intent = True
        if not self._store.save_state(state):
            return False
        # Wake through requeue_parent: when no entry is currently claimed
        # (e.g. formerly ACKed), requeue_parent falls back to enqueue_parent.
        # When a claim is tracked, it atomically ACKs the old + enqueues the new.
        if not self._store.requeue_parent(state):
            return False
        self._store.append_event(run_id, "cancellation_requested", {})
        return True

    def get_state(self, run_id: str) -> PipelineRunState | None:
        return self._store.load_state(run_id)
