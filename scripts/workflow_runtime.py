"""Durable, provider-agnostic state for typed GPU Manager workflows.

This module is deliberately an execution boundary, not an execution engine.
It pins a workflow/service revision, tracks stage attempts and provider handles,
and applies only checked state transitions.  Adapters (n8n, OpenDesign,
ComfyUI, audio.cpp, cloud providers, or local QC) remain outside this module.
The in-memory store is a deterministic fixture for the same records a durable
store can persist.  ``FileWorkflowStore`` is an explicit single-writer local
persistence boundary; neither store performs network, process, or provider
execution.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping
import tempfile
from typing import Any

from gpu_manager_contracts import apply_priority_contract
from workflow_contracts import validate_workflow_definition, workflow_fingerprint


class WorkflowRuntimeError(ValueError):
    """A requested workflow transition violates the durable contract."""


class StageStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    SUBMITTED = "submitted"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"
    AWAITING_REVIEW = "awaiting_review"
    SKIPPED = "skipped"


class WorkflowStatus(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"
    AWAITING_REVIEW = "awaiting_review"


_ACTIVE_STAGE_STATES = {
    StageStatus.READY,
    StageStatus.SUBMITTED,
    StageStatus.IN_FLIGHT,
}
_TERMINAL_WORKFLOW_STATES = {
    WorkflowStatus.COMPLETED,
    WorkflowStatus.FAILED,
    WorkflowStatus.CANCELLED,
    WorkflowStatus.OUTCOME_UNKNOWN,
    WorkflowStatus.AWAITING_REVIEW,
}


def _now(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def _payload_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_map(definition: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    stages = definition.get("stages", [])
    return {
        str(stage["id"]): stage
        for stage in stages
        if isinstance(stage, Mapping) and isinstance(stage.get("id"), str)
    }


def _stage_enabled(stage: Mapping[str, Any]) -> bool:
    return stage.get("enabled_by_default", True) is not False


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage_id: str
    attempt: int = 1
    status: StageStatus = StageStatus.PENDING
    idempotency_key: str = ""
    provider_handle: str | None = None
    output_refs: tuple[str, ...] = ()
    error: str | None = None
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class WorkflowJob:
    job_id: str
    service_name: str
    service_revision: str
    workflow_id: str
    workflow_revision: int
    workflow_fingerprint: str
    payload_digest: str
    idempotency_key: str
    priority: int
    priority_class: str
    broker_priority: int
    status: WorkflowStatus = WorkflowStatus.ACCEPTED
    cancel_requested: bool = False
    terminal_reason: str | None = None
    stages: tuple[StageRecord, ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0

    def stage(self, stage_id: str) -> StageRecord:
        for record in self.stages:
            if record.stage_id == stage_id:
                return record
        raise WorkflowRuntimeError(f"unknown stage {stage_id!r}")


def _replace_stage(job: WorkflowJob, record: StageRecord, *, now: float) -> WorkflowJob:
    stages = tuple(
        record if current.stage_id == record.stage_id else current
        for current in job.stages
    )
    return replace(job, stages=stages, updated_at=now)


def _derive_status(job: WorkflowJob, definition: Mapping[str, Any]) -> WorkflowStatus:
    statuses = {record.status for record in job.stages}
    if StageStatus.OUTCOME_UNKNOWN in statuses:
        return WorkflowStatus.OUTCOME_UNKNOWN
    if StageStatus.AWAITING_REVIEW in statuses:
        return WorkflowStatus.AWAITING_REVIEW
    if StageStatus.FAILED in statuses:
        return WorkflowStatus.FAILED
    if job.cancel_requested:
        if any(record.status in _ACTIVE_STAGE_STATES for record in job.stages):
            return WorkflowStatus.RUNNING
        return WorkflowStatus.CANCELLED
    enabled = [
        record for record in job.stages
        if _stage_enabled(_stage_map(definition).get(record.stage_id, {}))
    ]
    if enabled and all(record.status in {StageStatus.SUCCEEDED, StageStatus.SKIPPED} for record in enabled):
        return WorkflowStatus.COMPLETED
    if any(
        record.status in _ACTIVE_STAGE_STATES or record.status is StageStatus.SUCCEEDED
        for record in job.stages
    ):
        return WorkflowStatus.RUNNING
    return WorkflowStatus.ACCEPTED


def _with_status(job: WorkflowJob, definition: Mapping[str, Any], *, now: float) -> WorkflowJob:
    status = _derive_status(job, definition)
    terminal_reason = job.terminal_reason
    if status in _TERMINAL_WORKFLOW_STATES and terminal_reason is None:
        terminal_reason = status.value
    return replace(job, status=status, terminal_reason=terminal_reason, updated_at=now)


def ready_stage_ids(job: WorkflowJob, definition: Mapping[str, Any]) -> tuple[str, ...]:
    """Return pending stages whose dependencies have all succeeded/skipped."""
    stages = _stage_map(definition)
    ready: list[str] = []
    for record in job.stages:
        stage = stages.get(record.stage_id)
        if stage is None or record.status is not StageStatus.PENDING:
            continue
        if not _stage_enabled(stage):
            continue
        dependencies = stage.get("depends_on", [])
        if all(
            job.stage(str(dep)).status in {StageStatus.SUCCEEDED, StageStatus.SKIPPED}
            for dep in dependencies
        ):
            ready.append(record.stage_id)
    return tuple(ready)


def accept_workflow(
    definition: Mapping[str, Any],
    *,
    job_id: str,
    service_name: str,
    service_revision: str,
    payload: Any,
    idempotency_key: str,
    priority: Mapping[str, Any] | None = None,
    now: float = 0.0,
) -> WorkflowJob:
    """Pin a validated workflow/service revision at parent admission."""
    errors = validate_workflow_definition(definition)
    if errors:
        raise WorkflowRuntimeError("invalid workflow: " + "; ".join(errors))
    for field, value in {
        "job_id": job_id,
        "service_name": service_name,
        "service_revision": service_revision,
        "idempotency_key": idempotency_key,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise WorkflowRuntimeError(f"{field} must be a non-empty string")
    if not isinstance(priority, Mapping):
        priority = {}
    try:
        contracted = apply_priority_contract(priority)
    except (TypeError, ValueError) as exc:
        raise WorkflowRuntimeError(f"invalid priority: {exc}") from exc
    stages = tuple(
        StageRecord(
            stage_id=str(stage["id"]),
            status=StageStatus.PENDING
            if _stage_enabled(stage)
            else StageStatus.SKIPPED,
            updated_at=_now(now),
        )
        for stage in definition["stages"]
    )
    return WorkflowJob(
        job_id=job_id,
        service_name=service_name,
        service_revision=service_revision,
        workflow_id=str(definition["id"]),
        workflow_revision=int(definition.get("revision", 1)),
        workflow_fingerprint=workflow_fingerprint(definition),
        payload_digest=_payload_digest(payload),
        idempotency_key=idempotency_key,
        priority=int(contracted["priority"]),
        priority_class=str(contracted["priority_class"]),
        broker_priority=int(contracted["broker_priority"]),
        stages=stages,
        created_at=_now(now),
        updated_at=_now(now),
    )


def _require_mutable(job: WorkflowJob) -> None:
    if job.status in _TERMINAL_WORKFLOW_STATES:
        raise WorkflowRuntimeError(
            f"job {job.job_id} is terminal ({job.status.value}); late completion rejected"
        )


def _require_pinned_definition(
    job: WorkflowJob, definition: Mapping[str, Any]
) -> None:
    """Reject a transition evaluated against a different workflow revision."""
    errors = validate_workflow_definition(definition)
    if errors:
        raise WorkflowRuntimeError("invalid workflow: " + "; ".join(errors))
    observed = workflow_fingerprint(definition)
    if (
        definition.get("id") != job.workflow_id
        or int(definition.get("revision", 1)) != job.workflow_revision
        or observed != job.workflow_fingerprint
    ):
        raise WorkflowRuntimeError(
            "workflow revision changed after admission; refresh the pinned definition"
        )


def mark_stage_ready(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    stage_id: str,
    *,
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    _require_mutable(job)
    if stage_id not in ready_stage_ids(job, definition):
        raise WorkflowRuntimeError(f"stage {stage_id!r} is not dependency-ready")
    return _with_status(
        _replace_stage(job, replace(job.stage(stage_id), status=StageStatus.READY), now=now),
        definition,
        now=_now(now),
    )


def submit_stage(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    stage_id: str,
    *,
    idempotency_key: str,
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    _require_mutable(job)
    current = job.stage(stage_id)
    if current.status is not StageStatus.READY:
        raise WorkflowRuntimeError(f"stage {stage_id!r} is not ready")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise WorkflowRuntimeError("stage idempotency_key must be non-empty")
    return _with_status(
        _replace_stage(
            job,
            replace(
                current,
                status=StageStatus.SUBMITTED,
                idempotency_key=idempotency_key,
                error=None,
            ),
            now=now,
        ),
        definition,
        now=_now(now),
    )


def accept_stage(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    stage_id: str,
    *,
    provider_handle: str,
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    _require_mutable(job)
    current = job.stage(stage_id)
    if current.status is not StageStatus.SUBMITTED:
        raise WorkflowRuntimeError(f"stage {stage_id!r} is not submitted")
    if not isinstance(provider_handle, str) or not provider_handle.strip():
        raise WorkflowRuntimeError("provider_handle must be non-empty")
    return _with_status(
        _replace_stage(
            job,
            replace(
                current,
                status=StageStatus.IN_FLIGHT,
                provider_handle=provider_handle,
            ),
            now=now,
        ),
        definition,
        now=_now(now),
    )


def complete_stage(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    stage_id: str,
    *,
    output_refs: tuple[str, ...] | list[str] = (),
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    _require_mutable(job)
    current = job.stage(stage_id)
    if current.status not in {StageStatus.SUBMITTED, StageStatus.IN_FLIGHT}:
        raise WorkflowRuntimeError(f"stage {stage_id!r} has no accepted provider attempt")
    outputs = tuple(str(value) for value in output_refs)
    return _with_status(
        _replace_stage(
            job,
            replace(current, status=StageStatus.SUCCEEDED, output_refs=outputs, error=None),
            now=now,
        ),
        definition,
        now=_now(now),
    )


def fail_stage(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    stage_id: str,
    *,
    error: str,
    retry: bool = False,
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    _require_mutable(job)
    current = job.stage(stage_id)
    if current.status not in {StageStatus.SUBMITTED, StageStatus.IN_FLIGHT}:
        raise WorkflowRuntimeError(f"stage {stage_id!r} has no active provider attempt")
    if retry:
        next_record = StageRecord(
            stage_id=stage_id,
            attempt=current.attempt + 1,
            status=StageStatus.READY,
            error=str(error)[:1000],
            updated_at=_now(now),
        )
    else:
        next_record = replace(
            current,
            status=StageStatus.FAILED,
            error=str(error)[:1000],
            updated_at=_now(now),
        )
    return _with_status(
        _replace_stage(job, next_record, now=now), definition, now=_now(now)
    )


def mark_outcome_unknown(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    stage_id: str,
    *,
    error: str,
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    _require_mutable(job)
    current = job.stage(stage_id)
    if current.status not in {StageStatus.SUBMITTED, StageStatus.IN_FLIGHT}:
        raise WorkflowRuntimeError(f"stage {stage_id!r} has no ambiguous provider attempt")
    return _with_status(
        _replace_stage(
            job,
            replace(current, status=StageStatus.OUTCOME_UNKNOWN, error=str(error)[:1000]),
            now=now,
        ),
        definition,
        now=_now(now),
    )


def request_cancel(
    job: WorkflowJob,
    definition: Mapping[str, Any],
    *,
    reason: str = "cancel_requested",
    now: float = 0.0,
) -> WorkflowJob:
    _require_pinned_definition(job, definition)
    if job.status in _TERMINAL_WORKFLOW_STATES:
        return job
    stages = tuple(
        replace(record, status=StageStatus.CANCELLED, error=reason, updated_at=_now(now))
        if record.status in {StageStatus.PENDING, StageStatus.READY}
        else record
        for record in job.stages
    )
    return _with_status(
        replace(job, cancel_requested=True, terminal_reason=reason, stages=stages),
        definition,
        now=_now(now),
    )


class IdempotencyConflict(WorkflowRuntimeError):
    """The same admission key was reused for a different payload."""


class InMemoryWorkflowStore:
    """Durable-store-shaped fixture with atomic admission and transitions."""

    def __init__(self):
        self._jobs: dict[str, WorkflowJob] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        definition: Mapping[str, Any],
        *,
        job_id: str,
        service_name: str,
        service_revision: str,
        payload: Any,
        idempotency_key: str,
        priority: Mapping[str, Any] | None = None,
        now: float = 0.0,
    ) -> tuple[WorkflowJob, bool]:
        payload_hash = _payload_digest(payload)
        lookup = (service_name, idempotency_key)
        async with self._lock:
            existing = self._idempotency.get(lookup)
            if existing is not None:
                prior_hash, prior_job_id = existing
                if prior_hash != payload_hash:
                    raise IdempotencyConflict(
                        "idempotency key reused with a different payload"
                    )
                return self._jobs[prior_job_id], False
            if job_id in self._jobs:
                raise WorkflowRuntimeError(f"job_id already exists: {job_id}")
            job = accept_workflow(
                definition,
                job_id=job_id,
                service_name=service_name,
                service_revision=service_revision,
                payload=payload,
                idempotency_key=idempotency_key,
                priority=priority,
                now=now,
            )
            self._jobs[job_id] = job
            self._idempotency[lookup] = (payload_hash, job_id)
            return job, True

    async def get(self, job_id: str) -> WorkflowJob:
        async with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise WorkflowRuntimeError(f"unknown job {job_id!r}") from exc

    async def put(self, job: WorkflowJob) -> WorkflowJob:
        async with self._lock:
            if job.job_id not in self._jobs:
                raise WorkflowRuntimeError(f"unknown job {job.job_id!r}")
            self._jobs[job.job_id] = job
            return job


_STORE_SCHEMA = "workflow-runtime-store.v1"


def _stage_to_dict(record: StageRecord) -> dict[str, Any]:
    return {
        "stage_id": record.stage_id,
        "attempt": record.attempt,
        "status": record.status.value,
        "idempotency_key": record.idempotency_key,
        "provider_handle": record.provider_handle,
        "output_refs": list(record.output_refs),
        "error": record.error,
        "updated_at": record.updated_at,
    }


def _job_to_dict(job: WorkflowJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "service_name": job.service_name,
        "service_revision": job.service_revision,
        "workflow_id": job.workflow_id,
        "workflow_revision": job.workflow_revision,
        "workflow_fingerprint": job.workflow_fingerprint,
        "payload_digest": job.payload_digest,
        "idempotency_key": job.idempotency_key,
        "priority": job.priority,
        "priority_class": job.priority_class,
        "broker_priority": job.broker_priority,
        "status": job.status.value,
        "cancel_requested": job.cancel_requested,
        "terminal_reason": job.terminal_reason,
        "stages": [_stage_to_dict(record) for record in job.stages],
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise WorkflowRuntimeError(f"workflow store record is missing {key!r}") from exc


def _stage_from_dict(value: Any) -> StageRecord:
    if not isinstance(value, Mapping):
        raise WorkflowRuntimeError("workflow store stage record must be an object")
    try:
        output_refs = tuple(str(item) for item in _required(value, "output_refs"))
        return StageRecord(
            stage_id=str(_required(value, "stage_id")),
            attempt=int(_required(value, "attempt")),
            status=StageStatus(str(_required(value, "status"))),
            idempotency_key=str(_required(value, "idempotency_key")),
            provider_handle=(
                None
                if value.get("provider_handle") is None
                else str(value.get("provider_handle"))
            ),
            output_refs=output_refs,
            error=None if value.get("error") is None else str(value.get("error")),
            updated_at=float(_required(value, "updated_at")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowRuntimeError("invalid workflow store stage record") from exc


def _job_from_dict(value: Any) -> WorkflowJob:
    if not isinstance(value, Mapping):
        raise WorkflowRuntimeError("workflow store job record must be an object")
    try:
        stages = tuple(_stage_from_dict(item) for item in _required(value, "stages"))
        return WorkflowJob(
            job_id=str(_required(value, "job_id")),
            service_name=str(_required(value, "service_name")),
            service_revision=str(_required(value, "service_revision")),
            workflow_id=str(_required(value, "workflow_id")),
            workflow_revision=int(_required(value, "workflow_revision")),
            workflow_fingerprint=str(_required(value, "workflow_fingerprint")),
            payload_digest=str(_required(value, "payload_digest")),
            idempotency_key=str(_required(value, "idempotency_key")),
            priority=int(_required(value, "priority")),
            priority_class=str(_required(value, "priority_class")),
            broker_priority=int(_required(value, "broker_priority")),
            status=WorkflowStatus(str(_required(value, "status"))),
            cancel_requested=bool(_required(value, "cancel_requested")),
            terminal_reason=(
                None
                if value.get("terminal_reason") is None
                else str(value.get("terminal_reason"))
            ),
            stages=stages,
            created_at=float(_required(value, "created_at")),
            updated_at=float(_required(value, "updated_at")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowRuntimeError("invalid workflow store job record") from exc


class FileWorkflowStore:
    """Small atomic JSON store for restart/resume fixtures and local deployments.

    The file is a persistence boundary for the reducer above; it does not claim
    to replace a transactional queue database or coordinate provider calls.
    Writes use a same-directory temporary file, fsync and atomic replacement so
    a process restart cannot observe a half-written document.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._jobs: dict[str, WorkflowJob] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    def _read_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping) or raw.get("schema") != _STORE_SCHEMA:
                raise WorkflowRuntimeError("unsupported workflow store schema")
            jobs_raw = raw.get("jobs")
            idem_raw = raw.get("idempotency")
            if not isinstance(jobs_raw, Mapping) or not isinstance(idem_raw, list):
                raise WorkflowRuntimeError("invalid workflow store envelope")
            jobs = {str(job_id): _job_from_dict(value) for job_id, value in jobs_raw.items()}
            if any(job_id != job.job_id for job_id, job in jobs.items()):
                raise WorkflowRuntimeError("workflow store job key does not match record")
            idempotency: dict[tuple[str, str], tuple[str, str]] = {}
            for item in idem_raw:
                if not isinstance(item, Mapping):
                    raise WorkflowRuntimeError("invalid workflow idempotency record")
                service = str(_required(item, "service_name"))
                key = str(_required(item, "idempotency_key"))
                payload_digest = str(_required(item, "payload_digest"))
                job_id = str(_required(item, "job_id"))
                if job_id not in jobs:
                    raise WorkflowRuntimeError("idempotency record references an unknown job")
                job = jobs[job_id]
                if (
                    service != job.service_name
                    or key != job.idempotency_key
                    or payload_digest != job.payload_digest
                ):
                    raise WorkflowRuntimeError("workflow idempotency record does not match job")
                if (service, key) in idempotency:
                    raise WorkflowRuntimeError("duplicate workflow idempotency record")
                idempotency[(service, key)] = (payload_digest, job_id)
        except OSError as exc:
            raise WorkflowRuntimeError("unable to read workflow store") from exc
        except json.JSONDecodeError as exc:
            raise WorkflowRuntimeError("workflow store is not valid JSON") from exc
        self._jobs = jobs
        self._idempotency = idempotency

    def _write_locked(
        self,
        jobs: Mapping[str, WorkflowJob],
        idempotency: Mapping[tuple[str, str], tuple[str, str]],
    ) -> None:
        document = {
            "schema": _STORE_SCHEMA,
            "jobs": {job_id: _job_to_dict(job) for job_id, job in jobs.items()},
            "idempotency": [
                {
                    "service_name": service,
                    "idempotency_key": key,
                    "payload_digest": payload_digest,
                    "job_id": job_id,
                }
                for (service, key), (payload_digest, job_id) in idempotency.items()
            ],
        }
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self._path.name}.", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise WorkflowRuntimeError("unable to persist workflow store") from exc

    async def create(
        self,
        definition: Mapping[str, Any],
        *,
        job_id: str,
        service_name: str,
        service_revision: str,
        payload: Any,
        idempotency_key: str,
        priority: Mapping[str, Any] | None = None,
        now: float = 0.0,
    ) -> tuple[WorkflowJob, bool]:
        payload_hash = _payload_digest(payload)
        lookup = (service_name, idempotency_key)
        async with self._lock:
            self._read_locked()
            existing = self._idempotency.get(lookup)
            if existing is not None:
                prior_hash, prior_job_id = existing
                if prior_hash != payload_hash:
                    raise IdempotencyConflict(
                        "idempotency key reused with a different payload"
                    )
                return self._jobs[prior_job_id], False
            if job_id in self._jobs:
                raise WorkflowRuntimeError(f"job_id already exists: {job_id}")
            job = accept_workflow(
                definition,
                job_id=job_id,
                service_name=service_name,
                service_revision=service_revision,
                payload=payload,
                idempotency_key=idempotency_key,
                priority=priority,
                now=now,
            )
            jobs = dict(self._jobs)
            jobs[job_id] = job
            idempotency = dict(self._idempotency)
            idempotency[lookup] = (payload_hash, job_id)
            self._write_locked(jobs, idempotency)
            self._jobs, self._idempotency = jobs, idempotency
            return job, True

    async def get(self, job_id: str) -> WorkflowJob:
        async with self._lock:
            self._read_locked()
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise WorkflowRuntimeError(f"unknown job {job_id!r}") from exc

    async def put(self, job: WorkflowJob) -> WorkflowJob:
        async with self._lock:
            self._read_locked()
            if job.job_id not in self._jobs:
                raise WorkflowRuntimeError(f"unknown job {job.job_id!r}")
            current = self._jobs[job.job_id]
            for field in (
                "service_name",
                "service_revision",
                "workflow_id",
                "workflow_revision",
                "workflow_fingerprint",
                "payload_digest",
                "idempotency_key",
            ):
                if getattr(job, field) != getattr(current, field):
                    raise WorkflowRuntimeError(
                        f"workflow identity field {field!r} cannot change"
                    )
            lookup = (job.service_name, job.idempotency_key)
            if self._idempotency.get(lookup, (None, None))[1] != job.job_id:
                raise WorkflowRuntimeError("workflow idempotency index does not match job")
            jobs = dict(self._jobs)
            jobs[job.job_id] = job
            self._write_locked(jobs, self._idempotency)
            self._jobs = jobs
            return job


__all__ = [
    "IdempotencyConflict",
    "FileWorkflowStore",
    "InMemoryWorkflowStore",
    "StageRecord",
    "StageStatus",
    "WorkflowJob",
    "WorkflowRuntimeError",
    "WorkflowStatus",
    "accept_stage",
    "accept_workflow",
    "complete_stage",
    "fail_stage",
    "mark_outcome_unknown",
    "mark_stage_ready",
    "ready_stage_ids",
    "request_cancel",
    "submit_stage",
]
