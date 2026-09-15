"""Reviewed MiniMax Music 3 custom coordinator callbacks.

The MiniMax graph contains two operations that are deliberately not generic
HTTP providers:

* ``local-audio-qc`` runs the pinned CPU CLAP/Whisper checker and must never
  turn a missing model or hash mismatch into a pass.
* ``artifact-store`` uses a capability-agent reserve/upload/finalize/abort
  port.  The controller owns the destination and grant policy; a workflow
  request may not choose a URL, filename, or credential.

This module contains only the checked coordinator contract.  Deployment code
injects the qualified QC runner and capability port.  With either injection
missing the factory returns an unsupported binding rather than silently
creating a generic callback.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pipeline_coordinator import PipelineRunState, _now_iso


class CustomCallbackContractError(RuntimeError):
    """A reviewed MiniMax callback cannot safely execute this attempt."""


class CustomCallbackUnavailable(CustomCallbackContractError):
    """A deployment-owned dependency is not qualified or is unavailable."""


@runtime_checkable
class ArtifactCapabilityPort(Protocol):
    """Capability-agent reserve/upload/finalize/reconcile contract.

    Implementations may be async or return values directly.  The port owns
    bearer handling and sink authority; this module never receives a bearer.
    """

    def reserve(self, *, grant_id: str, grant_metadata: Mapping[str, Any]) -> Any: ...

    def upload(
        self,
        *,
        reservation: Mapping[str, Any],
        local_path: str,
        byte_size: int,
        sha256: str,
        media_type: str,
        role: str,
    ) -> Any: ...

    def finalize(
        self,
        *,
        reservation: Mapping[str, Any],
        byte_size: int,
        sha256: str,
    ) -> Any: ...

    def abort(self, *, reservation: Mapping[str, Any]) -> Any: ...

    def reconcile(self, *, provider_handle: str) -> Any: ...


AudioQCRunner = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class MiniMaxCustomCallbackPolicy:
    """Deployment-owned locality and delivery policy.

    ``audio_root`` is intentionally required.  It is the only source tree
    accepted by QC and upload, so a request cannot turn an artifact reference
    into arbitrary host-file access.
    """

    audio_root: Path
    max_audio_bytes: int = 262_144_000
    qc_timeout_seconds: float = 900.0
    grant_prefix: str = "CG-MINIMAX"
    media_type: str = "audio/flac"
    role: str = "minimax-music3-review"

    def __post_init__(self) -> None:
        root = Path(self.audio_root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("audio_root must be an existing non-symlink directory")
        if self.max_audio_bytes <= 0 or self.max_audio_bytes > 262_144_000:
            raise ValueError("max_audio_bytes is outside the bounded policy")
        if self.qc_timeout_seconds <= 0 or self.qc_timeout_seconds > 3_600:
            raise ValueError("qc_timeout_seconds is outside the bounded policy")
        if not re.fullmatch(r"CG-[A-Za-z0-9_-]{1,64}", self.grant_prefix):
            raise ValueError("grant_prefix must be a capability-agent CG identifier")
        if not isinstance(self.media_type, str) or "/" not in self.media_type:
            raise ValueError("media_type must be a MIME type")
        if not isinstance(self.role, str) or not self.role or len(self.role) > 128:
            raise ValueError("role must be a bounded non-empty string")

    @property
    def resolved_audio_root(self) -> Path:
        return Path(self.audio_root).resolve(strict=True)


def _bounded_text(value: Any, field: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise CustomCallbackContractError(f"{field} is required")
        return ""
    if not isinstance(value, str):
        raise CustomCallbackContractError(f"{field} must be a string")
    value = value.strip()
    if required and not value:
        raise CustomCallbackContractError(f"{field} is required")
    if len(value) > 16_000:
        raise CustomCallbackContractError(f"{field} exceeds 16000 characters")
    return value


def _resolved_inputs(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    values = stage.get("resolved_inputs")
    if not isinstance(values, Mapping):
        raise CustomCallbackContractError("stage has no resolved_inputs")
    return values


def _raw_path(value: Any, field: str) -> str:
    if isinstance(value, Mapping):
        value = value.get("path") or value.get("artifact_path") or value.get("path_or_url")
    if not isinstance(value, str) or not value.strip():
        raise CustomCallbackContractError(f"{field} has no local artifact path")
    value = value.strip()
    if "://" in value or value.startswith(("~", "\\")):
        raise CustomCallbackContractError(f"{field} must be a local path")
    return value


def _validated_audio_file(value: Any, policy: MiniMaxCustomCallbackPolicy) -> tuple[Path, int, str]:
    """Validate one regular, non-symlink file and hash its exact bytes."""

    raw = _raw_path(value, "original_audio")
    root = policy.resolved_audio_root
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    # Check every path component before resolving; a symlink inside the root
    # is not an acceptable source even if it points back inside the root.
    try:
        lexical = Path(os.path.abspath(candidate))
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise CustomCallbackContractError("audio path is outside the reviewed root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                raise CustomCallbackContractError("audio path contains a symlink")
        except OSError as exc:
            raise CustomCallbackContractError("audio path could not be inspected") from exc
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
        info = resolved.stat()
    except (OSError, ValueError) as exc:
        raise CustomCallbackContractError("audio file is missing or escapes the reviewed root") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CustomCallbackContractError("audio file must be a regular single-link file")
    if info.st_size <= 0 or info.st_size > policy.max_audio_bytes:
        raise CustomCallbackContractError("audio file size is outside the reviewed bound")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return resolved, info.st_size, digest.hexdigest()


def _native_brief(values: Mapping[str, Any], state: PipelineRunState) -> tuple[str, str, str]:
    brief = values.get("native_brief")
    if not isinstance(brief, Mapping):
        raise CustomCallbackContractError("native_brief must be an object")
    request = state.request_params if isinstance(state.request_params, Mapping) else {}
    original = brief.get("original_prompt") or brief.get("brief") or request.get("brief")
    original_prompt = _bounded_text(original, "original_prompt")
    caption = _bounded_text(brief.get("caption") or brief.get("prompt") or original_prompt, "caption")
    lyrics = _bounded_text(brief.get("lyrics"), "lyrics", required=False) or "[instrumental]"
    return original_prompt, caption, lyrics


def _safe_json(value: Any, *, limit: int = 10_000) -> str:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


_QC_REDACT_KEYS = frozenset(
    {
        "path",
        "audio_path",
        "model_path",
        "weights_path",
        "source_path",
        "credential",
        "credentials",
        "token",
        "secret",
        "authorization",
    }
)


def _redact_qc(value: Any, *, depth: int = 0) -> Any:
    """Keep QC evidence portable without persisting host paths or secrets."""

    if depth > 6:
        return "[truncated]"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:200]:
            name = str(key)
            if name.lower() in _QC_REDACT_KEYS or name.lower().endswith("_path"):
                continue
            result[name] = _redact_qc(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_qc(item, depth=depth + 1) for item in list(value)[:200]]
    if isinstance(value, str):
        return value[:2_000]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:2_000]


def _is_dependency_failure(exc: BaseException) -> bool:
    message = f"{exc.__class__.__name__}:{exc}".lower()
    return any(
        token in message
        for token in (
            "model_missing",
            "model_hash_mismatch",
            "model missing",
            "hash mismatch",
            "dependency",
            "not installed",
            "unavailable",
        )
    )


class _MiniMaxAudioQCCallback:
    def __init__(self, policy: MiniMaxCustomCallbackPolicy, runner: AudioQCRunner):
        if not callable(runner):
            raise TypeError("audio QC runner must be callable")
        self._policy = policy
        self._runner = runner

    def __call__(self, stage: dict, state: PipelineRunState, _idempotency_key: str) -> Mapping[str, Any]:
        return asyncio.run(self._execute(stage, state))

    async def _execute(self, stage: dict, state: PipelineRunState) -> Mapping[str, Any]:
        try:
            values = _resolved_inputs(stage)
            path, size, digest = _validated_audio_file(values.get("original_audio"), self._policy)
            original_prompt, caption, lyrics = _native_brief(values, state)
        except CustomCallbackContractError as exc:
            return {"status": "failed", "error": f"local_audio_qc_contract: {str(exc)[:500]}"}
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    self._runner,
                    audio_path=path,
                    original_prompt=original_prompt,
                    caption=caption,
                    lyrics=lyrics,
                ),
                timeout=self._policy.qc_timeout_seconds,
            )
            if inspect.isawaitable(raw):
                raw = await asyncio.wait_for(raw, timeout=self._policy.qc_timeout_seconds)
        except asyncio.TimeoutError:
            report = {
                "status": "blocked",
                "qc_state": "blocked",
                "semantic_pass": False,
                "failures": ["qc_timeout"],
                "audio_bytes": size,
                "audio_sha256": digest,
            }
            return self._result(stage, report, passed=False)
        except Exception as exc:
            if not _is_dependency_failure(exc):
                return {"status": "failed", "error": f"local_audio_qc_failed:{exc.__class__.__name__}"}
            report = {
                "status": "blocked",
                "qc_state": "blocked",
                "semantic_pass": False,
                "failures": ["qc_dependency_unavailable"],
                "audio_bytes": size,
                "audio_sha256": digest,
                "error_class": exc.__class__.__name__,
            }
            return self._result(stage, report, passed=False)
        if not isinstance(raw, Mapping):
            return {"status": "failed", "error": "local_audio_qc returned a non-object"}
        verdict = raw.get("verdict")
        passed = bool(verdict.get("semantic_pass")) if isinstance(verdict, Mapping) else bool(raw.get("semantic_pass"))
        report = _redact_qc(raw)
        report["qc_state"] = "completed"
        report["audio_bytes"] = size
        report["audio_sha256"] = digest
        report["semantic_pass"] = passed
        return self._result(stage, report, passed=passed)

    @staticmethod
    def _result(stage: Mapping[str, Any], report: Mapping[str, Any], *, passed: bool) -> Mapping[str, Any]:
        report_text = _safe_json(report)
        return {
            "status": "completed",
            "output": {
                "qc_report": copy.deepcopy(dict(report)),
                "qc_pass": passed,
                "report": report_text,
                "qc_state": str(report.get("qc_state") or "completed"),
            },
            "qc_verdict": {
                "stage_id": str(stage.get("id") or "musical-qc"),
                "passed": passed,
                "report": report_text,
                "raw_data": report_text,
                "scored_at": _now_iso(),
            },
        }


def _callable_result(value: Any) -> Any:
    return value


async def _await_call(method: Callable[..., Any], **kwargs: Any) -> Any:
    result = _callable_result(method(**kwargs))
    if inspect.isawaitable(result):
        return await result
    return result


def _reservation(value: Any, grant_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CustomCallbackContractError("artifact reserve returned a non-object")
    result = {str(key): copy.deepcopy(item) for key, item in value.items()}
    observed = str(result.get("grant_id") or grant_id)
    if observed != grant_id:
        raise CustomCallbackContractError("artifact reservation grant identity mismatch")
    result["grant_id"] = grant_id
    nonce = result.get("reservation_nonce")
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_.~-]{8,256}", nonce):
        raise CustomCallbackContractError("artifact reservation nonce is invalid")
    return result


def _artifact_ref(value: Any, *, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise CustomCallbackContractError("artifact finalize returned a non-object")
    uri = value.get("path_or_url") or value.get("uri") or value.get("artifact_uri")
    digest = value.get("sha256") or value.get("digest")
    if not isinstance(uri, str) or not uri.strip() or len(uri) > 2_000:
        raise CustomCallbackContractError("artifact finalize returned no bounded URI")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CustomCallbackContractError("artifact finalize returned an invalid sha256")
    if digest != expected_sha256:
        raise CustomCallbackContractError("artifact finalize hash does not match the uploaded file")
    size = value.get("byte_size", value.get("bytes_count", expected_size))
    if not isinstance(size, int) or size != expected_size:
        raise CustomCallbackContractError("artifact finalize size does not match the uploaded file")
    result = {
        "path_or_url": uri[:2_000],
        "sha256": digest,
        "byte_size": size,
    }
    for key in ("artifact_id", "media_type", "role", "expires_at"):
        if isinstance(value.get(key), (str, int, float, bool)):
            result[key] = value[key]
    return result


def _unknown_finalize(exc: BaseException) -> bool:
    return "unknown" in exc.__class__.__name__.lower() or "outcome_unknown" in str(exc).lower()


class _MiniMaxArtifactCallback:
    def __init__(self, policy: MiniMaxCustomCallbackPolicy, port: ArtifactCapabilityPort):
        if not isinstance(port, ArtifactCapabilityPort):
            # Runtime protocol checks only verify method names and keep the
            # failure at bootstrap, before any workflow can be admitted.
            raise TypeError("artifact port does not implement the reviewed capability contract")
        self._policy = policy
        self._port = port

    def __call__(self, stage: dict, state: PipelineRunState, idempotency_key: str) -> Mapping[str, Any]:
        try:
            return asyncio.run(self._execute(stage, state, idempotency_key))
        except CustomCallbackContractError as exc:
            return {"status": "failed", "error": f"artifact_delivery_contract: {str(exc)[:500]}"}
        except Exception as exc:
            return {"status": "failed", "error": f"artifact_delivery_unavailable:{exc.__class__.__name__}"}

    async def _execute(self, stage: dict, state: PipelineRunState, idempotency_key: str) -> Mapping[str, Any]:
        values = _resolved_inputs(stage)
        path, size, digest = _validated_audio_file(values.get("original_audio"), self._policy)
        qc_report = values.get("qc_report")
        if not isinstance(qc_report, Mapping):
            raise CustomCallbackContractError("qc_report must be an object")
        if not bool(qc_report.get("semantic_pass", qc_report.get("qc_pass", False))):
            return {"status": "failed", "error": "artifact delivery refused because QC did not pass"}

        grant_hash = hashlib.sha256(str(idempotency_key).encode("utf-8")).hexdigest()[:32]
        grant_id = f"{self._policy.grant_prefix}-{grant_hash}"
        provider_handle = self._provider_handle(grant_id, digest)
        metadata = {
            "service": "minimax-music3-full",
            "operation": "artifact.deliver_for_review",
            "stage_id": str(stage.get("id") or "deliver"),
            "media_type": self._policy.media_type,
            "role": self._policy.role,
            "byte_size": size,
            "sha256": digest,
            "qc_sha256": hashlib.sha256(_safe_json(qc_report).encode("utf-8")).hexdigest(),
        }
        try:
            reservation = _reservation(
                await _await_call(self._port.reserve, grant_id=grant_id, grant_metadata=metadata),
                grant_id,
            )
        except Exception:
            # A previous attempt may have reserved/uploaded the same grant
            # before the coordinator received its receipt. Reconcile the
            # deterministic handle instead of issuing a second upload.
            recovered = await self._reconcile_existing(
                provider_handle, expected_size=size, expected_sha256=digest
            )
            if recovered is not None:
                return recovered
            raise
        try:
            await _await_call(
                self._port.upload,
                reservation=reservation,
                local_path=str(path),
                byte_size=size,
                sha256=digest,
                media_type=self._policy.media_type,
                role=self._policy.role,
            )
            try:
                finalized = await _await_call(
                    self._port.finalize,
                    reservation=reservation,
                    byte_size=size,
                    sha256=digest,
                )
            except Exception as exc:
                if _unknown_finalize(exc):
                    return {
                        "status": "in_flight",
                        "provider_handle": provider_handle,
                        "output": {
                            "delivery_state": "finalize_unknown",
                            "grant_id": grant_id,
                            "sha256": digest,
                        },
                    }
                raise
            ref = _artifact_ref(finalized, expected_size=size, expected_sha256=digest)
            return self._completed(ref, grant_id)
        except Exception as exc:
            # A rejected/ordinary failure is compensatable. Unknown finalize
            # deliberately bypasses this block so a durable reservation can be
            # reconciled instead of being aborted after a possible commit.
            try:
                await _await_call(self._port.abort, reservation=reservation)
            except Exception:
                return {"status": "failed", "error": "artifact delivery failed and abort outcome is unknown"}
            if isinstance(exc, CustomCallbackContractError):
                raise
            return {"status": "failed", "error": f"artifact delivery failed:{exc.__class__.__name__}"}

    @staticmethod
    def _provider_handle(grant_id: str, digest: str) -> str:
        return f"artifact-finalize:{grant_id}:{digest}"

    async def _reconcile_existing(
        self,
        provider_handle: str,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> Mapping[str, Any] | None:
        try:
            result = await _await_call(
                self._port.reconcile, provider_handle=provider_handle
            )
        except Exception:
            return None
        if not isinstance(result, Mapping):
            return None
        status = str(result.get("status") or "").lower()
        if status in {"queued", "pending", "in_flight", "running"}:
            return {
                "status": "in_flight",
                "provider_handle": provider_handle,
                "output": {"delivery_state": "reconcile_pending"},
            }
        if status in {"failed", "rejected"}:
            return {
                "status": "failed",
                "error": "artifact reconciliation reported a terminal failure",
            }
        ref_value = result.get("artifact") or result.get("artifact_ref") or result
        try:
            ref = _artifact_ref(
                ref_value,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except (TypeError, ValueError, CustomCallbackContractError):
            return None
        grant_id = provider_handle.split(":", 2)[1]
        return self._completed(ref, grant_id)

    @staticmethod
    def _completed(ref: Mapping[str, Any], grant_id: str) -> Mapping[str, Any]:
        manifest = {
            "delivery_state": "finalized",
            "grant_id": grant_id,
            "artifact": dict(ref),
        }
        return {
            "status": "completed",
            "output": {"delivery_manifest": manifest},
            "artifact": {
                "path_or_url": ref["path_or_url"],
                "sha256": ref["sha256"],
            },
        }

    def poll(self, _stage: dict, _state: PipelineRunState, _idempotency_key: str, provider_handle: str) -> Mapping[str, Any]:
        return asyncio.run(self._poll(provider_handle))

    async def _poll(self, provider_handle: str) -> Mapping[str, Any]:
        if not isinstance(provider_handle, str) or not re.fullmatch(r"artifact-finalize:CG-[A-Za-z0-9_-]{1,64}-[0-9a-f]{32}:[0-9a-f]{64}", provider_handle):
            return {"status": "failed", "error": "artifact provider handle is invalid"}
        try:
            result = await _await_call(self._port.reconcile, provider_handle=provider_handle)
        except Exception as exc:
            return {"status": "in_flight", "provider_handle": provider_handle, "error": f"artifact reconciliation pending:{exc.__class__.__name__}"}
        if not isinstance(result, Mapping):
            return {"status": "failed", "error": "artifact reconciliation returned a non-object"}
        status = str(result.get("status") or "").lower()
        if status in {"queued", "pending", "in_flight", "running"}:
            return {"status": "in_flight", "provider_handle": provider_handle}
        if status in {"failed", "rejected"}:
            return {"status": "failed", "error": str(result.get("error") or "artifact finalize failed")[:500]}
        ref_value = result.get("artifact") or result.get("artifact_ref") or result
        try:
            if not isinstance(ref_value, Mapping):
                raise CustomCallbackContractError("artifact reconciliation has no artifact object")
            expected_size = int(result.get("byte_size") or ref_value.get("byte_size"))
            expected_sha256 = str(result.get("sha256") or ref_value.get("sha256"))
            ref = _artifact_ref(ref_value, expected_size=expected_size, expected_sha256=expected_sha256)
        except (TypeError, ValueError, CustomCallbackContractError) as exc:
            return {"status": "failed", "error": f"artifact reconciliation invalid: {str(exc)[:500]}"}
        grant_id = provider_handle.split(":", 2)[1]
        return self._completed(ref, grant_id)


def _catalog_bindings(
    config: Mapping[str, Any],
    operation_catalogs: Mapping[Any, Any] | None,
) -> dict[str, list[tuple[str, str]]]:
    catalogs = operation_catalogs if operation_catalogs is not None else config.get("operation_catalogs", {})
    result: dict[str, list[tuple[str, str]]] = {}
    if not isinstance(catalogs, Mapping):
        return result
    for catalog in catalogs.values():
        if not isinstance(catalog, Mapping) or not isinstance(catalog.get("operations"), Mapping):
            continue
        for operation_id, operation in catalog["operations"].items():
            if not isinstance(operation, Mapping):
                continue
            adapter = operation.get("adapter")
            refs = operation.get("provider_refs")
            if not isinstance(adapter, str) or not isinstance(refs, list):
                continue
            for provider in refs:
                result.setdefault(str(provider), []).append((str(adapter), str(operation_id)))
    return result


def build_minimax_custom_callbacks(
    config: Mapping[str, Any],
    *,
    operation_catalogs: Mapping[Any, Any] | None = None,
    policy: MiniMaxCustomCallbackPolicy | None = None,
    qc_runner: AudioQCRunner | None = None,
    artifact_port: ArtifactCapabilityPort | None = None,
) -> tuple[
    dict[tuple[str, str], Callable[..., Mapping[str, Any]]],
    dict[tuple[str, str], Callable[..., Mapping[str, Any]]],
    tuple[tuple[str, str, str], ...],
]:
    """Build only qualified custom callbacks and report the rest as unsupported.

    The returned tuple is ``(execute_handlers, poll_handlers, unsupported)``.
    ``unsupported`` deliberately keeps the existing bridge identity shape so
    bootstrap diagnostics can show which custom binding still needs a private
    overlay without exposing paths, endpoints, or credentials.
    """

    providers = config.get("pipeline_providers", {})
    if not isinstance(providers, Mapping):
        raise CustomCallbackContractError("pipeline_providers must be an object")
    bindings = _catalog_bindings(config, operation_catalogs)
    handlers: dict[tuple[str, str], Callable[..., Mapping[str, Any]]] = {}
    poll_handlers: dict[tuple[str, str], Callable[..., Mapping[str, Any]]] = {}
    unsupported: list[tuple[str, str, str]] = []
    for provider, provider_cfg in providers.items():
        if not isinstance(provider_cfg, Mapping) or provider_cfg.get("managed") != "external":
            continue
        adapter = str(provider_cfg.get("adapter") or "").strip()
        if adapter not in {"local_audio_qc", "artifact_store"}:
            continue
        operations = bindings.get(str(provider), [])
        if not operations:
            unsupported.append((str(provider), adapter, adapter))
            continue
        callback_keys = [adapter]
        callback_keys.extend(operation_adapter for operation_adapter, _ in operations)
        callback_keys = list(dict.fromkeys(callback_keys))
        callback: Callable[..., Mapping[str, Any]] | None = None
        if policy is not None and adapter == "local_audio_qc" and callable(qc_runner):
            callback = _MiniMaxAudioQCCallback(policy, qc_runner)
        elif policy is not None and adapter == "artifact_store" and artifact_port is not None:
            callback = _MiniMaxArtifactCallback(policy, artifact_port)
        for operation_adapter in callback_keys:
            key = (str(provider), operation_adapter)
            if callback is None:
                unsupported.append((str(provider), operation_adapter, adapter))
                continue
            handlers[key] = callback
            if adapter == "artifact_store":
                poll_handlers[key] = callback.poll  # type: ignore[attr-defined]
    return handlers, poll_handlers, tuple(sorted(set(unsupported)))


__all__ = [
    "ArtifactCapabilityPort",
    "AudioQCRunner",
    "CustomCallbackContractError",
    "CustomCallbackUnavailable",
    "MiniMaxCustomCallbackPolicy",
    "build_minimax_custom_callbacks",
]
