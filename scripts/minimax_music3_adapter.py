#!/usr/bin/env python3
"""Checked adapter for the recovered MiniMax Music 3 audio.cpp workflow.

The GPU Manager worker calls :func:`run_pipeline` after claiming a durable
orchestrator queue entry.  The actual workflow runs in an isolated child
process: preparation/research produces a frozen brief, then the accepted
audio.cpp Q8 language-model/BF16 DiT route is invoked from that frozen input.
The recovered sources are hash checked before either stage runs. The
controller owns the audio.cpp leaf admission; this adapter never edits runtime
configuration.

No user-supplied command, source path, environment override, or output path is
accepted by the request contract.  Those are deployment-owned settings.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from minimax_music3_preparation_contract import install_contract


ADAPTER_ID = "minimax-music3-pinned-cli"
SOURCE_ROOT_ENV = "GPU_MANAGER_MINIMAX_MUSIC3_SOURCE_ROOT"
EVIDENCE_ROOT_ENV = "GPU_MANAGER_MINIMAX_MUSIC3_EVIDENCE_ROOT"
AUDIO_CPP_RUNNER_ENV = "GPU_MANAGER_MINIMAX_AUDIO_CPP_RUNNER"
DEFAULT_SOURCE_LOCK = (
    Path(__file__).resolve().parents[1]
    / "config/workflows/minimax-music3-source-lock.v1.json"
)
RUNNER_REFERENCE = "qualification/run_minimax_input.py"
REQUIRED_SOURCE_NAMES = {
    "preparation",
    "shared_workflow_support",
    "musical_qc",
    "production_entrypoint",
    "production_generation_runner",
}
_MAX_WORKER_RESULT_BYTES = 4 * 1024 * 1024
_MAX_WORKER_DIAGNOSTIC_BYTES = 64 * 1024


class MiniMaxAdapterError(RuntimeError):
    """The checked workflow could not be safely started or completed."""


@dataclass(frozen=True, slots=True)
class MiniMaxAdapterConfig:
    source_root: Path
    evidence_root: Path
    source_lock: Path = DEFAULT_SOURCE_LOCK
    timeout_seconds: int = 7200
    audio_cpp_runner: Path | None = None

    @classmethod
    def from_environment(cls) -> "MiniMaxAdapterConfig":
        source = os.environ.get(SOURCE_ROOT_ENV, "").strip()
        evidence = os.environ.get(EVIDENCE_ROOT_ENV, "").strip()
        audio_cpp_runner = os.environ.get(AUDIO_CPP_RUNNER_ENV, "").strip()
        if not source or not evidence or not audio_cpp_runner:
            missing = [
                name
                for name, value in (
                    (SOURCE_ROOT_ENV, source),
                    (EVIDENCE_ROOT_ENV, evidence),
                    (AUDIO_CPP_RUNNER_ENV, audio_cpp_runner),
                )
                if not value
            ]
            raise MiniMaxAdapterError(
                "MiniMax Music 3 deployment overlay is incomplete: "
                + ", ".join(missing)
            )
        return cls(
            source_root=Path(source),
            evidence_root=Path(evidence),
            audio_cpp_runner=Path(audio_cpp_runner),
        )


def _resolve_audio_cpp_runner(config: MiniMaxAdapterConfig) -> Path:
    """Resolve the deployment-owned audio.cpp runner, never from job input."""
    value = config.audio_cpp_runner
    if value is None:
        configured = os.environ.get(AUDIO_CPP_RUNNER_ENV, "").strip()
        value = Path(configured) if configured else None
    if value is None:
        raise MiniMaxAdapterError(
            f"{AUDIO_CPP_RUNNER_ENV} is required for the approved audio.cpp route"
        )
    runner = Path(value).expanduser().resolve()
    if not runner.is_file():
        raise MiniMaxAdapterError(f"audio.cpp runner is missing: {runner}")
    return runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise MiniMaxAdapterError("source lock contains an empty reference")
    normalized = reference.strip().replace("\\", "/")
    if normalized.startswith(("/", "~")) or "://" in normalized:
        raise MiniMaxAdapterError(f"source reference must be relative: {reference!r}")
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise MiniMaxAdapterError(
            f"source reference escapes the pinned source root: {reference!r}"
        ) from exc
    return candidate


def verify_source_lock(config: MiniMaxAdapterConfig) -> dict[str, Any]:
    """Verify the complete production path against the reviewed source lock."""
    try:
        lock = json.loads(config.source_lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MiniMaxAdapterError(f"cannot load source lock: {exc}") from exc
    if lock.get("schema_version") != "workflow-source-lock.v1":
        raise MiniMaxAdapterError("unsupported MiniMax source-lock schema")
    sources = lock.get("sources")
    if not isinstance(sources, Mapping):
        raise MiniMaxAdapterError("MiniMax source lock has no sources map")
    missing = sorted(REQUIRED_SOURCE_NAMES - set(sources))
    if missing:
        raise MiniMaxAdapterError(
            "MiniMax source lock is incomplete: " + ", ".join(missing)
        )

    verified: dict[str, dict[str, str]] = {}
    for name in sorted(REQUIRED_SOURCE_NAMES):
        record = sources[name]
        if not isinstance(record, Mapping):
            raise MiniMaxAdapterError(f"invalid source-lock record: {name}")
        reference = record.get("reference")
        expected = record.get("sha256")
        path = _source_path(config.source_root, str(reference or ""))
        if not path.is_file():
            raise MiniMaxAdapterError(f"pinned MiniMax source is missing: {reference}")
        actual = _sha256(path)
        if actual != expected:
            raise MiniMaxAdapterError(
                f"pinned MiniMax source hash mismatch: {reference}"
            )
        verified[name] = {"reference": str(reference), "sha256": actual}

    local_contract = lock.get("local_preparation_contract")
    if not isinstance(local_contract, Mapping):
        raise MiniMaxAdapterError("MiniMax source lock has no local preparation contract")
    local_reference = local_contract.get("reference")
    local_expected = local_contract.get("sha256")
    local_path = Path(__file__).resolve().with_name(
        Path(str(local_reference or "")).name
    )
    if (
        local_reference != "scripts/minimax_music3_preparation_contract.py"
        or not local_path.is_file()
        or _sha256(local_path) != local_expected
    ):
        raise MiniMaxAdapterError("local MiniMax preparation contract hash mismatch")

    runner = _source_path(config.source_root, RUNNER_REFERENCE)
    return {
        "adapter": ADAPTER_ID,
        "status": "verified",
        "source_root": str(config.source_root.resolve()),
        "source_lock": str(config.source_lock.resolve()),
        "source_lock_sha256": _sha256(config.source_lock),
        "runner": str(runner),
        "sources": verified,
        "local_preparation_contract": {
            "reference": local_reference,
            "sha256": local_expected,
            "contract": local_contract.get("contract"),
        },
    }


def preflight(config: MiniMaxAdapterConfig, *, probe_cli: bool = True) -> dict[str, Any]:
    """Run the source-integrity and non-generating CLI availability checks."""
    result = verify_source_lock(config)
    evidence_root = config.evidence_root.resolve()
    evidence_parent = evidence_root if evidence_root.exists() else evidence_root.parent
    if not evidence_parent.is_dir():
        raise MiniMaxAdapterError(
            f"MiniMax evidence parent does not exist: {evidence_parent}"
        )
    if not os.access(evidence_parent, os.W_OK):
        raise MiniMaxAdapterError(
            f"MiniMax evidence parent is not writable: {evidence_parent}"
        )
    result["evidence_root"] = str(evidence_root)
    result["evidence_parent_writable"] = True

    if probe_cli:
        audio_cpp_runner = _resolve_audio_cpp_runner(config)
        runner_probe = subprocess.run(
            [sys.executable, str(audio_cpp_runner), "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if runner_probe.returncode != 0:
            raise MiniMaxAdapterError(
                "audio.cpp runner CLI probe failed: "
                + runner_probe.stderr[-500:]
            )
        for label, reference in (
            ("production_entrypoint", RUNNER_REFERENCE),
            ("production_generation_runner", RUNNER_REFERENCE),
        ):
            path = _source_path(config.source_root, reference)
            completed = subprocess.run(
                [sys.executable, str(path), "--help"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                raise MiniMaxAdapterError(
                    f"{label} CLI probe failed: {completed.stderr[-500:]}"
                )
        result["cli_probe"] = "passed"
        result["audio_cpp_runner"] = str(audio_cpp_runner)
    return result


def validate_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "brief",
        "prompt",
        "lyrics",
        "duration_seconds",
        "research_policy",
        "seed",
        "diffusion_seed",
    }
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        raise MiniMaxAdapterError(
            "unsupported MiniMax request fields: " + ", ".join(unsupported)
        )
    brief = payload.get("brief")
    prompt = payload.get("prompt")
    if brief and prompt and str(brief) != str(prompt):
        raise MiniMaxAdapterError("brief and prompt disagree; provide only one")
    prompt = brief or prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise MiniMaxAdapterError("brief must be a non-empty string")
    if len(prompt) > 12000:
        raise MiniMaxAdapterError("brief exceeds 12000 characters")

    lyrics = payload.get("lyrics")
    if lyrics is not None and (not isinstance(lyrics, str) or len(lyrics) > 16000):
        raise MiniMaxAdapterError("lyrics must be a string of at most 16000 characters")
    duration = payload.get("duration_seconds", 60)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise MiniMaxAdapterError("duration_seconds must be numeric")
    if not 10 <= float(duration) <= 60:
        raise MiniMaxAdapterError(
            "the approved audio.cpp MiniMax route supports 10-60 seconds"
        )
    research_policy = payload.get("research_policy", "required_live")
    if research_policy not in {"required_live", "no_web"}:
        raise MiniMaxAdapterError(
            "research_policy must be 'required_live' or 'no_web'"
        )
    for name in ("seed", "diffusion_seed"):
        value = payload.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise MiniMaxAdapterError(f"{name} must be an integer")
    return {
        # Preserve the workflow contract's canonical input name while keeping
        # ``prompt`` for compatibility with the recovered CLI adapter.
        "brief": prompt.strip(),
        "prompt": prompt.strip(),
        "lyrics": lyrics,
        "duration_seconds": float(duration),
        "research_policy": research_policy,
        "seed": payload.get("seed"),
        "diffusion_seed": payload.get("diffusion_seed"),
    }


def _load_production_runner(config: MiniMaxAdapterConfig):
    runner_path = _source_path(config.source_root, RUNNER_REFERENCE)
    spec = importlib.util.spec_from_file_location(
        "gpu_manager_minimax_music3_production", runner_path
    )
    if spec is None or spec.loader is None:
        raise MiniMaxAdapterError(f"cannot load MiniMax runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_run(
    config: MiniMaxAdapterConfig,
    payload: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute inside the isolated adapter child process."""
    request = validate_request(payload)
    integrity = verify_source_lock(config)
    parent_job_id = control.get("parent_job_id")
    priority = control.get("priority")
    priority_class = control.get("priority_class")
    if not isinstance(parent_job_id, str) or not parent_job_id:
        raise MiniMaxAdapterError("parent_job_id is required")
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
        raise MiniMaxAdapterError("parent priority must be an integer from 0 to 100")
    if priority_class not in {"interactive", "normal", "background"}:
        raise MiniMaxAdapterError("parent priority_class is invalid")
    audio_cpp_runner = _resolve_audio_cpp_runner(config)
    os.environ["GPU_MANAGER_MINIMAX_PARENT_JOB_ID"] = parent_job_id
    os.environ["GPU_MANAGER_MINIMAX_PARENT_PRIORITY"] = str(priority)
    os.environ["GPU_MANAGER_MINIMAX_PARENT_PRIORITY_CLASS"] = str(priority_class)
    os.environ["MINIMAX_AUDIOCPP_RUNNER"] = str(audio_cpp_runner)
    config.evidence_root.mkdir(parents=True, exist_ok=True)
    runner = _load_production_runner(config)
    minimax_source = runner._load_minimax_module()
    preparation_contract = install_contract(minimax_source)
    runner._load_minimax_module = lambda: minimax_source

    lyrics_path: Path | None = None
    try:
        if request["lyrics"] is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="minimax-lyrics-",
                suffix=".txt",
                dir=config.evidence_root,
                delete=False,
            ) as handle:
                handle.write(request["lyrics"])
                lyrics_path = Path(handle.name)

        # Some recovered preparation helpers emit progress on stdout. Keep the
        # adapter protocol stdout as one JSON document by redirecting that
        # progress to stderr inside the isolated child.
        with contextlib.redirect_stdout(sys.stderr):
            prepared = runner._prepare_only(
                request["prompt"],
                request["duration_seconds"],
                lyrics_path,
                request["seed"],
                request["diffusion_seed"],
                request["research_policy"] == "no_web",
                config.evidence_root,
                runtime="audio-cpp",
                audio_cpp_variant="bf16",
            )
            frozen = Path(prepared["frozen_brief"])
            generated = runner._execute_from_frozen(
                frozen,
                config.evidence_root,
                config.timeout_seconds,
                precision="current",
                runtime="audio-cpp",
                audio_cpp_runner=audio_cpp_runner,
                audio_cpp_variant="bf16",
            )
        generation_success = generated.get("submission_status") == "success"
        qc_summary = generated.get("qc_summary") or {}
        qc_status = str(
            qc_summary.get("classification")
            or generated.get("qc_status")
            or "unavailable"
        )
        hard_qc_failure = bool(qc_summary.get("hard_failure")) or qc_status in {
            "corrupt",
            "corruption_suspected",
        }
        completed = generation_success and not hard_qc_failure
        return {
            "adapter": ADAPTER_ID,
            "status": "completed" if completed else "failed",
            "preparation": prepared,
            "generation": generated,
            "source_lock_sha256": integrity["source_lock_sha256"],
            "preparation_contract": preparation_contract,
            "review_status": (
                "awaiting_human_review"
                if completed
                else "not_ready"
            ),
            "qc_disposition": qc_status,
            "human_review_required": bool(generation_success),
        }
    finally:
        if lyrics_path is not None:
            lyrics_path.unlink(missing_ok=True)


async def _terminate_owned_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _drain_bounded(
    stream: asyncio.StreamReader,
    *,
    limit: int,
    keep_tail: bool = False,
) -> tuple[bytes, int]:
    """Drain a child stream without retaining unbounded output in memory."""
    retained = bytearray()
    total = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if keep_tail:
            retained.extend(chunk)
            if len(retained) > limit:
                del retained[:-limit]
        elif len(retained) < limit:
            retained.extend(chunk[: limit - len(retained)])
    return bytes(retained), total


async def run_pipeline(
    payload: Mapping[str, Any],
    *,
    parent_job_id: str,
    priority: int,
    priority_class: str,
    config: MiniMaxAdapterConfig | None = None,
) -> dict[str, Any]:
    """Run one claimed workflow through the isolated pinned adapter process."""
    request = validate_request(payload)
    selected = config or MiniMaxAdapterConfig.from_environment()
    verify_source_lock(selected)
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--source-root",
        str(selected.source_root),
        "--source-lock",
        str(selected.source_lock),
        "--evidence-root",
        str(selected.evidence_root),
        "--timeout",
        str(selected.timeout_seconds),
    ]
    if selected.audio_cpp_runner is not None:
        argv.extend(["--audio-cpp-runner", str(selected.audio_cpp_runner)])
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdout_task = asyncio.create_task(
        _drain_bounded(process.stdout, limit=_MAX_WORKER_RESULT_BYTES)
    )
    stderr_task = asyncio.create_task(
        _drain_bounded(
            process.stderr,
            limit=_MAX_WORKER_DIAGNOSTIC_BYTES,
            keep_tail=True,
        )
    )
    try:
        request_bytes = json.dumps(
            {
                "request": request,
                "control": {
                    "parent_job_id": parent_job_id,
                    "priority": priority,
                    "priority_class": priority_class,
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        assert process.stdin is not None
        process.stdin.write(request_bytes)
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()
        await asyncio.wait_for(process.wait(), timeout=selected.timeout_seconds + 900)
        (stdout, stdout_total), (_stderr_tail, stderr_total) = await asyncio.gather(
            stdout_task, stderr_task
        )
    except BaseException:
        await asyncio.shield(_terminate_owned_process(process))
        for task in (stdout_task, stderr_task):
            task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise

    if stdout_total > _MAX_WORKER_RESULT_BYTES:
        raise MiniMaxAdapterError(
            "MiniMax adapter result exceeded the 4 MiB protocol limit"
        )

    try:
        result = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MiniMaxAdapterError(
            f"MiniMax adapter returned invalid JSON (exit={process.returncode})"
        ) from exc
    if not isinstance(result, Mapping):
        raise MiniMaxAdapterError("MiniMax adapter result must be an object")
    response = dict(result)
    response["_adapter_exit_code"] = process.returncode
    response["_adapter_diagnostic_bytes"] = stderr_total
    response["_adapter_diagnostics_truncated"] = (
        stderr_total > _MAX_WORKER_DIAGNOSTIC_BYTES
    )
    return response


def _config_from_args(args: argparse.Namespace) -> MiniMaxAdapterConfig:
    return MiniMaxAdapterConfig(
        source_root=args.source_root,
        evidence_root=args.evidence_root,
        source_lock=args.source_lock,
        timeout_seconds=args.timeout,
        audio_cpp_runner=args.audio_cpp_runner,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "worker"):
        command = subparsers.add_parser(name)
        command.add_argument("--source-root", required=True, type=Path)
        command.add_argument(
            "--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK
        )
        command.add_argument("--evidence-root", required=True, type=Path)
        command.add_argument("--timeout", type=int, default=7200)
        command.add_argument("--audio-cpp-runner", type=Path, default=None)
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    try:
        if args.command == "preflight":
            result = preflight(config)
        else:
            raw = sys.stdin.buffer.read(65537)
            if len(raw) > 65536:
                raise MiniMaxAdapterError("MiniMax request exceeds 64 KiB")
            envelope = json.loads(raw.decode("utf-8"))
            if not isinstance(envelope, Mapping):
                raise MiniMaxAdapterError("MiniMax request must be an object")
            payload = envelope.get("request")
            control = envelope.get("control")
            if not isinstance(payload, Mapping) or not isinstance(control, Mapping):
                raise MiniMaxAdapterError("MiniMax worker envelope is invalid")
            result = _worker_run(config, payload, control)
    except Exception as exc:
        print(
            json.dumps(
                {"adapter": ADAPTER_ID, "status": "error", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {"verified", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
