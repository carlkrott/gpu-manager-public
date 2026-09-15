"""Production callbacks for the hash-pinned MiniMax Music 3 workflow.

The coordinator owns stage progression.  This adapter imports the reviewed
September 2026 sources and exposes their individual preparation, research and
QC functions; it never invokes the recovered all-in-one CLI.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from pipeline_coordinator import PipelineRunState, QCVerdict, _now_iso

def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


SOURCE_ROOT = _optional_path("GPU_MANAGER_MINIMAX_SOURCE_ROOT")
RUN_ROOT = Path(os.environ.get("GPU_MANAGER_MINIMAX_COORDINATOR_ROOT", "state/coordinator"))
SEAM_ROOT = _optional_path("GPU_MANAGER_MINIMAX_AUDIO_CPP_ROOT")

_LOCKS = {
    "hermes/music-generate/minimax.py": "9a95c8db489e357a32281f7c77e60e999540de07fa381e29b4461cc10fc96c7d",
    "hermes/music-generate/workflow.py": "0815318bc3b8206ef1f289c8489d44476aed97118ed540ec16961bab09092eaa",
    "hermes/music-generate/local_audio_qc.py": "b8d10cdb5838520f0ead0698136edf22e17808b33bd8c5218ba07c3a12f0f6bd",
    "qualification/run_minimax_input.py": "acef78c96df2f4686a5772e0ba0b8201abe9b888d63cec8a9e166d4f3ba89ce1",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources():
    if SOURCE_ROOT is None or SEAM_ROOT is None:
        raise RuntimeError("MINIMAX_SOURCE_ROOT_REQUIRED")
    for relative, expected in _LOCKS.items():
        path = SOURCE_ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"MINIMAX_SOURCE_LOCK_MISMATCH:{relative}")
    root = str(SOURCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    minimax = importlib.import_module("hermes.music-generate.minimax")
    contract = importlib.import_module("minimax_music3_preparation_contract")
    if not getattr(minimax, "_gpu_manager_contract_installed", False):
        contract.install_contract(minimax)
        minimax._gpu_manager_contract_installed = True
    seam_path = str(SEAM_ROOT / "runner")
    if seam_path not in sys.path:
        sys.path.insert(0, seam_path)
    seam = importlib.import_module("minimax_music3_seam_gate")
    return minimax, seam


def _inputs(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    value = stage.get("resolved_inputs")
    if not isinstance(value, Mapping):
        raise RuntimeError("MINIMAX_STAGE_INPUTS_MISSING")
    return value


def _run_dir(state: PipelineRunState, stage_id: str) -> Path:
    safe = hashlib.sha256(state.run_id.encode()).hexdigest()[:24]
    path = RUN_ROOT / safe / stage_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _completed(output: Mapping[str, Any], *, verdict: QCVerdict | None = None) -> dict[str, Any]:
    return {"status": "completed", "output": dict(output), "qc_verdict": verdict}


def build_recovered_minimax_callbacks(config: Mapping[str, Any], **_kwargs):
    """Return exact reviewed callbacks for non-leaf MiniMax operations."""
    if not {"research-http", "audio-analysis"}.intersection(config.get("pipeline_providers") or {}):
        # Text-only deployments use the same coordinator without installing
        # MiniMax's private model/source overlay.
        return {}, {}, ()
    minimax, seam = _sources()
    research_config = (config.get("pipeline_providers") or {}).get("research-http") or {}
    search_endpoint = str(research_config.get("endpoint") or os.environ.get(
        "GPU_MANAGER_SEARXNG_URL", "http://127.0.0.1:8888"
    )).rstrip("/")
    # This is a controller-owned integration. Never ask Hermes to discover
    # its environment or profile credentials for an ordinary workflow run.
    minimax.workflow._searxng_url = lambda: search_endpoint
    # Extend the recovered resolver's bounded live-page fallback to the
    # optional lanes the reviewed planner permits. These are fetched live;
    # neither their descriptions nor search snippets qualify as page evidence.
    minimax._CANONICAL_LIVE_LANE_PAGES.update({
        "instrument": ("Musical instrument", "Musical instruments, instrumentation and performance."),
        "recording": ("Sound recording and reproduction", "Audio recording, production and sound reproduction."),
        "effects": ("Audio signal processing", "Audio effects, reverb, delay and signal processing."),
    })
    minimax._LANE_RELEVANCE_TERMS.update({
        "recording": frozenset({"recording", "audio", "sound", "production"}),
        "effects": frozenset({"audio", "effect", "reverb", "delay", "processing"}),
    })

    # The active Combined Gemma route emits private reasoning tokens alongside
    # the requested JSON.  The recovered September workflow predates that
    # transport behaviour and its 4096-token ceiling can truncate the actual
    # JSON after a valid reasoning pass. Keep the reviewed prompts/validators
    # unchanged while enforcing enough response budget for both channels.
    if not getattr(minimax.workflow, "_gpu_manager_token_budget_installed", False):
        original_combined_json = minimax.workflow._combined_json

        def combined_json_with_output_room(stage, system, payload, max_tokens=3072):
            identity = hashlib.sha256(json.dumps(
                {"stage": stage, "system": system, "payload": payload},
                sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest()
            evidence_path = RUN_ROOT / "preparation-evidence" / f"{identity}.json"
            # An exact, validated expansion can be reused when repairing a
            # later stage of the same brief. Research is always executed live.
            if stage in {"brief_expand", "reference_plan"} and evidence_path.is_file():
                prior = json.loads(evidence_path.read_text())["response"]
                from minimax_music3_preparation_contract import validate_expanded_brief, validate_query_plan
                validator = validate_expanded_brief if stage == "brief_expand" else validate_query_plan
                if validator(prior):
                    return prior
            result = original_combined_json(
                stage, system, payload, max_tokens=max(int(max_tokens), 8192)
            )
            # Preserve the actual structured response so a rejected schema can
            # be repaired from evidence without another blind generation run.
            minimax.workflow._atomic_json(
                evidence_path,
                {"stage": stage, "request": payload, "response": result},
            )
            return result

        minimax.workflow._combined_json = combined_json_with_output_room
        minimax.workflow._gpu_manager_token_budget_installed = True

    def expand(stage, state, _key):
        values = _inputs(stage)
        brief = str(values.get("brief") or state.request_params.get("brief") or "")
        duration = values.get("duration_seconds") or state.request_params.get("duration_seconds") or 60
        return _completed({"expanded_brief": minimax.workflow._expand_brief(brief, duration, target="minimax")})

    def plan(stage, state, _key):
        values = _inputs(stage)
        brief = str(values.get("brief") or state.request_params.get("brief") or "")
        return _completed({"query_plan": minimax.workflow._plan_references(brief, dict(values["expanded_brief"]), target="minimax")})

    def resolve(stage, state, _key):
        values = _inputs(stage)
        enabled = str(values.get("research_policy") or "required_live") != "no_web"
        plan_value = json.loads(json.dumps(values["query_plan"]))
        aliases = {"key_harmony": "key", "instrument_direction": "instrument",
                   "mixing_levels": "mixing", "mastering_loudness": "mastering"}
        for query in plan_value.get("queries", []):
            component = str(query.get("component") or "").lower()
            if component in aliases:
                query["original_component"] = component
                query["component"] = aliases[component]
        manifest = minimax._resolve_text_references(plan_value, _run_dir(state, "research"), enable_web=enabled)
        manifest_path = _run_dir(state, "research") / "manifest.json"
        minimax.workflow._atomic_json(manifest_path, manifest)
        if enabled and manifest.get("research_status") != "success":
            return {"status": "failed", "error": f"MINIMAX_RESEARCH_{str(manifest.get('research_status')).upper()}",
                    "output": {"research_manifest": manifest, "manifest_path": str(manifest_path)}}
        return _completed({"research_manifest": manifest})

    def validate(stage, _state, _key):
        manifest = dict(_inputs(stage)["research_manifest"])
        if manifest.get("research_status") not in {"success", "disabled"}:
            return {"status": "failed", "error": "MINIMAX_RESEARCH_VALIDATION_FAILED"}
        if manifest.get("errors"):
            return {"status": "failed", "error": "MINIMAX_RESEARCH_HAS_ERRORS"}
        return _completed({"validated_research": manifest})

    def finalize(stage, state, _key):
        values = _inputs(stage)
        brief = str(values.get("brief") or state.request_params.get("brief") or "")
        duration = state.request_params.get("duration_seconds") or 60
        # The validated manifest retains its query plan for audit when the
        # resolver emits it; otherwise use the completed planning output.
        attempts = state.stage_attempts.get("plan-research") or []
        plan_value = (attempts[-1].output or {}).get("query_plan", {}) if attempts else {}
        result = minimax._finalize_minimax_brief(
            brief, dict(values["expanded_brief"]), dict(plan_value),
            dict(values["validated_research"]), duration,
            values.get("lyrics") if values.get("lyrics") is not None else state.request_params.get("lyrics"),
        )
        native = {"original_prompt": brief, "caption": result["caption"], "lyrics": result["lyrics"]}
        return _completed({"native_brief": native, "caption": result["caption"], "lyrics": result["lyrics"]})

    def seam_check(stage, state, _key):
        values = _inputs(stage)
        artifact = values["original_audio"]
        path = Path(artifact.get("path_or_url") or artifact.get("path")) if isinstance(artifact, Mapping) else Path(str(artifact))
        expected = artifact.get("sha256") if isinstance(artifact, Mapping) else None
        report = seam.analyze(path, expected_sha256=expected)
        report_path = _run_dir(state, "seam") / "report.json"
        minimax.workflow._atomic_json(report_path, report)
        if not report.get("may_enter_qc"):
            return {"status": "failed", "error": "MINIMAX_SEAM_GATE_BLOCKED", "output": {"integrity_report": report}}
        return _completed({"integrity_report": report, "report_path": str(report_path)})

    def musical_qc(stage, state, _key):
        values = _inputs(stage)
        artifact = values["original_audio"]
        path = Path(artifact.get("path_or_url") or artifact.get("path")) if isinstance(artifact, Mapping) else Path(str(artifact))
        native = dict(values["native_brief"])
        raw = minimax.local_audio_qc.analyze_audio(
            audio_path=path,
            original_prompt=native.get("original_prompt"),
            caption=str(native.get("caption") or ""),
            lyrics=str(native.get("lyrics") or "[Instrumental]"),
        )
        report_path = _run_dir(state, "musical-qc") / "report.json"
        minimax.workflow._atomic_json(report_path, raw)
        passed = bool((raw.get("verdict") or {}).get("semantic_pass"))
        report = json.dumps(raw, sort_keys=True, default=str)[:10000]
        verdict = QCVerdict(str(stage.get("id") or "musical-qc"), passed, report, report, _now_iso())
        return _completed({"qc_report": raw, "qc_pass": passed, "report": report,
                           "report_path": str(report_path)}, verdict=verdict)

    def deliver(stage, state, _key):
        values = _inputs(stage)
        artifact = values["original_audio"]
        source = Path(artifact.get("path_or_url") or artifact.get("path")) if isinstance(artifact, Mapping) else Path(str(artifact))
        if not source.is_file():
            return {"status": "failed", "error": "MINIMAX_DELIVERY_SOURCE_MISSING"}
        destination_dir = _run_dir(state, "delivery")
        destination = destination_dir / f"{state.parent_job_id}{source.suffix.lower()}"
        descriptor, temporary = tempfile.mkstemp(prefix=".delivery-", dir=destination_dir)
        os.close(descriptor)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        digest = _sha(destination)
        receipt = _completed({"delivery_manifest": {
            "status": "delivered", "path_or_url": str(destination),
            "sha256": digest, "byte_size": destination.stat().st_size,
            "delivered_at": _now_iso(), "destination_kind": "controller-artifact-store",
        }, "artifact": {"path_or_url": str(destination), "sha256": digest}})
        receipt["artifact"] = {"path_or_url": str(destination), "sha256": digest}
        return receipt

    handlers = {
        ("combined-gemma", "llm"): expand,
        ("combined-gemma", "minimax.preparation"): finalize,
        ("research-http", "research.http"): resolve,
        ("research-validator", "research.validator"): validate,
        ("audio-analysis", "audio.analysis"): seam_check,
        ("local-audio-qc", "local-audio-qc"): musical_qc,
        ("artifact-store", "artifact-store"): deliver,
    }
    # research.plan shares the llm adapter key; dispatch by operation.
    def llm(stage, state, key):
        return plan(stage, state, key) if stage.get("operation") == "research.plan" else expand(stage, state, key)
    handlers[("combined-gemma", "llm")] = llm
    return handlers, {}, ()


__all__ = ["build_recovered_minimax_callbacks"]
