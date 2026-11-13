"""Retired controller is gone: raw prompt in, grounded local QC out.

Every case is hermetic. No retired endpoint is contacted: the outbound HTTP
session factory is replaced by a tripwire, so any attempt to reach a retired
controller fails the test instead of silently degrading.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


@pytest.fixture(scope="module")
def controller():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("retired_controller_facade", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_outbound(controller, monkeypatch):
    """Any attempt to open an HTTP session during these cases is a failure."""
    import aiohttp

    class Tripwire:
        def __init__(self, *args, **kwargs):
            raise AssertionError("retired outbound controller call attempted")

    monkeypatch.setattr(aiohttp, "ClientSession", Tripwire)
    yield


VISION_URL = "http://vision.test/v1/chat/completions"


def _workflow(prompt: str) -> dict:
    return {
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt},
        }
    }


def test_workflow_entry_preserves_raw_prompt_and_omits_enhancer(
    controller, no_outbound, monkeypatch
):
    raw_prompt = "exactly two kelp crabs on wet slate, no text, no watermark"
    template = {
        "prompt_enhance": {
            "enabled": True,
            "project_id": "production-image-qc",
            "agent": "opencode",
            "model": "minimax/MiniMax-M3",
        },
        "inject_keys": [["prompt", "6", "text"]],
        "workflow": _workflow(raw_prompt),
        "endpoint": "http://127.0.0.1:8188",
        "http_path": "/prompt",
    }
    monkeypatch.setattr(controller, "_load_services_config", lambda: {})

    entry, error = asyncio.run(
        controller._build_comfyui_workflow_entry(
            service_name="comfyui",
            tmpl=template,
            params={"prompt": raw_prompt},
            job_id="job-raw-1",
            priority=50,
            callback_url="",
            metadata={},
        )
    )
    assert error is None
    payload = json.loads(base64.b64decode(entry["body_b64"]).decode("utf-8"))
    assert payload["prompt"]["6"]["inputs"]["text"] == raw_prompt

    metadata = json.loads(entry["metadata"])
    evidence = metadata["prompt_evidence"]
    assert evidence["effective_prompt"] == raw_prompt
    assert evidence["original_prompt_chars"] == len(raw_prompt)
    assert evidence["original_prompt_sha256"] == evidence["effective_prompt_sha256"]
    assert evidence["enhancer"] == {"provider": "none"}
    worker_pipeline = metadata.get("worker_pipeline") or {}
    assert "prompt_enhance" not in worker_pipeline


def test_post_claim_body_preparation_keeps_raw_prompt_and_reports_provenance(
    controller, no_outbound
):
    raw_prompt = "raw prompt preserved through claim"
    body = json.dumps({"prompt": _workflow(raw_prompt)}).encode("utf-8")
    job = {
        "metadata": json.dumps(
            {
                "worker_pipeline": {
                    "schema": "image-pipeline.v1",
                    "original_prompt": raw_prompt,
                    "prompt_target": "6.inputs.text",
                    "prompt_enhance": {
                        "enabled": True,
                        "project_id": "production-image-qc",
                    },
                }
            }
        )
    }

    prepared, evidence = asyncio.run(
        controller._prepare_comfyui_body_after_claim(job, body)
    )
    assert prepared == body
    assert evidence["effective_prompt"] == raw_prompt
    assert evidence["original_prompt_sha256"] == evidence["effective_prompt_sha256"]
    assert evidence["enhancer"] == {"provider": "none"}
    assert evidence["prompt_target"] == "6.inputs.text"


def test_post_claim_body_preparation_refuses_a_rewritten_prompt(
    controller, no_outbound
):
    """A body whose prompt target no longer carries the admitted raw prompt is
    rejected instead of being forwarded or silently re-enhanced."""
    raw_prompt = "raw prompt preserved through claim"
    workflow = _workflow(raw_prompt)
    workflow["6"]["inputs"]["text"] = "some enhancement wrote over this"
    body = json.dumps({"prompt": workflow}).encode("utf-8")
    job = {
        "metadata": json.dumps(
            {
                "worker_pipeline": {
                    "schema": "image-pipeline.v1",
                    "original_prompt": raw_prompt,
                    "prompt_target": "6.inputs.text",
                }
            }
        )
    }

    with pytest.raises(RuntimeError, match="raw prompt"):
        asyncio.run(controller._prepare_comfyui_body_after_claim(job, body))


def test_grounded_qc_without_external_critique_never_invents_a_pass(
    controller, no_outbound, monkeypatch
):
    observations = {
        "observations": ["subject count is ambiguous"],
        "atomic_checks": [
            {
                "requirement": "exactly two kelp crabs",
                "status": "uncertain",
                "evidence": "second crab partly occluded",
            }
        ],
    }

    async def fake_observations(image_bytes, *, prompt, policy):
        assert image_bytes
        return json.loads(json.dumps(observations))

    monkeypatch.setattr(
        controller, "_analyze_image_with_local_vision", fake_observations
    )

    result = asyncio.run(
        controller._run_grounded_image_qc(
            b"fake-image-bytes",
            prompt="exactly two kelp crabs",
            policy={"requirements": ["exactly two kelp crabs"], "vision_url": VISION_URL},
            image_name="krea-output.png",
        )
    )
    assert result["decision"] != "pass"
    assert result["decision"] in {"review", "reject"}
    assert result["provider"] != "open-design"
    assert result["grounded_observations"]["atomic_checks"] == (
        observations["atomic_checks"]
    )
    assert result.get("qc_status") != "unavailable"


def test_grounded_qc_local_vision_failure_is_typed_unavailable_not_pass(
    controller, no_outbound, monkeypatch
):
    async def broken_observations(image_bytes, *, prompt, policy):
        raise RuntimeError("local vision endpoint refused the connection")

    monkeypatch.setattr(
        controller, "_analyze_image_with_local_vision", broken_observations
    )

    result = asyncio.run(
        controller._run_grounded_image_qc(
            b"fake-image-bytes",
            prompt="a fox",
            policy={"requirements": ["a fox"], "vision_url": VISION_URL},
            image_name="krea-output.png",
        )
    )
    assert result["decision"] == "review"
    assert result["qc_status"] == "unavailable"
    assert result["qc_failure"] is True
    assert result["failure_reason"] == "qc_unavailable"


def test_image_critique_calls_vision_helper_and_returns_its_verdict(
    controller, no_outbound, monkeypatch
):
    observations = {
        "observations": ["a single fox is visible"],
        "atomic_checks": [
            {"requirement": "a fox", "status": "fail", "evidence": "no fox visible"}
        ],
    }
    calls = []

    async def fake_observations(image_bytes, *, prompt, policy):
        calls.append(policy.get("vision_url"))
        return json.loads(json.dumps(observations))

    monkeypatch.setattr(
        controller, "_analyze_image_with_local_vision", fake_observations
    )

    result = asyncio.run(
        controller._run_grounded_image_qc(
            b"fake-image-bytes",
            prompt="a fox",
            policy={"requirements": ["a fox"], "vision_url": VISION_URL},
            image_name="krea-output.png",
        )
    )
    assert calls == [VISION_URL]
    assert result["grounded_observations"]["atomic_checks"] == (
        observations["atomic_checks"]
    )
    assert result["decision"] != "pass"
    assert result.get("qc_status") != "unavailable"


def test_image_critique_without_vision_url_is_explicit_unavailable(
    controller, no_outbound, monkeypatch
):
    calls = []

    async def fake_observations(image_bytes, *, prompt, policy):
        calls.append(policy)
        raise AssertionError("vision helper must not run without a vision_url")

    monkeypatch.setattr(
        controller, "_analyze_image_with_local_vision", fake_observations
    )

    result = asyncio.run(
        controller._run_grounded_image_qc(
            b"fake-image-bytes",
            prompt="a fox",
            policy={"requirements": ["a fox"]},
            image_name="krea-output.png",
        )
    )
    assert calls == []
    assert result["decision"] == "review"
    assert result["qc_status"] == "unavailable"
    assert result["qc_failure"] is True
    assert result["failure_reason"] == "qc_unavailable"
    assert result["error"] == "vision_url_not_configured"


def test_grounded_qc_uncertain_observation_cannot_become_terminal_pass(
    controller,
):
    reduced = controller._reduce_visual_qc_decision(
        {
            "atomic_checks": [
                {"requirement": "text legibility", "status": "uncertain",
                 "evidence": "blurred"}
            ]
        },
        {"decision": "pass", "score": 10},
        {"p0_requirements": ["text legibility"]},
    )
    assert reduced["terminal_status"] == "review"
    assert reduced["human_review_required"] is True


def test_n8n_prompt_enhance_provider_is_preserved(controller):
    """Removing the retired controller must not drop the surviving n8n provider.

    An unsupported provider is refused outright by ``_reconcile_pipeline_evidence``
    (returns ``reconciled=False``), so deleting the whole ``prompt_enhance`` key
    would silently disable every n8n enhancement stage.
    """
    import inspect

    source = inspect.getsource(controller._reconcile_pipeline_evidence)
    assert '"prompt_enhance": {"n8n"}' in source
    assert "open-design" not in source


def test_retired_override_id_is_not_emitted_by_the_reducer(controller):
    """The reducer's pass-block override id carries no retired-controller name."""
    reduced = controller._reduce_visual_qc_decision(
        {
            "atomic_checks": [
                {"requirement": "two subjects", "status": "fail", "evidence": "blurred"}
            ]
        },
        {"decision": "pass", "score": 10},
        {"p0_requirements": ["two subjects"]},
    )
    assert reduced["terminal_status"] != "pass"
    overrides = reduced.get("overrides") or []
    assert overrides
    assert not any("od_" in str(item) for item in overrides), overrides


def test_controller_source_has_no_retired_controller_identifier():
    source = FACADE.read_text(encoding="utf-8")
    offenders = sorted(
        {
            match.group(0)
            for match in re.finditer(
                r"_OD_BASE_URL|_ensure_od_project|_od_\w+|_enhance_prompt_with_od"
                r"|_critique_visual_observations_with_od|open[-_ ]?design|17456",
                source,
                flags=re.IGNORECASE,
            )
        }
    )
    assert offenders == []