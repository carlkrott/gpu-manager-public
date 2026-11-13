"""Retired prompt enhancer is absent; image QC pipeline remains intact."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_image_templates_use_raw_prompt_and_pipeline_keeps_qc_correction():
    config = json.loads((ROOT / "config/services.json").read_text())
    for template_id in ("image", "semantic-scene-krea-candidate"):
        template = config["generation_templates"][template_id]
        assert "prompt_enhance" not in template
        assert template["param_map"]["prompt"] == "6.inputs.text"

    pipeline = config["service_pipelines"]["krea-image-qc"]
    stages = pipeline["stages"]
    assert [(stage["id"], stage["kind"], stage["provider"]) for stage in stages] == [
        ("krea-generate", "generate", "comfyui"),
        ("gemma-qc", "qc", "combined-gemma"),
        ("qwen-correct", "correct", "comfyui-qwen"),
    ]
    assert stages[0].get("depends_on", []) == []
    assert stages[1]["depends_on"] == ["krea-generate"]
    assert stages[2]["depends_on"] == ["gemma-qc"]
    assert pipeline["loops"] == [{
        "correction_stage_id": "qwen-correct",
        "id": "qc-correction-loop",
        "max_iterations": 3,
        "on_correction": ["gemma-qc"],
        "trigger_stage_id": "gemma-qc",
    }]


def test_retired_open_design_registry_and_capability_are_absent():
    services = json.loads((ROOT / "config/services.json").read_text())
    capabilities = json.loads((ROOT / "config/capabilities.json").read_text())
    assert "open_design" not in services["services"]
    assert "open-design" not in services["pipeline_providers"]
    assert "open_design" not in services.get("endpoints", {})
    assert "open-design" not in services["services"]["minimax-music3-full"]["metadata"]["optional_integrations"]
    assert "n8n" in services["services"]["minimax-music3-full"]["metadata"]["optional_integrations"]
    assert "open-design-7995x" not in json.dumps(capabilities)
    assert "n8n-amp-spot-intro-7995x" in capabilities["providers"]


def test_retired_acestep_and_sulphur_entries_are_absent():
    services = json.loads((ROOT / "config/services.json").read_text())
    capabilities = json.loads((ROOT / "config/capabilities.json").read_text())
    for name in ("acestep", "acestep-candidate", "sulphur_unet"):
        assert name not in services["services"]
    for name in ("acestep-candidate", "acestep-candidate-mi50"):
        assert name not in services["bundles"]
    for name in ("music", "music-acestep-candidate", "music-acestep-xl-candidate", "sulphur-video"):
        assert name not in services["generation_templates"]
    for name in ("acestep", "acestep-candidate", "sulphur-unet"):
        assert name not in services["pipeline_providers"]
    for name in ("acestep-candidate", "music-candidate-gen", "music-gen", "video-gen"):
        assert name not in services["routing_groups"]
    assert "acestep" not in json.dumps(capabilities)
    assert "minimax-music3-audiocpp-bf16-dit-qualification" in services["generation_templates"]
