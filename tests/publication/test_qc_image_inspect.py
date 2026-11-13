"""Fail closed when visual QC has no image to inspect."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_provider_runtime import (  # noqa: E402
    ProviderRuntimeError,
    _OpenAICompatibleTransport,
)
from stage_dispatch import StageContext  # noqa: E402

LIVE_ARTIFACT = Path(
    "/mnt/agent-storage-b/ai-dev/minimax-music3/production/comfyui-output/"
    "gm-6f4dff9a5b61-image-0.png"
)
LIVE_FAILURE = "QC unavailable: image could not be inspected."


async def _capture_request(artifacts, *, stage_kind="qc", max_inline_image_bytes=None):
    calls = []

    async def inner(headers, method, url, body, timeout):
        calls.append((headers, method, url, body, timeout))
        return 200, json.dumps({"choices": [{"message": {"content": '{"qc_pass": true, "report": "ok"}'}}]}).encode(), {}

    transport = _OpenAICompatibleTransport(
        inner, model="gemma-4", max_inline_image_bytes=max_inline_image_bytes
    )
    await transport.request(
        "POST",
        "http://provider.invalid/v1/chat/completions",
        json={
            "stage": {"id": "gemma-qc", "kind": stage_kind, "params": {"vision_model": "gemma-4"}},
            "context": {"original_prompt": "a test image", "artifacts": artifacts},
        },
    )
    return json.loads(calls[0][3]), calls


@pytest.mark.parametrize("retry_count", [0, 1], ids=["first-attempt", "generate-retry-2"])
def test_qc_request_includes_decodable_image_for_live_artifact(retry_count):
    if not LIVE_ARTIFACT.is_file():
        pytest.skip("live artifact is installation data, not part of the public export")
    encoded = base64.b64encode(LIVE_ARTIFACT.read_bytes()).decode("ascii")
    produced = {
        "images": [{"filename": LIVE_ARTIFACT.name, "subfolder": "", "type": "output", "data_b64": encoded}],
        "artifact_count": 1,
        "evidence_source": "live_comfyui",
    }
    ctx = StageContext(original_prompt="a test image")
    if retry_count:
        ctx = ctx.with_retry("krea-generate", 1)
    ctx = ctx.with_artifact("krea-generate", produced)
    assert ctx.retry_count.get("krea-generate", 0) == retry_count
    body, calls = asyncio.run(
        _capture_request(ctx.artifacts, max_inline_image_bytes=524288)
    )

    assert len(calls) == 1
    content = body["messages"][1]["content"]
    image = next(part for part in content if part.get("type") == "image_url")
    data_url = image["image_url"]["url"]
    mime, encoded = data_url.split(";base64,", 1)
    image_bytes = base64.b64decode(encoded, validate=True)
    assert mime == "data:image/jpeg"
    assert 0 < len(image_bytes) <= 524288
    import io
    with Image.open(io.BytesIO(image_bytes)) as preview:
        preview.load()
        assert preview.width > 0 and preview.height > 0
        assert preview.getpixel((0, 0)) is not None


def test_prompt_enhance_remains_text_only():
    body, calls = asyncio.run(_capture_request({}, stage_kind="prompt_enhance"))
    assert len(calls) == 1
    assert isinstance(body["messages"][1]["content"], str)
    assert "Preparation requirements:" in body["messages"][1]["content"]


@pytest.mark.parametrize("encoded", ["%%%not-base64%%%", ""], ids=["malformed", "empty"])
def test_malformed_inline_image_is_refused_before_transport(encoded):
    calls = []

    async def inner(*args):
        calls.append(args)
        return 200, b"{}", {}

    transport = _OpenAICompatibleTransport(inner, model="gemma-4")
    artifacts = {
        "krea-generate": {
            "images": [{"filename": "image.png", "data_b64": encoded}]
        }
    }
    with pytest.raises(ProviderRuntimeError, match="QC inline image"):
        asyncio.run(transport.request(
            "POST",
            "http://provider.invalid/v1/chat/completions",
            json={
                "stage": {"id": "gemma-qc", "kind": "qc", "params": {}},
                "context": {"original_prompt": "a test image", "artifacts": artifacts},
            },
        ))
    assert calls == []


@pytest.mark.parametrize("artifact_ref", [None, "not-an-image.txt"], ids=["missing", "unusable"])
def test_missing_or_unusable_qc_image_raises_before_transport_with_live_failure_message(
    tmp_path, artifact_ref
):
    calls = []

    async def inner(*args):
        calls.append(args)
        return 200, b"{}", {}

    artifacts = {}
    expected_error = LIVE_FAILURE
    if artifact_ref is not None:
        unusable = tmp_path / artifact_ref
        unusable.write_text("not an image")
        artifacts = {"krea-generate": {"artifact_path": str(unusable)}}
        expected_error = "QC artifact is not an image"

    transport = _OpenAICompatibleTransport(inner, model="gemma-4")
    with pytest.raises(ProviderRuntimeError, match=expected_error):
        asyncio.run(transport.request(
            "POST",
            "http://provider.invalid/v1/chat/completions",
            json={
                "stage": {"id": "gemma-qc", "kind": "qc", "params": {}},
                "context": {"original_prompt": "a test image", "artifacts": artifacts},
            },
        ))
    assert calls == []
