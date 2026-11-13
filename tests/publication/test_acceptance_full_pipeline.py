"""The live acceptance gate against a local fake manager backend.

The fake speaks only the endpoints the gate uses (/v1/submit/generation,
/v1/job/{id}, /health, /vram) and is scripted per test, so each way the gate
can fail is proven to fail. No live service, GPU or Redis is contacted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import acceptance_full_pipeline as gate  # noqa: E402

TOKEN = "fake-gate-token-0123456789"
FAST = dict(job_timeout_s=1.0, release_timeout_s=0.6, poll_interval_s=0.02)


class FakeManager:
    """Scriptable manager: job outcome, bundle release and card VRAM."""

    def __init__(self):
        self.job_status = "completed"
        self.job_error_type = None
        self.job_polls_until_terminal = 2
        self.job_record_extra = {"stages": [{"name": "qwen-correct", "selected": True}]}
        self.bundle_release_after_polls = 2
        self.card_gb = {"card0": 14.3, "card1": 16.6}
        self.restore_after_polls = 2
        self.llm_evicted = False
        self.runtime_mode = "normal"
        self.contradictions: list[str] = []
        self.submitted: list[dict] = []
        self.auth_seen: list[str | None] = []
        self._job_polls = 0
        self._health_polls = 0
        self._vram_polls = 0

    def _auth(self, request):
        self.auth_seen.append(request.headers.get("Authorization"))
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            raise web.HTTPUnauthorized()

    async def submit(self, request):
        self._auth(request)
        self.submitted.append(await request.json())
        return web.json_response({"job_id": "job-gate-1", "status": "queued"}, status=202)

    async def job(self, request):
        self._auth(request)
        self._job_polls += 1
        if self._job_polls < self.job_polls_until_terminal:
            return web.json_response({"job_id": "job-gate-1", "status": "in_flight"})
        doc = {"job_id": "job-gate-1", "status": self.job_status,
               "error_type": self.job_error_type, **self.job_record_extra}
        return web.json_response(doc)

    async def health(self, request):
        self._auth(request)
        self._health_polls += 1
        loaded = [{"gpu_id": "mi50", "bundle": "aux-mi50"}]
        if self._health_polls <= self.bundle_release_after_polls:
            loaded.append({"gpu_id": "rx6950xt", "bundle": "comfyui"})
        return web.json_response({
            "loaded_runtime_bundles": loaded,
            "vram": {"llm_evicted": self.llm_evicted},
            "runtime_mode": self.runtime_mode,
            "contradictions": self.contradictions,
        })

    async def vram(self, request):
        self._auth(request)
        self._vram_polls += 1
        busy = self._vram_polls <= self.restore_after_polls
        gpus = []
        for card, gb in self.card_gb.items():
            gpus.append({"card": card, "vram_used_mib": int((22.0 if busy else gb) * 1024)})
        return web.json_response({"gpus": gpus})


@pytest_asyncio.fixture
async def backend():
    fake = FakeManager()
    app = web.Application()
    app.router.add_post("/v1/submit/generation", fake.submit)
    app.router.add_get("/v1/job/{job_id}", fake.job)
    app.router.add_get("/health", fake.health)
    app.router.add_get("/vram", fake.vram)
    server = TestServer(app)
    await server.start_server()
    try:
        yield SimpleNamespace(fake=fake, url=str(server.make_url("")).rstrip("/"))
    finally:
        await server.close()


def _by_name(result):
    return {c.name: c for c in result.checks}


@pytest.mark.asyncio
async def test_gate_passes_when_job_completes_bundle_released_and_baselines_restored(backend):
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    assert result.ok, result.report()
    assert [c.name for c in result.checks] == [
        "job_submitted", "job_completed", "qwen_correct_stage_ran",
        "generation_bundle_released", "dgpu_baselines_restored", "manager_not_degraded",
    ]
    assert result.job_id == "job-gate-1"
    # The default payload really is a qwen-correct generation submit.
    assert backend.fake.submitted[0]["type"] == "image-edit-q5-benchmark"
    assert set(backend.fake.auth_seen) == {f"Bearer {TOKEN}"}


@pytest.mark.asyncio
async def test_gate_fails_when_job_ends_failed_typed(backend):
    backend.fake.job_status = "failed"
    backend.fake.job_error_type = "lost_completion"
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    assert not result.ok
    done = _by_name(result)["job_completed"]
    assert not done.ok and "lost_completion" in done.detail
    # Release/baseline checks are not reached: a failed job can never pass.
    assert "generation_bundle_released" not in _by_name(result)


@pytest.mark.asyncio
async def test_gate_fails_when_job_never_reaches_a_terminal_state(backend):
    backend.fake.job_polls_until_terminal = 10_000  # always in_flight
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    assert not result.ok
    assert not _by_name(result)["job_completed"].ok


@pytest.mark.asyncio
async def test_gate_fails_when_comfyui_bundle_is_not_released(backend):
    backend.fake.bundle_release_after_polls = 10_000
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    checks = _by_name(result)
    assert not result.ok
    assert not checks["generation_bundle_released"].ok
    assert "comfyui" in checks["generation_bundle_released"].detail
    assert checks["job_completed"].ok  # failure is attributed to the release, not the job


@pytest.mark.asyncio
async def test_gate_fails_when_only_one_dgpu_baseline_is_restored(backend):
    # The incident: MI50 comes back but the RX idle Gemma is never reloaded.
    backend.fake.card_gb = {"card0": 3.1, "card1": 16.6}
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    checks = _by_name(result)
    assert not result.ok
    assert not checks["dgpu_baselines_restored"].ok
    assert "card0" in checks["dgpu_baselines_restored"].detail
    assert checks["generation_bundle_released"].ok


@pytest.mark.asyncio
async def test_gate_fails_when_baselines_never_restore_after_release(backend):
    backend.fake.restore_after_polls = 10_000  # cards stay at generation load
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    assert not _by_name(result)["dgpu_baselines_restored"].ok


@pytest.mark.asyncio
async def test_gate_fails_without_the_qwen_correct_stage_in_the_job_record(backend):
    backend.fake.job_record_extra = {"stages": []}
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    assert not result.ok
    assert not _by_name(result)["qwen_correct_stage_ran"].ok
    # ...and an empty marker deliberately skips only that check.
    backend.fake._job_polls = 0
    skipped = await gate.run_gate(backend.url, TOKEN, stage_marker="", **FAST)
    assert "qwen_correct_stage_ran" not in _by_name(skipped)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("llm_evicted", True),
    ("runtime_mode", "degraded"),
    ("contradictions", ["llm_evicted_with_llm_loaded_state"]),
])
async def test_gate_fails_when_manager_is_left_evicted_or_degraded(backend, field, value):
    setattr(backend.fake, field, value)
    result = await gate.run_gate(backend.url, TOKEN, **FAST)
    assert not result.ok
    assert not _by_name(result)["manager_not_degraded"].ok


@pytest.mark.asyncio
async def test_gate_fails_on_rejected_credentials(backend):
    result = await gate.run_gate(backend.url, "wrong-token", **FAST)
    assert not result.ok
    assert result.checks[0].name == "job_submitted" and not result.checks[0].ok


def test_parse_baselines_defaults_and_overrides():
    assert gate.parse_baselines(None) == {"card0": 14.3, "card1": 16.6}
    assert gate.parse_baselines(["card0=10", "card1=12.5"]) == {"card0": 10.0, "card1": 12.5}
    with pytest.raises(ValueError):
        gate.parse_baselines(["card0"])


@pytest.mark.asyncio
async def test_cli_exit_codes_and_report_never_contain_the_token(backend, tmp_path, capsys):
    import asyncio

    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n")
    argv = ["--base-url", backend.url, "--token-file", str(token_file),
            "--job-timeout-s", "1", "--release-timeout-s", "0.6", "--poll-interval-s", "0.02"]
    # main() owns its event loop, so run it off the loop that serves the fake.
    rc = await asyncio.to_thread(gate.main, argv)
    out = capsys.readouterr()
    assert rc == 0
    assert json.loads(out.out)["verdict"] == "pass"
    assert TOKEN not in out.out and TOKEN not in out.err
    missing = ["--base-url", backend.url, "--token-file", str(tmp_path / "missing")]
    assert await asyncio.to_thread(gate.main, missing) == 2
