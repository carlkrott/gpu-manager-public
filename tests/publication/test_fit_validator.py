"""Phase 4b fit validator: report-only findings, never blocks activation."""
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_manager import fit_validator as fv  # noqa: E402
from test_idle_restore_http import AUTH, lifecycle, manager  # noqa: E402,F401  (fixtures)

DEVICES = {
    "rx6950xt": {"rocm_device": 0, "role": "primary", "vram_mib": 16384},
    "mi50": {"rocm_device": 1, "role": "secondary", "vram_mib": 16384},
}
NO_FILES = lambda name: None  # noqa: E731


def _cfg(**extra):
    return {"gpu_devices": DEVICES, "services": {}, "bundles": {}, **extra}


def _rules(findings, owner=None):
    return {(f["bundle"], f["rule"]) for f in findings
            if f["severity"] == "error" and (owner is None or f["bundle"] == owner)}


def test_over_budget_component_is_flagged_and_fit_one_is_not():
    cfg = _cfg(bundles={
        "big": {"gpu_id": "rx6950xt", "components": [
            {"name": "unet", "device": "rx", "vram_peak_gb": 14.0}]},
        "small": {"gpu_id": "mi50", "components": [
            {"name": "enc", "device": "mi50", "vram_peak_gb": 6.0}]},
    })
    findings = fv.evaluate(cfg, None, size_lookup=NO_FILES)
    over = [f for f in findings if f["rule"] == "over_budget"]
    assert [(f["bundle"], f["budget_gb"]) for f in over] == [("big", 13.5)]
    assert over[0]["budget_source"] == "fallback"


def test_live_qwen_edit_distorch_workflow_is_flagged():
    cfg = _cfg(bundles={"qwen-edit-dualgpu": {"gpu_id": "rx6950xt", "services": []}},
               generation_templates={"image-edit-q5-dualgpu": {
                   "bundle": "qwen-edit-dualgpu",
                   "workflow_template": "workflows/qwen-image-edit-2511-q5-dualgpu.json"}})
    split = [f for f in fv.evaluate(cfg, None, size_lookup=NO_FILES)
             if f["rule"] == "split_offload"]
    assert len(split) == 1
    assert split[0]["device"] == "rx6950xt"  # compute_device cuda:0
    for needle in ("DisTorch", "virtual_vram_gb=3.0", "donor_device=cuda:1"):
        assert needle in split[0]["detail"]


def test_moe_split_is_exempt_but_dense_split_is_not():
    def svc(**kw):
        return {"port": 1, "gpu_id": "rx6950xt", "vram_peak_gb": 40.0,
                "launch_args": ["-m", "x.gguf", "-ts", "9,15", "--cpu-moe"], **kw}
    cfg = _cfg(services={"moe-llm": svc(moe=True), "dense-llm": svc()},
               bundles={"a": {"gpu_id": "rx6950xt", "services": ["moe-llm"]},
                        "b": {"gpu_id": "rx6950xt", "services": ["dense-llm"]}})
    rules = _rules(fv.evaluate(cfg, None, size_lookup=NO_FILES))
    assert not {r for r in rules if r[0] == "a"}
    assert ("b", "split_offload") in rules and ("b", "over_budget") in rules


def test_weights_outside_models_ssd_are_flagged(tmp_path):
    ssd, other = tmp_path / "ssd", tmp_path / "home"
    ssd.mkdir(), other.mkdir()
    (ssd / "in.gguf").write_bytes(b"x")
    (other / "out.gguf").write_bytes(b"x")
    (other / "link-in.gguf").symlink_to(ssd / "in.gguf")  # realpath under the SSD: fine
    cfg = _cfg(services={
        n: {"port": i, "gpu_id": "mi50", "model_path": str(other / f)}
        for i, (n, f) in enumerate([("inside", "link-in.gguf"), ("outside", "out.gguf")])})
    rules = _rules(fv.evaluate(cfg, None, models_root=str(ssd)))
    assert rules == {("outside", "weights_off_models_ssd")}


def test_live_protected_resident_reduces_mi50_budget():
    cfg = _cfg(services={"D1-MI50": {"protected": True, "port": 8149, "gpu_id": "mi50"}})
    live = {"gpus": [{"gpu_id": "mi50", "processes": [
        {"pid": 7, "service_port": 8149, "vram_mib": 2048}]}],
        "vulkan_tenants": {"mi50": [{"pid": 7, "amdgpu_vram_mib": 3072}]}}
    assert fv.card_budget(cfg, live, "mi50")[:2] == (13.0, "live")  # 16 - 3 (vulkan max)
    assert fv.card_budget(cfg, None, "mi50")[:2] == (round(16 - 3929 / 1024, 2), "fallback")
    gone = {"gpus": [{"gpu_id": "mi50", "processes": []}]}
    assert fv.card_budget(cfg, gone, "mi50")[:2] == (16.0, "live")
    assert fv.card_budget(cfg, live, "mi50", owner_services={"D1-MI50"})[0] == 16.0


def test_revision_activation_hook_reports_revision_findings(caplog):
    cfg = _cfg(service_revision_catalog={"services": {"h3": [{"revision": 1, "definition": {
        "gpu_id": "mi50", "model_path": "",
        "resource_requirements": {"text_encoder_gpu_split": ["mi50", "rx6950xt"]}}}]}})
    caplog.set_level(logging.WARNING)
    errors = fv.warn_if_unfit(logging.getLogger("fit-test"), cfg, None, ["h3@r1"], service="h3")
    assert [(f["bundle"], f["rule"]) for f in errors] == [("h3@r1", "split_offload")]
    assert "activation of h3@r1 would be refused" in caplog.text


def test_warn_if_unfit_never_raises(monkeypatch):
    log = logging.getLogger("fit-test")
    monkeypatch.setattr(fv, "evaluate", lambda *a, **k: 1 / 0)
    assert fv.warn_if_unfit(log, {}, None, ["x"]) == []


def _make_gemma_rx_split(env):
    env.cfg["services"]["gemma-rx"]["launch_args"] = ["-ts", "9,15"]


@pytest.mark.asyncio
async def test_activation_succeeds_and_logs_when_findings_exist(lifecycle, caplog):
    _make_gemma_rx_split(lifecycle)
    caplog.set_level(logging.WARNING)
    r = await lifecycle.client.post("/v1/bundles/preload", headers=AUTH,
                                    json={"bundle_name": "Gemma-RX6950XT-idle"})
    assert r.status == 200, await r.text()
    assert (await r.json())["preload_state"] == "loaded"
    assert any("FitValidator (report-only): activation of Gemma-RX6950XT-idle would be refused"
               in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_activation_succeeds_when_validator_crashes(lifecycle):
    lifecycle.mp.setattr(lifecycle.m._fit_validator, "evaluate", lambda *a, **k: 1 / 0)
    r = await lifecycle.client.post("/v1/bundles/preload", headers=AUTH,
                                    json={"bundle_name": "Gemma-RX6950XT-idle"})
    assert r.status == 200, await r.text()


@pytest.mark.asyncio
async def test_fit_report_route_is_admin_read_only(lifecycle):
    _make_gemma_rx_split(lifecycle)
    assert (await lifecycle.client.get("/v1/fit-report")).status in (401, 403)
    r = await lifecycle.client.get("/v1/fit-report", headers=AUTH)
    assert r.status == 200
    body = await r.json()
    assert body["report_only"] is True
    assert "Gemma-RX6950XT-idle" in body["would_refuse"]


def test_foreign_tenants_reduce_live_budget():
    cfg = _cfg(services={"D1-MI50": {"protected": True, "port": 8149, "gpu_id": "mi50"},
                         "Evictable": {"port": 8100, "gpu_id": "mi50"}})
    live = {"gpus": [{"gpu_id": "mi50", "processes": [
        {"pid": 7, "service_port": 8149, "vram_mib": 2048},
        {"pid": 8, "service_port": 8100, "vram_mib": 4096},
        {"pid": 9, "service_port": 8166, "vram_mib": 1024}]}],
        "vulkan_tenants": {"mi50": [{"pid": 9, "amdgpu_vram_mib": 5120},
                                    {"pid": 11, "amdgpu_vram_mib": 1024}]}}
    budget, _src, residents = fv.card_budget(cfg, live, "mi50")
    assert residents == {"D1-MI50": 2.0, "foreign": 6.0}  # breccia pid 9 (vulkan max) + pid 11
    assert budget == 8.0
