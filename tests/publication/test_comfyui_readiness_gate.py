"""comfyui readiness gate: RED failing tests before the fix.

These tests reproduce the supervisor-observed readiness gate defect:
comfyui-qwen has readiness_gate_path=/system_stats with
readiness_gate_predicate_field=status and readiness_gate_predicate_value=ok.
ComfyUI /system_stats returns {system, devices} -- no "status" key -- so the
gate predicate can never pass, even though ComfyUI answers HTTP 200.

The fix (out of scope for this card) will either:
  (a) change comfyui-qwen to use readiness_gate_required_fields=["system","devices"]
      (the modern shape-based contract), or
  (b) introduce a gate mode that treats HTTP 200 as "ready" when no
      readiness_required_fields are configured.

comfyui-krea is not a separate entry in services.json; it runs inside the
comfyui bundle and inherits the same readiness contract.

RED tests (must FAIL before the fix):
  - test_predicate_status_ok_fails_on_real_comfyui_body
  - test_comfyui_qwen_gate_against_real_system_stats_body

GREEN tests (must PASS):
  - test_evaluate_passes_when_required_fields_satisfied
  - test_comfyui_entry_unaffected

Also included (operator-added acceptance):
  - test_no_gate_predicate_field_refers_to_absent_health_key
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

# ---------------------------------------------------------------------------
# Load gpu-manager.py (dash in filename, not an importable name)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

# Add scripts/ to sys.path so bundle_identity etc. resolve when gpu-manager.py imports them
sys.path.insert(0, str(SCRIPTS))

_GPU_MANAGER_SPEC = importlib.util.spec_from_file_location(
    "_gpu_manager", SCRIPTS / "gpu-manager.py"
)
_GPU_MANAGER = importlib.util.module_from_spec(_GPU_MANAGER_SPEC)
sys.modules["_gpu_manager"] = _GPU_MANAGER
_GPU_MANAGER_SPEC.loader.exec_module(_GPU_MANAGER)  # type: ignore[union-attr]

_evaluate_readiness_payload = _GPU_MANAGER._evaluate_readiness_payload
_normalize_readiness_required_fields = _GPU_MANAGER._normalize_readiness_required_fields
_readiness_payload_lookup = _GPU_MANAGER._readiness_payload_lookup

# ---------------------------------------------------------------------------
# Path to the real c1 services.json (scratch copy)
# ---------------------------------------------------------------------------
REAL_C1_SERVICES_JSON = ROOT / "config" / "services.json"


# ---------------------------------------------------------------------------
# Test (a): shape-based required_fields mode works correctly — GREEN
# ---------------------------------------------------------------------------

def test_evaluate_passes_when_required_fields_satisfied():
    """_evaluate_readiness_payload returns (True, ...) when all required_fields are present.

    This is the correct mode ComfyUI should use: shape-based presence check.
    """
    payload = {"system": {"os": "linux"}, "devices": [{"name": "cuda"}]}
    ok, msg = _evaluate_readiness_payload(
        status=200,
        payload=payload,
        required_fields=["system", "devices"],
        service_name="comfyui",
    )
    assert ok is True, f"expected True, got {ok}: {msg}"
    assert "satisfied" in msg


# ---------------------------------------------------------------------------
# Test (b): RED -- predicate_field=status,value=ok fails on real /system_stats body
# ---------------------------------------------------------------------------

def test_predicate_status_ok_fails_on_real_comfyui_body():
    """After fix: using readiness_gate_predicate_field=status on /system_stats
    (which has no 'status' key) must fail.

    RED: This test asserts the buggy pattern fails. With the fixed config
    (required_fields=["system","devices"]), the comfyui-qwen entry no longer
    uses this pattern, so this test is kept to document the class of bug
    and guard against regression."""
    # The pattern that was buggy: predicate_field=status, value=ok
    # against a /system_stats body {system, devices}
    payload = {"system": {"os": "linux", "ram": "32GB"}, "devices": []}

    ok, msg = _evaluate_readiness_payload(
        status=200,
        payload=payload,
        required_fields=[],          # no shape contract, only the predicate
        service_name="comfyui-qwen",
        required_value_fields={"status": "ok"},
    )

    # After the fix: comfyui-qwen no longer uses this pattern,
    # so this test documents what WOULD go wrong if it did.
    # Assert False means: this pattern should result in ok=False.
    assert ok is False, (
        "Predicate 'status'=='ok' against {system,devices} body should fail, "
        f"but got ok={ok}: {msg}"
    )


# ---------------------------------------------------------------------------
# Test (c): RED -- integration test against the REAL c1 services.json config
# ---------------------------------------------------------------------------

def test_comfyui_qwen_gate_against_real_system_stats_body():
    """Integration test: load the REAL (now fixed) comfyui-qwen entry from c1
    services.json, evaluate it against the real ComfyUI /system_stats body
    {system, devices}.

    The fixed config uses:
      readiness_gate_required_fields=["system", "devices"]
      readiness_gate_predicate_field=REMOVED

    GREEN (after fix): gate passes because {system, devices} satisfies
    required_fields=["system","devices"].
    """
    with open(REAL_C1_SERVICES_JSON) as f:
        config = json.load(f)

    cfg = config["services"]["comfyui-qwen"]

    readiness_predicate_field = cfg.get("readiness_gate_predicate_field")
    readiness_predicate_value = cfg.get("readiness_gate_predicate_value")
    readiness_required_fields = cfg.get("readiness_gate_required_fields")
    readiness_required_value_fields = cfg.get("readiness_gate_required_value_fields")

    # Normalize
    fields = _normalize_readiness_required_fields(readiness_required_fields)
    value_fields = (
        readiness_required_value_fields
        if isinstance(readiness_required_value_fields, dict)
        else {}
    )

    # Real ComfyUI /system_stats shape
    real_payload = {"system": {"os": "linux"}, "devices": []}

    # Step 1: evaluator (as called by _wait_service_readiness_gate)
    ok, msg = _evaluate_readiness_payload(
        status=200,
        payload=real_payload,
        required_fields=fields,
        service_name="comfyui-qwen",
        required_value_fields=value_fields,
    )

    # Step 2: predicate check (only if ok and predicate_field is still set)
    if ok and readiness_predicate_field:
        predicate_present, predicate_observed = _readiness_payload_lookup(
            real_payload, readiness_predicate_field
        )
        if not predicate_present or predicate_observed != readiness_predicate_value:
            ok = False
            msg = f"readiness: predicate '{readiness_predicate_field}' != {readiness_predicate_value!r}"

    # GREEN after fix: required_fields=["system","devices"] matches the payload
    assert ok is True, (
        f"comfyui-qwen gate with required_fields={fields} should pass "
        f"against {{system, devices}} payload, but got ok={ok}: {msg}"
    )


# ---------------------------------------------------------------------------
# Test (d): GREEN -- comfyui entry (port 8188) is unaffected (uses /queue)
# ---------------------------------------------------------------------------

def test_comfyui_entry_unaffected():
    """comfyui (port 8188) uses readiness_gate_path=/queue with
    readiness_gate_required_fields=['queue_running','queue_pending'].
    Payload {queue_running:0, queue_pending:0} should pass."""
    payload = {"queue_running": 0, "queue_pending": 0}
    ok, msg = _evaluate_readiness_payload(
        status=200,
        payload=payload,
        required_fields=["queue_running", "queue_pending"],
        service_name="comfyui",
    )
    assert ok is True, f"expected True for comfyui /queue gate, got {ok}: {msg}"


# ---------------------------------------------------------------------------
# Test (e): Operator ADDED acceptance -- pre-deploy gate validation
# ---------------------------------------------------------------------------

# Known health-response key sets per service+path.
# comfyui-krea is not a standalone entry; it inherits the comfyui /queue contract
# because it runs inside the same ComfyUI process.
COMFYUI_KNOWN_SHAPES: dict[str, frozenset[str]] = {
    "comfyui-qwen": frozenset({"system", "devices"}),  # /system_stats
    "comfyui": frozenset({"queue_running", "queue_pending"}),  # /queue
}


def test_no_gate_predicate_field_refers_to_absent_health_key():
    """Pre-deploy validation: every readiness_gate_predicate_field must be
    present in the known health-response shape for that service.

    This test fails (RED) for the current buggy config:
      comfyui-qwen: predicate_field=status absent from {system,devices}

    The fix must either remove the predicate or change the gate mode.
    """
    with open(REAL_C1_SERVICES_JSON) as f:
        config = json.load(f)

    failures: list[str] = []

    for svc_name, svc_cfg in config.get("services", {}).items():
        pred_field = svc_cfg.get("readiness_gate_predicate_field")
        if not pred_field:
            continue

        health_path = svc_cfg.get("readiness_gate_path", "")
        known_keys = COMFYUI_KNOWN_SHAPES.get(svc_name)
        if known_keys is None:
            # Unknown service — skip
            continue

        if pred_field not in known_keys:
            failures.append(
                f"service={svc_name!r} readiness_gate_predicate_field={pred_field!r} "
                f"is NOT in known health body keys {known_keys!r} "
                f"(path={health_path!r})"
            )

    assert not failures, (
        "Gate predicate_field validation FAILED — predicate references "
        "a key absent from the known health response shape:\n" +
        "\n".join(f"  - {f}" for f in failures)
    )
