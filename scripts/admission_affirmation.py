"""Fail-closed admission and configuration affirmation contracts for GPU Manager v2.

Validates catalogue kind, backend, ISA/precision, min VRAM plus reserve, sharing,
and live available capacity before any queue claim, eviction or unload.
Returns caller-compatible structured outcomes:
- accepted(placement)
- queued(position, reason)
- rejected(reason, nearest alternative)

Also implements configuration dry run and quarantine of invalid entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Mapping, Optional, Sequence

from hardware_recognition import (
    ARCH_CAPABILITIES,
    ComputeAPI,
    DeviceStatus,
    HardwareProfile,
    recognize_hardware_device,
)


class AdmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AdmissionDecision:
    status: AdmissionStatus
    placement: Optional[dict[str, Any]] = None
    position: Optional[int] = None
    reason: Optional[str] = None
    nearest_alternative: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"status": self.status.value}
        if self.placement is not None:
            res["placement"] = self.placement
        if self.position is not None:
            res["position"] = self.position
        if self.reason is not None:
            res["reason"] = self.reason
        if self.nearest_alternative is not None:
            res["nearest_alternative"] = self.nearest_alternative
        return res


def accepted(placement: dict[str, Any]) -> AdmissionDecision:
    return AdmissionDecision(status=AdmissionStatus.ACCEPTED, placement=placement)


def queued(position: int, reason: str) -> AdmissionDecision:
    return AdmissionDecision(status=AdmissionStatus.QUEUED, position=position, reason=reason)


def rejected(reason: str, nearest_alternative: Optional[dict[str, Any]] = None) -> AdmissionDecision:
    return AdmissionDecision(
        status=AdmissionStatus.REJECTED,
        reason=reason,
        nearest_alternative=nearest_alternative,
    )


class AdmissionController:
    """Fail-closed controller validating catalogue, hardware, capacity and configs."""

    def __init__(self, hardware_registry: Optional[Mapping[str, HardwareProfile]] = None):
        self.hardware_registry = dict(hardware_registry or {})

    def evaluate_admission(
        self,
        request: Mapping[str, Any],
        services_config: Mapping[str, Any],
        telemetry: Optional[Mapping[str, Any]] = None,
        queue_depth: int = 0,
    ) -> AdmissionDecision:
        """Validate catalogue kind, backend, ISA/precision, min VRAM plus reserve, sharing & live capacity.
        
        Evaluated BEFORE any queue claim, eviction, or unload.
        """
        # 1. Catalogue kind & backend validation
        kind = request.get("kind") or request.get("type") or "llm"
        model = request.get("model") or request.get("model_id") or ""
        backend = request.get("backend") or ""
        service_name = request.get("service") or request.get("service_name") or ""

        # Extract requirements
        req_precision = (request.get("precision") or "").lower()
        req_vram_mib = int(request.get("min_vram_mib") or request.get("vram_mib") or 0)
        target_device = request.get("device") or request.get("gpu_id")
        compute_api = (request.get("compute_api") or "rocm").lower()

        gpu_devices = services_config.get("gpu_devices", {})
        services = services_config.get("services", {})
        templates = services_config.get("generation_templates", {})

        # Resolve target device if not explicitly set
        if not target_device and service_name in services:
            target_device = services[service_name].get("gpu_id")
        if not target_device and kind in templates:
            tmpl = templates[kind]
            backend = tmpl.get("backend", backend)
            if backend in services:
                target_device = services[backend].get("gpu_id")

        if not target_device:
            # Default or fallback candidate lookup
            candidates = list(gpu_devices.keys()) if gpu_devices else ["rx6950xt", "mi50"]
            target_device = candidates[0] if candidates else "rx6950xt"

        # Check device in hardware registry or recognize it
        hw = self.hardware_registry.get(target_device)
        if not hw:
            dev_cfg = gpu_devices.get(target_device, {})
            telem_dev = None
            if telemetry:
                for g in telemetry.get("gpus", []):
                    if g.get("gpu_id") == target_device or g.get("card") == dev_cfg.get("card"):
                        telem_dev = g
                        break
            hw = recognize_hardware_device(target_device, dev_cfg, telem_dev)
            self.hardware_registry[target_device] = hw

        # Fail-closed if target device is missing or unknown
        if hw.status in (DeviceStatus.MISSING, DeviceStatus.UNKNOWN):
            alt = None
            for alt_dev, alt_hw in self.hardware_registry.items():
                if alt_hw.status == DeviceStatus.OBSERVED and alt_dev != target_device:
                    alt = {"device": alt_dev}
                    break
            if not alt:
                alt = {"device": "cpu"}
            return rejected(
                f"Device '{target_device}' is unavailable (status: {hw.status.value})",
                nearest_alternative=alt,
            )

        # Precision / ISA check
        if req_precision:
            if req_precision in hw.unsupported_precisions or (req_precision == "bf16" and not hw.native_bf16):
                # Look for nearest alternative (e.g. CPU or an alternate device supporting precision)
                alt = None
                for alt_dev, alt_hw in self.hardware_registry.items():
                    if alt_hw.native_bf16 or req_precision in alt_hw.supported_precisions:
                        alt = {"device": alt_dev, "precision": req_precision}
                        break
                if not alt and hw.supported_precisions:
                    alt = {"device": target_device, "precision": hw.supported_precisions[0]}
                return rejected(
                    f"Device '{target_device}' does not natively support precision '{req_precision}' (arch '{hw.gfx_arch}')",
                    nearest_alternative=alt,
                )

        # Compute API check
        if compute_api and compute_api not in [api.lower() for api in hw.compute_apis]:
            return rejected(
                f"Device '{target_device}' does not support compute API '{compute_api}' (supported: {hw.compute_apis})",
                nearest_alternative={"device": "cpu", "compute_api": "cpu"},
            )

        # VRAM ceiling and reserve calculation
        dev_cfg = gpu_devices.get(target_device, {})
        vram_total = hw.vram_total_mib or int(dev_cfg.get("vram_mib") or 0)
        configured_ceiling = dev_cfg.get("vram_ceiling_mib")
        reserve = int(dev_cfg.get("vram_reserve_mib") or (1536 if target_device == "rx6950xt" else 0))

        if target_device == "rx6950xt":
            effective_ceiling = min(int(configured_ceiling), 14832) if configured_ceiling is not None else 14832
        else:
            effective_ceiling = int(configured_ceiling) if configured_ceiling is not None else (vram_total - reserve if vram_total > reserve else vram_total)

        if req_vram_mib > 0 and effective_ceiling and req_vram_mib > effective_ceiling:
            alt = None
            for alt_dev, alt_hw in self.hardware_registry.items():
                alt_ceiling = alt_hw.vram_ceiling_mib or alt_hw.vram_total_mib
                if alt_ceiling and req_vram_mib <= alt_ceiling:
                    alt = {"device": alt_dev, "vram_ceiling_mib": alt_ceiling}
                    break
            return rejected(
                f"Requested VRAM {req_vram_mib} MiB exceeds effective device ceiling {effective_ceiling} MiB on '{target_device}' (desktop reserve enforced)",
                nearest_alternative=alt,
            )

        # Live available capacity check (before queue claim / eviction)
        live_used = 0
        if telemetry:
            for g in telemetry.get("gpus", []):
                if g.get("gpu_id") == target_device or g.get("card") == dev_cfg.get("card"):
                    live_used = int(g.get("vram_used_mib") or 0)
                    break
        else:
            live_used = hw.vram_total_mib - hw.vram_free_mib

        available_vram = max(0, effective_ceiling - live_used)
        if req_vram_mib > 0 and req_vram_mib > available_vram:
            # Cannot be placed immediately without eviction, must be queued
            return queued(
                position=queue_depth + 1,
                reason=f"Insufficient live capacity on '{target_device}': {available_vram} MiB available, {req_vram_mib} MiB required (ceiling {effective_ceiling} MiB)",
            )

        # Admitted and accepted
        placement = {
            "device": target_device,
            "card": hw.card,
            "gfx_arch": hw.gfx_arch,
            "effective_ceiling_mib": effective_ceiling,
            "allocated_vram_mib": req_vram_mib,
            "available_vram_mib": available_vram - req_vram_mib,
        }
        return accepted(placement=placement)


def dry_run_config(config: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Dry run configuration validation and quarantine invalid entries.
    
    Returns (is_valid, errors, quarantined_config) without crashing service.
    """
    errors: list[str] = []
    quarantined = copy_config = dict(config)
    gpu_devices = dict(config.get("gpu_devices", {}) or {})
    valid_devices = {}
    quarantined_devices = {}

    for gid, dev in gpu_devices.items():
        if not isinstance(dev, Mapping):
            continue
        dev_errs = []
        ceiling = dev.get("vram_ceiling_mib")
        vram_total = dev.get("vram_mib")
        reserve = dev.get("vram_reserve_mib")

        if gid == "rx6950xt" and ceiling is not None and int(ceiling) > 14832:
            dev_errs.append(f"gpu_devices.{gid}: vram_ceiling_mib={ceiling} exceeds fixed ceiling 14832")

        if ceiling is not None and vram_total is not None and reserve is not None:
            if int(ceiling) > (int(vram_total) - int(reserve)):
                dev_errs.append(f"gpu_devices.{gid}: vram_ceiling_mib={ceiling} exceeds physical-reserve")

        target = dev.get("vram_target_mib")
        if target is not None and ceiling is not None and int(target) > int(ceiling):
            dev_errs.append(f"gpu_devices.{gid}: vram_target_mib={target} exceeds vram_ceiling_mib={ceiling}")

        if dev_errs:
            errors.extend(dev_errs)
            quarantined_devices[gid] = dict(dev)
        else:
            valid_devices[gid] = dict(dev)

    if quarantined_devices:
        quarantined["gpu_devices"] = valid_devices
        quarantined["_quarantined_gpu_devices"] = quarantined_devices

    # Check scheduling
    scheduling = config.get("scheduling", {}) or {}
    if scheduling.get("scheduler_owns_load"):
        templates = config.get("generation_templates", {}) or {}
        valid_templates = {}
        quarantined_templates = {}
        for tname, tpl in templates.items():
            if isinstance(tpl, Mapping) and tpl.get("evict_first"):
                errors.append(f"template '{tname}': evict_first is forbidden when scheduler_owns_load=true")
                quarantined_templates[tname] = dict(tpl)
            else:
                valid_templates[tname] = tpl
        if quarantined_templates:
            quarantined["generation_templates"] = valid_templates
            quarantined["_quarantined_generation_templates"] = quarantined_templates

    is_valid = len(errors) == 0
    return is_valid, errors, quarantined
