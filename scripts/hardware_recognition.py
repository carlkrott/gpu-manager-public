"""Hardware recognition and model capability contracts for GPU Manager v2.

Recognizes AMD GPUs (gfx1030, gfx906, etc.), CPU fallbacks, VRAM totals/free,
compute APIs (ROCm, Vulkan, CPU), precision constraints (e.g. lack of native BF16
on gfx1030/gfx906), and device loss/hotplug observation states (observed, missing, unknown).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import re
from typing import Any, Mapping, Optional, Sequence


class ComputeAPI(str, Enum):
    ROCM = "rocm"
    VULKAN = "vulkan"
    CPU = "cpu"
    UNKNOWN = "unknown"


class DeviceStatus(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"
    UNKNOWN = "unknown"


# Hardware architecture traits
# gfx1030 (RX 6950 XT / Navi 21) & gfx906 (MI50 / Vega 20) do NOT have native BF16 arithmetic.
ARCH_CAPABILITIES: dict[str, dict[str, Any]] = {
    "gfx1030": {
        "vendor": "AMD",
        "family": "Navi 21 / RDNA2",
        "native_bf16": False,
        "supported_precisions": ["fp32", "fp16", "int8", "int4"],
        "unsupported_precisions": ["bf16"],
        "compute_apis": [ComputeAPI.ROCM, ComputeAPI.VULKAN],
        "default_ceiling_mib": 14832,  # Desktop primary GPU fixed safe ceiling
    },
    "gfx906": {
        "vendor": "AMD",
        "family": "Vega 20 / GCN5",
        "native_bf16": False,
        "supported_precisions": ["fp32", "fp16", "int8", "int4"],
        "unsupported_precisions": ["bf16"],
        "compute_apis": [ComputeAPI.ROCM, ComputeAPI.VULKAN],
        "default_ceiling_mib": None,
    },
    "cpu": {
        "vendor": "Generic",
        "family": "CPU",
        "native_bf16": True,
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4", "q8_0", "q4_k_m"],
        "unsupported_precisions": [],
        "compute_apis": [ComputeAPI.CPU],
        "default_ceiling_mib": None,
    },
}

# PCI Device ID mapping to architecture and model name
AMD_DEVICE_PCI_MAP: dict[str, dict[str, Any]] = {
    "0x73a5": {
        "name": "AMD Radeon RX 6950 XT",
        "gfx_arch": "gfx1030",
        "vendor": "AMD",
        "role": "primary",
    },
    "0x66af": {
        "name": "AMD Radeon VII (MI50)",
        "gfx_arch": "gfx906",
        "vendor": "AMD",
        "role": "secondary",
    },
    # 7995x MI50 runs the Radeon Pro VII VBIOS since 2026-10-29 (66af -> 66a1).
    "0x66a1": {
        "name": "AMD Radeon Pro VII (MI50)",
        "gfx_arch": "gfx906",
        "vendor": "AMD",
        "role": "secondary",
    },
    "0x164e": {
        "name": "AMD Raphael",
        "gfx_arch": "gfx1036",
        "vendor": "AMD",
        "role": "igpu",
    },
}


@dataclass
class HardwareProfile:
    device_id: str
    card: Optional[str]
    pci_address: Optional[str]
    vendor: str
    gfx_arch: Optional[str]
    vram_total_mib: int
    vram_free_mib: int
    status: DeviceStatus
    compute_apis: list[str] = field(default_factory=list)
    native_bf16: bool = False
    supported_precisions: list[str] = field(default_factory=list)
    unsupported_precisions: list[str] = field(default_factory=list)
    vram_ceiling_mib: Optional[int] = None
    vram_reserve_mib: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "card": self.card,
            "pci_address": self.pci_address,
            "vendor": self.vendor,
            "gfx_arch": self.gfx_arch,
            "vram_total_mib": self.vram_total_mib,
            "vram_free_mib": self.vram_free_mib,
            "status": self.status.value if isinstance(self.status, DeviceStatus) else str(self.status),
            "compute_apis": self.compute_apis,
            "native_bf16": self.native_bf16,
            "supported_precisions": self.supported_precisions,
            "unsupported_precisions": self.unsupported_precisions,
            "vram_ceiling_mib": self.vram_ceiling_mib,
            "vram_reserve_mib": self.vram_reserve_mib,
        }


def recognize_hardware_device(
    device_id: str,
    device_config: Optional[Mapping[str, Any]] = None,
    telemetry_gpu: Optional[Mapping[str, Any]] = None,
) -> HardwareProfile:
    """Build a comprehensive HardwareProfile combining registry config and observed telemetry."""
    cfg = dict(device_config or {})
    telem = dict(telemetry_gpu or {})

    # Preserve CPU route
    if device_id.lower() in ("cpu", "gemma-cpu") or cfg.get("device") == "cpu":
        return HardwareProfile(
            device_id=device_id,
            card=None,
            pci_address=None,
            vendor="Generic",
            gfx_arch="cpu",
            vram_total_mib=0,
            vram_free_mib=0,
            status=DeviceStatus.OBSERVED,
            compute_apis=[ComputeAPI.CPU.value],
            native_bf16=True,
            supported_precisions=ARCH_CAPABILITIES["cpu"]["supported_precisions"],
            unsupported_precisions=[],
            vram_ceiling_mib=None,
            vram_reserve_mib=None,
        )

    card = cfg.get("card") or telem.get("card")
    pci_address = cfg.get("pci") or telem.get("pci_address")
    gfx_arch = cfg.get("gfx_arch")
    vendor = cfg.get("vendor", "AMD")

    # Detect gfx_arch if unknown
    if not gfx_arch and card:
        # Check sysfs device ID
        dev_id_path = f"/sys/class/drm/{card}/device/device"
        if os.path.exists(dev_id_path):
            try:
                with open(dev_id_path) as f:
                    pci_dev = f.read().strip().lower()
                    if pci_dev in AMD_DEVICE_PCI_MAP:
                        gfx_arch = AMD_DEVICE_PCI_MAP[pci_dev]["gfx_arch"]
            except Exception:
                pass

    if not gfx_arch and device_id == "rx6950xt":
        gfx_arch = "gfx1030"
    elif not gfx_arch and device_id == "mi50":
        gfx_arch = "gfx906"

    arch_info = ARCH_CAPABILITIES.get(gfx_arch or "", {})
    native_bf16 = arch_info.get("native_bf16", False)
    supported_precisions = list(arch_info.get("supported_precisions", ["fp32", "fp16"]))
    unsupported_precisions = list(arch_info.get("unsupported_precisions", ["bf16"] if not native_bf16 else []))
    compute_apis = [api.value if isinstance(api, ComputeAPI) else str(api) for api in arch_info.get("compute_apis", [ComputeAPI.ROCM, ComputeAPI.VULKAN])]

    vram_total = int(cfg.get("vram_mib") or telem.get("vram_total_mib") or 0)
    vram_used = int(telem.get("vram_used_mib") or 0)
    vram_free = max(0, vram_total - vram_used)

    # Determine status (observed, missing, unknown)
    if telem:
        status = DeviceStatus.OBSERVED
    elif card and os.path.exists(f"/sys/class/drm/{card}"):
        status = DeviceStatus.OBSERVED
    elif cfg:
        status = DeviceStatus.MISSING
    else:
        status = DeviceStatus.UNKNOWN

    vram_ceiling = cfg.get("vram_ceiling_mib")
    if device_id == "rx6950xt" and vram_ceiling is None:
        vram_ceiling = 14832
    elif device_id == "rx6950xt" and vram_ceiling is not None:
        vram_ceiling = min(int(vram_ceiling), 14832)

    vram_reserve = cfg.get("vram_reserve_mib")

    return HardwareProfile(
        device_id=device_id,
        card=card,
        pci_address=pci_address,
        vendor=vendor,
        gfx_arch=gfx_arch,
        vram_total_mib=vram_total,
        vram_free_mib=vram_free,
        status=status,
        compute_apis=compute_apis,
        native_bf16=native_bf16,
        supported_precisions=supported_precisions,
        unsupported_precisions=unsupported_precisions,
        vram_ceiling_mib=vram_ceiling,
        vram_reserve_mib=vram_reserve,
    )


def validate_model_hardware_compatibility(
    model_set: Mapping[str, Any],
    hardware_registry: Mapping[str, HardwareProfile],
) -> list[str]:
    """Validate that a model-set can run on the target hardware without BF16 or unsupportable constraints."""
    errors = []
    components = model_set.get("components", [])
    if isinstance(components, list):
        for idx, comp in enumerate(components):
            if not isinstance(comp, Mapping):
                continue
            comp_id = comp.get("id", f"comp_{idx}")
            dev_id = comp.get("device", "")
            precision = comp.get("precision", "").lower()
            compute_api = comp.get("compute_api")

            if dev_id in ("cpu", "CPU"):
                continue

            hw = hardware_registry.get(dev_id)
            if hw is None:
                errors.append(f"component '{comp_id}' references unknown or missing hardware device '{dev_id}'")
                continue

            # Check precision constraints
            if precision in hw.unsupported_precisions or (precision == "bf16" and not hw.native_bf16):
                errors.append(
                    f"component '{comp_id}' requires precision '{precision}' which is not natively supported "
                    f"by device '{dev_id}' (arch '{hw.gfx_arch}')"
                )

            # Check compute_api constraints
            if compute_api and str(compute_api).lower() not in [api.lower() for api in hw.compute_apis]:
                errors.append(
                    f"component '{comp_id}' requires compute_api '{compute_api}' not supported by device '{dev_id}' "
                    f"(supported: {hw.compute_apis})"
                )

    # Check memory budget vs ceiling / total
    budgets = model_set.get("memory_budget_mib", {})
    if isinstance(budgets, Mapping):
        for dev_id, budget in budgets.items():
            if dev_id.lower() == "cpu":
                continue
            hw = hardware_registry.get(dev_id)
            if hw and budget is not None:
                effective_limit = hw.vram_ceiling_mib or hw.vram_total_mib
                if effective_limit and budget > effective_limit:
                    errors.append(
                        f"model-set budget for '{dev_id}' ({budget} MiB) exceeds device ceiling/capacity ({effective_limit} MiB)"
                    )

    return errors
