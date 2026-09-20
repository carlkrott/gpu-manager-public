"""Neutral controller defaults for source-only deployments."""
from __future__ import annotations

import os
import re

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _identifier(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        return default
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must contain only letters, digits, '.', '_', ':', or '-'")
    return value


PRIMARY_GPU_ID = _identifier("GPU_MANAGER_PRIMARY_GPU_ID", "gpu_primary")
SECONDARY_GPU_ID = _identifier("GPU_MANAGER_SECONDARY_GPU_ID", "gpu_secondary")
DEFAULT_LLM_SERVICE_NAME = _identifier("GPU_MANAGER_DEFAULT_LLM_SERVICE", "") if os.environ.get("GPU_MANAGER_DEFAULT_LLM_SERVICE", "").strip() else ""


def empty_services_config() -> dict:
    """Return a safe registry with no host-specific services enabled."""
    return {
        "services": {},
        "bundles": {},
        "gpu_devices": {},
        "routing_groups": {},
        "generation_templates": {},
        "pipeline_providers": {},
        "scheduling": {
            "queue_owner": "durable",
            "scheduler_owns_load": False,
            "scheduler_dry_run_mode": True,
            "proactive_scheduling_enabled": False,
            "maintenance_mode": True,
            "idle_service": "",
            "idle_services": {},
        },
    }
