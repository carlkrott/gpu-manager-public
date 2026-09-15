"""Read-only host/container preflight for GPU Manager profiles.

The preflight is intentionally conservative.  It reports what the host can
prove (container engine availability, render nodes, memory and disk) and
requires a private hardware overlay before mapping stable GPU IDs to PCI or
render devices.  It never starts/stops a service, changes permissions, mounts
devices, downloads models, or writes a file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Any, Callable, Mapping

from runtime_contracts import validate_runtime_profile


PREFLIGHT_SCHEMA = "gpu-manager-preflight.v1"
OVERLAY_SCHEMA = "hardware-overlay.v1"
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RENDER_NODE_RE = re.compile(r"^renderD[0-9]+$")
_PCI_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


def _read_mem_total_mib(path: Path) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def collect_host_facts(
    *,
    which: Callable[[str], str | None] = shutil.which,
    drm_root: Path = Path("/dev/dri"),
    sysfs_root: Path = Path("/sys/class/drm"),
    meminfo: Path = Path("/proc/meminfo"),
    disk_root: Path = Path("/"),
) -> dict[str, Any]:
    """Collect non-secret, read-only host facts with injectable roots."""

    render_nodes = sorted(
        path.name
        for path in drm_root.glob("renderD*")
        if _RENDER_NODE_RE.fullmatch(path.name)
    )
    drm_cards = sorted(
        path.name
        for path in sysfs_root.glob("card*")
        if re.fullmatch(r"card[0-9]+", path.name)
    )
    try:
        disk_free_gib = round(shutil.disk_usage(disk_root).free / (1024**3), 2)
    except OSError:
        disk_free_gib = None
    commands = {
        name: bool(which(name))
        for name in ("docker", "podman", "systemctl", "rocminfo")
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "platform": platform.system().lower(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "container_engines": [name for name in ("docker", "podman") if commands[name]],
        "commands": commands,
        "render_nodes": render_nodes,
        "drm_cards": drm_cards,
        "memory_total_mib": _read_mem_total_mib(meminfo),
        "disk_free_gib": disk_free_gib,
    }


def validate_hardware_overlay(overlay: Mapping[str, Any]) -> list[str]:
    """Validate a private stable-device overlay without exposing its values."""

    errors: list[str] = []
    if not isinstance(overlay, Mapping):
        return ["hardware overlay must be an object"]
    if overlay.get("schema_version") != OVERLAY_SCHEMA:
        errors.append(f"schema_version must be {OVERLAY_SCHEMA!r}")
    devices = overlay.get("devices")
    if not isinstance(devices, Mapping) or not devices:
        return errors + ["devices must be a non-empty object"]
    for device_id, device in devices.items():
        path = f"devices.{device_id}"
        if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
            errors.append(f"{path}: invalid stable device id")
        if not isinstance(device, Mapping):
            errors.append(f"{path} must be an object")
            continue
        pci = device.get("pci")
        if pci is not None and (not isinstance(pci, str) or not _PCI_RE.fullmatch(pci)):
            errors.append(f"{path}.pci must be a PCI address")
        render_node = device.get("render_node")
        if render_node is not None and (
            not isinstance(render_node, str) or not _RENDER_NODE_RE.fullmatch(render_node)
        ):
            errors.append(f"{path}.render_node must be a renderD node")
        architecture = device.get("architecture")
        if architecture is not None and not isinstance(architecture, str):
            errors.append(f"{path}.architecture must be a string")
    return errors


def preflight_profile(
    profile_name: str,
    profile: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    hardware_overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a profile has enough checked host evidence to run."""

    blockers = list(validate_runtime_profile(profile_name, profile))
    warnings: list[str] = []
    requirements = profile.get("resource_requirements", {}) if isinstance(profile, Mapping) else {}
    if not isinstance(requirements, Mapping):
        requirements = {}
    required_gpu_ids = requirements.get("gpu_ids", requirements.get("gpus", []))
    if not isinstance(required_gpu_ids, list):
        required_gpu_ids = []

    overlay_devices: Mapping[str, Any] = {}
    if hardware_overlay is not None:
        overlay_errors = validate_hardware_overlay(hardware_overlay)
        blockers.extend(f"hardware_overlay: {error}" for error in overlay_errors)
        if not overlay_errors and isinstance(hardware_overlay.get("devices"), Mapping):
            overlay_devices = hardware_overlay["devices"]
    if required_gpu_ids:
        if hardware_overlay is None:
            blockers.append("gpu_identity_unresolved")
        else:
            missing = sorted(
                str(device_id)
                for device_id in required_gpu_ids
                if device_id not in overlay_devices
            )
            if missing:
                blockers.append("missing_gpu_ids:" + ",".join(missing))
            render_nodes = set(str(node) for node in facts.get("render_nodes", []))
            missing_nodes = sorted(
                str(device.get("render_node"))
                for device_id, device in overlay_devices.items()
                if device_id in required_gpu_ids
                and isinstance(device, Mapping)
                and device.get("render_node")
                and device.get("render_node") not in render_nodes
            )
            if missing_nodes:
                blockers.append("missing_render_nodes:" + ",".join(missing_nodes))

    if not facts.get("container_engines"):
        warnings.append("no_container_engine_detected")
    activation_status = profile.get("metadata", {}).get("activation_status") if isinstance(profile.get("metadata"), Mapping) else None
    if activation_status == "draft":
        blockers.append("profile_is_draft")
    elif activation_status:
        warnings.append(f"activation_status:{activation_status}")

    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "profile": profile_name,
        "ready": not blockers,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "required_gpu_ids": [str(item) for item in required_gpu_ids],
        "observed_render_nodes": sorted(str(item) for item in facts.get("render_nodes", [])),
        "container_engines": list(facts.get("container_engines", [])),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="runtime-profile JSON file")
    parser.add_argument("--overlay", type=Path, help="private hardware-overlay JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = _load_json(args.profile)
    profile_name = args.profile.name.removesuffix(".runtime-profile.json")
    overlay = _load_json(args.overlay) if args.overlay else None
    facts = collect_host_facts()
    result = preflight_profile(profile_name, profile, facts, hardware_overlay=overlay)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "OVERLAY_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "collect_host_facts",
    "preflight_profile",
    "validate_hardware_overlay",
]
