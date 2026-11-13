"""Facade-bound GPU topology and tenant-detection helpers.

Function bodies are templates: collaborators, regexes, caches, and telemetry
resolve from each facade's globals at call time.
"""
from __future__ import annotations

from types import FunctionType

NAMES = (
    "_normalise_pcie",
    "_scan_render_node_fdinfo_cached",
    "_build_render_node_to_gpu_map_cached",
    "_read_proc_name",
    "_scan_render_node_fdinfo",
    "_render_node_to_gpu_id",
    "_build_render_node_to_gpu_map",
    "_gpu_has_active_generation_tenant",
)


async def _normalise_pcie(addr: str | None) -> str:
    """Canonicalise a PCI Express BDF address from operator-owned input.

    Driver and sysfs sources may emit the same slot in different case,
    prefix, or short-form representations. This normaliser produces a single
    canonical form
    ``dddd:bb:dd.f`` (4-hex domain, 2-hex bus, 2-hex device, 1-hex
    function) so equality comparisons across these forms succeed.

    Returns the empty string for empty / unparseable input.
    """
    if not addr:
        return ""
    s = str(addr).strip()
    if not s:
        return ""
    # Find the first substring of the form hex:hex:hex(.hex)?.
    # The string may contain extra text (sysfs path, etc.); we
    # search the whole string for the first BDF-like sequence.
    import re as _re
    m = _re.search(
        r"([0-9a-fA-F]{1,4}):([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})(?:\.([0-9a-fA-F]))?",
        s,
    )
    if not m:
        return ""
    domain_s, bus_s, dev_s, fn_s = m.groups()
    try:
        domain = int(domain_s, 16)
        bus = int(bus_s, 16)
        dev = int(dev_s, 16)
        func = int(fn_s, 16) if fn_s else 0
    except ValueError:
        return ""
    return f"{domain:04x}:{bus:02x}:{dev:02x}.{func:x}"
def _scan_render_node_fdinfo_cached() -> dict:
    """Cached wrapper around _scan_render_node_fdinfo (1s TTL)."""
    import time as _time
    now = _time.monotonic()
    if _vulkan_scan_cache["data"] is not None and (now - _vulkan_scan_cache["ts"]) < _VULKAN_SCAN_TTL:
        return _vulkan_scan_cache["data"]
    data = _scan_render_node_fdinfo()
    _vulkan_scan_cache["data"] = data
    _vulkan_scan_cache["ts"] = now
    return data
def _build_render_node_to_gpu_map_cached(gpu_devices: dict) -> dict:
    """Cached wrapper around _build_render_node_to_gpu_map (1s TTL)."""
    import time as _time
    now = _time.monotonic()
    if _vulkan_rn_map_cache["data"] is not None and (now - _vulkan_rn_map_cache["ts"]) < _VULKAN_RN_MAP_TTL:
        return _vulkan_rn_map_cache["data"]
    data = _build_render_node_to_gpu_map(gpu_devices)
    _vulkan_rn_map_cache["data"] = data
    _vulkan_rn_map_cache["ts"] = now
    return data
def _read_proc_name(pid: int) -> str:
    """Read the process name from /proc/<pid>/comm. Returns '' on error."""
    try:
        with open(f"/proc/{pid}/comm", "rb") as f:
            return f.read().decode("utf-8", errors="replace").strip()
    except (OSError, IOError):
        return ""
def _scan_render_node_fdinfo() -> dict:
    """Walk /proc/*/fd/ and return per-pid per-rendernode drm-vm + amdgpu VRAM usage.

    The drm-vm-size line in /proc/<pid>/fdinfo/<fd> is per GEM-object fd,
    regardless of whether the user-space driver is ROCm (KFD), Vulkan
    (RADV/amdvlk), OpenGL, or any other GEM-using API. This means a Vulkan
    process holding render-node GEM fds will be visible here even though
    amdgpu_top's per-process fdinfo only attributes to the KFD-visible
    allocation.

    Returns: { pid: { render_node_path: { drm_vm_bytes,
                                         amdgpu_vram_bytes, fd_count } } }
    No sudo required — /proc/<pid>/fdinfo/* is world-readable.
    """
    import os as _os
    result: dict = {}
    seen_clients: set = set()  # dup'd fds share one drm-client-id and report the same memory
    # shortcut: keyed per pid, so a render fd inherited across fork counts once per pid; drop pid from the key if forked GPU tenants appear
    try:
        for entry in _os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            fd_dir = f"/proc/{entry}/fd"
            try:
                fd_names = _os.listdir(fd_dir)
            except (OSError, IOError):
                continue
            for fd_name in fd_names:
                try:
                    target = _os.readlink(f"{fd_dir}/{fd_name}")
                except (OSError, IOError):
                    continue
                if not _RENDER_NODE_RE.match(target):
                    continue
                try:
                    with open(f"/proc/{entry}/fdinfo/{fd_name}", "rb") as f:
                        info = f.read().decode("utf-8", errors="replace")
                except (OSError, IOError):
                    continue
                drm_vm = 0
                amdgpu_vram = 0
                # Standard DRM fdinfo (kernel 7.x amdgpu prints only these): "drm-total-vram: 13439248 KiB".
                units = {"": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}  # functions are rebound into the facade: no module globals
                std = {}
                client = None
                for line in info.splitlines():
                    key, _, value = line.partition(":")
                    parts = value.split()
                    try:
                        num = int(parts[0]) * units.get(parts[1] if len(parts) > 1 else "", 1)
                    except (ValueError, IndexError):
                        continue
                    if key == "drm-vm-size":
                        drm_vm += num
                    elif key == "amdgpu-memory-vram":
                        amdgpu_vram += num
                    elif key in ("drm-total-vram", "drm-total-gtt"):
                        std[key] = num
                    elif key == "drm-client-id":
                        client = num
                if not drm_vm and not amdgpu_vram and std:
                    amdgpu_vram = std.get("drm-total-vram", 0)
                    drm_vm = amdgpu_vram + std.get("drm-total-gtt", 0)
                if drm_vm == 0 and amdgpu_vram == 0:
                    continue
                if client is not None:
                    if (pid, target, client) in seen_clients:
                        continue
                    seen_clients.add((pid, target, client))
                result.setdefault(pid, {}).setdefault(target, {
                    "drm_vm_bytes": 0,
                    "amdgpu_vram_bytes": 0,
                    "fd_count": 0,
                })
                result[pid][target]["drm_vm_bytes"] += drm_vm
                result[pid][target]["amdgpu_vram_bytes"] += amdgpu_vram
                result[pid][target]["fd_count"] += 1
    except OSError:
        pass
    return result
def _render_node_to_gpu_id(render_node: str, gpu_devices: dict) -> str | None:
    """Map an operator-owned render-node path to a configured logical GPU via PCI metadata."""
    import os as _os
    try:
        device_link = _os.readlink(f"/sys/class/drm/{_os.path.basename(render_node)}/device")
    except OSError:
        return None
    m = _PCI_BDF_RE.search(device_link)
    if not m:
        return None
    target_pci = m.group(1)
    for gid, cfg in gpu_devices.items():
        if cfg.get("pci") == target_pci:
            return gid
    return None
def _build_render_node_to_gpu_map(gpu_devices: dict) -> dict:
    """Build /dev/dri/renderD* -> gpu_id map for all known render nodes."""
    import os as _os
    result: dict = {}
    try:
        for entry in _os.listdir("/sys/class/drm"):
            if not entry.startswith("renderD"):
                continue
            gid = _render_node_to_gpu_id(f"/dev/dri/{entry}", gpu_devices)
            if gid:
                result[f"/dev/dri/{entry}"] = gid
    except OSError:
        pass
    return result
async def _gpu_has_active_generation_tenant(gpu_id: str | None) -> tuple[bool, str]:
    """Return (is_blocked, reason) — True if a generation backend is using this GPU.
    Checks: (a) any generation_backend with matching gpu_id has active systemd unit,
    (b) total VRAM used on the GPU > 2 GiB,
    (c) Vulkan/OpenGL/CL tenant holding GEM fds (drm-vm > 512 MiB)."""
    if not gpu_id:
        return False, ""
    config = _load_services_config()
    services = config.get("services", {})
    for name, svc in services.items():
        if svc.get("type") != "generation_backend":
            continue
        if svc.get("gpu_id") != gpu_id:
            continue
        unit = svc.get("systemd_unit", "")
        if not unit:
            continue
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "is-active", unit,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if stdout.decode().strip() == "active":
            return True, f"generation backend '{name}' ({unit}) is active on GPU {gpu_id}"
    # VRAM threshold check
    # B3 BUGFIX: build a normalized lookup from gpu_devices config so we
    # match by pcie address (or card name) from the vram poll, NOT by a
    # fragile substring match on a free-form gpu_name.  PR10 also
    # routes every comparison through ``_normalise_pcie()`` so that
    # the various BDF encodings (lspci -D, sysfs, sysfs path,
    # short-form) all collapse to a single canonical form before
    # comparison.
    try:
        gpu_devices = config.get("gpu_devices", {}) or {}
        # target_pcie: set of normalised pcie addresses.
        target_pcie: set[str] = set()
        # card -> gpu_id (fallback match for hosts where the
        # gpu_devices config has no ``pci`` field).
        target_card: dict[str, str] = {}
        for dev_id, dev_cfg in gpu_devices.items():
            raw_pci = dev_cfg.get("pci") or dev_cfg.get("pcie") or ""
            n_pci = await _normalise_pcie(raw_pci)
            if n_pci:
                target_pcie.add(n_pci)
            card = dev_cfg.get("card") or ""
            if card:
                target_card[card] = dev_id

        # Direct match: gpu_id itself is a normalised pcie address.
        gpu_id_norm = await _normalise_pcie(gpu_id)
        vram_data = await _poll_vram()
        # Find the GPU entry that matches the caller's gpu_id, either
        # by pcie address or by card name.
        target_gpu = None
        for gpu in vram_data.get("gpus", []):
            gpu_pci = await _normalise_pcie(gpu.get("pci_address") or "")
            gpu_card = gpu.get("card") or ""
            if gpu_id_norm and gpu_pci and gpu_pci == gpu_id_norm:
                target_gpu = gpu
                break
            if gpu_card and gpu_card in target_card and target_card[gpu_card] == gpu_id:
                target_gpu = gpu
                break
        if target_gpu is not None:
            used = target_gpu.get("vram_used_mib", 0) or 0
            if used > 2048:
                return True, f"GPU {gpu_id} VRAM {used} MiB > 2 GiB threshold"
    except Exception as e:
        logger.debug(f"B3 gate: VRAM threshold check failed: {e}")
    # Vulkan-tenant check: walk render-node fdinfo to find processes
    # holding GEM fds on the target GPU. ROCm processes won't show up
    # here (they use KFD, not GEM fds), but Vulkan/OpenGL/CL processes
    # will. This catches cases like a multi-GPU Vulkan backend where
    # the user-space allocation is ~8 GB of Vulkan and invisible to
    # amdgpu_top's KFD fdinfo path.
    try:
        config_v = _load_services_config()
        gpu_devices_v = config_v.get("gpu_devices", {}) or {}
        rn_map = _build_render_node_to_gpu_map_cached(gpu_devices_v)
        if gpu_id in rn_map.values():
            scan = await asyncio.get_event_loop().run_in_executor(
                None, _scan_render_node_fdinfo_cached
            )
            for pid, rn_data in scan.items():
                for rn, info in rn_data.items():
                    if rn_map.get(rn) != gpu_id:
                        continue
                    drm_vm_mib = info["drm_vm_bytes"] // (1024 * 1024)
                    if drm_vm_mib > 512:
                        name = _read_proc_name(pid) or f"pid={pid}"
                        return True, (
                            f"Vulkan tenant {name} (pid={pid}, "
                            f"{drm_vm_mib} MiB GEM-fd) on GPU {gpu_id}"
                        )
    except Exception as e:
        logger.debug(f"B3 gate: Vulkan tenant check failed: {e}")
    return False, ""


def bind(facade_globals):
    """Return topology helpers rebuilt against one facade namespace."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    bound = []
    for name in NAMES:
        function = namespace[name]
        rebuilt = FunctionType(
            function.__code__, facade_globals, function.__name__,
            function.__defaults__, function.__closure__,
        )
        rebuilt.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
        rebuilt.__annotations__ = function.__annotations__.copy()
        rebuilt.__doc__ = function.__doc__
        rebuilt.__qualname__ = function.__qualname__
        rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
        rebuilt.__dict__.update(function.__dict__)
        bound.append(rebuilt)
    return tuple(bound)
