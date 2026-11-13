"""Hermetic binding proof for facade-owned GPU topology helpers."""
from __future__ import annotations

import importlib.util
import inspect
import io
import os
import socket
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"
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


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def bound_facades(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from gpu_manager import gpu_topology

    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    import aiohttp.web  # noqa: F401 -- load socket subclasses before the guard
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(subprocess, "run", _deny)
    monkeypatch.setattr(os, "system", _deny)

    facades = []
    for name in ("topology_facade_one", "topology_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        facade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(facade)
        facades.append(facade)
    return gpu_topology, facades


def test_functions_bind_to_independent_facade_globals_and_preserve_metadata(bound_facades):
    module, (one, two) = bound_facades
    assert module.NAMES == NAMES
    assert len(module.bind({})) == len(NAMES)
    for facade in (one, two):
        for name in NAMES:
            function, template = getattr(facade, name), getattr(module, name)
            assert function is not template
            assert function.__globals__ is vars(facade)
            assert function.__module__ == facade.__name__
            assert function.__qualname__ == template.__qualname__
            assert function.__doc__ == template.__doc__
            assert function.__annotations__ == template.__annotations__
            assert inspect.signature(function) == inspect.signature(template)
            assert function.__code__ is template.__code__
        assert facade.__name__ not in __import__("sys").modules
    for name in NAMES:
        assert getattr(one, name) is not getattr(two, name)


def test_binding_uses_facade_local_caches_and_late_replacements(bound_facades):
    _, (one, two) = bound_facades
    calls = {"one": 0, "two": 0}
    one._scan_render_node_fdinfo = lambda: calls.__setitem__("one", calls["one"] + 1) or {"one": 1}
    two._scan_render_node_fdinfo = lambda: calls.__setitem__("two", calls["two"] + 1) or {"two": 2}
    assert one._scan_render_node_fdinfo_cached() == {"one": 1}
    assert one._scan_render_node_fdinfo_cached() == {"one": 1}
    assert two._scan_render_node_fdinfo_cached() == {"two": 2}
    assert calls == {"one": 1, "two": 1}
    assert vars(one)["_vulkan_scan_cache"] is not vars(two)["_vulkan_scan_cache"]


def test_proc_and_render_node_scans_use_synthetic_inputs(bound_facades, monkeypatch):
    _, (facade, _) = bound_facades
    listing = {
        "/proc": ["42", "self", "43", "44"],
        "/proc/42/fd": ["3", "4"],
        "/proc/43/fd": ["5"],
        "/proc/44/fd": ["6", "7"],
    }
    links = {
        "/proc/42/fd/3": "/dev/dri/renderD128",
        "/proc/42/fd/4": "/dev/null",
        "/proc/43/fd/5": "/dev/dri/renderD129",
        "/proc/44/fd/6": "/dev/dri/renderD129",
        "/proc/44/fd/7": "/dev/dri/renderD129",
    }
    contents = {
        "/proc/42/fdinfo/3": b"drm-vm-size: 1048576\namdgpu-memory-vram: 2048\n",
        "/proc/43/fdinfo/5": b"drm-vm-size: invalid\namdgpu-memory-vram: 0\n",
        # kernel 7.x amdgpu: only the standard DRM keys, with units (Breccia, 11-13)
        "/proc/44/fdinfo/6": b"drm-driver:\tamdgpu\ndrm-client-id:\t88\ndrm-total-gtt:\t1024 KiB\ndrm-total-vram:\t13439248 KiB\n",
        # dup'd fd of the same DRM client (Xwayland has 4): counted once
        "/proc/44/fdinfo/7": b"drm-driver:\tamdgpu\ndrm-client-id:\t88\ndrm-total-gtt:\t1024 KiB\ndrm-total-vram:\t13439248 KiB\n",
        "/proc/42/comm": b"synthetic-renderer\n",
    }
    monkeypatch.setattr(os, "listdir", lambda path: listing[path])
    monkeypatch.setattr(os, "readlink", lambda path: links[path])
    facade.open = lambda path, mode="r", **kwargs: io.BytesIO(contents[path])

    assert facade._scan_render_node_fdinfo() == {
        42: {"/dev/dri/renderD128": {"drm_vm_bytes": 1048576, "amdgpu_vram_bytes": 2048, "fd_count": 1}},
        44: {"/dev/dri/renderD129": {"drm_vm_bytes": (13439248 + 1024) * 1024,
                                     "amdgpu_vram_bytes": 13439248 * 1024, "fd_count": 1}},
    }
    assert facade._read_proc_name(42) == "synthetic-renderer"


def test_render_node_mapping_and_gpu_tenant_poll_are_stubbed(bound_facades, monkeypatch):
    _, (facade, _) = bound_facades
    monkeypatch.setattr(os, "listdir", lambda path: ["renderD128", "card0"])
    monkeypatch.setattr(os, "readlink", lambda path: "/devices/pci0000:01/0000:01:00.0" if path.endswith("/device") else "")
    facade._gpu_has_active_generation_tenant  # binding presence; no host probe
    assert facade._render_node_to_gpu_id("/dev/dri/renderD128", {"gpu0": {"pci": "0000:01:00.0"}}) == "gpu0"
    assert facade._build_render_node_to_gpu_map({"gpu0": {"pci": "0000:01:00.0"}}) == {"/dev/dri/renderD128": "gpu0"}

    polls = []
    async def poll():
        polls.append(True)
        return {"gpus": []}
    facade._load_services_config = lambda: {"services": {}, "gpu_devices": {}}
    facade._poll_vram = poll
    facade._build_render_node_to_gpu_map_cached = lambda devices: {}
    coroutine = facade._gpu_has_active_generation_tenant("gpu0")
    with pytest.raises(StopIteration) as result:
        coroutine.send(None)
    assert result.value.value == (False, "")
    assert polls == [True]


def test_binding_rejects_non_dict(bound_facades):
    module, _ = bound_facades
    with pytest.raises(TypeError):
        module.bind(object())
