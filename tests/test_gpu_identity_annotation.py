"""GPU identity in /vram: PCI identity from device/uevent, configured gpu_id/name.

Live symptom: every /vram row showed gpu_name "Unknown AMD
GPU" and pci_address null, so no row got a gpu_id and every dashboard GPU chip
said "No bundle".  Cause: ``/sys/class/drm/cardN/uevent`` is the DRM minor's
uevent (MAJOR/MINOR/DEVNAME) and the reader preferred it over
``cardN/device/uevent``, which holds PCI_ID and PCI_SLOT_NAME.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"


def _deny(*args, **kwargs):
    raise AssertionError("process creation forbidden in GPU identity tests")


@pytest.fixture
def facade(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    spec = importlib.util.spec_from_file_location("gpu_identity_facade", FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRM = "/sys/class/drm"

# Real sysfs layout: the direct cardN/uevent is the DRM minor's uevent.
FAKE_SYSFS = {
    f"{DRM}/card0/uevent": "MAJOR=226\nMINOR=0\nDEVNAME=dri/card0\nDEVTYPE=drm_minor",
    f"{DRM}/card0/device/uevent": (
        "DRIVER=amdgpu\nPCI_CLASS=30000\nPCI_ID=1002:AAA1\n"
        "PCI_SUBSYS_ID=1002:0001\nPCI_SLOT_NAME=0000:0a:00.0"
    ),
    f"{DRM}/card0/device/vendor": "0x1002",
    f"{DRM}/card0/device/device": "0xaaa1",
    f"{DRM}/card0/device/mem_info_vram_total": str(16 * 1024**3),
    f"{DRM}/card0/device/mem_info_vram_used": str(15 * 1024**3),
    f"{DRM}/card1/uevent": "MAJOR=226\nMINOR=1\nDEVNAME=dri/card1\nDEVTYPE=drm_minor",
    f"{DRM}/card1/device/uevent": (
        "DRIVER=amdgpu\nPCI_CLASS=30000\nPCI_ID=1002:BBB2\n"
        "PCI_SUBSYS_ID=1002:0002\nPCI_SLOT_NAME=0000:0b:00.0"
    ),
    f"{DRM}/card1/device/vendor": "0x1002",
    f"{DRM}/card1/device/device": "0xbbb2",
    f"{DRM}/card1/device/mem_info_vram_total": str(16 * 1024**3),
    f"{DRM}/card1/device/mem_info_vram_used": str(14 * 1024**3),
}

GPU_DEVICES = {
    "gpu_a": {"card": "card0", "pci": "0000:0a:00.0", "name": "Accelerator A"},
    "gpu_b": {"card": "card1", "pci": "0000:0b:00.0", "name": "Accelerator B"},
}


@pytest.fixture
def fake_sysfs(facade, monkeypatch):
    real_exists, real_isfile, real_listdir = os.path.exists, os.path.isfile, os.listdir

    def exists(path):
        p = str(path)
        if p.startswith(DRM):
            return p in FAKE_SYSFS
        return real_exists(path)

    def isfile(path):
        p = str(path)
        if p.startswith(DRM):
            return p in FAKE_SYSFS
        return real_isfile(path)

    def listdir(path):
        p = str(path)
        if p == DRM:
            return ["card0", "card1", "card0-DP-1", "renderD128"]
        if p.startswith(DRM):
            raise FileNotFoundError(p)
        return real_listdir(path)

    def read_text(path):
        return FAKE_SYSFS.get(str(path)) if path else None

    def read_int(path):
        text = read_text(path)
        try:
            return int(str(text).strip(), 0) if text is not None else None
        except ValueError:
            return None

    monkeypatch.setattr(os.path, "exists", exists)
    monkeypatch.setattr(os.path, "isfile", isfile)
    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(facade, "_read_sysfs_text", read_text)
    monkeypatch.setattr(facade, "_read_sysfs_int", read_int)
    return facade


def test_pci_uevent_prefers_device_uevent_over_drm_minor(fake_sysfs):
    text = fake_sysfs._read_card_pci_uevent("card0")
    assert "PCI_SLOT_NAME=0000:0a:00.0" in text
    assert fake_sysfs._read_card_pci_uevent("card9") == ""


def test_detect_gpus_reports_pci_address_and_device_id(fake_sysfs):
    rows = {row["card"]: row for row in fake_sysfs._detect_gpus()}
    assert set(rows) == {"card0", "card1"}
    assert rows["card0"]["pci_address"] == "0000:0a:00.0"
    assert rows["card1"]["pci_address"] == "0000:0b:00.0"
    # Without product_name the name falls back to the PCI device id, never
    # to "Unknown AMD GPU" when the PCI identity is readable.
    assert rows["card0"]["gpu_name"] == "AMD GPU (AAA1)"
    assert rows["card1"]["gpu_name"] == "AMD GPU (BBB2)"


def test_annotate_sets_configured_gpu_id_and_name(fake_sysfs):
    gpus = fake_sysfs._annotate_gpu_identity(fake_sysfs._detect_gpus(), GPU_DEVICES)
    rows = {row["card"]: row for row in gpus}
    assert rows["card0"]["gpu_id"] == "gpu_a"
    assert rows["card0"]["gpu_name"] == "Accelerator A"
    assert rows["card1"]["gpu_id"] == "gpu_b"
    assert rows["card1"]["gpu_name"] == "Accelerator B"


def test_annotate_never_card_matches_against_a_disagreeing_pci(facade):
    gpus = [{"card": "card0", "pci_address": "0000:0c:00.0", "gpu_name": "AMD GPU (CCC3)"}]
    facade._annotate_gpu_identity(gpus, GPU_DEVICES)
    assert "gpu_id" not in gpus[0]
    assert gpus[0]["gpu_name"] == "AMD GPU (CCC3)"


def test_annotate_falls_back_to_card_without_pci_and_keeps_real_names(facade):
    gpus = [
        {"card": "card1", "pci_address": None, "gpu_name": "Unknown AMD GPU"},
        {"card": "card0", "pci_address": "0000:0a:00.0", "gpu_name": "Vendor Product A"},
        {"card": "card2", "pci_address": "0000:0c:00.0", "gpu_name": "AMD GPU (CCC3)"},
    ]
    facade._annotate_gpu_identity(gpus, GPU_DEVICES)
    assert gpus[0]["gpu_id"] == "gpu_b"
    assert gpus[0]["gpu_name"] == "Accelerator B"
    assert gpus[1]["gpu_id"] == "gpu_a"
    assert gpus[1]["gpu_name"] == "Vendor Product A"  # a real detected name wins
    assert "gpu_id" not in gpus[2]  # unconfigured third GPU stays unmapped


def test_dashboard_bundle_chip_uses_gpu_id_and_rebuilds_on_bundle_change():
    source = FACADE.read_text()
    assert "_bundleIndicatorHtml(bundleKey)" in source
    assert "const bundleKey = gpu.gpu_id || gpuId;" in source
    assert "tab.dataset.sig !== _gpuTabSig(gpus[i], i)" in source
