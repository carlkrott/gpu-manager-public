from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


def _load_controller():
    spec = importlib.util.spec_from_file_location("gfx_arch_probe_gpu_manager", CONTROLLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_kfd(sys_root: Path, nodes: dict[str, tuple[int, int]]) -> None:
    """Create a fake /sys/class/kfd topology: node name -> (gfx_target_version, drm_render_minor)."""
    for name, (gfx, render_minor) in nodes.items():
        node_dir = sys_root / "class" / "kfd" / "kfd" / "topology" / "nodes" / name
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "properties").write_text(
            f"cpu_cores_count 0\ngfx_target_version {gfx}\ndrm_render_minor {render_minor}\n",
            encoding="utf-8",
        )


def _write_fake_drm_card(sys_root: Path, card: str, render_minor: int) -> None:
    render = sys_root / "class" / "drm" / card / "device" / "drm" / f"renderD{render_minor}"
    render.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def fake_sysfs(tmp_path: Path) -> Path:
    sys_root = tmp_path / "sys"
    _write_fake_kfd(sys_root, {
        "0": (0, 0),          # CPU node, no GPU target
        "1": (100300, 128),   # gfx10.3.0 part on render minor 128
        "2": (90006, 129),    # gfx9.0.6 part on render minor 129
        "3": (100306, 130),   # gfx10.3.6 part on render minor 130
    })
    _write_fake_drm_card(sys_root, "card1", 128)
    _write_fake_drm_card(sys_root, "card2", 129)
    _write_fake_drm_card(sys_root, "card3", 130)
    return sys_root


def test_kfd_target_version_converts_to_gfx_arch():
    module = _load_controller()
    assert module._kfd_gfx_arch(100300) == "gfx1030"
    assert module._kfd_gfx_arch(90006) == "gfx906"
    assert module._kfd_gfx_arch(100306) == "gfx1036"
    assert module._kfd_gfx_arch(0) is None


def test_card_arch_is_read_from_faked_kfd_topology(fake_sysfs: Path):
    module = _load_controller()
    assert module._gfx_arch_for_card("card1", "", sys_root=str(fake_sysfs)) == "gfx1030"
    assert module._gfx_arch_for_card("card2", "", sys_root=str(fake_sysfs)) == "gfx906"
    assert module._gfx_arch_for_card("card3", "", sys_root=str(fake_sysfs)) == "gfx1036"


def test_missing_render_node_yields_no_kfd_arch(fake_sysfs: Path):
    module = _load_controller()
    assert module._kfd_gfx_arch_for_render_minor(None, str(fake_sysfs)) is None
    assert module._kfd_gfx_arch_for_render_minor(999, str(fake_sysfs)) is None
    assert module._drm_render_minor("card9", str(fake_sysfs)) is None


def test_hint_file_is_empty_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_controller()
    monkeypatch.delenv("GPU_MANAGER_GFX_ARCH_MAP", raising=False)
    assert module._load_gfx_arch_map() == {}
    assert module._gfx_arch_bf16_unsupported() == set()

    hints = tmp_path / "hints.json"
    hints.write_text('{"bf16_unsupported": ["gfx1030"]}', encoding="utf-8")
    monkeypatch.setenv("GPU_MANAGER_GFX_ARCH_MAP", str(hints))
    assert module._gfx_arch_bf16_unsupported() == {"gfx1030"}
