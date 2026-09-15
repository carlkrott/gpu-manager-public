from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"
PORTABLE_DEFAULTS = ROOT / "scripts" / "portable_defaults.py"


def _source() -> str:
    return CONTROLLER.read_text(encoding="utf-8") + PORTABLE_DEFAULTS.read_text(encoding="utf-8")


def test_controller_has_neutral_default_identifiers_and_no_private_topology_literals():
    source = _source()
    assert 'GPU_MANAGER_PRIMARY_GPU_ID' in source
    assert 'GPU_MANAGER_DEFAULT_LLM_SERVICE' in source
    assert not re.search(r"(?i)radeon|\bgfx\d{3,4}\b|\b(?:rx|mi)\d{2,4}\b", source)
    assert "PCI_ID=1002:" not in source
    assert "PR4 /" not in source
    assert "PR6 (" not in source
    assert "PR10 /" not in source
    assert "0000:0a:00.0" not in source
    assert "renderD128" not in source
    assert "/home/" not in source
    assert "/mnt/" not in source


def test_default_registry_is_empty_and_durable_only():
    tree = ast.parse(_source())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEFAULT_SERVICES_CONFIG"
                for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "empty_services_config"


def test_fixed_hardware_ceiling_is_not_in_controller():
    source = _source()
    assert "_RX6950XT_VRAM_CEILING_MIB" not in source
    assert "fixed ceiling for the RX" not in source
    assert "physical-reserve" in source
