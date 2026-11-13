"""The scoped token may name configured local routing groups, nothing else."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def test_scoped_backends_are_members_plus_configured_groups(monkeypatch):
    spec = importlib.util.spec_from_file_location("candidate_gpm_scoped", ROOT / "scripts" / "gpu-manager.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_services_config", {"combined_gemma_broker": {"ordered_members": ["A"]}})
    assert m._scoped_token_backends() == {"A"}
    monkeypatch.setattr(m, "_services_config", {
        "combined_gemma_broker": {"ordered_members": ["A"]},
        "scoped_token_routing_groups": ["small.chat", "", 3]})
    assert m._scoped_token_backends() == {"A", "small.chat"}
    # the broker ingress still only pins real members
    assert "small.chat" not in m._combined_gemma_member_names()
