"""ModelValidator must never suggest a model absent from its own registry."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

_CONTROLLER = Path(__file__).resolve().parents[2] / "scripts" / "gpu-manager.py"
sys.path.insert(0, str(_CONTROLLER.parent))


def _load():
    spec = importlib.util.spec_from_file_location("mv_registry_gpu_manager", _CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inst(ctx, cx="high", vision=False):
    return {"context_length": ctx, "capabilities": {"min_complexity": cx, "vision": vision}}


def _exists(cfg, name):
    p = cfg["providers"]
    if name in p.get("local", {}).get("instances", {}):
        return True
    if "/" in name:
        prov, model = name.split("/", 1)
        return model in p.get(prov, {}).get("models", {})
    return False


def _configs():
    small = {"small-a": _inst(4096, "low"), "small-b": _inst(8192, "low")}
    big = {"big-a": _inst(32768), "big-b": _inst(131072)}
    cloud = {"cloud": {"type": "cloud", "models": {"m1": _inst(1000)}}}
    rules = [
        {"hint": "h1", "require": "retired-model"},
        {"hint": "h2", "never_fallback_to": ["small-a"], "prefer": "retired-model"},
        {"hint": "h3", "never_fallback_to": ["small-a"], "require": "big-b"},
    ]
    for inst in (small, {**small, **big}):
        yield {"providers": {"local": {"instances": inst}, **cloud}, "routing_rules": rules}


def test_suggested_model_always_in_registry():
    mod = _load()
    seen_suggestion = False
    for cfg in _configs():
        v = mod.ModelValidator(cfg)
        names = list(cfg["providers"]["local"]["instances"])
        for name, hint, req in itertools.product(
            names, [None, "h1", "h2", "h3"], [None, {"min_complexity": "high"}]
        ):
            res = v.validate(name, task_hint=hint, requirements=req)
            sug = res.get("suggested_model")
            if sug is not None:
                seen_suggestion = True
                assert _exists(cfg, sug), (name, hint, req, res)
    assert seen_suggestion


def test_no_eligible_member_does_not_invent_one():
    mod = _load()
    cfg = {"providers": {"local": {"instances": {"s": _inst(4096, "low")}}}, "routing_rules": []}
    res = mod.ModelValidator(cfg).validate("s", requirements={"min_complexity": "high"})
    assert res == {"ok": True, "action": "route"}


def test_picks_largest_context_eligible_member():
    mod = _load()
    cfg = {"providers": {"local": {"instances": {
        "s": _inst(4096, "low"), "a": _inst(8192), "b": _inst(65536)}}}, "routing_rules": []}
    res = mod.ModelValidator(cfg).validate("s", requirements={"min_complexity": "high"})
    assert res["suggested_model"] == "b" and res["action"] == "upgrade"
