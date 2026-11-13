"""Binding proof for the extracted ``VRAMLifecycleManager`` (tranche 4, Set 1).

Drives the real ``gpu_manager.vram_lifecycle.bind`` through the real facade: the facade is loaded (twice, for
isolation) and every check reads the class the facade publishes.  Hermetic: no ``.git``, no network, no
subprocess, no ``torch``, no event loop.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import inspect
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"
MODULE = SCRIPTS / "gpu_manager" / "vram_lifecycle.py"
CLASS_NAME = "VRAMLifecycleManager"

# Frozen from the pre-extraction facade at da4c62d (1,234 lines, 688d87ea...); re-frozen after the
# intended audit 2026-11-12 B1-B4 edits (926d3c4, 60551e3). Re-freeze only for a deliberate class change.
ORIGINAL_CLASS_SHA256 = "4a23de2bd0957386bb82a5b805e7da681b804d6c9d584b29e0ccfd1783324474"
ORIGINAL_CLASS_LINES = 1278
METHODS = (
    "__init__", "mark_llm_evicted", "status", "drain_llm", "evict_llm", "restore_llm", "_comfyui_systemd_state",
    "_checked_comfyui_runtime_configured", "_configured_comfyui_unit", "_comfyui_unit_state",
    "_all_comfyui_units_stopped", "_any_comfyui_unit_active", "mark_comfyui_running_after_commit",
    "_unload_comfyui_models_for_restore", "_restart_acestep_for_restore", "stop_comfyui", "start_comfyui",
    "_ensure_comfyui_running", "_check_service_active", "_systemctl", "_wait_for_port_free",
    "_wait_for_vram_below", "wait_for_free_mib", "_detect_backend_for_gpu", "_pci_for_gpu", "_free_mib_for_async",
    "_wait_for_llm_healthy", "_free_comfyui", "_wait_for_comfyui_port_free", "_wait_for_comfyui_ready",
    "get_gpu_services", "evict_for_vram",
)
KINDS = {
    "_checked_comfyui_runtime_configured": staticmethod, "_configured_comfyui_unit": staticmethod,
    "_comfyui_unit_state": staticmethod, "_all_comfyui_units_stopped": classmethod,
    "_any_comfyui_unit_active": classmethod,
}
# Import-time values of the facade constants that the two definition-time defaults snapshot.
DRAIN_TIMEOUT_SNAPSHOT = 60
LLM_RESTORE_TIMEOUT_SNAPSHOT = 90
PUBLICATION_PATCHES = ("_wait_for_vram_below", "stop_comfyui", "wait_for_free_mib")


def _deny(*args, **kwargs):
    raise AssertionError("socket/process creation forbidden in binding proof")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FACADE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def scripts_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))


@pytest.fixture
def facades(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    return [_load("vram_lifecycle_facade_one"), _load("vram_lifecycle_facade_two")]


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("coroutine suspended")


def _raw(cls, name):
    return cls.__dict__[name]


def _function(cls, name):
    value = _raw(cls, name)
    return value.__func__ if isinstance(value, (staticmethod, classmethod)) else value


def _template():
    from gpu_manager import vram_lifecycle
    return vram_lifecycle.VRAMLifecycleManager


def check_binding(cls, facade_globals):
    """Raise AssertionError unless ``cls`` is a faithful class bound to ``facade_globals``."""
    template = _template()
    assert cls is not template
    assert cls.__name__ == template.__name__ == CLASS_NAME
    assert cls.__qualname__ == CLASS_NAME
    assert cls.__module__ == facade_globals["__name__"]
    assert cls.__doc__ == template.__doc__
    assert cls.__bases__ == (object,)
    names = tuple(k for k, v in vars(cls).items() if isinstance(v, (types.FunctionType, staticmethod, classmethod)))
    assert names == METHODS and len(names) == 32
    for name in METHODS:
        raw, ref = _raw(cls, name), _raw(template, name)
        assert type(raw) is KINDS.get(name, types.FunctionType), name
        assert type(ref) is type(raw)
        bound, original = _function(cls, name), _function(template, name)
        assert bound is not original
        assert bound.__globals__ is facade_globals, name
        assert bound.__code__ is original.__code__
        assert bound.__closure__ is None
        assert bound.__name__ == original.__name__ and bound.__qualname__ == original.__qualname__
        assert bound.__module__ == facade_globals["__name__"]
        assert inspect.signature(bound) == inspect.signature(original)
        assert bound.__defaults__ == original.__defaults__
        assert bound.__kwdefaults__ == original.__kwdefaults__
        assert bound.__annotations__ == original.__annotations__
        assert bound.__doc__ == original.__doc__
        assert inspect.iscoroutinefunction(bound) == inspect.iscoroutinefunction(original)
    assert _function(cls, "drain_llm").__defaults__ == (DRAIN_TIMEOUT_SNAPSHOT,)
    assert _function(cls, "_wait_for_llm_healthy").__defaults__ == (LLM_RESTORE_TIMEOUT_SNAPSHOT,)


def test_template_is_the_unchanged_original_class_source():
    source = MODULE.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == CLASS_NAME)
    segment = "\n".join(source.split("\n")[node.lineno - 1:node.end_lineno])
    assert len(segment.split("\n")) == ORIGINAL_CLASS_LINES
    assert hashlib.sha256(segment.encode("utf-8")).hexdigest() == ORIGINAL_CLASS_SHA256
    assert not node.decorator_list and not node.bases and not node.keywords


def test_template_is_unusable_unbound():
    with pytest.raises(NameError):
        _template()(None, None)  # __init__ resolves GpuState in the template module


def test_facade_owns_one_bound_class_assignment_and_no_class_statement(facades):
    tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    assert CLASS_NAME not in {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == CLASS_NAME for t in n.targets)
    ]
    assert len(assigns) == 1
    assert ast.unparse(assigns[0].value) == "_bind_vram_lifecycle(globals())"
    binds = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "_bind_vram_lifecycle"]
    assert len(binds) == 1  # the call only; the import alias is not a Name node
    for facade in facades:
        check_binding(facade.VRAMLifecycleManager, facade.__dict__)
        assert facade.__dict__["VRAMLifecycleManager"] is facade.VRAMLifecycleManager


def test_facades_get_distinct_classes_and_share_no_state(facades):
    one, two = facades
    assert one.VRAMLifecycleManager is not two.VRAMLifecycleManager
    assert _function(one.VRAMLifecycleManager, "stop_comfyui").__globals__ is one.__dict__
    assert _function(two.VRAMLifecycleManager, "stop_comfyui").__globals__ is two.__dict__
    assert _template().__dict__["stop_comfyui"].__globals__ is not one.__dict__
    assert "vram_lifecycle_facade_one" not in sys.modules


def constructor_sites(facade_tree):
    sites = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scopes = []

        def visit_FunctionDef(self, node):
            self.scopes.append(node.name)
            self.generic_visit(node)
            self.scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == CLASS_NAME:
                sites.append((self.scopes[-1], node))
            self.generic_visit(node)

    Visitor().visit(facade_tree)
    return sites


def test_all_five_constructor_sites_build_instances_of_the_published_class(facades, monkeypatch):
    facade = facades[0]
    sites = constructor_sites(ast.parse(FACADE.read_text(encoding="utf-8")))
    assert [scope for scope, _ in sites] == ["_mcp_call_tool", "_rollback_started_units", "_load_bundle", "_load_bundle", "create_app"]
    marker_session = object()
    for scope, call in sites:
        code = compile(ast.Expression(call), "<constructor-site>", "eval")
        instance = eval(code, facade.__dict__, {"session": marker_session})
        assert type(instance) is facade.VRAMLifecycleManager
        assert instance.logger is facade.logger
        assert instance.session is (None if ast.unparse(call.args[0]) == "None" else marker_session)
        assert instance.gpu_state is facade.GpuState.LLM_LOADED
        assert instance._gpu_evicted == {} and instance._llm_evicted is False
    # the facade-global rebinding pattern replaces what every site constructs
    class Fake:
        def __init__(self, session, logger):
            self.fake = True
    monkeypatch.setattr(facade, CLASS_NAME, Fake)
    for scope, call in sites:
        built = eval(compile(ast.Expression(call), "<constructor-site>", "eval"), facade.__dict__, {"session": None})
        assert type(built) is Fake


@pytest.mark.parametrize("name", PUBLICATION_PATCHES)
def test_publication_style_class_patch_reaches_actual_instances(facades, monkeypatch, name):
    facade = facades[0]
    instance = facade.VRAMLifecycleManager(None, facade.logger)
    before = getattr(instance, name)
    sentinel = lambda self, *args, **kwargs: ("patched", name)  # noqa: E731
    monkeypatch.setattr(facade.VRAMLifecycleManager, name, sentinel)
    assert getattr(instance, name)() == ("patched", name)
    assert facade.VRAMLifecycleManager.__dict__[name] is sentinel
    monkeypatch.undo()
    assert getattr(instance, name).__func__ is before.__func__


def test_internal_call_sees_class_patch(facades, monkeypatch):
    facade = facades[0]
    cls = facade.VRAMLifecycleManager
    seen = []

    async def fake_stop(self, *args, **kwargs):
        seen.append(("stop", type(self) is cls))
        return True

    monkeypatch.setattr(cls, "stop_comfyui", fake_stop)
    instance = cls(None, facade.logger)
    assert _run(instance.stop_comfyui()) is True
    assert seen == [("stop", True)]


def test_methods_resolve_facade_globals_late_and_write_back(facades, monkeypatch):
    one, two = facades
    patched = types.SimpleNamespace(LLM_LOADED="patched-state")
    monkeypatch.setattr(one, "GpuState", patched)
    assert one.VRAMLifecycleManager(None, one.logger).gpu_state == "patched-state"
    assert two.VRAMLifecycleManager(None, two.logger).gpu_state is two.GpuState.LLM_LOADED
    # `global _services_config` writes land in the facade mapping and only there
    config = {"services": {
        "b": {"enabled": True, "systemd_unit": "b.service", "priority": 2},
        "a": {"enabled": True, "systemd_unit": "a.service", "priority": 1},
        "cpu": {"enabled": True, "systemd_unit": "c.service", "cpu_only": True},
        "off": {"enabled": False, "systemd_unit": "o.service"},
    }}
    monkeypatch.setattr(one, "_services_config", None)
    monkeypatch.setattr(one, "_load_services_config", lambda: config)
    before_two = two._services_config
    manager = one.VRAMLifecycleManager(None, one.logger)
    assert [name for name, _ in manager.get_gpu_services()] == ["a", "b"]
    assert one._services_config is config
    assert two._services_config is before_two
    assert "_services_config" not in _template().__dict__ and not hasattr(sys.modules.get("gpu_manager.vram_lifecycle"), "_services_config")
    # a later facade rebinding is observed by the already-built instance
    monkeypatch.setattr(one, "_services_config", {"services": {}})
    assert manager.get_gpu_services() == []


def test_definition_time_defaults_are_import_time_snapshots(facades):
    one, two = facades
    facade_values = (one.DRAIN_TIMEOUT, one.LLM_RESTORE_TIMEOUT)
    assert facade_values == (DRAIN_TIMEOUT_SNAPSHOT, LLM_RESTORE_TIMEOUT_SNAPSHOT)
    one.DRAIN_TIMEOUT, one.LLM_RESTORE_TIMEOUT = 7, 8
    cls = one.VRAMLifecycleManager
    assert _function(cls, "drain_llm").__defaults__ == (DRAIN_TIMEOUT_SNAPSHOT,)
    assert _function(cls, "_wait_for_llm_healthy").__defaults__ == (LLM_RESTORE_TIMEOUT_SNAPSHOT,)
    assert inspect.signature(cls.drain_llm).parameters["timeout"].default == DRAIN_TIMEOUT_SNAPSHOT
    assert inspect.signature(cls._wait_for_llm_healthy).parameters["timeout"].default == LLM_RESTORE_TIMEOUT_SNAPSHOT
    # a second bind of the same template keeps the snapshots and never reads the (changed) facade values
    from gpu_manager import vram_lifecycle
    rebound = vram_lifecycle.bind(two.__dict__)
    assert _function(rebound, "drain_llm").__defaults__ == (DRAIN_TIMEOUT_SNAPSHOT,)
    two_values = (two.DRAIN_TIMEOUT, two.LLM_RESTORE_TIMEOUT)
    assert two_values == facade_values


def test_bind_requires_the_facade_dictionary_and_returns_a_fresh_class():
    from gpu_manager import vram_lifecycle
    with pytest.raises(TypeError):
        vram_lifecycle.bind(types.SimpleNamespace())
    with pytest.raises(TypeError):
        vram_lifecycle.bind(types.MappingProxyType({}))
    mapping = {"__name__": "isolated_vram_mapping"}
    first, second = vram_lifecycle.bind(mapping), vram_lifecycle.bind(mapping)
    assert first is not second and first is not vram_lifecycle.VRAMLifecycleManager
    check_binding(first, mapping)
    assert vram_lifecycle.NAMES == (CLASS_NAME,)


def test_polling_helpers_stay_in_the_facade(facades):
    tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vram_loop_should_poll"] == ["_vram_loop_should_poll"]
    assert any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VRAM_IDLE_POLL_EVERY" for t in n.targets)
        for n in tree.body
    )
    module_tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    defined = set()
    for node in module_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert not {"_vram_loop_should_poll", "VRAM_IDLE_POLL_EVERY", "_poll_vram"} & defined
    assert facades[0]._vram_loop_should_poll.__globals__ is facades[0].__dict__


def test_no_template_consumers_outside_the_facade_and_this_proof():
    owners = []
    for path in sorted(list((REPO / "scripts").rglob("*.py")) + list((REPO / "tests").rglob("*.py"))):
        if path == MODULE or path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if "vram_lifecycle" in text:
            owners.append(path.relative_to(REPO).as_posix())
    # only the facade imports the template module; nothing reads the unbound template class
    assert owners == ["scripts/gpu-manager.py"], owners
    facade_tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    imports = [
        n for n in ast.walk(facade_tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("vram_lifecycle")
    ]
    assert [(n.module, [(a.name, a.asname) for a in n.names]) for n in imports] == [
        ("gpu_manager.vram_lifecycle", [("bind", "_bind_vram_lifecycle")])
    ]


# --- negative controls: defective bindings must fail the same check the real binding passes -------------------

def _defective(mutate):
    template = _template()
    mapping = {"__name__": "defective_mapping"}
    from gpu_manager import vram_lifecycle
    cls = vram_lifecycle.bind(mapping)
    mutate(template, cls, mapping)
    return cls, mapping


def test_negative_control_template_globals_are_detected():
    def mutate(template, cls, mapping):
        setattr(cls, "stop_comfyui", template.__dict__["stop_comfyui"])
    cls, mapping = _defective(mutate)
    with pytest.raises(AssertionError):
        check_binding(cls, mapping)


def test_negative_control_dropped_defaults_are_detected():
    def mutate(template, cls, mapping):
        _function(cls, "drain_llm").__defaults__ = None
    cls, mapping = _defective(mutate)
    with pytest.raises(AssertionError):
        check_binding(cls, mapping)


def test_negative_control_changed_snapshot_is_detected():
    def mutate(template, cls, mapping):
        _function(cls, "_wait_for_llm_healthy").__defaults__ = (LLM_RESTORE_TIMEOUT_SNAPSHOT + 1,)
    cls, mapping = _defective(mutate)
    with pytest.raises(AssertionError):
        check_binding(cls, mapping)


def test_negative_control_stripped_descriptor_is_detected():
    def mutate(template, cls, mapping):
        setattr(cls, "_configured_comfyui_unit", _function(cls, "_configured_comfyui_unit"))
    cls, mapping = _defective(mutate)
    with pytest.raises(AssertionError):
        check_binding(cls, mapping)


def test_negative_control_copied_mapping_misses_late_facade_rebinding():
    facade_globals = {"__name__": "copy_probe", "GpuState": types.SimpleNamespace(LLM_LOADED="first"), "asyncio": asyncio}
    from gpu_manager import vram_lifecycle
    good = vram_lifecycle.bind(facade_globals)
    copied = vram_lifecycle.bind(dict(facade_globals))
    facade_globals["GpuState"] = types.SimpleNamespace(LLM_LOADED="second")
    assert good(None, None).gpu_state == "second"
    assert copied(None, None).gpu_state == "first"  # a snapshot mapping would be caught by the late-rebinding checks


def test_negative_control_template_class_construction_is_detected():
    with pytest.raises(NameError):
        _template()(None, None)
