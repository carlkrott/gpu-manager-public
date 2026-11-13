"""Binding proof for the extracted memory policy (scripts/gpu_manager/memory_policy.py).

The second half of this file is the class-binding proof that gates the scheduler
extractions (sets 5-6).  It extracts nothing: ``bind_class`` below is the
candidate mechanism, applied to templates compiled from the *real* facade source
of ``_RestoreAdmissionGuard`` and ``_TargetStateUnreadable``.  See
binding-compat-classes.md for the mechanism, its limits and the test names.
"""
from __future__ import annotations

import ast
import asyncio
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

MIB = 1024 * 1024
GOOD = dict(
    memory_max_bytes=2000 * MIB,
    memory_high_bytes=1000 * MIB,
    memory_swap_max_bytes=100 * MIB,
    working_set_p99_mib=950,
    working_set_swap_p99_mib=90,
)


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during import proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
    """Load two fresh, unregistered facades under hermetic guards."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    # aiohttp subclasses socket.socket at import time; load it before the guard.
    import aiohttp.web  # noqa: F401

    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)

    loaded = []
    for name in ("memory_policy_facade_one", "memory_policy_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def test_functions_are_bound_to_their_own_facade(facades):
    one, two = facades
    for facade in facades:
        for name in (
            "_finite_positive_cap",
            "_positive_measurement",
            "_memory_cap_provenance",
            "_normalized_unit_name",
            "_is_cpu_llm_unit",
            "_validate_production_parallel_invariant",
        ):
            assert getattr(facade, name).__globals__ is vars(facade)
    assert one._memory_cap_provenance is not two._memory_cap_provenance
    assert one._memory_cap_provenance.__module__ == one.__name__


def test_signatures_and_defaults_are_preserved(facades):
    import inspect

    one = facades[0]
    sig = inspect.signature(one._memory_cap_provenance)
    assert list(sig.parameters) == [
        "memory_max_bytes",
        "memory_high_bytes",
        "memory_swap_max_bytes",
        "working_set_p99_mib",
        "working_set_swap_p99_mib",
        "aliases",
    ]
    assert all(
        sig.parameters[n].default is None for n in list(sig.parameters)[:5]
    )
    assert sig.parameters["aliases"].kind is inspect.Parameter.VAR_KEYWORD
    assert one._finite_positive_cap.__doc__.startswith("Normalize one cgroup")


def test_rebinding_facade_globals_after_import_is_seen(facades, monkeypatch):
    one = facades[0]
    assert one._memory_cap_provenance(**GOOD).valid is True
    # Shrink the facade MiB unit: the same byte caps now exceed the p99 envelope.
    monkeypatch.setattr(one, "_TASK8A_MIB", 1)
    assert one._memory_cap_provenance(**GOOD).valid is False
    # Replace a facade helper that the policy function reads at call time.
    monkeypatch.setattr(one, "_TASK8A_MIB", MIB)
    monkeypatch.setattr(one, "_finite_positive_cap", lambda value: None)
    result = one._memory_cap_provenance(**GOOD)
    assert result.reason == "max_not_finite"
    # The rebinding of the facade global is what the next call used.
    assert one._memory_cap_provenance.__globals__["_finite_positive_cap"] is one._finite_positive_cap


def test_two_facades_keep_isolated_state(facades, monkeypatch):
    one, two = facades
    monkeypatch.setattr(one, "_TASK8A_UINT64_MAX", 10)
    assert one._finite_positive_cap(11) is None
    assert two._finite_positive_cap(11) == 11
    assert one._memory_cap_provenance(**GOOD).reason == "max_not_finite"
    assert two._memory_cap_provenance(**GOOD).valid is True
    monkeypatch.setattr(two, "_TASK8A_MIB", 1)
    assert two._memory_cap_provenance(**GOOD).valid is False
    monkeypatch.setattr(one, "_TASK8A_UINT64_MAX", 2**64 - 1)
    assert one._memory_cap_provenance(**GOOD).valid is True


def test_existing_class_usage_is_compatible(facades):
    one, two = facades
    # Leaf classes are shared; identity and isinstance hold across facades.
    assert one._MemoryCapProvenance is two._MemoryCapProvenance
    assert one._MemoryCapRejectReason is two._MemoryCapRejectReason
    result = one._memory_cap_provenance(**GOOD)
    assert isinstance(result, one._MemoryCapProvenance)
    assert isinstance(result, dict)
    assert result.as_dict() == {
        "valid": True,
        "reason": "",
        "headroom_mib": 1000,
        "swap_envelope_mib": 112,
    }
    assert result.headroom_mib == 1000 and result.swap_envelope_mib == 112
    rejected = one._memory_cap_provenance(MemoryMax=1, memory_max_bytes=2)
    assert rejected.reason == one._MemoryCapRejectReason.CONFLICTING_ALIASES.value
    assert one._MemoryCapRejectReason("high_geq_max") is one._MemoryCapRejectReason.HIGH_GEQ_MAX


def test_cpu_unit_and_parallel_invariant_behaviour(facades):
    one = facades[0]
    assert one._normalized_unit_name(" Gemma-CPU.service ") == "gemma-cpu"
    assert one._normalized_unit_name(None) == ""
    assert one._is_cpu_llm_unit("gemma-cpu-vision.service") is True
    assert one._is_cpu_llm_unit("llm-primary") is False
    one._validate_production_parallel_invariant({"services": {"a": {"parallel": 4}}})
    cfg = {
        "strict_production_parallel": True,
        "services": {"a": {"type": "generation_backend", "queue_parallel": 2}},
    }
    with pytest.raises(ValueError, match="exceeds strict maximum 1"):
        one._validate_production_parallel_invariant(cfg)


def test_bind_requires_a_globals_dict():
    sys.path.insert(0, str(SCRIPTS))
    try:
        from gpu_manager import memory_policy
    finally:
        sys.path.remove(str(SCRIPTS))
    with pytest.raises(TypeError):
        memory_policy.bind(object())
    assert len(memory_policy.bind({})) == len(memory_policy.NAMES) == 8


# ---------------------------------------------------------------------------
# Class-binding proof (no extraction): per-bind factory class.
# ---------------------------------------------------------------------------


def bind_class(template, facade_globals):
    """Return a fresh class equivalent to ``template`` bound to ``facade_globals``.

    Every function reachable from the class dictionary (plain, staticmethod,
    classmethod, property accessors) is rebuilt with ``types.FunctionType`` so its
    ``__globals__`` is exactly ``facade_globals``.  A ``__class__`` closure cell
    (zero-argument ``super()``) is re-pointed at the new class.  Nothing is cached
    and ``sys.modules`` is never consulted.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    class_cell = types.CellType()

    def rebuild(function):
        closure = function.__closure__
        if closure is not None:
            closure = tuple(
                class_cell if name == "__class__" else cell
                for name, cell in zip(function.__code__.co_freevars, closure)
            )
        rebuilt = types.FunctionType(
            function.__code__,
            facade_globals,
            function.__name__,
            function.__defaults__,
            closure,
        )
        rebuilt.__kwdefaults__ = (
            function.__kwdefaults__.copy() if function.__kwdefaults__ else None
        )
        rebuilt.__annotations__ = function.__annotations__.copy()
        rebuilt.__doc__ = function.__doc__
        rebuilt.__qualname__ = function.__qualname__
        rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
        rebuilt.__dict__.update(function.__dict__)
        return rebuilt

    namespace = {}
    for key, value in vars(template).items():
        if key in ("__dict__", "__weakref__"):
            continue
        if isinstance(value, types.FunctionType):
            value = rebuild(value)
        elif isinstance(value, (staticmethod, classmethod)):
            value = type(value)(rebuild(value.__func__))
        elif isinstance(value, property):
            value = property(
                rebuild(value.fget) if value.fget else None,
                rebuild(value.fset) if value.fset else None,
                rebuild(value.fdel) if value.fdel else None,
                value.__doc__,
            )
        namespace[key] = value
    namespace["__module__"] = facade_globals.get("__name__", template.__module__)
    bound = type(template)(template.__name__, template.__bases__, namespace)
    class_cell.cell_contents = bound
    return bound


def _real_class_templates():
    """Compile the two real facade class definitions into scratch templates."""
    # A class moved out of the facade (scheduler-state) is read from its implementation module.
    wanted = {"_RestoreAdmissionGuard", "_TargetStateUnreadable"}
    nodes = []
    for path in (FACADE, SCRIPTS / "gpu_manager" / "scheduler_state.py"):
        found = {node.name for node in nodes}
        if found == wanted:
            break
        nodes += [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name in wanted - found
        ]
    assert {node.name for node in nodes} == wanted
    scratch = {"__name__": "class_model_scratch"}
    exec(compile(ast.Module(nodes, []), "<facade-class-models>", "exec"), scratch)
    return scratch["_RestoreAdmissionGuard"], scratch["_TargetStateUnreadable"]


class _FakeManager:
    def __init__(self):
        self._evict_lock = asyncio.Lock()


def _run(coro):
    """Drive a coroutine that never suspends (uncontended locks) without an event loop.

    The module guard denies socket creation and an event loop needs a socketpair.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("coroutine suspended; the proof only uses uncontended locks")


def test_class_models_come_from_real_facade_source_and_match_real_behaviour(facades):
    one = facades[0]
    guard, unreadable = _real_class_templates()
    # Same shape as the real classes defined in the facade.
    for model, real in ((guard, one._RestoreAdmissionGuard), (unreadable, one._TargetStateUnreadable)):
        assert model.__name__ == real.__name__
        assert model.__bases__ == real.__bases__
        assert model.__doc__ == real.__doc__
        assert sorted(vars(model)) == sorted(vars(real))
    # The unbound guard resolves _get_dual_swap_lock in the scratch namespace, not a facade.
    assert guard.__init__.__globals__ is not vars(one)
    with pytest.raises(NameError):
        guard(_FakeManager())
    # The facade-owned original reads the facade global: this is the precedent defect.
    assert one._RestoreAdmissionGuard.__init__.__globals__ is vars(one)


def test_bound_class_preserves_bases_metadata_and_function_globals(facades):
    one = facades[0]
    guard, unreadable = _real_class_templates()
    bound_guard = bind_class(guard, vars(one))
    bound_exc = bind_class(unreadable, vars(one))
    assert bound_guard is not guard and bound_exc is not unreadable
    assert bound_exc.__bases__ == (Exception,) and issubclass(bound_exc, Exception)
    assert bound_guard.__bases__ == (object,)
    assert bound_guard.__name__ == "_RestoreAdmissionGuard"
    assert bound_guard.__qualname__ == guard.__qualname__
    assert bound_guard.__module__ == one.__name__
    assert one.__name__ not in sys.modules  # unregistered facade: no sys.modules dependency
    assert bound_guard.__doc__ == guard.__doc__
    for name in ("__init__", "__aenter__", "__aexit__"):
        method = vars(bound_guard)[name]
        assert method.__globals__ is vars(one)
        assert inspect.iscoroutinefunction(method) == inspect.iscoroutinefunction(vars(guard)[name])
        assert str(inspect.signature(method)) == str(inspect.signature(vars(guard)[name]))
        assert method.__qualname__ == f"_RestoreAdmissionGuard.{name}"
    assert vars(bound_exc)["__init__"].__globals__ is vars(one)
    assert str(inspect.signature(bound_exc.__init__)) == "(self, group: 'str', gpu_id: 'str')"


class _DescriptorModel:
    """Synthetic model of every descriptor kind a scheduler class could carry."""

    shared_tag = ("immutable",)

    def __init__(self, value: int = 3, *, scale: float = 2.0) -> None:
        self._value = value
        self._scale = scale

    @staticmethod
    def static(x, y=4):
        return _MARKER_NAME + str(x + y)

    @classmethod
    def make(cls, *, scale: float = 5.0):
        return cls(scale=scale)

    @property
    def doubled(self):
        """Doubled value."""
        return _MARKER_NAME + str(self._value * 2)

    @doubled.setter
    def doubled(self, new):
        self._value = new // 2

    def read_marker(self):
        return _MARKER_NAME

    def lambda_marker(self):
        return (lambda: _MARKER_NAME)()


class _SuperBase:
    def describe(self):
        return "base"


class _SuperModel(_SuperBase):
    def describe(self):
        return "child+" + super().describe() + "+" + _MARKER_NAME


_MARKER_NAME = "template"


def test_descriptors_defaults_annotations_and_closure_cell_are_preserved():
    facade = {"__name__": "synthetic_facade", "_MARKER_NAME": "facade"}
    other = {"__name__": "other_facade", "_MARKER_NAME": "other"}
    bound = bind_class(_DescriptorModel, facade)
    other_bound = bind_class(_DescriptorModel, other)
    assert isinstance(vars(bound)["static"], staticmethod)
    assert isinstance(vars(bound)["make"], classmethod)
    assert isinstance(vars(bound)["doubled"], property)
    assert bound.static(1) == "facade5"
    assert other_bound.static(1) == "other5"
    assert bound.static.__defaults__ == (4,)
    assert bound.__init__.__defaults__ == (3,)
    assert bound.__init__.__kwdefaults__ == {"scale": 2.0}
    assert bound.__init__.__kwdefaults__ is not _DescriptorModel.__init__.__kwdefaults__
    assert bound.__init__.__annotations__ == _DescriptorModel.__init__.__annotations__
    assert bound.__init__.__annotations__ is not _DescriptorModel.__init__.__annotations__
    made = bound.make()
    assert type(made) is bound and made._scale == 5.0
    obj = bound(8)
    assert obj.doubled == "facade16"
    obj.doubled = 20
    assert obj._value == 10
    assert vars(bound)["doubled"].__doc__ == "Doubled value."
    assert obj.read_marker() == "facade" and obj.lambda_marker() == "facade"
    assert bound.shared_tag is _DescriptorModel.shared_tag
    assert _DescriptorModel(1).read_marker() == "template"  # template untouched
    # zero-argument super() uses the __class__ cell: it must name the bound class.
    super_bound = bind_class(_SuperModel, facade)
    assert super_bound.describe.__closure__[0].cell_contents is super_bound
    assert super_bound().describe() == "child+base+facade"
    assert _SuperModel().describe() == "child+base+template"


def test_target_state_unreadable_binding_keeps_exception_semantics(facades):
    one = facades[0]
    _, unreadable = _real_class_templates()
    bound = bind_class(unreadable, vars(one))
    # super().__init__ works only if the __class__ cell names the bound class.
    assert vars(bound)["__init__"].__closure__[0].cell_contents is bound
    exc = bound("comfyui-gen", "gpu1")
    assert isinstance(exc, bound) and isinstance(exc, Exception)
    assert (exc.group, exc.gpu_id) == ("comfyui-gen", "gpu1")
    assert str(exc) == "target state unreadable for group 'comfyui-gen' gpu 'gpu1'"
    assert exc.args == (str(exc),)
    assert bound.__mro__ == (bound, Exception, BaseException, object)
    with pytest.raises(bound):
        raise exc
    assert not isinstance(exc, unreadable)  # the template is not the bound class
    # traceback/chaining behave like a native exception class.
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise bound("g", "x") from inner
    except bound as caught:
        assert isinstance(caught.__cause__, ValueError)


def _warm_probe_patch(monkeypatch, facade):
    """Make the real _target_gpus_hold_warm_bundle hit its unreadable-state branch."""

    def boom(gpu_id):
        raise RuntimeError("cannot read " + gpu_id)

    monkeypatch.setattr(facade, "_group_reservations", {"grp": {"requires_gpus": ["gpuX"]}})
    monkeypatch.setattr(facade, "_resolve_group_target_gpus", lambda group: ["gpuX"])
    monkeypatch.setattr(facade, "_resolve_target_bundle_components", lambda group: (None, []))
    monkeypatch.setattr(facade, "_get_gpu_state", boom)


def test_exception_catching_uses_the_bound_facade_class(facades, monkeypatch):
    one, two = facades
    _, unreadable = _real_class_templates()
    bound_one = bind_class(unreadable, vars(one))
    bound_two = bind_class(unreadable, vars(two))
    monkeypatch.setattr(one, "_TargetStateUnreadable", bound_one)
    monkeypatch.setattr(two, "_TargetStateUnreadable", bound_two)
    for facade in (one, two):
        _warm_probe_patch(monkeypatch, facade)
    assert bound_one is not bound_two
    # The real facade function resolves the global at raise time: it raises each facade's class.
    with pytest.raises(bound_one) as caught_one:
        one._target_gpus_hold_warm_bundle("grp")
    assert type(caught_one.value) is bound_one
    assert (caught_one.value.group, caught_one.value.gpu_id) == ("grp", "gpuX")
    with pytest.raises(bound_two) as caught_two:
        two._target_gpus_hold_warm_bundle("grp")
    assert type(caught_two.value) is bound_two
    # An except clause naming the other facade's class does not catch it.
    with pytest.raises(bound_one):
        try:
            one._target_gpus_hold_warm_bundle("grp")
        except bound_two:  # pragma: no cover - must not match
            pytest.fail("caught by the wrong facade's exception class")
    assert not isinstance(caught_one.value, bound_two)
    # The facade's own evaluator path (`except _TargetStateUnreadable`) sees the bound class.
    assert one._TargetStateUnreadable is bound_one
    assert two._TargetStateUnreadable is bound_two


def test_returned_class_identity_is_stable_within_one_binding(facades, monkeypatch):
    one = facades[0]
    guard, unreadable = _real_class_templates()
    bound_guard = bind_class(guard, vars(one))
    monkeypatch.setattr(one, "_RestoreAdmissionGuard", bound_guard)
    manager = _FakeManager()
    first, second = one._RestoreAdmissionGuard(manager), one._RestoreAdmissionGuard(manager)
    assert type(first) is type(second) is bound_guard is one._RestoreAdmissionGuard
    assert isinstance(first, one._RestoreAdmissionGuard)
    assert not isinstance(first, guard)
    # Identity is per bind() call: the facade must call bind once and keep the result.
    assert bind_class(guard, vars(one)) is not bound_guard


def test_restore_guard_sees_replaced_dual_swap_lock_on_next_instance(facades, monkeypatch):
    one = facades[0]
    guard, _ = _real_class_templates()
    bound = bind_class(guard, vars(one))
    manager = _FakeManager()

    async def scenario():
        first = bound(manager)
        real_lock = first._admission_lock
        assert real_lock is one._get_dual_swap_lock()  # facade-global lookup, not a copy
        replacement = asyncio.Lock()
        monkeypatch.setattr(one, "_get_dual_swap_lock", lambda: replacement)
        second = bound(manager)  # constructed AFTER bind and AFTER the replacement
        assert second._admission_lock is replacement
        assert first._admission_lock is real_lock  # earlier instance is unchanged
        async with second:
            assert replacement.locked() and manager._evict_lock.locked()
            assert not real_lock.locked()
        assert not replacement.locked() and not manager._evict_lock.locked()
        # Failure path releases the admission lock and re-raises.
        class Boom:
            class _L:
                async def acquire(self):
                    raise RuntimeError("evict lock failed")

            _evict_lock = _L()

        third = bound(Boom())
        with pytest.raises(RuntimeError, match="evict lock failed"):
            await third.__aenter__()
        assert not replacement.locked()

    _run(scenario())


def test_two_facade_mappings_stay_isolated_for_the_guard(facades, monkeypatch):
    one, two = facades
    guard, _ = _real_class_templates()
    bound_one, bound_two = bind_class(guard, vars(one)), bind_class(guard, vars(two))
    assert bound_one is not bound_two
    sentinel = object()
    monkeypatch.setattr(one, "_get_dual_swap_lock", lambda: sentinel)
    manager = _FakeManager()
    assert bound_one(manager)._admission_lock is sentinel
    lock_two = bound_two(manager)._admission_lock
    assert isinstance(lock_two, asyncio.Lock) and lock_two is not sentinel
    assert lock_two is two._get_dual_swap_lock()
    assert lock_two is not one._dual_swap_lock
    # Facade two's lazily created lock lives in facade two's globals only.
    assert vars(two)["_dual_swap_lock"] is lock_two
    assert vars(one)["_dual_swap_lock"] is None


def test_bind_class_rejects_non_dict_and_does_not_touch_sys_modules(facades):
    guard, _ = _real_class_templates()
    with pytest.raises(TypeError):
        bind_class(guard, object())
    before = set(sys.modules)
    bind_class(guard, vars(facades[0]))
    assert set(sys.modules) == before
    assert facades[0].__name__ not in sys.modules
