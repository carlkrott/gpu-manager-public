"""Binding proof for Enum classes and plain classes with facade-global method dependencies.

``bind_class`` below is the candidate mechanism: it rebuilds a template class so every
function in it resolves globals through a supplied facade dictionary, with an
Enum-specific path that drives ``EnumType.__prepare__`` instead of handing the metaclass a
plain dict.  The ``GpuState`` template is compiled from ``GPU_STATE_SOURCE``, an immutable
literal of the pre-extraction facade class, so the proof does not move with the facade or
with ``bundle_model.py``; small synthetic sources add the shapes ``GpuState`` lacks
(methods, aliases, ``super()``, a ``str`` mixin).  Every assertion compares the candidate
with the *baseline*: the class defined directly in a facade's globals.  The proof is
location-independent: it holds before extraction (``GpuState`` defined in the facade) and
after it (``bundle_model`` bound into the facade) and rejects any hybrid.  See
binding-compat-enum.md for the mechanism, limits and test names.
"""
from __future__ import annotations

import ast
import copy
import enum
import importlib.util
import os
import pickle
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FACADE = SCRIPTS / "gpu-manager.py"
BUNDLE_MODEL = SCRIPTS / "gpu_manager" / "bundle_model.py"
BUNDLE_CLASSES = ("GpuState", "GpuBundleState", "BundleCoordinator")
BIND_STATEMENT = "GpuState, GpuBundleState, BundleCoordinator = _bind_bundle_model(globals())"


def _probe_enum_internals():
    """Keys this interpreter's EnumType synthesises on every enum class.

    They are rebuilt by the metaclass, never copied.  The set differs between versions
    (3.14 adds ``_hashable_values_``, ``__firstlineno__`` ...), so it is probed from an
    empty enum rather than listed.  ``_generate_next_value_`` is left out: a user-supplied
    one must be kept, so ``bind_class`` compares it with the default instead.
    """

    class Probe(enum.Enum):
        MEMBER = "member"

    keys = set(vars(Probe)) - {"MEMBER", "__module__", "__doc__", "_generate_next_value_"}
    assert {"_member_map_", "_member_names_", "_value2member_map_", "__new__"} <= keys
    return frozenset(keys)


_ENUM_INTERNAL = _probe_enum_internals()

ENUM_SOURCE = '''
class Mode(enum.Enum):
    """Mode doc."""
    FAST = "fast"
    QUICK = "fast"
    SLOW = "slow"

    def label(self):
        return helper(("label", self.value))

    @classmethod
    def _missing_(cls, value):
        return cls.SLOW if value == "default" else None

    @property
    def shout(self):
        return helper(("shout", self.name))

    @staticmethod
    def static(x):
        return helper(("static", x))

    def parent_repr(self):
        return super().__repr__()


class Tag(str, enum.Enum):
    RED = "red"
    BLUE = "blue"

    def tagged(self):
        return helper(("tagged", str(self.value)))
'''

# Complete GpuState class text copied from the pre-set facade (tranche 2 sets 0-4 HEAD,
# scripts/gpu-manager.py).  An immutable literal: the comparison baseline is never read
# from bundle_model.py or from a facade that may have moved.
GPU_STATE_SOURCE = '''
class GpuState(Enum):
    """GPU state machine states.

    LLM_LOADED    → Qwen3.6 on GPU, serving requests (default idle state)
    DRAINING      → Waiting for active LLM requests to finish before evicting
    GPU_WORK      → ComfyUI or ACE-Step using GPU
    LLM_RELIEF    → Temporary LLM restore to clear queued request backlog
    RESTORING     → Qwen3.6 reloading to GPU after GPU work complete
    """
    LLM_LOADED = "llm_loaded"
    DRAINING = "draining"
    GPU_WORK = "gpu_work"
    LLM_RELIEF = "llm_relief"
    RESTORING = "restoring"
'''

CLASS_SOURCE = '''
class Meter:
    """Plain class."""
    unit = "m"

    def __init__(self, n):
        self.n = n
        self.label = helper(("init", n))

    def __repr__(self):
        return "Meter(%r)" % (self.n,)

    def show(self):
        return helper(("show", self.n))

    @classmethod
    def make(cls, n):
        return cls(helper(("make", n)))

    @staticmethod
    def static(x):
        return helper(("static", x))

    @property
    def doubled(self):
        return helper(("doubled", self.n * 2))

    def parent_repr(self):
        return super().__repr__()
'''


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
    """Two fresh, unregistered real facades under hermetic guards."""
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401 -- aiohttp subclasses socket.socket at import time

    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    loaded = []
    for name in ("enum_binding_facade_one", "enum_binding_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


# ---------------------------------------------------------------------------
# Candidate mechanism
# ---------------------------------------------------------------------------


def _unwrap(value):
    """The function behind ``value``; 3.14 stores ``_generate_next_value_`` as a staticmethod."""
    return getattr(value, "__func__", value)


def bind_class(template, facade_globals):
    """Return a fresh class equivalent to ``template`` bound to ``facade_globals``."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    class_cell = types.CellType()
    module_name = facade_globals.get("__name__", template.__module__)

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
        rebuilt.__module__ = module_name
        rebuilt.__dict__.update(function.__dict__)
        return rebuilt

    def convert(value):
        if isinstance(value, types.FunctionType):
            return rebuild(value)
        if isinstance(value, (staticmethod, classmethod)):
            return type(value)(rebuild(value.__func__))
        if type(value) is property:
            return property(
                rebuild(value.fget) if value.fget else None,
                rebuild(value.fset) if value.fset else None,
                rebuild(value.fdel) if value.fdel else None,
                value.__doc__,
            )
        return value

    meta = type(template)
    if isinstance(template, enum.EnumMeta):
        namespace = meta.__prepare__(template.__name__, template.__bases__)
        members = template._member_map_
        default_next = _unwrap(vars(enum.Enum)["_generate_next_value_"]).__code__
        for key, value in vars(template).items():
            if key in _ENUM_INTERNAL or key in members:
                continue
            if key == "_generate_next_value_" and _unwrap(value).__code__ is default_next:
                continue
            namespace[key] = convert(value)
        namespace["__qualname__"] = template.__qualname__
        namespace["__module__"] = module_name
        # Members last, in definition order; an alias re-states its canonical value.
        for name, member in members.items():
            namespace[name] = member._value_
    else:
        namespace = {}
        for key, value in vars(template).items():
            if key in ("__dict__", "__weakref__"):
                continue
            namespace[key] = convert(value)
        namespace["__module__"] = module_name
    bound = meta(template.__name__, template.__bases__, namespace)
    class_cell.cell_contents = bound
    return bound


def _compile_templates(source, names, **extra):
    scratch = {"__name__": "binding_template_scratch", "enum": enum, **extra}
    exec(compile(source, "<binding-template>", "exec"), scratch)
    return {name: scratch[name] for name in names}


def _real_gpu_state_template():
    """GpuState compiled from the literal pre-set facade text, never from a live file."""
    scratch = {"__name__": "binding_template_scratch", "Enum": enum.Enum}
    exec(compile(GPU_STATE_SOURCE, "<gpu-state-literal>", "exec"), scratch)
    return scratch["GpuState"]


def _class_defs(path, names=None):
    """Names of every class definition (at any depth) in ``path``, in source order."""
    tree = ast.parse(path.read_text())
    found = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return found if names is None else [n for n in found if n in names]


def _bind_statements(path):
    tree = ast.parse(path.read_text())
    return [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and ast.unparse(n) == BIND_STATEMENT
    ]


def _fake_facade(name):
    return {"__name__": name, "enum": enum, "helper": lambda value: (name, value)}


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _outcome(call):
    try:
        return ("ok", call())
    except Exception as exc:  # noqa: BLE001 -- the failure itself is the observation
        return ("raised", type(exc).__name__, str(exc))


def _enum_shape(cls):
    return {
        "name": cls.__name__,
        "qualname": cls.__qualname__,
        "module": cls.__module__,
        "doc": cls.__doc__,
        "metaclass": type(cls),
        "bases": cls.__bases__,
        "member_names": [m.name for m in cls],
        "member_values": [m.value for m in cls],
        "member_map_keys": list(cls.__members__),
        "alias_targets": {k: v.name for k, v in cls.__members__.items()},
        "member_type": cls._member_type_,
        "use_args": cls._use_args_,
        "value_map_keys": list(cls._value2member_map_),
        "class_repr": repr(cls),
        "member_reprs": [repr(m) for m in cls],
        "member_strs": [str(m) for m in cls],
        "member_formats": [format(m) for m in cls],
        "len": len(cls),
        "dir_members": sorted(set(dir(cls)) - set(dir(enum.Enum))),
        "reduce_ex": [m.__reduce_ex__(2)[1] for m in cls],
    }


def _pickle_outcome(obj):
    return _outcome(lambda: pickle.dumps(obj))


# ---------------------------------------------------------------------------
# Proof: real GpuState on two real facades
# ---------------------------------------------------------------------------


def test_real_source_shape_and_no_extraction(facades):
    """Exactly one of two states holds; anything in between fails.

    Before extraction the facade defines ``GpuState`` once and there is no
    ``bundle_model``.  After extraction the facade defines none of the three classes, binds
    them with one assignment and ``bundle_model.py`` defines exactly those three.
    """
    literal_tree = ast.parse(GPU_STATE_SOURCE)
    extracted = BUNDLE_MODEL.exists()
    if not extracted:
        assert _class_defs(FACADE, ("GpuState",)) == ["GpuState"]
        assert _bind_statements(FACADE) == []
        facade_class = [
            n
            for n in ast.walk(ast.parse(FACADE.read_text()))
            if isinstance(n, ast.ClassDef) and n.name == "GpuState"
        ]
        # the literal is a verbatim copy of the facade class while it is still there
        assert ast.dump(facade_class[0]) == ast.dump(literal_tree.body[0])
    else:
        assert _class_defs(FACADE, BUNDLE_CLASSES) == []
        assert _bind_statements(FACADE) == [BIND_STATEMENT]
        assert sorted(_class_defs(BUNDLE_MODEL, BUNDLE_CLASSES)) == sorted(BUNDLE_CLASSES)
    template = _real_gpu_state_template()
    assert [m.name for m in template] == [
        "LLM_LOADED",
        "DRAINING",
        "GPU_WORK",
        "LLM_RELIEF",
        "RESTORING",
    ]
    template_shape = _enum_shape(template)
    template_shape.pop("module")  # scratch module name; each facade has its own
    assert template_shape.pop("bases") == (enum.Enum,)
    for facade in facades:
        assert facade.GpuState.__module__ == facade.__name__
        assert facade.__name__ not in sys.modules
        # the real facade class equals the independently compiled literal baseline
        real_shape = _enum_shape(facade.GpuState)
        real_shape.pop("module")
        assert real_shape.pop("bases") == (enum.Enum,)
        assert real_shape == template_shape


def test_gpu_state_binding_matches_baseline_shape_and_identity(facades):
    one, two = facades
    template = _real_gpu_state_template()
    for facade in facades:
        baseline = facade.GpuState
        candidate = bind_class(template, vars(facade))
        assert candidate is not baseline and candidate is not template
        # Everything except the class objects themselves is equal to the baseline.
        base_shape, cand_shape = _enum_shape(baseline), _enum_shape(candidate)
        assert cand_shape.pop("bases") == base_shape.pop("bases") == (enum.Enum,)
        assert cand_shape == base_shape
        assert candidate.__members__.keys() == baseline.__members__.keys()
        for name in baseline.__members__:
            base_member, cand_member = baseline[name], candidate[name]
            # identity relationships inside one class
            assert candidate[name] is getattr(candidate, name)
            assert candidate(base_member.value) is cand_member
            assert type(cand_member) is candidate
            assert isinstance(cand_member, candidate) and isinstance(cand_member, enum.Enum)
            # members never cross between baseline and candidate
            assert cand_member is not base_member
            assert not isinstance(cand_member, baseline)
            assert not isinstance(base_member, candidate)
            assert cand_member != base_member
            assert cand_member.name == base_member.name
            assert cand_member.value == base_member.value
            assert hash(cand_member) == hash(candidate[name]) and cand_member in set(candidate)
            assert copy.copy(cand_member) is cand_member
            assert copy.deepcopy(cand_member) is cand_member
        # lookup failures match the baseline text
        assert _outcome(lambda: baseline("missing"))[:2] == ("raised", "ValueError")
        assert _outcome(lambda: candidate("missing")) == _outcome(lambda: baseline("missing"))
        assert _outcome(lambda: baseline["MISSING"])[:2] == ("raised", "KeyError")
        assert _outcome(lambda: candidate["MISSING"]) == _outcome(lambda: baseline["MISSING"])
        # immutability matches the baseline
        for cls in (baseline, candidate):
            assert _outcome(lambda cls=cls: setattr(cls, "DRAINING", 1))[:2] == (
                "raised",
                "AttributeError",
            )
            assert _outcome(lambda cls=cls: setattr(cls.DRAINING, "value", 1))[0] == "raised"
    # two bindings of the same template are isolated from each other
    cand_one = bind_class(template, vars(one))
    cand_two = bind_class(template, vars(two))
    assert cand_one is not cand_two
    assert cand_one.LLM_LOADED is not cand_two.LLM_LOADED
    assert not isinstance(cand_one.LLM_LOADED, cand_two)
    assert cand_one.__module__ == one.__name__ != cand_two.__module__ == two.__name__
    assert one.GpuState is not two.GpuState


def test_gpu_state_pickle_matches_baseline_when_facade_is_unregistered(facades):
    template = _real_gpu_state_template()
    for facade in facades:
        baseline = facade.GpuState
        candidate = bind_class(template, vars(facade))
        for name in baseline.__members__:
            base_result = _pickle_outcome(baseline[name])
            cand_result = _pickle_outcome(candidate[name])
            # The baseline itself cannot pickle (module not importable): a precondition,
            # not a candidate success.  The candidate must fail identically.
            assert base_result[0] == "raised"
            assert base_result[1] == "PicklingError", base_result
            assert facade.__name__ in base_result[2]
            assert cand_result == base_result
        assert _pickle_outcome(baseline) == _pickle_outcome(candidate)
        assert facade.__name__ not in sys.modules


# ---------------------------------------------------------------------------
# Proof: Enum and plain class with facade-global method dependencies
# ---------------------------------------------------------------------------


def _baseline_and_candidates(source, names):
    """Baseline classes (defined in the facade dict) and two candidate bindings."""
    one, two = _fake_facade("enum_fake_facade_one"), _fake_facade("enum_fake_facade_two")
    exec(compile(source, "<baseline>", "exec"), one)
    templates = _compile_templates(source, names, helper=None)
    baseline = {n: one[n] for n in names}
    cand_one = {n: bind_class(templates[n], one) for n in names}
    cand_two = {n: bind_class(templates[n], two) for n in names}
    return one, two, templates, baseline, cand_one, cand_two


def test_enum_with_facade_global_methods_matches_baseline():
    one, two, templates, baseline, cand_one, cand_two = _baseline_and_candidates(
        ENUM_SOURCE, ("Mode", "Tag")
    )
    for name in ("Mode", "Tag"):
        base, cand = baseline[name], cand_one[name]
        base_shape, cand_shape = _enum_shape(base), _enum_shape(cand)
        base_bases, cand_bases = base_shape.pop("bases"), cand_shape.pop("bases")
        assert cand_bases == base_bases
        assert cand_shape == base_shape
        assert cand is not base and cand is not templates[name]
    mode, bmode, mode2 = cand_one["Mode"], baseline["Mode"], cand_two["Mode"]
    # alias identity: QUICK is FAST, not a separate member, in baseline and candidate
    assert bmode.QUICK is bmode.FAST and mode.QUICK is mode.FAST
    assert list(mode.__members__) == list(bmode.__members__) == ["FAST", "QUICK", "SLOW"]
    assert [m.name for m in mode] == [m.name for m in bmode] == ["FAST", "SLOW"]
    # facade-global resolution: baseline and candidate agree for the same facade ...
    assert mode.FAST.label() == bmode.FAST.label() == ("enum_fake_facade_one", ("label", "fast"))
    assert mode.SLOW.shout == bmode.SLOW.shout == ("enum_fake_facade_one", ("shout", "SLOW"))
    assert mode.static(3) == bmode.static(3) == ("enum_fake_facade_one", ("static", 3))
    assert mode.FAST.parent_repr() == bmode.FAST.parent_repr() == "<Mode.FAST: 'fast'>"
    # ... classmethod _missing_ runs again and returns this binding's member
    assert mode("default") is mode.SLOW and bmode("default") is bmode.SLOW
    assert _outcome(lambda: mode("nope"))[:2] == _outcome(lambda: bmode("nope"))[:2]
    # str mixin keeps str behaviour and identity rules
    tag, btag = cand_one["Tag"], baseline["Tag"]
    assert tag.RED == "red" and btag.RED == "red"
    assert isinstance(tag.RED, str) and tag("red") is tag.RED
    assert tag.RED.tagged() == btag.RED.tagged() == ("enum_fake_facade_one", ("tagged", "red"))
    # second binding resolves its own facade global and is a distinct class
    assert mode2 is not mode and mode2.FAST is not mode.FAST
    assert mode2.FAST.label() == ("enum_fake_facade_two", ("label", "fast"))
    assert mode.FAST.label() == ("enum_fake_facade_one", ("label", "fast"))
    assert mode.FAST.label.__func__.__globals__ is one
    assert mode2.FAST.label.__func__.__globals__ is two
    # replacing the facade global after bind is seen on the next call (call-time lookup)
    one["helper"] = lambda value: ("replaced", value)
    assert mode.FAST.label() == bmode.FAST.label() == ("replaced", ("label", "fast"))
    assert mode2.FAST.label() == ("enum_fake_facade_two", ("label", "fast"))


def test_plain_class_with_facade_global_methods_matches_baseline():
    one, two, templates, baseline, cand_one, cand_two = _baseline_and_candidates(
        CLASS_SOURCE, ("Meter",)
    )
    base, cand, cand2 = baseline["Meter"], cand_one["Meter"], cand_two["Meter"]
    assert cand is not base and cand is not templates["Meter"] and cand2 is not cand
    assert (cand.__name__, cand.__qualname__, cand.__module__, cand.__doc__, cand.__bases__) == (
        base.__name__,
        base.__qualname__,
        base.__module__,
        base.__doc__,
        base.__bases__,
    )
    b, c, c2 = base(2), cand(2), cand2(2)
    assert c.label == b.label == ("enum_fake_facade_one", ("init", 2))
    assert c2.label == ("enum_fake_facade_two", ("init", 2))
    assert (c.show(), c.doubled, repr(c)) == (b.show(), b.doubled, repr(b))
    assert cand.make(5).n == base.make(5).n == ("enum_fake_facade_one", ("make", 5))
    assert type(cand.make(5)) is cand
    assert cand.static(1) == base.static(1)
    assert c.parent_repr().split(" object at ")[0] == b.parent_repr().split(" object at ")[0]
    assert c.parent_repr().startswith("<enum_fake_facade_one.Meter object at ")
    assert c.__class__ is cand and isinstance(c, cand) and not isinstance(c, base)
    assert cand.unit is base.unit
    assert cand.show.__globals__ is one and cand2.show.__globals__ is two
    # call-time lookup of a facade global replaced after bind
    one["helper"] = lambda value: ("replaced", value)
    assert c.show() == b.show() == ("replaced", ("show", 2))
    assert c2.show() == ("enum_fake_facade_two", ("show", 2))
    # pickle: baseline itself is unregistered, so equivalence means equal failure
    base_result, cand_result = _pickle_outcome(b), _pickle_outcome(c)
    assert base_result[:2] == ("raised", "PicklingError"), base_result
    assert cand_result == base_result
    assert "enum_fake_facade_one" not in sys.modules


def test_enum_pickle_matches_baseline_when_unregistered():
    one, two, templates, baseline, cand_one, cand_two = _baseline_and_candidates(
        ENUM_SOURCE, ("Mode", "Tag")
    )
    for name in ("Mode", "Tag"):
        for member in baseline[name]:
            base_result = _pickle_outcome(member)
            cand_result = _pickle_outcome(cand_one[name][member.name])
            assert base_result[:2] == ("raised", "PicklingError"), base_result
            assert cand_result == base_result
        # Second binding fails the same way, naming its own module.
        second = _pickle_outcome(cand_two[name][next(iter(baseline[name])).name])
        assert second[:2] == ("raised", "PicklingError")
        assert "enum_fake_facade_two" in second[2]
    assert not {"enum_fake_facade_one", "enum_fake_facade_two"} & set(sys.modules)


# ---------------------------------------------------------------------------
# Expected negative results (what the mechanism must do to work)
# ---------------------------------------------------------------------------


def test_naive_dict_rebuild_of_an_enum_fails():
    """Handing EnumType a plain dict (the class-binding recipe for plain classes) fails."""
    template = _compile_templates(ENUM_SOURCE, ("Mode",), helper=None)["Mode"]
    outcome = _outcome(
        lambda: type(template)(template.__name__, template.__bases__, dict(vars(template)))
    )
    assert outcome[:2] == ("raised", "AttributeError"), outcome


def test_unbound_template_and_unrebuilt_functions_do_not_see_the_facade():
    templates = _compile_templates(ENUM_SOURCE + CLASS_SOURCE, ("Mode", "Meter"), helper=None)
    # the template's own globals have no working helper: the call fails with TypeError (None)
    assert _outcome(lambda: templates["Mode"].FAST.label())[:2] == ("raised", "TypeError")
    assert _outcome(lambda: templates["Meter"](1))[:2] == ("raised", "TypeError")
    # a template compiled without the name at all raises NameError, the t_9bc85058 precedent
    bare = {"__name__": "bare_scratch", "enum": enum}
    exec(compile(ENUM_SOURCE, "<bare>", "exec"), bare)
    assert _outcome(lambda: bare["Mode"].FAST.label())[:2] == ("raised", "NameError")
    # a bound class has the facade as __globals__ for every method; the template does not
    facade = _fake_facade("enum_negative_facade")
    bound = bind_class(templates["Mode"], facade)
    assert bound.FAST.label.__func__.__globals__ is facade
    assert templates["Mode"].FAST.label.__func__.__globals__ is not facade


def test_bind_class_rejects_non_dict_and_does_not_touch_sys_modules():
    before = dict(sys.modules)
    template = _compile_templates(ENUM_SOURCE, ("Mode",), helper=None)["Mode"]
    with pytest.raises(TypeError):
        bind_class(template, types.SimpleNamespace(helper=None))
    bind_class(template, _fake_facade("enum_untouched_modules_facade"))
    assert dict(sys.modules) == before
