"""Binding and behavior proof for facade-bound job model classes."""
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


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
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
    result = []
    for name in ("job_models_facade_one", "job_models_facade_two"):
        spec = importlib.util.spec_from_file_location(name, FACADE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("coroutine unexpectedly suspended in hermetic proof")


def test_real_source_shape_and_one_time_facade_binding(facades):
    source = FACADE.read_text()
    tree = ast.parse(source)
    top_level_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    assert not {"Job", "AceStepJob", "LLMProxy"} & top_level_classes
    assert "Job, AceStepJob, LLMProxy = _bind_job_models(globals())" in source
    module_path = SCRIPTS / "gpu_manager" / "job_models.py"
    module_tree = ast.parse(module_path.read_text())
    templates = {n.name for n in module_tree.body if isinstance(n, ast.ClassDef)}
    assert templates == {"Job", "AceStepJob", "LLMProxy"}
    one, two = facades
    for facade in facades:
        for cls in (facade.Job, facade.AceStepJob, facade.LLMProxy):
            assert cls.__module__ == facade.__name__
            for value in vars(cls).values():
                funcs = []
                if isinstance(value, types.FunctionType):
                    funcs.append(value)
                elif isinstance(value, property):
                    funcs.extend(f for f in (value.fget, value.fset, value.fdel) if f)
                for function in funcs:
                    assert function.__globals__ is vars(facade)
    assert one.Job is not two.Job
    assert one.AceStepJob is not two.AceStepJob
    assert one.LLMProxy is not two.LLMProxy
    assert one.Job is one.Job and one.Job.__bases__ == (object,)


def test_properties_metadata_and_job_collaborators_are_rebound(facades, monkeypatch):
    one = facades[0]
    calls = []
    monkeypatch.setattr(one, "extract_model_key", lambda workflow: calls.append("key") or "k")
    monkeypatch.setattr(one, "extract_model_fingerprint", lambda workflow: calls.append("fingerprint") or "f")
    monkeypatch.setattr(one, "priority_to_tier", lambda priority: types.SimpleNamespace(value=f"tier:{priority}"))
    job = one.Job({"nodes": {}}, priority=7, metadata={"source": "test"})
    assert job.model_key == "k" and job.fingerprint == "f"
    assert calls == ["key", "fingerprint"]
    assert job.priority_tier.value == "tier:7"
    assert job.to_dict()["priority_tier"] == "tier:7"
    assert job.to_dict()["metadata"] == {"source": "test"}
    assert isinstance(vars(one.Job)["priority_tier"], property)
    assert vars(one.Job)["priority_tier"].fget.__globals__ is vars(one)
    # The second facade resolves its own late-bound dependencies.
    monkeypatch.setattr(two := facades[1], "extract_model_key", lambda workflow: "other")
    monkeypatch.setattr(two, "extract_model_fingerprint", lambda workflow: "other-f")
    assert two.Job({}).model_key == "other"
    assert one.Job({}).model_key == "k"


def test_acestep_service_is_late_bound_and_serialization_is_preserved(facades, monkeypatch):
    one = facades[0]
    service = {"service_name": "first", "endpoint": "http://first", "audio_dir": "/first", "bundle_name": "bundle"}
    monkeypatch.setattr(one, "_resolve_acestep_service", lambda name: service)
    monkeypatch.setattr(one, "priority_to_tier", lambda priority: types.SimpleNamespace(value="normal"))
    job = one.AceStepJob({"prompt": "x"}, job_id="fixed")
    assert job.id == "fixed" and job.service_name == "first"
    assert job.to_dict()["result"] == {}
    job.result = {"audio": "ok", "lyrics": "private"}
    assert job.to_dict()["result"] == {"audio": "ok"}
    replacement = {"service_name": "second", "endpoint": "http://second", "audio_dir": "/second"}
    monkeypatch.setattr(one, "_resolve_acestep_service", lambda name: replacement)
    assert one.AceStepJob({}).service_name == "second"


def test_timeout_default_snapshots_bind_value_and_coroutine_runs_hermetically(facades, monkeypatch):
    one, two = facades
    for facade in facades:
        signature = inspect.signature(facade.LLMProxy.wait_for_llm)
        assert signature.parameters["timeout"].default == 120
        proxy = facade.LLMProxy()
        proxy._llm_healthy = True
        assert _run(proxy.wait_for_llm()) is True
    monkeypatch.setattr(one, "LLM_WAIT_TIMEOUT", 3)
    monkeypatch.setattr(two, "LLM_WAIT_TIMEOUT", 9)
    assert inspect.signature(one.LLMProxy.wait_for_llm).parameters["timeout"].default == 120
    assert inspect.signature(two.LLMProxy.wait_for_llm).parameters["timeout"].default == 120
    assert inspect.signature(one.LLMProxy.wait_for_llm).parameters["timeout"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_bind_rejects_non_dict_and_returns_three_isolated_classes(facades):
    sys.path.insert(0, str(SCRIPTS))
    try:
        from gpu_manager import job_models
    finally:
        sys.path.remove(str(SCRIPTS))
    with pytest.raises(TypeError):
        job_models.bind(object())
    one, two = facades
    assert job_models.bind(vars(one))[0] is not one.Job
    assert len(job_models.bind(vars(two))) == len(job_models.NAMES) == 3
