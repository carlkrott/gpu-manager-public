"""Facade binding and behavior proof for service event helpers."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
import importlib.util
import logging
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
MODULE = SCRIPTS / "gpu_manager" / "service_events.py"
NAMES = ("_queue_depth_for_group", "ServiceEventBus", "_force_recover_failed_service", "_persist_audio_cpp_result", "_persist_generated_image", "_required_native_helper_token_file")


def _deny(*args, **kwargs):
    raise AssertionError("network/process creation forbidden during binding proof")


@pytest.fixture
def facades(tmp_path, monkeypatch):
    home, xdg = tmp_path / "home", tmp_path / "xdg"
    home.mkdir(); xdg.mkdir()
    monkeypatch.setenv("HOME", str(home)); monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aiohttp.web  # noqa: F401
    monkeypatch.setattr(socket, "socket", _deny)
    monkeypatch.setattr(subprocess, "Popen", _deny)
    monkeypatch.setattr(os, "system", _deny)
    result=[]
    for name in ("service_events_facade_one", "service_events_facade_two"):
        spec=importlib.util.spec_from_file_location(name,FACADE)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); result.append(module)
    return result


def _module():
    sys.path.insert(0,str(SCRIPTS))
    try:
        from gpu_manager import service_events
    finally:
        sys.path.remove(str(SCRIPTS))
    return service_events


def _baseline_bus():
    """Pre-extraction event-bus contract literals, independent of candidate class."""
    class BaselineBus:
        def __init__(self):
            self._events={}; self._states={}; self._any_change=asyncio.Event(); self._planned_stops={}; self._PLANNED_STOP_GRACE_S=600.0
        def mark_planned_stop(self, service_name): self._planned_stops[service_name]=100.0
        def clear_planned_stop(self, service_name): self._planned_stops.pop(service_name,None)
        def is_planned_stop(self, service_name): return service_name in self._planned_stops
        def emit(self, service_name,new_state):
            self._states[service_name]=new_state
            if service_name not in self._events: self._events[service_name]=asyncio.Event()
            self._events[service_name].set(); self._any_change.set()
        async def wait_for_change(self,timeout=5.0):
            try: await asyncio.wait_for(self._any_change.wait(),timeout=timeout); self._any_change.clear()
            except asyncio.TimeoutError: pass
        def get_changed_services(self):
            changed=[]
            for svc,event in self._events.items():
                if event.is_set(): event.clear(); changed.append((svc,self._states.get(svc,"unknown")))
            return changed
        def get_state(self,service_name): return self._states.get(service_name,"unknown")
    return BaselineBus


def test_two_instances_are_isolated_and_bus_matches_baseline(facades):
    for facade in facades:
        candidate=facade.ServiceEventBus(); baseline=_baseline_bus()()
        assert candidate._events is not baseline._events
        assert candidate._states is not baseline._states
        assert candidate._planned_stops is not baseline._planned_stops
        assert candidate._any_change is not baseline._any_change
        for bus in (candidate,baseline):
            bus.emit("svc","starting"); bus.emit("svc","healthy")
            assert bus.get_state("svc")=="healthy"
            assert bus.get_changed_services()==[("svc","healthy")]
            assert bus.get_changed_services()==[]
            bus.mark_planned_stop("svc"); assert bus.is_planned_stop("svc")
            bus.clear_planned_stop("svc"); assert not bus.is_planned_stop("svc")
        assert (candidate.get_state("missing"), candidate.get_changed_services()) == ("unknown",[])
    one,two=facades
    assert one.ServiceEventBus is not two.ServiceEventBus
    a,b=one.ServiceEventBus(),two.ServiceEventBus(); a.emit("x","healthy")
    assert b.get_state("x")=="unknown" and b._events=={}


def test_binding_surface_and_call_time_facade_globals(facades,monkeypatch):
    mod=_module()
    assert mod.NAMES==NAMES
    for facade in facades:
        for name in NAMES:
            value=getattr(facade,name)
            if name=="ServiceEventBus":
                assert value.__module__==facade.__name__
                for method in ("emit","get_state","get_changed_services","mark_planned_stop"):
                    assert vars(value)[method].__globals__ is vars(facade)
            else: assert value.__globals__ is vars(facade)
    one,two=facades
    members=types.SimpleNamespace(get_group_members=lambda group:[{"queue_depth":2}])
    monkeypatch.setattr(one,"routing_engine",members); monkeypatch.setattr(one,"logger",logging.getLogger("late-one"))
    monkeypatch.setattr(two,"routing_engine",types.SimpleNamespace(get_group_members=lambda group:[{"queue_depth":9}]))
    assert one._queue_depth_for_group("g")==2 and two._queue_depth_for_group("g")==9
    # Recovery observes the facade's replacement collaborators at invocation time.
    calls=[]
    monkeypatch.setattr(one,"_load_services_config",lambda:{"services":{"svc":{"type":"generation_backend","gpu_id":None,"systemd_unit":"unit"}}})
    monkeypatch.setattr(one,"_resolve_bundles",lambda:{})
    monkeypatch.setattr(one,"vram",types.SimpleNamespace(_systemctl=lambda *a: _completed(calls,a)))
    result=_run(one._force_recover_failed_service("svc","job","boom"))
    assert result["ok"] and calls==[("stop","unit")]
    # No real file or credential is accessed: only the missing-env branch is exercised.
    monkeypatch.delenv("GPU_MANAGER_NATIVE_TASK_TOKEN_FILE",raising=False)
    monkeypatch.delenv("NATIVE_TASK_TOKEN_FILE",raising=False)
    with pytest.raises(RuntimeError,match="credential is required"):
        one._required_native_helper_token_file()


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("coroutine unexpectedly suspended in hermetic proof")


def _completed(calls,args):
    calls.append(args)
    async def done(): return None
    return done()


def test_persistence_and_token_helpers_use_isolated_fakes(facades,monkeypatch,tmp_path):
    one=facades[0]
    audio=io.BytesIO()
    with wave.open(audio,"wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(8000); handle.writeframes(b"\\x00\\x00"*8)
    monkeypatch.setenv("GPU_MANAGER_GENERATION_OUTPUT_DIR",str(tmp_path))
    payload=json.dumps({"model":"fake-model","audio":base64.b64encode(audio.getvalue()).decode()})
    artifact=json.loads(one._persist_audio_cpp_result("job-a",payload))["artifact"]
    assert Path(artifact["path_or_url"]).read_bytes()==audio.getvalue()
    assert artifact["byte_size"]==len(audio.getvalue()) and artifact["media_type"]=="audio/wav"

    writes=[]
    monkeypatch.setitem(sys.modules,"native_task_host",types.SimpleNamespace(atomic_write=lambda path,data:writes.append((Path(path),data))))
    monkeypatch.setattr(one,"_verified_image_bytes",lambda data:data)
    image=one._persist_generated_image("job-i",{"data_b64":base64.b64encode(b"fake-image").decode(),"filename":"x.png"},0)
    assert writes==[(Path(image["path_or_url"]),b"fake-image")]
    assert image["byte_size"]==10 and len(image["sha256"])==64

    checks=[]
    fake_runtime=types.SimpleNamespace(HELPER_TOKEN_FILE_ENV="GPU_MANAGER_HELPER_TOKEN_FILE",NATIVE_TASK_TOKEN_FILE_ENV="GPU_MANAGER_NATIVE_TASK_TOKEN_FILE",resolve_required_service_credential=lambda **kw:checks.append(kw))
    monkeypatch.setitem(sys.modules,"runtime_host_client",fake_runtime)
    monkeypatch.setenv("GPU_MANAGER_NATIVE_TASK_TOKEN_FILE","/fake/token/path")
    assert one._required_native_helper_token_file()=="/fake/token/path"
    assert checks==[{"service_token_file":"/fake/token/path"}]


def test_bind_rejects_non_dict_and_exports_six_facade_bound_definitions():
    mod=_module()
    with pytest.raises(TypeError): mod.bind(object())
    ns={"__name__":"isolated_facade","asyncio":asyncio,"logger":logging.getLogger("isolated"),"time":__import__("time"),"Path":Path,"Mapping":dict,"Any":object,"json":__import__("json"),"os":os,"hashlib":__import__("hashlib"),"PROJECT_ROOT":Path("/tmp"),"_load_services_config":lambda:{},"_resolve_bundles":lambda:{},"vram":None}
    bound=mod.bind(ns)
    assert len(bound)==len(NAMES)
    assert tuple(x.__name__ for x in bound)==NAMES
    assert bound[1].__module__=="isolated_facade"
    assert all(getattr(bound[1],name).__globals__ is ns for name in ("emit","wait_for_change","get_changed_services"))
