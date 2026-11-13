"""Phase 4d KV slot affinity: session pinning, save/restore order, fail-open, retention."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aiohttp import ClientSession, web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402
from gemma_broker import kv_affinity as kv  # noqa: E402
from gemma_broker.runtime import BufferedAiohttpTransport  # noqa: E402

MEMBER = SimpleNamespace(name="Qwen-Dual")


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def read(self):
        return b"{}"

    async def text(self):
        return ""


class _FakeLlama:
    """Records slot API calls in order; save writes the file like llama-server would."""

    def __init__(self, save_dir, slots, *, save_status=200, slots_status=200):
        self.save_dir, self.slots = save_dir, slots
        self.save_status, self.slots_status = save_status, slots_status
        self.calls = []

    def get(self, url, **_):
        self.calls.append(("GET", url.rsplit("/", 1)[-1]))
        return _Resp(self.slots_status, self.slots)

    def post(self, url, json=None, **_):
        action = url.rsplit("action=", 1)[-1]
        self.calls.append((action, json["filename"]))
        if action == "save" and self.save_status == 200:
            path = os.path.join(self.save_dir, json["filename"])
            Path(path).write_bytes(b"kv")
            os.chmod(path, 0o644)
        return _Resp(self.save_status if action == "save" else 200, {})


class _Inner:
    def __init__(self, log):
        self.log, self.bodies = log, []

    async def send(self, member, body, on_accepted):
        self.log.append(("dispatch", body.get("id_slot")))
        self.bodies.append(body)
        return {"ok": True}


def _setup(tmp_path, slots, **kw):
    llama = _FakeLlama(str(tmp_path), slots, **kw)
    affinity = kv.KvAffinity(base_url="http://127.0.0.1:8160", slot_save_dir=str(tmp_path), fp="fp",
                             session=llama)
    inner = _Inner(llama.calls)
    return llama, affinity, inner, kv.KvAffinityTransport(inner, {"Qwen-Dual": affinity})


def _body(session=None, text="hi", principal="api"):
    body = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]}
    if session:
        body[kv.SESSION_FIELD] = {"principal": principal, "session": session}
    return body


def _key(session, principal="api"):
    return kv.session_key({}, session, principal)


def _send(transport, body):
    return asyncio.run(transport.send(MEMBER, body, lambda: None))


def _idle(n_tokens=(0, 0)):
    return [{"id": i, "is_processing": False, "n_prompt_tokens": t} for i, t in enumerate(n_tokens)]


def test_same_session_gets_same_slot(tmp_path):
    llama, _, inner, transport = _setup(tmp_path, _idle())
    _send(transport, _body("s1"))
    first = inner.bodies[0]["id_slot"]
    # Without affinity the empty, unowned slot would win the second pick.
    llama.slots = _idle((9000, 0) if first == 0 else (0, 9000))
    _send(transport, _body("s1", text="hi again"))
    assert inner.bodies[1]["id_slot"] == first
    assert ("save", f"fp-{_key('s1')}.bin") not in llama.calls
    assert kv.SESSION_FIELD not in inner.bodies[1]


def test_moved_session_forgets_old_slot(tmp_path):
    llama, affinity, inner, transport = _setup(tmp_path, [{"id": 0, "is_processing": True},
                                                         {"id": 1, "is_processing": False}])
    s = _key("s1")
    affinity.occupant[0], affinity.session_slot[s] = s, 0
    _send(transport, _body("s1"))  # slot 0 busy -> moves to slot 1
    assert inner.bodies[0]["id_slot"] == 1 and affinity.occupant == {1: s}


def test_fallback_session_key_is_system_plus_first_user():
    a = kv.session_key({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "x"}]})
    b = kv.session_key({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]})
    assert a == b and a != kv.session_key({}, "hdr") and len(a) == 24


def test_occupied_slot_saves_then_restores(tmp_path):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 9000},
             {"id": 1, "is_processing": True, "n_prompt_tokens": 100}]
    llama, affinity, inner, transport = _setup(tmp_path, slots)
    newcomer = _key("B")
    (tmp_path / f"fp-{newcomer}.bin").write_bytes(b"old")
    affinity.occupant[0] = "A"
    _send(transport, _body("B"))
    assert llama.calls == [("GET", "slots"), ("save", "fp-A.bin"), ("restore", f"fp-{newcomer}.bin"),
                           ("dispatch", 0)]
    assert (os.stat(tmp_path / "fp-A.bin").st_mode & 0o777) == 0o600
    assert affinity.occupant[0] == newcomer and not affinity.in_flight


def test_small_occupant_not_saved_and_resident_not_restored(tmp_path):
    llama, affinity, _, transport = _setup(tmp_path, _idle((500, 0)))
    s = _key("B")
    (tmp_path / f"fp-{s}.bin").write_bytes(b"x")
    affinity.occupant[0], affinity.session_slot[s] = s, 0
    _send(transport, _body("B"))
    assert llama.calls == [("GET", "slots"), ("dispatch", 0)]


def test_flag_off_body_unchanged(tmp_path):
    services = {"Qwen-Dual": {"port": 8160, "kv_affinity": {"enabled": False}}, "Luna": {}}
    assert kv.build_kv_affinity(services, ["Qwen-Dual", "Luna"], session=None) == {}
    log = []
    inner = _Inner(log)
    body = _body()
    asyncio.run(kv.KvAffinityTransport(inner, {}).send(MEMBER, body, lambda: None))
    assert inner.bodies[0] is body and "id_slot" not in body


def test_save_failure_still_dispatches(tmp_path):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 9000},
             {"id": 1, "is_processing": True}]
    llama, affinity, inner, transport = _setup(tmp_path, slots, save_status=500)
    affinity.occupant[0] = "A"
    assert _send(transport, _body("B")) == {"ok": True}
    assert llama.calls[-1] == ("dispatch", 0)


def test_slots_unavailable_dispatches_unpinned(tmp_path):
    llama, _, inner, transport = _setup(tmp_path, [], slots_status=503)
    _send(transport, _body("B"))
    assert "id_slot" not in inner.bodies[0] and llama.calls[-1] == ("dispatch", None)


def test_save_all_saves_large_known_slots_and_resets(tmp_path):
    llama, affinity, _, _ = _setup(tmp_path, _idle((9000, 9000)) + [{"id": 2, "n_prompt_tokens": 9000}])
    affinity.occupant.update({0: "A", 2: "C"})  # slot 1 unknown occupant -> skipped
    assert asyncio.run(affinity.save_all()) == 2
    assert [c for c in llama.calls if c[0] == "save"] == [("save", "fp-A.bin"), ("save", "fp-C.bin")]
    assert affinity.occupant == {} and affinity.session_slot == {}


def test_runtime_transport_strips_session_field():
    seen = {}

    class _Session:
        def post(self, url, json=None, **_):
            seen.update(json)
            return _Resp(500)

    transport = BufferedAiohttpTransport(session=_Session(), endpoint_for_member=lambda m: "http://x")
    with pytest.raises(RuntimeError):
        asyncio.run(transport.send(MEMBER, _body("S"), lambda: None))
    assert kv.SESSION_FIELD not in seen and seen["stream"] is True


def test_fingerprint_tracks_shards_and_kv_types(tmp_path):
    shard1 = tmp_path / "m-00001-of-00002.gguf"
    shard2 = tmp_path / "m-00002-of-00002.gguf"
    shard1.write_bytes(b"a")
    shard2.write_bytes(b"b")
    service = {"model_path": str(shard1), "cache_type_k": "q4_0", "cache_type_v": "q4_0", "launch_args": []}
    base = kv.fingerprint(service)
    os.utime(shard2, (1, 1))
    assert kv.fingerprint(service) != base
    assert kv.fingerprint({**service, "cache_type_k": "q8_0"}) != kv.fingerprint(service)


def test_storage_rejects_syncthing_folder(tmp_path):
    (tmp_path / ".stfolder").mkdir()
    (tmp_path / "kv").mkdir()
    assert kv.storage_rejection(str(tmp_path / "kv")) == "syncthing folder"


def test_session_scoped_by_principal_and_skipped_without_one(tmp_path):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 0}]
    llama, _, inner, transport = _setup(tmp_path, slots)
    (tmp_path / f"fp-{_key('s', 'kda')}.bin").write_bytes(b"theirs")
    assert _key("s", "kda") != _key("s", "api")
    _send(transport, _body("s", principal="api"))  # same session id, other principal
    assert [c[0] for c in llama.calls] == ["GET", "dispatch"]  # never restores the other key's file
    llama.calls.clear()
    _send(transport, {**_body(), kv.SESSION_FIELD: {"principal": "", "session": "s"}})
    assert llama.calls == [("dispatch", None)] and kv.SESSION_FIELD not in inner.bodies[-1]


def test_save_enforces_budget_and_session_cap(tmp_path):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 9000}]
    llama, affinity, _, _ = _setup(tmp_path, slots)
    affinity.max_files, affinity.max_bytes, affinity.max_age_s = 2, 100, 10**12
    for i, name in enumerate(("a.bin", "b.bin")):
        _bin(tmp_path / name, 4, 1000 + i)
    assert asyncio.run(affinity._slot_action(0, "save", "A"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.bin", "fp-A.bin"]  # oldest went for the cap
    affinity.max_bytes = 1  # the new file alone (2 bytes) can never fit
    assert not asyncio.run(affinity._slot_action(0, "save", "C"))
    assert not (tmp_path / "fp-C.bin").exists()


def test_compat_carries_session_header_into_body_and_key():
    from gemma_broker.compat import CompatibilityWrapper

    submitted = []

    class _Service:
        def submit_compat(self, body, *, idempotency_key, kv_scope=None, **_):
            submitted.append((body, idempotency_key, kv_scope))
            return SimpleNamespace(job_id="j1", state="queued")

    wrapper = CompatibilityWrapper(_Service(), timeout=5)
    forged = b'{"messages":[],"_kv_session_id":{"principal":"honcho","session":"victim"}}'

    def call(headers, request=None, body=b'{"messages":[]}'):
        return asyncio.run(wrapper.handle_proxy(
            method="POST", path="/v1/chat/completions", body=body,
            headers={"Prefer": "respond-async", **headers}, request=request))

    assert call({"X-Session-Id": "t3-abc"}, {"kv_principal": "api"}).status == 202
    call({"X-Session-Id": "t3-abc"}, {"kv_principal": "kda"}, body=forged)
    call({"X-Session-Id": "t3-abc"}, body=forged)  # no verified principal: forged scope dropped
    (api, key_a, scope_a), (kda, key_b, scope_b), (anon, key_c, scope_c) = submitted
    assert scope_a == {"principal": "api", "session": "t3-abc"}
    assert scope_b == {"principal": "kda", "session": "t3-abc"} and scope_c is None
    assert all(kv.SESSION_FIELD not in body for body in (api, kda, anon))
    assert len({key_a, key_b, key_c}) == 3


def test_broker_submit_drops_client_scope_and_takes_only_out_of_band(monkeypatch):
    from gemma_broker import api

    monkeypatch.setattr(api, "evaluate", lambda *_a, **_k: SimpleNamespace(compatible=True))
    repo = SimpleNamespace(submit=lambda job, caller_scope: job)
    service = api.DurableBrokerService(repo, members=lambda: [SimpleNamespace(name="m", state=None, observed_at=0.0)], clock=lambda: 1.0)
    forged = {"principal": "honcho", "session": "victim"}

    def submit(key, **kw):  # every non-compat entry point (HTTP /v1/gemma/jobs, facades) lands here
        body = {"messages": [{"role": "user", "content": "x"}], kv.SESSION_FIELD: forged}
        return service.submit({"idempotency_key": key, "request": body, "caller_scope": "compat"}, **kw)

    assert kv.SESSION_FIELD not in submit("a").request_body
    scope = {"principal": "api", "session": "s"}
    assert submit("b", kv_scope=scope).request_body[kv.SESSION_FIELD] == scope


def _bin(path, size, mtime):
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))


def test_prune_age_then_lru(tmp_path):
    now = 1_000_000.0
    _bin(tmp_path / "old.bin", 10, now - 8 * 86400)
    _bin(tmp_path / "a.bin", 40, now - 300)
    _bin(tmp_path / "b.bin", 40, now - 200)
    _bin(tmp_path / "c.bin", 40, now - 100)
    _bin(tmp_path / "keep.txt", 999, now - 9 * 86400)
    os.symlink(tmp_path / "c.bin", tmp_path / "link.bin")
    doomed = kv.prune(str(tmp_path), max_bytes=90, now=now)
    assert [Path(p).name for p in doomed] == ["old.bin", "a.bin"]
    assert (tmp_path / "a.bin").exists()  # dry run by default
    kv.prune(str(tmp_path), max_bytes=90, now=now, dry_run=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.bin", "c.bin", "keep.txt", "link.bin"]


def test_controller_principal_comes_only_from_verified_token(monkeypatch):
    """Trust boundary: no or wrong bearer -> no principal; flag off -> nothing set."""
    import importlib.util
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    spec = importlib.util.spec_from_file_location("kv_principal_gpm", SCRIPTS / "gpu-manager.py")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    seen = []

    async def ingress(**kw):
        seen.append(kw["request"].get("kv_principal", "unset"))
        return web.json_response({})

    scoped = {"on": False}
    monkeypatch.setattr(gm, "_maybe_handle_combined_gemma_ingress", ingress)
    monkeypatch.setattr(gm, "_honcho_gemma_request_is_authorized", lambda r, b: False)
    monkeypatch.setattr(gm, "_scoped_gemma_request_is_authorized", lambda r: scoped["on"])
    monkeypatch.setattr(gm, "_scoped_gemma_pin_is_unpinned", lambda r: False)
    monkeypatch.setattr(gm, "_model_validator", None)
    monkeypatch.setattr(gm, "_api_token", lambda: "synthetic-admin")
    broker = SimpleNamespace(transport=SimpleNamespace(kv_affinity={"Qwen-Dual": object()}))
    monkeypatch.setattr(gm, "_combined_gemma_broker", broker)

    def run(auth=None):
        headers = {"X-Backend": "Gemma", **({"Authorization": auth} if auth else {})}
        request = make_mocked_request("POST", "/v1/chat/completions", headers=headers)
        asyncio.run(gm.handle_llm_proxy(request))

    run()  # loopback / node-forwarded: no bearer
    run("Bearer wrong")
    run("Bearer ")
    run("Bearer synthetic-admin")
    scoped["on"] = True
    run()
    broker.transport.kv_affinity = None  # flag off
    run("Bearer synthetic-admin")
    assert seen == [None, None, None, "api", "kda", "unset"]


def test_close_session_deletes_every_fingerprint_and_forgets_slot(tmp_path):
    llama, affinity, _, _ = _setup(tmp_path, _idle((9000, 100)))
    s, other = _key("S"), _key("OTHER")
    (tmp_path / f"fp1-{s}.bin").write_bytes(b"x")
    (tmp_path / f"fp2-{s}.bin").write_bytes(b"x")
    (tmp_path / f"fp1-{other}.bin").write_bytes(b"x")
    affinity.occupant.update({0: s, 1: "OTHER"})
    affinity.session_slot[s] = 0
    assert asyncio.run(affinity.close_session(s)) == 2
    assert [p.name for p in tmp_path.iterdir()] == [f"fp1-{other}.bin"]
    assert affinity.occupant == {1: "OTHER"} and s not in affinity.session_slot
    llama.calls.clear()
    assert asyncio.run(affinity.save_all()) == 0  # the closed session is never saved back
    assert [c for c in llama.calls if c[0] != "GET"] == []


NAS = ("/mnt/nas",)  # host config supplies this in production; the published module carries no host paths


def test_storage_rejects_forbidden_roots_and_backup_paths(tmp_path):
    assert kv.storage_rejection("/mnt/nas/kv", NAS) == "forbidden root"
    assert kv.storage_rejection("/mnt/nas", NAS) == "forbidden root"
    assert kv.storage_rejection("/mnt/nasty/kv", NAS) is None  # whole path parts, not prefixes
    os.symlink("/mnt/nas/kv", tmp_path / "link")
    assert kv.storage_rejection(str(tmp_path / "link"), NAS) == "forbidden root"
    assert kv.storage_rejection(str(tmp_path / "backups" / "kv"), NAS) == "backup path"
    assert kv.storage_rejection(str(tmp_path / "kv"), NAS) is None
    cfg = {"port": 8160, "kv_affinity": {"enabled": True, "slot_save_dir": "/mnt/nas/kv",
                                         "forbidden_roots": list(NAS)}}
    built = kv.build_kv_affinity({"Qwen-Dual": cfg}, ["Qwen-Dual"], session=None)["Qwen-Dual"]
    assert built.slot_save_dir == ""  # affinity only, nothing is ever written to the NAS


def test_fingerprint_change_refuses_restore(tmp_path):
    llama = _FakeLlama(str(tmp_path), _idle((0, 0)))
    svc = {"port": 8160, "kv_affinity": {"enabled": True, "slot_save_dir": str(tmp_path)}, "cache_type_k": "q4_0"}
    s = _key("S")
    before = kv.build_kv_affinity({"Qwen-Dual": svc}, ["Qwen-Dual"], session=llama)["Qwen-Dual"]
    assert asyncio.run(before._slot_action(0, "save", s))  # written under the old fingerprint
    after = kv.build_kv_affinity({"Qwen-Dual": {**svc, "cache_type_k": "q8_0"}}, ["Qwen-Dual"],
                                 session=llama)["Qwen-Dual"]
    assert after.fp != before.fp
    llama.calls.clear()
    asyncio.run(after.prepare({}, s))
    assert not any(c[0] == "restore" for c in llama.calls)  # the old file is never restored
    assert (tmp_path / f"{before.fp}-{s}.bin").exists()  # kept for the build it belongs to


def test_build_identity_tracks_libraries_and_commit(tmp_path):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    (tmp_path / "build" / "common").mkdir()
    server, lib = bin_dir / "llama-server", bin_dir / "libllama.so.0.6.0"
    info = tmp_path / "build" / "common" / "build-info.cpp"
    server.write_bytes(b"s")
    lib.write_bytes(b"l" * 8)
    info.write_text('LLAMA_COMMIT = "1b70350c9"')
    service = {"build_path": str(server)}
    base = kv.fingerprint(service)
    lib.write_bytes(b"l" * 16)  # a library size change
    assert kv.fingerprint(service) != base
    base = kv.fingerprint(service)
    os.utime(lib, (1, 1))  # a rebuilt library with the same bytes
    assert kv.fingerprint(service) != base
    base = kv.fingerprint(service)
    info.write_text('LLAMA_COMMIT = "deadbeef"')  # the commit recorded in build-info.cpp
    assert kv.fingerprint(service) != base
    assert len(kv.fingerprint({})) == 12  # no build path at all still yields a key


def test_d1_members_never_get_kv_affinity(tmp_path):
    services = {"D1-MI50": {"port": 8149, "kv_affinity": {"enabled": True, "slot_save_dir": str(tmp_path)}}}
    assert kv.build_kv_affinity(services, ["D1-MI50"], session=None) == {}


class _HttpLlama:
    """Real HTTP server with llama-server's slot routes: save writes a file, restore reads one."""

    def __init__(self, save_dir, slots):
        self.save_dir, self.slots, self.calls = save_dir, slots, []

    def app(self):
        app = web.Application()
        app.router.add_get("/slots", self._slots)
        app.router.add_post("/slots/{id}", self._action)
        return app

    async def _slots(self, request):
        return web.json_response(self.slots)

    async def _action(self, request):
        slot, action = int(request.match_info["id"]), request.query.get("action")
        name = (await request.json())["filename"]
        self.calls.append((action, slot, name))
        path = os.path.join(self.save_dir, name)
        if action == "save":
            Path(path).write_bytes(b"kv")
            os.chmod(path, 0o644)  # llama-server's own umask; the broker must tighten it
            return web.json_response({"n_saved": 2})
        if not os.path.isfile(path):
            return web.json_response({"error": "missing"}, status=404)
        return web.json_response({"n_restored": 2})


@asynccontextmanager
async def _real_llama(tmp_path, slots):
    fake = _HttpLlama(str(tmp_path), slots)
    server = TestServer(fake.app())  # ephemeral port
    await server.start_server()
    session = ClientSession()
    affinity = kv.KvAffinity(base_url=f"http://127.0.0.1:{server.port}", slot_save_dir=str(tmp_path),
                             fp="fp", session=session)
    try:
        yield fake, affinity
    finally:
        await session.close()
        await server.close()


@pytest.mark.parametrize("tokens, saved", [(8192, False), (8193, True)])
def test_save_threshold_is_above_8k_over_real_http(tmp_path, tokens, saved):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": tokens},
             {"id": 1, "is_processing": True, "n_prompt_tokens": 0}]

    async def run():
        async with _real_llama(tmp_path, slots) as (fake, affinity):
            affinity.occupant[0] = "A"
            payload = {}
            await affinity.prepare(payload, "B")
            return fake.calls, payload

    calls, payload = asyncio.run(run())
    assert payload["id_slot"] == 0
    assert (("save", 0, "fp-A.bin") in calls) is saved
    assert (tmp_path / "fp-A.bin").exists() is saved
    if saved:
        assert (os.stat(tmp_path / "fp-A.bin").st_mode & 0o777) == 0o600


def test_restore_over_real_http(tmp_path):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 0}]
    s = _key("B")

    async def run():
        async with _real_llama(tmp_path, slots) as (fake, affinity):
            (tmp_path / f"fp-{s}.bin").write_bytes(b"old")
            payload = {}
            await affinity.prepare(payload, s)
            return fake.calls, payload

    calls, payload = asyncio.run(run())
    assert calls == [("restore", 0, f"fp-{s}.bin")] and payload["id_slot"] == 0


def test_save_dir_created_0700_and_fails_open(tmp_path, monkeypatch):
    cfg = {"port": 8160, "kv_affinity": {"enabled": True, "slot_save_dir": str(tmp_path / "kv")}}
    made = kv.build_kv_affinity({"Qwen-Dual": cfg}, ["Qwen-Dual"], session=None)["Qwen-Dual"]
    assert made.slot_save_dir == str(tmp_path / "kv")
    assert (os.stat(tmp_path / "kv").st_mode & 0o777) == 0o700

    def refuse(*_args, **_kwargs):
        raise PermissionError("chmod refused")

    monkeypatch.setattr(kv.os, "chmod", refuse)
    cfg["kv_affinity"]["slot_save_dir"] = str(tmp_path / "kv2")
    fallback = kv.build_kv_affinity({"Qwen-Dual": cfg}, ["Qwen-Dual"], session=None)["Qwen-Dual"]
    assert fallback.slot_save_dir == ""  # fail-open: affinity only, broker still starts


def test_declared_kv_identity_replaces_argv_layout(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"a")
    base = {"model_path": str(model), "launch_args": ["-ctk", "q4_0"],
            "kv_identity": {"engine": "breccia", "arch": "gemma", "k_layout": "f16", "v_layout": "f16", "state_version": 1}}
    fp = kv.fingerprint(base)
    # argv is ignored once the engine declares its layout (Breccia accepts -ctk but runs f16 K)
    assert kv.fingerprint({**base, "launch_args": ["-ctk", "q8_0"]}) == fp
    for field, value in (("v_layout", "q8_0"), ("engine", "llama"), ("state_version", 2), ("arch", "qwen")):
        changed = {**base, "kv_identity": {**base["kv_identity"], field: value}}
        assert kv.fingerprint(changed) != fp, field
    assert kv.fingerprint({k: v for k, v in base.items() if k != "kv_identity"}) != fp


def test_non_slot_engine_gets_no_pinning_or_save_restore(tmp_path):
    llama = _FakeLlama(str(tmp_path), _idle())
    affinity = kv.KvAffinity(base_url="http://127.0.0.1:1", slot_save_dir=str(tmp_path), fp="fp",
                             session=llama, engine="breccia")
    payload = {}
    assert asyncio.run(affinity.prepare(payload, "s")) is None
    assert "id_slot" not in payload and llama.calls == []
    assert asyncio.run(affinity.save_all()) == 0
    assert asyncio.run(affinity._slot_action(0, "save", "s")) is False
    assert llama.calls == []


def test_build_passes_engine_through(tmp_path):
    services = {"B": {"port": 1, "kv_affinity": {"enabled": True, "engine": "breccia", "slot_save_dir": str(tmp_path)}},
                "Q": {"port": 2, "kv_affinity": {"enabled": True, "slot_save_dir": str(tmp_path)}}}
    built = kv.build_kv_affinity(services, ["B", "Q"], object())
    assert built["B"].engine == "breccia" and built["Q"].engine == "llama"
