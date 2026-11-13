"""Catalog phase 2: per-client tokens, subscriber leases and the advisory share."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from aiohttp.test_utils import TestClient, TestServer
import pytest

from api_auth import CATALOG_CLIENT_TOKENS_DIR_ENV, CATALOG_READ_TOKEN_FILE_ENV


ROOT = Path(__file__).parents[2]
CONTROLLER = ROOT / "scripts" / "gpu-manager.py"
CLIENTS = ("zeroclaw-msi", "zeroclaw-pi", "honcho", "t3")


@pytest.fixture
def module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("candidate_gpu_manager_subscribe", CONTROLLER)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    cliproxy = tmp_path / "cliproxy.yaml"
    cliproxy.write_text("validation:\n  max_message_length: 250000\n", encoding="utf-8")
    monkeypatch.setenv("GPU_MANAGER_CLIPROXY_CONFIG", str(cliproxy))
    clock = [1000.0]
    monkeypatch.setattr(loaded, "_catalog_clock", lambda: clock[0])
    loaded._test_clock = clock
    return loaded


def _config(module) -> dict:
    config = module.empty_services_config()
    config["services"].update({
        "combined_gemma_broker": {"type": "embedded_broker", "enabled": True},
        "Cloud-Haiku": {"enabled": True, "model": "claude-haiku-5-5", "parallel": 4,
                        "context_per_slot": 200000},
        "Local-Qwen": {"enabled": True, "parallel": 4, "context_per_slot": 150000},
        "comfyui": {"type": "comfyui", "enabled": True},
    })
    config["combined_gemma_broker"] = {"ordered_members": ["Cloud-Haiku", "Local-Qwen"]}
    return config


def _metrics(free: int = 8, walled: tuple[str, ...] = ()) -> dict:
    members = [
        {"name": "Cloud-Haiku", "state": "ready_accepting", "accepting": True,
         "configured_slots": 4, "compatible_free_slots": min(free, 4)},
        {"name": "Local-Qwen", "state": "ready_accepting", "accepting": True,
         "configured_slots": 4, "compatible_free_slots": max(free - 4, 0)},
    ]
    return {"members": members, "dispatch_loop": {"quota_walls": {n: {} for n in walled}}}


def _token(path: Path, value: str) -> Path:
    path.write_text(value, encoding="ascii")
    path.chmod(0o600)
    return path


@pytest.fixture
def tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(_token(tmp_path / "admin", "admin-synthetic")))
    monkeypatch.setenv(CATALOG_READ_TOKEN_FILE_ENV, str(_token(tmp_path / "cat", "catalog-synthetic")))
    directory = tmp_path / "clients"
    directory.mkdir(mode=0o700)
    for name in CLIENTS:
        _token(directory / name, f"{name}-synthetic")
    monkeypatch.setenv(CATALOG_CLIENT_TOKENS_DIR_ENV, str(directory))
    return directory


def _app(module, monkeypatch, tmp_path, metrics):
    config = _config(module)
    services_path = tmp_path / "services.json"
    services_path.write_text(json.dumps(config), encoding="utf-8")
    module.SERVICES_CONFIG_PATH = str(services_path)
    providers_path = tmp_path / "providers.json"
    providers_path.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    module.PROVIDERS_CONFIG_PATH = str(providers_path)
    monkeypatch.setattr(module, "_load_services_config", lambda: copy.deepcopy(config))
    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    fake_redis.connection_pool.connection_kwargs = {"decode_responses": True}
    monkeypatch.setattr(module._redis_mod, "Redis", MagicMock(return_value=fake_redis))
    monkeypatch.setattr(module._combined_gemma_api_proxy, "broker_metrics", lambda: metrics[0])
    app = module.create_app()
    app.on_startup.clear()
    app.on_cleanup.clear()
    return app


def _run(app, scenario):
    async def wrapper():
        async with TestClient(TestServer(app)) as client:
            async def call(method, path, token=None, body=None):
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                response = await client.request(method, path, headers=headers, json=body)
                text = await response.text()
                return response.status, (json.loads(text) if text.startswith("{") else text)
            return await scenario(call)
    return asyncio.run(wrapper())


def _sub(want_max=8, priority="kanban", **extra):
    return {"want_max": want_max, "priority": priority, **extra}


def test_identity_comes_from_token_not_body(module, monkeypatch, tmp_path, tokens):
    app = _app(module, monkeypatch, tmp_path, [_metrics()])

    async def scenario(call):
        mine = await call("POST", "/v1/catalog/llm.chat/subscribe", "zeroclaw-msi-synthetic",
                          _sub(client="zeroclaw-msi"))
        spoof = await call("POST", "/v1/catalog/llm.chat/subscribe", "zeroclaw-msi-synthetic",
                           _sub(client="t3"))
        admin = await call("POST", "/v1/catalog/llm.chat/subscribe", "admin-synthetic", _sub())
        reader = await call("POST", "/v1/catalog/llm.chat/subscribe", "catalog-synthetic", _sub())
        sid = mine[1]["subscriber_id"]
        other_hb = await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "t3-synthetic", {})
        other_del = await call("DELETE", f"/v1/catalog/subscriptions/{sid}", "t3-synthetic")
        read = await call("GET", "/v1/catalog/llm.chat", "zeroclaw-msi-synthetic")
        other_read = await call("GET", "/v1/catalog/llm.chat", "t3-synthetic")
        out_of_scope = [await call("GET", "/services", "zeroclaw-msi-synthetic"),
                        await call("POST", "/v1/chat/completions", "zeroclaw-msi-synthetic",
                                   {"model": "gemma"}),
                        await call("GET", "/v1/catalog/llm.chat?x=1", "zeroclaw-msi-synthetic")]
        own_del = await call("DELETE", f"/v1/catalog/subscriptions/{sid}", "zeroclaw-msi-synthetic")
        gone = await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat",
                          "zeroclaw-msi-synthetic", {})
        return mine, spoof, admin, reader, other_hb, other_del, read, other_read, out_of_scope, own_del, gone

    mine, spoof, admin, reader, other_hb, other_del, read, other_read, out_of_scope, own_del, gone = _run(app, scenario)
    assert mine[0] == 201 and mine[1]["client"] == "zeroclaw-msi"
    assert len(mine[1]["subscriber_id"]) == 32 and mine[1]["heartbeat_s"] == 60
    assert spoof[0] == 403 and spoof[1]["code"] == "client_label_mismatch"
    assert admin[0] == 403 and admin[1]["code"] == "catalog_client_token_required"
    assert reader[0] in {401, 403}
    assert other_hb[0] == other_del[0] == 404
    assert read[0] == 200 and read[1]["you"]["subscriber_id"] == mine[1]["subscriber_id"]
    assert other_read[0] == 200 and "you" not in other_read[1]
    assert all(status in {401, 403} for status, _ in out_of_scope), out_of_scope
    assert own_del == (200, {"released": True, "subscriber_id": mine[1]["subscriber_id"]})
    assert gone[0] == 404


def test_lease_expires_without_heartbeat_and_resubscribe_replaces(module, monkeypatch, tmp_path, tokens):
    app = _app(module, monkeypatch, tmp_path, [_metrics()])
    clock = module._test_clock

    async def scenario(call):
        first = await call("POST", "/v1/catalog/llm.chat/subscribe", "honcho-synthetic", _sub())
        second = await call("POST", "/v1/catalog/llm.chat/subscribe", "honcho-synthetic", _sub())
        old_hb = await call("POST", f"/v1/catalog/subscriptions/{first[1]['subscriber_id']}/heartbeat",
                            "honcho-synthetic", {})
        sid = second[1]["subscriber_id"]
        clock[0] += 170
        alive = await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "honcho-synthetic", {})
        clock[0] += 170  # heartbeat renewed the 180 s lease
        still = await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "honcho-synthetic", {})
        clock[0] += 181
        expired = await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "honcho-synthetic", {})
        return first, second, old_hb, alive, still, expired

    first, second, old_hb, alive, still, expired = _run(app, scenario)
    assert first[0] == second[0] == 201
    assert old_hb[0] == 404  # one lease per (client, service)
    assert alive[0] == still[0] == 200 and still[1]["expires_in_s"] == 180
    assert expired[0] == 404 and module._catalog_leases == {}


def _lease(module, client, priority, want_max=64, in_flight=0):
    return {"subscriber_id": client, "client": client, "public_id": "llm.chat",
            "priority": priority, "want_max": want_max, "in_flight": in_flight}


def test_share_math_weights_cap_and_floor(module):
    entry = next(e for e in module._build_service_catalog(_config(module), _metrics(free=8))["services"]
                 if e["public_id"] == "llm.chat")
    t3 = _lease(module, "t3", "interactive")
    hermes = _lease(module, "hermes", "kanban", in_flight=2)
    zc = _lease(module, "zeroclaw-msi", "background")
    leases = [t3, hermes, zc]  # weights 4 + 2 + 1 = 7
    assert module._catalog_recommended(entry, t3, leases) == 8 * 4 // 7 == 4
    assert module._catalog_recommended(entry, hermes, leases) == (8 + 2) * 2 // 7 == 2
    assert module._catalog_recommended(entry, zc, leases) == 1  # 8 // 7, and min 1
    assert module._catalog_recommended(entry, _lease(module, "t3", "interactive", want_max=2),
                                       leases) == 2  # want_max caps
    alone = _lease(module, "pibook", "background")
    assert module._catalog_recommended(entry, alone, [alone]) == 8

    full = next(e for e in module._build_service_catalog(_config(module), _metrics(free=0))["services"]
                if e["public_id"] == "llm.chat")
    assert module._catalog_recommended(full, zc, leases) == 1  # healthy but busy: still 1

    walled = next(e for e in module._build_service_catalog(
        _config(module), _metrics(free=0, walled=("Cloud-Haiku", "Local-Qwen")))["services"]
        if e["public_id"] == "llm.chat")
    assert walled["slots_healthy"] == 0
    assert module._catalog_recommended(walled, zc, leases) == 0  # nothing healthy

    static = next(e for e in module._build_service_catalog(_config(module), None)["services"]
                  if e["public_id"] == "llm.chat")
    assert module._catalog_recommended(static, zc, leases) is None


def test_over_share_logs_rate_limited_and_never_refuses(module, monkeypatch, tmp_path, tokens, caplog):
    app = _app(module, monkeypatch, tmp_path, [_metrics(free=2)])

    async def scenario(call):
        unknown = await call("POST", "/v1/catalog/llm.chat/subscribe", "pibook-synthetic", _sub())
        sub = await call("POST", "/v1/catalog/llm.chat/subscribe", "t3-synthetic",
                         _sub(want_max=2, priority="interactive", in_flight=6))
        sid = sub[1]["subscriber_id"]
        beats = [await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "t3-synthetic",
                            {"in_flight": 7}) for _ in range(2)]
        module._test_clock[0] += 61
        beats.append(await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "t3-synthetic",
                                {"in_flight": 50}))
        return unknown, sub, beats

    with caplog.at_level(logging.WARNING, logger="gpu-manager"):
        unknown, sub, beats = _run(app, scenario)
    assert unknown[0] == 401  # no token file for pibook: not an identity
    assert sub[0] == 201 and sub[1]["recommended_concurrency"] == 2  # want_max caps 2 + 6
    assert all(status == 200 and body["advisory"] is True for status, body in beats)
    assert [b["in_flight"] for _, b in beats] == [7, 7, 8]  # clamped to slots_total
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("catalog_over_share")]
    assert len(lines) == 2  # subscribe, then once more after the 60 s window
    record = json.loads(lines[0].split(" ", 1)[1])
    assert record == {"client": "t3", "public_id": "llm.chat", "subscriber_id": sub[1]["subscriber_id"],
                      "priority": "interactive", "in_flight": 6, "recommended_concurrency": 2}


def _enforce_run(module, monkeypatch, tmp_path, flag, *, other, in_flight=6, metrics=None):
    """t3 (want_max 2 -> share 2) subscribes then heartbeats; `other` = honcho's subscribe body or None.
    honcho's share here is 1, so in_flight 1 is saturated and 0 is idle."""
    if flag is None:
        monkeypatch.delenv("GPU_MANAGER_CATALOG_ENFORCE", raising=False)
    else:
        monkeypatch.setenv("GPU_MANAGER_CATALOG_ENFORCE", flag)
    app = _app(module, monkeypatch, tmp_path, [metrics or _metrics(free=2)])

    async def scenario(call):
        if other:
            await call("POST", "/v1/catalog/llm.chat/subscribe", "honcho-synthetic", other)
        sub = await call("POST", "/v1/catalog/llm.chat/subscribe", "t3-synthetic",
                         _sub(want_max=2, priority="interactive", in_flight=in_flight))
        sid = sub[1]["subscriber_id"]
        beat = await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "t3-synthetic",
                          {"in_flight": in_flight})
        return sub, beat

    return _run(app, scenario)


def test_enforce_off_by_default_never_429(module, monkeypatch, tmp_path, tokens):
    for flag in (None, "0"):
        sub, beat = _enforce_run(module, monkeypatch, tmp_path, flag, other=_sub(in_flight=1))
        assert (sub[0], beat[0]) == (201, 200)


def test_enforce_heartbeat_429_when_other_saturated_and_wants_more(module, monkeypatch, tmp_path, tokens):
    sub, beat = _enforce_run(module, monkeypatch, tmp_path, "1", other=_sub(want_max=8, in_flight=1))
    assert sub[0] == 201  # subscribe never enforces: a 429 there would still create the lease
    status, body = beat
    assert status == 429
    assert body["code"] == "catalog_over_share" and body["retry_after"] == 60
    assert body["lease"]["recommended_concurrency"] == 2 and body["lease"]["in_flight"] == 6
    # never classifiable as a provider plan wall (10-22 lesson)
    assert not any(word in json.dumps(body).lower() for word in ("rate_limited", "quota", "wall"))
    assert len(module._catalog_leases) == 2  # the lease stays held


def test_enforce_200_when_nobody_is_waiting(module, monkeypatch, tmp_path, tokens):
    cases = {
        "alone": None,
        "other idle": _sub(want_max=8, in_flight=0),
        "other saturated, want_max == in_flight": _sub(want_max=1, in_flight=1),
    }
    for name, other in cases.items():
        sub, beat = _enforce_run(module, monkeypatch, tmp_path, "1", other=other)
        assert (sub[0], beat[0]) == (201, 200), name
    sub, beat = _enforce_run(module, monkeypatch, tmp_path, "1", other=_sub(want_max=8, in_flight=1),
                             in_flight=1)
    assert (sub[0], beat[0]) == (201, 200)  # within share
    # nothing healthy: recommended is 0 for everyone, an idle other (0 >= 0) must still not count
    sub, beat = _enforce_run(module, monkeypatch, tmp_path, "1", other=_sub(want_max=8, in_flight=0),
                             metrics=_metrics(free=0, walled=("Cloud-Haiku", "Local-Qwen")))
    assert (sub[0], beat[0]) == (201, 200)


def test_invalid_bodies_are_400(module, monkeypatch, tmp_path, tokens):
    app = _app(module, monkeypatch, tmp_path, [_metrics()])

    async def scenario(call):
        token = "t3-synthetic"
        return [
            (await call("POST", "/v1/catalog/llm.chat/subscribe", token, {"priority": "kanban"}))[0],
            (await call("POST", "/v1/catalog/llm.chat/subscribe", token, _sub(want_max=True)))[0],
            (await call("POST", "/v1/catalog/llm.chat/subscribe", token, _sub(want_max=0)))[0],
            (await call("POST", "/v1/catalog/llm.chat/subscribe", token, _sub(priority="urgent")))[0],
            (await call("POST", "/v1/catalog/llm.chat/subscribe", token, _sub(in_flight=-1)))[0],
            (await call("POST", "/v1/catalog/nope.nope/subscribe", token, _sub()))[0],
        ]

    assert _run(app, scenario) == [400, 400, 400, 400, 400, 404]


def test_fail_closed_without_token_config(module, monkeypatch, tmp_path, tokens):
    from api_auth import catalog_client_tokens

    assert sorted(catalog_client_tokens()) == sorted(CLIENTS)
    # A client token equal to another principal's, or shared by two clients, is nobody.
    _token(tokens / "hermes", "catalog-synthetic")
    _token(tokens / "pibook", "t3-synthetic")
    _token(tokens / "stray.bak", "stray-synthetic")
    assert sorted(catalog_client_tokens()) == sorted(set(CLIENTS) - {"t3"})
    tokens.chmod(0o750)
    assert catalog_client_tokens() == {}
    tokens.chmod(0o700)
    monkeypatch.delenv(CATALOG_CLIENT_TOKENS_DIR_ENV)
    assert catalog_client_tokens() == {}

    app = _app(module, monkeypatch, tmp_path, [_metrics()])

    async def scenario(call):
        return [
            (await call("POST", "/v1/catalog/llm.chat/subscribe", "zeroclaw-msi-synthetic", _sub()))[0],
            (await call("GET", "/v1/catalog", "zeroclaw-msi-synthetic"))[0],
        ]

    assert _run(app, scenario) == [401, 401]


def test_subscriptions_list_counters_and_subscriber_count(module, monkeypatch, tmp_path, tokens):
    app = _app(module, monkeypatch, tmp_path, [_metrics(free=2)])
    clock = module._test_clock

    async def scenario(call):
        empty = await call("GET", "/v1/catalog/subscriptions", "catalog-synthetic")
        await call("POST", "/v1/catalog/llm.chat/subscribe", "honcho-synthetic", _sub())
        honcho = await call("POST", "/v1/catalog/llm.chat/subscribe", "honcho-synthetic", _sub())
        t3 = await call("POST", "/v1/catalog/llm.chat/subscribe", "t3-synthetic",
                        _sub(want_max=2, priority="interactive", in_flight=6))
        sid = t3[1]["subscriber_id"]
        clock[0] += 30
        await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "t3-synthetic", {"in_flight": 7})
        clock[0] += 10
        await call("POST", f"/v1/catalog/subscriptions/{sid}/heartbeat", "t3-synthetic", {"in_flight": 1})
        clock[0] += 5
        listed = await call("GET", "/v1/catalog/subscriptions", "zeroclaw-msi-synthetic")
        one = await call("GET", "/v1/catalog/llm.chat", "catalog-synthetic")
        clock[0] += 181  # both leases expire; client totals stay
        after = await call("GET", "/v1/catalog/subscriptions", "catalog-synthetic")
        none = await call("GET", "/v1/catalog/llm.chat", "catalog-synthetic")
        anon = await call("GET", "/v1/catalog/subscriptions")
        return empty, honcho, listed, one, after, none, anon

    empty, honcho, listed, one, after, none, anon = _run(app, scenario)
    assert empty[0] == 200 and empty[1]["subscriptions"] == [] and empty[1]["clients"] == {}
    assert anon[0] in {401, 403}
    assert listed[0] == 200
    rows = {row["client"]: row for row in listed[1]["subscriptions"]}
    assert rows["t3"] == {
        "client": "t3", "public_id": "llm.chat", "subscriber_id": rows["t3"]["subscriber_id"],
        "priority": "interactive", "want_max": 2, "in_flight": 1, "max_in_flight": 7,
        "recommended_concurrency": 2, "heartbeat_age_s": 5, "age_s": 45, "expires_in_s": 175,
        "heartbeats": 2, "over_share_events": 2,  # subscribe 6 > 2, beat 7 > 2, beat 1 ok
    }
    assert rows["honcho"]["subscriber_id"] == honcho[1]["subscriber_id"]
    assert rows["honcho"]["heartbeats"] == rows["honcho"]["over_share_events"] == 0
    assert "token" not in json.dumps(listed[1]) and "synthetic" not in json.dumps(listed[1])
    assert listed[1]["clients"] == {
        "honcho": {"subscribes": 2, "replacements": 1, "expiries": 0, "over_share_events": 0},
        "t3": {"subscribes": 1, "replacements": 0, "expiries": 0, "over_share_events": 2},
    }
    assert one[0] == 200 and one[1]["subscribers"] == 2
    assert after[1]["subscriptions"] == []
    assert after[1]["clients"]["honcho"]["expiries"] == after[1]["clients"]["t3"]["expiries"] == 1
    assert after[1]["clients"]["t3"]["over_share_events"] == 2
    assert none[1]["subscribers"] == 0
