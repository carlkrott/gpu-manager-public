from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from combined_gemma_broker_service import create_standalone_app


SCOPED_TOKEN = "synthetic-kda-gemma-token"
ADMIN_TOKEN = "synthetic-admin-token"


def _credential(tmp_path: Path, name: str, value: str) -> Path:
    path = tmp_path / name
    path.write_text(value, encoding="ascii")
    path.chmod(0o600)
    return path


def _service():
    service = MagicMock()
    service.settings.config.ordered_members = ("gemma-4-e4b", "Gemma-CPU")
    service.settings.registry = {}
    service.handle_proxy = AsyncMock(side_effect=lambda **_kwargs: web.json_response({"ok": True}))
    return service


def test_headerless_scoped_credential_is_pinned_to_its_default_model(monkeypatch, tmp_path):
    admin_file = _credential(tmp_path, "admin-token", ADMIN_TOKEN)
    scoped_file = _credential(tmp_path, "scoped-token", SCOPED_TOKEN)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(admin_file))
    monkeypatch.setenv("GPU_MANAGER_KDA_GEMMA_TOKEN_FILE", str(scoped_file))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    service = _service()
    app = create_standalone_app(service)

    async def run():
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {SCOPED_TOKEN}"}
            in_scope = await client.post(
                "/v1/chat/completions", json={"model": "gemma-4-e4b"}, headers=headers
            )
            assert in_scope.status == 200, in_scope.status
            forwarded = service.handle_proxy.await_args.kwargs["headers"]
            assert forwarded.get("X-Backend") == "Gemma"

            out_of_scope = await client.post(
                "/v1/chat/completions", json={"model": "gpt-6-luna"}, headers=headers
            )
            assert out_of_scope.status == 403, out_of_scope.status
            assert (await out_of_scope.json())["code"] == "scoped_default_model_mismatch"

            # The refused request never became a submission.
            assert service.handle_proxy.await_count == 1

            unscoped = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-6-luna"},
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "X-Backend": "Gemma"},
            )
            assert unscoped.status == 200, unscoped.status
            assert service.handle_proxy.await_args.kwargs["headers"].get("X-Backend") == "Gemma"

    asyncio.run(run())


def test_scoped_kda_token_is_limited_to_gemma_chat_route(monkeypatch, tmp_path):
    admin_file = _credential(tmp_path, "admin-token", ADMIN_TOKEN)
    scoped_file = _credential(tmp_path, "scoped-token", SCOPED_TOKEN)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(admin_file))
    monkeypatch.setenv("GPU_MANAGER_KDA_GEMMA_TOKEN_FILE", str(scoped_file))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    service = _service()
    app = create_standalone_app(service)

    async def run():
        async with TestClient(TestServer(app)) as client:
            scoped = {"Authorization": f"Bearer {SCOPED_TOKEN}", "X-Backend": "Gemma"}
            # "X-Backend: Gemma" is an UNPINNED submission, so it is held to the
            # credential's model scope: it is admitted only for the in-scope model.
            # (A different model here is refused by the gate; that is covered by
            # test_scoped_credential_x_backend_gemma_cannot_bypass_model_scope.)
            chat = await client.post("/v1/chat/completions", json={"model": "gemma-4-e4b"}, headers=scoped)
            assert chat.status == 200

            for method, path in (
                ("POST", "/v1/gemma/jobs"),
                ("POST", "/v1/gemma/members/Gemma-CPU/drain"),
                ("POST", "/v1/gemma/members/Gemma-CPU/rejoin"),
            ):
                response = await client.request(method, path, json={}, headers=scoped)
                assert response.status == 401, (method, path, response.status)

            admin = await client.post(
                "/v1/chat/completions",
                json={"model": "gemma"},
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "X-Backend": "Gemma"},
            )
            assert admin.status == 200

            no_bearer = await client.post("/v1/chat/completions", json={"model": "gemma"}, headers={"X-Backend": "Gemma"})
            assert no_bearer.status == 401

    asyncio.run(run())


# --- X-Backend: Gemma is an unpinned submission, so it must not skip the gate ---
#
# The logical routing-group name is deliberately exempt from the member check in
# api_auth.scoped_gemma_request_is_authorized, and gemma_broker/compat.py maps any
# backend whose name casefolds to "gemma" to pinned_member=None. So a scoped
# credential sending "X-Backend: Gemma" was admitted, skipped the default-pin gate
# (which was guarded by "X-Backend not in request.headers") and reached handle_proxy
# as an UNPINNED, model-unchecked submission. Such a request must be gated exactly
# like a header-less one.


def _gate_world(monkeypatch, tmp_path):
    admin_file = _credential(tmp_path, "admin-token", ADMIN_TOKEN)
    scoped_file = _credential(tmp_path, "scoped-token", SCOPED_TOKEN)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(admin_file))
    monkeypatch.setenv("GPU_MANAGER_KDA_GEMMA_TOKEN_FILE", str(scoped_file))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    service = _service()
    return service, create_standalone_app(service)


def test_scoped_credential_x_backend_gemma_cannot_bypass_model_scope(monkeypatch, tmp_path):
    service, app = _gate_world(monkeypatch, tmp_path)

    async def run():
        async with TestClient(TestServer(app)) as client:
            # The routing-group name is recognised case-insensitively here exactly as
            # it is in the controller gate, so casing cannot decide admission.
            for backend in ("Gemma", "gemma", "GEMMA"):
                scoped = {"Authorization": f"Bearer {SCOPED_TOKEN}", "X-Backend": backend}
                for model in ("gpt-6-luna", "gemini-3.8-flash", None):
                    payload: dict = {"messages": []}
                    if model is not None:
                        payload["model"] = model
                    refused = await client.post("/v1/chat/completions", json=payload, headers=scoped)
                    assert refused.status == 403, (backend, model, refused.status)
                    assert (await refused.json())["code"] == "scoped_default_model_mismatch"
                # The refused requests never became submissions.
                assert service.handle_proxy.await_count == 0, backend

                for model in ("gemma", "gemma-4-E4B", "gemma-4-e4b"):
                    admitted = await client.post(
                        "/v1/chat/completions", json={"model": model}, headers=scoped
                    )
                    assert admitted.status == 200, (backend, model, admitted.status)
                    # In scope means admitted AND sent to the default target (the Gemma group).
                    assert service.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == "Gemma"
                    service.handle_proxy.reset_mock()

    asyncio.run(run())


def test_explicit_member_pin_and_unscoped_credential_are_not_narrowed(monkeypatch, tmp_path):
    service, app = _gate_world(monkeypatch, tmp_path)

    async def run():
        async with TestClient(TestServer(app)) as client:
            # A real member pin is an explicit, checked choice: it still passes through
            # untouched, whatever the body model says.
            pinned = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-6-luna"},
                headers={"Authorization": f"Bearer {SCOPED_TOKEN}", "X-Backend": "Gemma-CPU"},
            )
            assert pinned.status == 200, pinned.status
            assert service.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == "Gemma-CPU"

            # An unscoped/admin credential keeps the logical group and any model.
            unscoped = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-6-luna"},
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "X-Backend": "Gemma"},
            )
            assert unscoped.status == 200, unscoped.status
            assert service.handle_proxy.await_args.kwargs["headers"]["X-Backend"] == "Gemma"

    asyncio.run(run())


def test_honcho_scope_routes_unpinned_and_reads_capacity_metrics(monkeypatch, tmp_path):
    admin_file = _credential(tmp_path, "admin-token", ADMIN_TOKEN)
    kda_file = _credential(tmp_path, "kda-token", SCOPED_TOKEN)
    honcho_token = "synthetic-honcho-gemma-token"
    honcho_file = _credential(tmp_path, "honcho-token", honcho_token)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(admin_file))
    monkeypatch.setenv("GPU_MANAGER_KDA_GEMMA_TOKEN_FILE", str(kda_file))
    monkeypatch.setenv("GPU_MANAGER_HONCHO_GEMMA_TOKEN_FILE", str(honcho_file))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    service = _service()
    service.broker_metrics = AsyncMock(return_value={"capacity": {"available": 2}})
    app = create_standalone_app(service)

    async def run():
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {honcho_token}"}
            for extra in ({}, {"X-Backend": "Gemma"}):
                response = await client.post(
                    "/v1/chat/completions", json={"model": "gemma"},
                    headers={**headers, **extra},
                )
                assert response.status == 200, response.status
                forwarded = service.handle_proxy.await_args.kwargs["headers"]
                assert forwarded.get("X-Backend") == extra.get("X-Backend")
                assert service.handle_proxy.await_args.kwargs["body"] == b'{"model": "gemma"}'

            metrics = await client.get(
                "/v1/metrics/broker?durable=0", headers=headers
            )
            assert metrics.status == 200, metrics.status
            assert await metrics.json() == {"capacity": {"available": 2}}
            service.broker_metrics.assert_awaited_once_with(include_durable_counts=False)

            # The KDA credential still goes to its default target and cannot read metrics.
            kda = {"Authorization": f"Bearer {SCOPED_TOKEN}"}
            kda_chat = await client.post(
                "/v1/chat/completions", json={"model": "gemma-4-e4b"}, headers=kda
            )
            assert kda_chat.status == 200
            assert service.handle_proxy.await_args.kwargs["headers"].get("X-Backend") == "Gemma"
            kda_metrics = await client.get(
                "/v1/metrics/broker?durable=0", headers=kda
            )
            assert kda_metrics.status == 403

    asyncio.run(run())


def test_honcho_scope_rejects_wrong_model_and_member_pin_before_proxy(monkeypatch, tmp_path):
    admin_file = _credential(tmp_path, "admin-token", ADMIN_TOKEN)
    kda_file = _credential(tmp_path, "kda-token", SCOPED_TOKEN)
    honcho_token = "synthetic-honcho-gemma-token"
    honcho_file = _credential(tmp_path, "honcho-token", honcho_token)
    monkeypatch.setenv("GPU_MANAGER_API_TOKEN_FILE", str(admin_file))
    monkeypatch.setenv("GPU_MANAGER_KDA_GEMMA_TOKEN_FILE", str(kda_file))
    monkeypatch.setenv("GPU_MANAGER_HONCHO_GEMMA_TOKEN_FILE", str(honcho_file))
    monkeypatch.delenv("GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK", raising=False)
    service = _service()
    app = create_standalone_app(service)

    async def run():
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {honcho_token}"}
            for payload, extra in (
                ({"model": "gemma-4-e4b"}, {}),
                ({"model": "other-model"}, {}),
                ({"model": "gemma"}, {"X-Backend": "Gemma-CPU"}),
                ({"model": "gemma"}, {"X-Backend": "unknown-member"}),
            ):
                response = await client.post(
                    "/v1/chat/completions", json=payload,
                    headers={**headers, **extra},
                )
                # The shared middleware maps out-of-scope principals to its
                # generic 401 response before the standalone handler runs.
                assert response.status == 401, (payload, extra, response.status)
            assert service.handle_proxy.await_count == 0

    asyncio.run(run())
