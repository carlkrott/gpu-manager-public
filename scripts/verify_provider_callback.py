#!/usr/bin/env python3
"""Qualify one restart-safe external provider callback over loopback HTTP.

This is a disposable mechanism check, not a provider test suite. It starts a
private HTTP server, submits one operation, persists the returned handle in the
coordinator state, replaces the coordinator, and polls that handle. The receipt
proves that an accepted operation is not submitted twice. No GPU, credential,
external endpoint, or shared queue is used.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any
from urllib.request import Request, urlopen

from pipeline_coordinator import (
    InMemoryCoordinatorStore,
    LeafGenerationClient,
    PipelineCoordinator,
    PipelineRunState,
)
from workflow_coordinator_runtime import (
    CoordinatorExternalExecutor,
    ReviewedAdapterRegistry,
)


class _ProviderHandler(BaseHTTPRequestHandler):
    submit_calls = 0
    poll_calls = 0

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/submit":
            self._json(404, {"error": "not_found"})
            return
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).submit_calls += 1
        self._json(202, {"status": "in_flight", "provider_handle": "op-1"})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/poll/op-1":
            self._json(404, {"error": "not_found"})
            return
        type(self).poll_calls += 1
        self._json(
            200,
            {"status": "completed", "output": {"provider_receipt": "op-1-complete"}},
        )

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _NoopLeaf(LeafGenerationClient):
    def submit(
        self,
        _stage: dict,
        _parent_run_id: str,
        _parent_state: PipelineRunState,
        _idempotency_key: str,
        _trusted_config: dict,
    ) -> str:
        raise AssertionError("provider callback fixture unexpectedly reached a leaf")

    def poll(self, _child_job_id: str) -> dict:
        raise AssertionError("provider callback fixture unexpectedly polled a leaf")

    def cancel(self, _child_job_id: str) -> bool:
        return False


def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=2.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("provider returned a non-object JSON response")
    return value


def verify_provider_callback() -> dict[str, Any]:
    _ProviderHandler.submit_calls = 0
    _ProviderHandler.poll_calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        registry = ReviewedAdapterRegistry(
            {("callback-provider", "callback.adapter"): "external"}
        )

        def submit_handler(
            stage: dict, _state: PipelineRunState, idempotency_key: str
        ) -> dict[str, Any]:
            return _http_json(
                "POST",
                f"{stage['params']['endpoint']}/submit",
                {"stage": stage["id"], "idempotency_key": idempotency_key},
            )

        def poll_handler(
            stage: dict,
            _state: PipelineRunState,
            _idempotency_key: str,
            provider_handle: str,
        ) -> dict[str, Any]:
            return _http_json(
                "GET", f"{stage['params']['endpoint']}/poll/{provider_handle}"
            )

        external = CoordinatorExternalExecutor(
            registry,
            {("callback-provider", "callback.adapter"): submit_handler},
            poll_handlers={("callback-provider", "callback.adapter"): poll_handler},
        )
        store = InMemoryCoordinatorStore()
        compiled = {
            "pipeline_id": "provider-callback-fixture",
            "workflow_fingerprint": "c" * 64,
            "stages": [
                {
                    "id": "research",
                    "kind": "preparation",
                    "provider": "callback-provider",
                    "adapter": "callback.adapter",
                    "params": {"endpoint": endpoint},
                }
            ],
        }
        first = PipelineCoordinator(store, registry, external, _NoopLeaf())
        if not first.submit(
            "job-callback", "run-callback", compiled, {"prompt": "fixture"}
        ):
            raise RuntimeError("initial parent admission failed")
        claimed = store.dequeue_parent("callback-1")
        if claimed is None:
            raise RuntimeError("callback parent was not queued")
        in_flight = first.advance(claimed)
        if in_flight.status != "running":
            raise RuntimeError(f"provider did not remain in flight: {in_flight.status}")
        persisted = store.load_state("run-callback")
        if persisted is None or not persisted.stage_attempts["research"][-1].provider_handle:
            raise RuntimeError("provider handle was not persisted")

        recovered = store.reclaim_orphaned_parents("callback-1")
        if len(recovered) != 1:
            raise RuntimeError("in-flight parent was not recoverable")
        replacement = PipelineCoordinator(store, registry, external, _NoopLeaf())
        terminal = replacement.advance(recovered[0])
        if terminal.status != "completed":
            raise RuntimeError(f"provider callback did not complete: {terminal.status}")
        if _ProviderHandler.submit_calls != 1 or _ProviderHandler.poll_calls != 1:
            raise RuntimeError(
                "callback replayed unexpectedly: "
                f"submits={_ProviderHandler.submit_calls}, polls={_ProviderHandler.poll_calls}"
            )
        return {
            "status": "passed",
            "provider": "callback-provider",
            "adapter": "callback.adapter",
            "submit_calls": _ProviderHandler.submit_calls,
            "poll_calls": _ProviderHandler.poll_calls,
            "provider_handle_persisted": True,
            "runtime_replaced": True,
            "shared_queue_touched": False,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main() -> int:
    try:
        print(json.dumps(verify_provider_callback(), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["verify_provider_callback"]
