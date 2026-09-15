# GPUManager

GPUManager is a small control plane for registering services, routing work, and coordinating durable queue operations. This repository is a source-only publication: it contains software, neutral fixtures and tests, not a configured workload or a model distribution.

## Safety boundary

A clean checkout starts with no configured services, no model paths, no credentials, no deployment units and no production state. The supplied examples are schema fixtures. The Combined Gemma example is disabled at the member level and its demo uses only an in-memory repository and synthetic member snapshots.

The controller and standalone broker use fail-closed bearer authentication by default. Protected reads and all mutations require a token loaded from `GPU_MANAGER_API_TOKEN_FILE` or the systemd `CREDENTIALS_DIRECTORY` contract. The opt-in `GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK=1` compatibility mode is off by default and is not enabled by the examples or CI.

The default bind is loopback. A wider bind does not disable authentication. Do not place a token in a URL, cookie, browser storage or source file.

## Capability map

- `scripts/gpu-manager.py` — controller HTTP API, dashboard, mesh and stdio MCP surface.
- `scripts/api_auth.py` — shared fail-closed HTTP and WebSocket authentication policy.
- `scripts/queue_engine.py`, `scripts/bundle_lifecycle.py` — durable queue and lifecycle coordination.
- `scripts/combined_gemma_broker_service.py` and `scripts/gemma_broker/` — standalone broker composition and durable broker contracts.
- `scripts/declarative_workflow_registration.py`, `scripts/execution_boundary.py` and pipeline modules — declarative registration and worker/external ownership boundaries.
- `scripts/runtime_host_supervisor.py` and runtime modules — checked host-runtime contracts.
- `scripts/check_public_payload.py` and `scripts/export_public_source.py` — local publication gates.
- `examples/empty/` — neutral empty registry.
- `examples/combined-gemma/` — candidate-mode broker schema and fixture-only demo.

Installation-adjacent adapters are optional integrations. They require explicit operator configuration when invoked and do not load private source roots at import time.

## Setup

Python 3.11 or newer is required. The hash-locked install is validated for
CPython 3.14 on the GitHub-hosted `ubuntu-24.04` x86_64 runner; regenerate
hash entries before using another interpreter or platform.

    python3 -m venv .venv
    .venv/bin/python -m pip install --requirement requirements-dev.lock

The lock files describe Python packages only. They do not download models, start services or configure a host.

## Verification

Run the source-only publication tests:

    PYTHONPATH=scripts .venv/bin/python -m pytest -q tests/publication

Run the public payload gate from the repository root:

    PYTHONPATH=scripts .venv/bin/python scripts/check_public_payload.py .

The exporter requires a new destination and binds the result to `release/public-files.json`:

    .venv/bin/python scripts/export_public_source.py /tmp/gpumanager-public-export --source-root .

The exported directory must be checked again with `check_public_payload.py`. Keep scanner reports and any exact-identifier screening outside the repository.

## Neutral examples

Validate the empty registry and Combined Gemma fixture without network access:

    python3 examples/combined-gemma/demo.py --dry-run

Run the bounded fixture-only lifecycle demonstration:

    python3 examples/combined-gemma/demo.py --mock-demo

The mock command does not produce model output or claim production health. See `examples/combined-gemma/README.md` for the exact scope.

## Operator configuration

Real deployments require an operator-owned registry and credential boundary. Supply configuration through reviewed environment or service-manager inputs; keep real service definitions, model paths, model hashes, host topology, keys, state, results and evidence outside this repository.

The broker has no implicit configuration-file discovery. Supply `--config` or `GPU_MANAGER_CONFIG_PATH` explicitly. Protected HTTP routes need a token file. Review endpoint ownership and external-provider configuration before enabling a service.

No internet-readiness, GPU qualification, model-quality or production-health claim is made by this source-only test suite.

## License status

This project is licensed under the MIT License; see `LICENSE`. Dependency and workflow terms are documented in `THIRD_PARTY_NOTICES.md`. Model, model-weight, private-workflow and production-service licenses are not redistributed or implied.
