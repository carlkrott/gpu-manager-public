# Security

This repository is a source-only control-plane publication. It is not a production deployment guide and does not contain a configured workload, model, credential, service unit or host state.

## Reporting route

Before publication, the repository owner must populate the repository security contact and private reporting route. Do not put suspected credentials, private paths, production snapshots or audit reports in a public issue.

## Authentication boundary

- The controller and standalone broker use the shared `scripts/api_auth.py` policy.
- Missing, unreadable or empty credential configuration fails closed on protected routes.
- Supplied bearer tokens are compared in constant time.
- There is no implicit loopback or proxy-header authentication exemption.
- `GPU_MANAGER_ALLOW_UNAUTHENTICATED_LOOPBACK=1` is an explicit compatibility opt-in and is off in examples and CI.
- `GET`/`HEAD /health/live` is the minimal liveness exception. Dashboard bootstrap, registry data, queue/result data, diagnostics, readiness, metrics, broker administration and mutations remain protected.
- WebSocket clients authenticate in the first application frame before application state is sent.
- Dashboard credentials are entered in memory only; tokens are not stored in URLs, cookies, local/session storage or rendered HTML.

Load credentials from a permission-controlled token file through `GPU_MANAGER_API_TOKEN_FILE` or the service-manager credential-directory contract. Never put a token in source, a command-line argument, a URL, a fixture, a log or a commit message.

## Configuration trust boundary

Operator-authored service and pipeline configuration is executable control-plane input. Review it as code. The declarative registration and execution-boundary checks reject unknown ownership, unsafe executable fields and conflicting worker/external ownership rather than silently choosing a fallback.

External provider endpoints require explicit provider ownership and endpoint policy. Keep real endpoints and credentials in operator configuration, not in examples or public documentation.

## Data handling

Do not commit:

- credentials, private keys, certificates or tokens;
- model files, model caches or model hashes from a private deployment;
- service registries, system state, queue state, results or evidence;
- production host paths, hardware identifiers, topology or private audit material.

Run the local payload checker and a secret scanner against the exact export before any release review. Keep detailed scanner findings and private exact-identifier screening outside the repository.

`.gitleaks.toml` extends the pinned scanner defaults with only two path-and-value exact matches for known non-secret public identifiers that trigger the generic-key heuristic. All other findings fail the scan; this is not a baseline or blanket scanner bypass.

## Safe reporting content

A useful report includes the affected file and a synthetic reproduction. Redact tokens, private addresses, hostnames, absolute paths, model identity and production state. If an issue could expose a credential or enable control-plane access, use the owner’s private route rather than a public issue.

## Scope

The local tests prove source portability, authentication seams, schema boundaries and fixture-only behavior. They do not prove internet exposure safety, production deployment correctness, GPU placement, model quality or live service health.
