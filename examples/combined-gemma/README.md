# Neutral Combined Gemma example

This directory is a source-only broker example. It is safe to inspect and test without credentials, Redis, models, GPUs, systemd, Docker, or internet access.

The persisted `services.json` is candidate-mode configuration:

- `combined_gemma_broker.enabled` is `true` because the candidate validator requires the broker contract to be enabled.
- The three example members are all `enabled: false`, so the example cannot activate a backend.
- `redis_namespace` uses the isolated `qual:combined-gemma:example:` candidate namespace.
- Member names, ports, and endpoints are synthetic schema fixtures. They are not deployment instructions.
- No model identifier, model path, model hash, GPU identity, PCI identifier, credential, or download URL is present.

The member order is explicit and stable:

1. `member-primary`
2. `member-secondary`
3. `member-fallback`

The broker configuration is a closed contract. Unknown or missing broker keys are rejected by `BrokerConfig.from_dict()`.

## Dry-run validation

From the repository root:

    python3 examples/combined-gemma/demo.py --dry-run

This parses the example, validates the registry, validates the candidate broker contract, builds the member configuration and prints JSON evidence. It does not open a socket or contact Redis.

The demo defaults to the same dry-run behavior when no flag is supplied.

## Fixture-only mock demo

    python3 examples/combined-gemma/demo.py --mock-demo

This constructs the real candidate `BrokerRuntime` with `InMemoryJobRepository`, injected synthetic member snapshots and a no-op transport. It demonstrates:

- rejection when all simulated members are unhealthy;
- a queued request;
- queued cancellation;
- deterministic terminal completion.

The command binds short-lived loopback fixture listeners on OS-assigned ports only so the output can show bounded fixture resources. It never connects to those listeners, the configured example ports, Redis, systemd, Docker, a model or a production service. Every event is labelled `simulated`, and the terminal result explicitly says that no model output was generated.

The mock run closes its fixture sockets and runtime before exiting. It is not an inference or production-health demonstration.

## Using the JSON as a starting point

Treat the file as a schema fixture. An operator who builds a real deployment must supply and review the external runtime, service ownership, credentials, authentication policy and lifecycle controls separately. Keep those values outside this public example and pass them through the supported operator configuration boundary.
