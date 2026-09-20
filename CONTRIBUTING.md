# Contributing

Contributions should keep this repository source-only, portable and fail-closed.

## Before editing

- Read `README.md` and `SECURITY.md`.
- Work only in a clean candidate checkout or a dedicated branch.
- Do not copy production registries, service units, model files, credentials, state, results or private audit material.
- Use synthetic names, loopback fixture ports and in-memory fakes in tests.

## Changes

- Keep runtime behavior and examples separate: examples must not load operator configuration or private runtime roots.
- Preserve explicit authentication and execution-ownership boundaries.
- Add a focused regression for each changed contract.
- Do not weaken a failing privacy or authorization test to make the suite green.
- Keep optional integrations importable without their deployment overlay and fail clearly only when invoked.

## Verification

From the repository root, export the public payload to a guaranteed-new temporary destination and then check that export together with the receipt the exporter just generated:

    PYTHONPATH=scripts python3 -m pytest -q tests/publication
    EXPORT_DIR="$(mktemp -d)/gpumanager-public-export"
    mkdir -p "$(dirname "$EXPORT_DIR")"
    PYTHONPATH=scripts python3 scripts/export_public_source.py --manifest release/public-files.json "$EXPORT_DIR"
    PYTHONPATH=scripts python3 scripts/check_public_payload.py "$EXPORT_DIR" --manifest "$EXPORT_DIR/release/export-manifest.json"
    python3 examples/combined-gemma/demo.py --dry-run
    python3 examples/combined-gemma/demo.py --mock-demo

The checker runs against the freshly built export and the receipt the exporter wrote next to it; the developer source tree itself is not checked or cleaned. For public file-set changes, update only the path/disposition entries in `release/public-files.json`, re-run the export, then rerun the checker against that exact export. Do not hand-edit receipt fields or reuse an old export directory. There is no numeric coverage gate.

## Pull requests

Describe:

- the contract changed;
- the tests and checker commands run;
- any warnings or limitations;
- whether the manifest changed;
- whether the project license or third-party notice inventory changed.

Never include secrets or private scanner reports in a pull request. Do not add deployment, publication or release automation without explicit owner authorization.
