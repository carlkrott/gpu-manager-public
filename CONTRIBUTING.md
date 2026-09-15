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

From the repository root:

    PYTHONPATH=scripts python3 -m pytest -q tests/publication
    PYTHONPATH=scripts python3 scripts/check_public_payload.py .
    python3 examples/combined-gemma/demo.py --dry-run
    python3 examples/combined-gemma/demo.py --mock-demo

For manifest-affecting changes, refresh the SHA-256 and size records in `release/public-files.json`, export to a new destination, and rerun the checker against the exact export. Do not reuse an old export directory.

## Pull requests

Describe:

- the contract changed;
- the tests and checker commands run;
- any warnings or limitations;
- whether the manifest changed;
- whether the project license or third-party notice inventory changed.

Never include secrets or private scanner reports in a pull request. Do not add deployment, publication or release automation without explicit owner authorization.
