# Third-party notices

No third-party source code is vendored into the candidate tree. Runtime and development dependencies are installed from the exact pins in `requirements.lock` and `requirements-dev.lock`; their upstream license texts remain the responsibility of the corresponding distributions.

The current dependency inventory is:

| Package | Locked version | Declared license expression |
|---|---:|---|
| aiohttp | 3.13.5 | Apache-2.0 AND MIT |
| redis | 8.1.0 | MIT |
| Pillow | 12.2.0 | MIT-CMU |
| jsonschema | 4.25.1 | MIT |
| PyYAML | 6.0.3 | MIT |
| xmltodict | 1.0.4 | MIT |
| pytest | 9.0.3 | MIT |
| iniconfig | 2.3.0 | MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.6.0 | MIT |
| Pygments | 2.20.0 | BSD-2-Clause |

Transitive runtime packages are pinned in `requirements.lock` with hashes. Development pins are explicit in `requirements-dev.lock`; resolve them only from the configured package index and review the installed distribution metadata before a release.

GitHub Actions used by the source-only workflow are referenced by immutable commit SHA and are not copied into this repository. Their upstream licenses and terms remain applicable when the workflow is run; the gitleaks action is configured to redact findings and not publish comments, summaries or scan artifacts.

This project is licensed under the MIT License; see `LICENSE`. No model license, model weight, private workflow, production service definition or hardware-specific asset is redistributed or implied.
