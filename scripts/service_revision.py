"""Immutable, content-addressed service revision records.

This module is a registry boundary for preview and future activation APIs.  It
does not write the live services registry or start a runtime.  A revision
contains the exact declarative service definition a job should pin; executable
fields are still rejected by ``validate_service_definition``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import MappingProxyType
from typing import Any

from gpu_manager_contracts import validate_service_definition


SERVICE_REVISION_SCHEMA = "service-revision.v1"


class ServiceRevisionError(ValueError):
    """A service revision is malformed or cannot be created."""


class RevisionConflict(ServiceRevisionError):
    """A compare-and-swap precondition did not match the catalog state."""


class RevisionState(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def service_revision_fingerprint(name: str, definition: Mapping[str, Any]) -> str:
    """Hash the service name and exact definition content."""
    try:
        encoded = json.dumps(
            {"name": name, "definition": _thaw(definition)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ServiceRevisionError("service definition must be JSON-compatible") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ServiceRevision:
    name: str
    revision: int
    fingerprint: str
    state: RevisionState
    definition: Mapping[str, Any]
    schema_version: str = SERVICE_REVISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "definition", _freeze(self.definition))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "state": self.state.value,
            "definition": _thaw(self.definition),
        }


def make_service_revision(
    name: str,
    definition: Mapping[str, Any],
    *,
    revision: int = 1,
    state: RevisionState | str = RevisionState.DRAFT,
) -> ServiceRevision:
    """Validate and create a content-addressed immutable revision."""
    errors = validate_service_definition(name, definition)
    if errors:
        raise ServiceRevisionError("invalid service definition: " + "; ".join(errors))
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ServiceRevisionError("revision must be a positive integer")
    try:
        revision_state = state if isinstance(state, RevisionState) else RevisionState(state)
    except ValueError as exc:
        raise ServiceRevisionError(f"unsupported revision state: {state!r}") from exc
    frozen = _freeze(definition)
    return ServiceRevision(
        name=name,
        revision=revision,
        fingerprint=service_revision_fingerprint(name, definition),
        state=revision_state,
        definition=frozen,
    )


def next_service_revision(existing: Mapping[str, Any] | None) -> int:
    """Return the next monotonically increasing revision number."""
    if existing is None:
        return 1
    value = existing.get("revision", existing.get("service_revision", 0))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ServiceRevisionError("existing revision must be a non-negative integer")
    return value + 1


def validate_service_revision(record: Mapping[str, Any]) -> list[str]:
    """Validate serialized revision metadata without mutating it."""
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["revision record must be an object"]
    if record.get("schema_version") != SERVICE_REVISION_SCHEMA:
        errors.append(f"schema_version must be {SERVICE_REVISION_SCHEMA!r}")
    name = record.get("name")
    definition = record.get("definition")
    revision = record.get("revision")
    if not isinstance(name, str) or not name:
        errors.append("name must be a non-empty string")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        errors.append("revision must be a positive integer")
    if not isinstance(definition, Mapping):
        errors.append("definition must be an object")
    elif isinstance(name, str) and name:
        errors.extend(validate_service_definition(name, definition))
    try:
        state = RevisionState(str(record.get("state")))
    except ValueError:
        errors.append("state must be draft, active or deprecated")
        state = None
    if state is not None and not isinstance(record.get("fingerprint"), str):
        errors.append("fingerprint must be a string")
    if (
        not errors
        and isinstance(name, str)
        and isinstance(definition, Mapping)
        and record.get("fingerprint") != service_revision_fingerprint(name, definition)
    ):
        errors.append("fingerprint does not match definition")
    return errors


def _state_copy(record: ServiceRevision, state: RevisionState) -> ServiceRevision:
    """Create a new immutable record with the same definition identity."""

    return ServiceRevision(
        name=record.name,
        revision=record.revision,
        fingerprint=record.fingerprint,
        state=state,
        definition=record.definition,
    )


class ServiceRevisionCatalog:
    """Small CAS-protected revision catalog for preview/activation fixtures.

    Definitions are immutable and content-addressed.  State changes create a
    new frozen record with the same revision/fingerprint, so a job that already
    pinned that identity remains valid while the catalog moves from draft to
    active or deprecated.  The catalog is intentionally in-memory; a durable
    registry store and controller integration remain separate release gates.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[int, ServiceRevision]] = {}
        self._lock = threading.RLock()

    def _ensure_loaded_locked(self) -> None:
        """Hook for persistent catalogs; the in-memory catalog is ready."""

    def _after_mutation_locked(self) -> None:
        """Hook for persistent catalogs; the in-memory catalog needs no I/O."""

    def _records_copy_locked(self) -> dict[str, dict[int, ServiceRevision]]:
        return {name: dict(revisions) for name, revisions in self._records.items()}

    def _commit_locked(
        self, previous: dict[str, dict[int, ServiceRevision]]
    ) -> None:
        """Persist a mutation and restore the prior view if persistence fails."""

        try:
            self._after_mutation_locked()
        except Exception:
            self._records = previous
            raise

    def latest(self, name: str) -> ServiceRevision | None:
        with self._lock:
            self._ensure_loaded_locked()
            revisions = self._records.get(name, {})
            return revisions[max(revisions)] if revisions else None

    def active(self, name: str) -> ServiceRevision | None:
        with self._lock:
            self._ensure_loaded_locked()
            for record in self._records.get(name, {}).values():
                if record.state is RevisionState.ACTIVE:
                    return record
        return None

    def get(self, name: str, revision: int) -> ServiceRevision:
        with self._lock:
            self._ensure_loaded_locked()
            try:
                return self._records[name][revision]
            except KeyError as exc:
                raise ServiceRevisionError(
                    f"unknown service revision {name!r}@{revision}"
                ) from exc

    def publish(
        self,
        name: str,
        definition: Mapping[str, Any],
        *,
        expected_latest_fingerprint: str | None = None,
    ) -> ServiceRevision:
        """Publish a new draft, enforcing an optional latest-revision CAS."""

        with self._lock:
            self._ensure_loaded_locked()
            current = self.latest(name)
            observed = current.fingerprint if current else None
            if (
                expected_latest_fingerprint is not None
                and expected_latest_fingerprint != observed
            ):
                raise RevisionConflict(
                    f"service {name!r} changed; expected {expected_latest_fingerprint!r}, observed {observed!r}"
                )
            candidate_fingerprint = service_revision_fingerprint(name, definition)
            if current is not None and current.fingerprint == candidate_fingerprint:
                return current
            revision = (current.revision + 1) if current else 1
            record = make_service_revision(
                name,
                definition,
                revision=revision,
                state=RevisionState.DRAFT,
            )
            previous = self._records_copy_locked()
            self._records.setdefault(name, {})[revision] = record
            self._commit_locked(previous)
            return record

    def activate(
        self,
        name: str,
        revision: int,
        *,
        expected_fingerprint: str | None = None,
        expected_active_fingerprint: str | None = None,
    ) -> ServiceRevision:
        """Atomically make one existing revision active and deprecate its peer."""

        with self._lock:
            self._ensure_loaded_locked()
            target = self.get(name, revision)
            if expected_fingerprint is not None and target.fingerprint != expected_fingerprint:
                raise RevisionConflict("target revision fingerprint does not match")
            current_active = self.active(name)
            observed_active = current_active.fingerprint if current_active else None
            if (
                expected_active_fingerprint is not None
                and expected_active_fingerprint != observed_active
            ):
                raise RevisionConflict(
                    f"active service {name!r} changed; expected {expected_active_fingerprint!r}, observed {observed_active!r}"
                )
            if target.state is RevisionState.DEPRECATED:
                raise ServiceRevisionError("deprecated revision cannot be activated")
            previous = self._records_copy_locked()
            if current_active is not None and current_active.revision != target.revision:
                self._records[name][current_active.revision] = _state_copy(
                    current_active, RevisionState.DEPRECATED
                )
            active = _state_copy(target, RevisionState.ACTIVE)
            self._records[name][revision] = active
            self._commit_locked(previous)
            return active

    def deprecate(
        self,
        name: str,
        revision: int,
        *,
        expected_fingerprint: str | None = None,
    ) -> ServiceRevision:
        """Deprecate a revision without deleting its pinned identity."""

        with self._lock:
            self._ensure_loaded_locked()
            target = self.get(name, revision)
            if expected_fingerprint is not None and target.fingerprint != expected_fingerprint:
                raise RevisionConflict("revision fingerprint does not match")
            if target.state is RevisionState.DEPRECATED:
                return target
            previous = self._records_copy_locked()
            deprecated = _state_copy(target, RevisionState.DEPRECATED)
            self._records[name][revision] = deprecated
            self._commit_locked(previous)
            return deprecated

    def export(self, *, sanitize: bool = False) -> dict[str, Any]:
        """Return a deterministic, optionally sanitized catalog export."""

        with self._lock:
            self._ensure_loaded_locked()
            document = {
                "schema_version": "service-revision-catalog.v1",
                "services": {
                    name: [
                        record.to_dict()
                        for _, record in sorted(revisions.items())
                    ]
                    for name, revisions in sorted(self._records.items())
                },
            }
        if sanitize:
            try:
                from sanitize_registry import sanitize_registry

                return sanitize_registry(document)
            except ImportError:
                pass
        return document


_REVISION_CATALOG_SCHEMA = "service-revision-catalog.v1"


def service_revision_catalog_from_document(
    raw: Mapping[str, Any] | None,
) -> ServiceRevisionCatalog:
    """Load and fully validate a serialized catalog without performing I/O."""
    if raw is None:
        return ServiceRevisionCatalog()
    if not isinstance(raw, Mapping) or raw.get("schema_version") != _REVISION_CATALOG_SCHEMA:
        raise ServiceRevisionError("unsupported service revision catalog schema")
    services = raw.get("services")
    if not isinstance(services, Mapping):
        raise ServiceRevisionError("service revision catalog services must be an object")

    loaded: dict[str, dict[int, ServiceRevision]] = {}
    for service_name, values in services.items():
        if not isinstance(service_name, str) or not service_name:
            raise ServiceRevisionError("service revision catalog has an invalid service name")
        if not isinstance(values, list):
            raise ServiceRevisionError(
                f"service revision catalog {service_name!r} must contain a list"
            )
        revisions: dict[int, ServiceRevision] = {}
        for value in values:
            if not isinstance(value, Mapping):
                raise ServiceRevisionError("service revision catalog record must be an object")
            errors = validate_service_revision(value)
            if errors:
                raise ServiceRevisionError(
                    f"invalid service revision {service_name!r}: " + "; ".join(errors)
                )
            if value.get("name") != service_name:
                raise ServiceRevisionError(
                    f"service revision key does not match {service_name!r}"
                )
            revision = int(value["revision"])
            if revision in revisions:
                raise ServiceRevisionError(
                    f"duplicate service revision {service_name!r}@{revision}"
                )
            revisions[revision] = ServiceRevision(
                name=service_name,
                revision=revision,
                fingerprint=str(value["fingerprint"]),
                state=RevisionState(str(value["state"])),
                definition=value["definition"],
            )
        if sum(record.state is RevisionState.ACTIVE for record in revisions.values()) > 1:
            raise ServiceRevisionError(
                f"service {service_name!r} has multiple active revisions"
            )
        loaded[service_name] = revisions

    catalog = ServiceRevisionCatalog()
    catalog._records = loaded
    return catalog


class FileServiceRevisionCatalog(ServiceRevisionCatalog):
    """Atomic single-writer JSON catalog for local restart/activation fixtures.

    This is deliberately not a distributed registry.  A caller must provide a
    single writer (or an external lock) when multiple processes can publish at
    once.  Every mutation is written through a same-directory temporary file,
    fsync and atomic replacement.  Corrupt, tampered or unknown-schema files
    fail closed before the catalog is exposed.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__()
        self._path = Path(path)
        self._loaded = False

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ServiceRevisionError("unable to read service revision catalog") from exc
        except json.JSONDecodeError as exc:
            raise ServiceRevisionError("service revision catalog is not valid JSON") from exc
        self._records = service_revision_catalog_from_document(raw)._records

    def _after_mutation_locked(self) -> None:
        document = {
            "schema_version": _REVISION_CATALOG_SCHEMA,
            "services": {
                name: [
                    record.to_dict()
                    for _, record in sorted(revisions.items())
                ]
                for name, revisions in sorted(self._records.items())
            },
        }
        parent = self._path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self._path.name}.", dir=parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(document, handle, sort_keys=True, separators=(",", ":"))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._path)
                try:
                    directory_fd = os.open(parent, os.O_RDONLY)
                except OSError:
                    directory_fd = None
                if directory_fd is not None:
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except OSError:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise ServiceRevisionError("unable to persist service revision catalog") from exc


__all__ = [
    "RevisionState",
    "RevisionConflict",
    "SERVICE_REVISION_SCHEMA",
    "FileServiceRevisionCatalog",
    "ServiceRevision",
    "ServiceRevisionCatalog",
    "ServiceRevisionError",
    "make_service_revision",
    "next_service_revision",
    "service_revision_fingerprint",
    "service_revision_catalog_from_document",
    "validate_service_revision",
]
