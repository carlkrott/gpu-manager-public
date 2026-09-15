"""Controller-side bridge from bundle configuration to checked host runtimes."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import time
import uuid
from typing import Any

from runtime_contracts import (
    model_set_fingerprint,
    validate_model_set,
    validate_runtime_profile,
)
from runtime_host_client import HostSupervisorRuntimeAdapter
from runtime_state import FileRuntimeStateStore, RuntimeDesiredState, RuntimeInstanceRecord
from runtime_supervisor import transition_runtime


class CheckedRuntimeError(RuntimeError):
    """A checked bundle binding is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class CheckedRuntimeBinding:
    instance_id: str
    profile_name: str
    profile: Mapping[str, Any]


_REQUIRED_CAPABILITIES = {
    "inspect", "prepare", "load", "health", "drain", "unload", "stop", "reconcile"
}


def _bundle_gpu_ids(config: Mapping[str, Any], bundle_name: str) -> set[str]:
    bundles = config.get("bundles") or {}
    if not isinstance(bundles, Mapping):
        return set()
    connected = {bundle_name}
    pending = [bundle_name]
    while pending:
        current = pending.pop()
        bundle = bundles.get(current) or {}
        if not isinstance(bundle, Mapping):
            continue
        neighbours = set()
        for field in ("linked_to", "linked_from", "linked_bundles"):
            value = bundle.get(field) or []
            if isinstance(value, list):
                neighbours.update(str(item) for item in value)
        for other_name, other in bundles.items():
            if not isinstance(other, Mapping):
                continue
            for field in ("linked_to", "linked_from", "linked_bundles"):
                if current in (other.get(field) or []):
                    neighbours.add(str(other_name))
        for neighbour in neighbours - connected:
            if neighbour in bundles:
                connected.add(neighbour)
                pending.append(neighbour)
    return {
        str((bundles.get(name) or {}).get("gpu_id"))
        for name in connected
        if (bundles.get(name) or {}).get("gpu_id")
    }


def resolve_checked_runtime_binding(
    config: Mapping[str, Any], bundle_name: str
) -> CheckedRuntimeBinding | None:
    bundles = config.get("bundles") or {}
    bundle = bundles.get(bundle_name) if isinstance(bundles, Mapping) else None
    if not isinstance(bundle, Mapping):
        raise CheckedRuntimeError(f"unknown bundle {bundle_name!r}")

    instance_id = bundle.get("runtime_instance")
    profile_name = bundle.get("runtime_profile")
    if instance_id is None and profile_name is None:
        return None
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise CheckedRuntimeError(
            f"bundle {bundle_name!r} must declare runtime_instance"
        )
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise CheckedRuntimeError(
            f"bundle {bundle_name!r} must declare runtime_profile"
        )

    profiles = config.get("runtime_profiles") or {}
    profile = profiles.get(profile_name) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        raise CheckedRuntimeError(
            f"bundle {bundle_name!r} references unknown runtime profile {profile_name!r}"
        )
    errors = validate_runtime_profile(profile_name, profile)
    if errors:
        raise CheckedRuntimeError("invalid runtime profile: " + "; ".join(errors))

    model_sets = config.get("model_sets")
    if not isinstance(model_sets, Mapping):
        raise CheckedRuntimeError(
            f"checked runtime profile {profile_name!r} requires a model_sets registry"
        )
    model_set_name = profile.get("model_set")
    model_set = (
        model_sets.get(model_set_name)
        if isinstance(model_set_name, str)
        else None
    )
    if not isinstance(model_set, Mapping):
        raise CheckedRuntimeError(
            f"runtime profile {profile_name!r} has no resolved model set"
        )
    model_errors = validate_model_set(model_set_name, model_set)
    if model_errors:
        raise CheckedRuntimeError("invalid model set: " + "; ".join(model_errors))
    if model_set.get("status") != "qualified":
        raise CheckedRuntimeError(
            f"model set {model_set_name!r} is not qualified"
        )
    if profile.get("model_set_revision") != model_set.get("revision") or profile.get(
        "model_set_fingerprint"
    ) != model_set_fingerprint(model_set):
        raise CheckedRuntimeError(
            f"runtime profile {profile_name!r} is not bound to the current model set revision"
        )

    capabilities = set(profile.get("capabilities") or [])
    missing = sorted(_REQUIRED_CAPABILITIES - capabilities)
    if missing:
        raise CheckedRuntimeError(
            f"runtime profile {profile_name!r} lacks required actions: {missing}"
        )

    topology_gpus = _bundle_gpu_ids(config, bundle_name)
    requirements = profile.get("resource_requirements") or {}
    profile_gpus = {
        str(gpu_id) for gpu_id in (requirements.get("gpu_ids") or [])
    } if isinstance(requirements, Mapping) else set()
    if topology_gpus != profile_gpus:
        raise CheckedRuntimeError(
            f"bundle {bundle_name!r} GPU topology {sorted(topology_gpus)} does not "
            f"match runtime profile {profile_name!r} {sorted(profile_gpus)}"
        )
    return CheckedRuntimeBinding(instance_id, profile_name, profile)


class CheckedRuntimeController:
    """Run configured bundle transitions through one durable fenced store."""

    def __init__(
        self,
        *,
        state_path: str | os.PathLike[str],
        socket_path: str | os.PathLike[str] | None = None,
        adapter_factory: Callable[..., Any] = HostSupervisorRuntimeAdapter,
    ) -> None:
        self.store = FileRuntimeStateStore(Path(state_path))
        self.socket_path = socket_path
        self.adapter_factory = adapter_factory

    async def transition_if_configured(
        self,
        config: Mapping[str, Any],
        bundle_name: str,
        *,
        desired_state: RuntimeDesiredState | str,
        timeout_seconds: float,
        owner: str | None = None,
    ) -> RuntimeInstanceRecord | None:
        binding = resolve_checked_runtime_binding(config, bundle_name)
        if binding is None:
            return None
        adapter = self.adapter_factory(
            profile_name=binding.profile_name,
            socket_path=self.socket_path,
            timeout_seconds=min(max(float(timeout_seconds), 1.0), 300.0),
        )
        transition_owner = owner or (
            f"gpu-manager:{os.getpid()}:{bundle_name}:{uuid.uuid4().hex}"
        )
        return await transition_runtime(
            self.store,
            adapter,
            instance_id=binding.instance_id,
            profile_name=binding.profile_name,
            profile=binding.profile,
            owner=transition_owner,
            desired_state=desired_state,
            ttl_seconds=max(180.0, float(timeout_seconds) + 30.0),
            readiness_timeout_seconds=max(1.0, float(timeout_seconds)),
            readiness_poll_interval_seconds=1.0,
        )

    async def admission_errors_if_configured(
        self, config: Mapping[str, Any], bundle_name: str
    ) -> list[str] | None:
        binding = resolve_checked_runtime_binding(config, bundle_name)
        if binding is None:
            return None
        return await self.store.admission_errors(
            binding.instance_id,
            binding.profile_name,
            binding.profile,
        )

    async def health_summary(self) -> dict[str, Any]:
        """Expose safe runtime state without host commands or private bindings."""
        now = time.time()
        records = await self.store.snapshots()
        items = []
        for record in records:
            transition_active = bool(
                record.transition_owner and record.transition_expires_at > now
            )
            items.append(
                {
                    "instance_id": record.instance_id,
                    "profile_name": record.profile_name,
                    "profile_fingerprint": record.profile_fingerprint,
                    "desired_state": record.desired_state.value,
                    "state": record.state.value,
                    "process_generation": record.process_generation,
                    "process_ready": record.process_ready,
                    "model_ready": record.model_ready,
                    "accepting": record.accepting,
                    "transition_fence": record.transition_fence,
                    "transition_active": transition_active,
                    "transition_expires_in_seconds": round(
                        max(0.0, record.transition_expires_at - now), 3
                    ),
                    "observation_age_seconds": round(
                        max(0.0, now - record.observed_at), 3
                    ) if record.observed_at else None,
                    "last_error": str(record.last_error)[:500]
                    if record.last_error
                    else None,
                }
            )
        blocked = sum(item["state"] == "blocked" for item in items)
        transitioning = sum(item["transition_active"] for item in items)
        return {
            "status": (
                "idle"
                if not items
                else "blocked"
                if blocked
                else "transitioning"
                if transitioning
                else "ready"
            ),
            "instance_count": len(items),
            "blocked_count": blocked,
            "transitioning_count": transitioning,
            "instances": items,
        }


__all__ = [
    "CheckedRuntimeBinding",
    "CheckedRuntimeController",
    "CheckedRuntimeError",
    "resolve_checked_runtime_binding",
]
