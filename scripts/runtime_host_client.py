"""Narrow client boundary between GPU Manager and a host runtime supervisor.

The controller sends only an allowlisted action and immutable runtime identity
over a local Unix socket.  It never sends a command, systemd unit, device path,
mount, environment override, or model path.  Those private bindings belong to
the host supervisor's separately reviewed overlay.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, UnixConnector

from runtime_contracts import runtime_profile_fingerprint, validate_runtime_profile


ACTION_SCHEMA = "runtime-host-action.v1"
RESULT_SCHEMA = "runtime-host-action-result.v1"
DEFAULT_SOCKET_PATH = "/run/gpu-manager-host/supervisor.sock"
_ACTIONS = frozenset(
    {"inspect", "prepare", "load", "drain", "unload", "stop", "health", "reconcile"}
)
_OBSERVATION_ACTIONS = frozenset({"inspect", "health", "reconcile"})
_MAX_RESPONSE_BYTES = 1024 * 1024


class HostRuntimeClientError(RuntimeError):
    """The host supervisor was unavailable or violated its protocol."""


class HostSupervisorRuntimeAdapter:
    """RuntimeAdapter implementation backed by a local Unix-socket supervisor."""

    def __init__(
        self,
        *,
        profile_name: str,
        socket_path: str | os.PathLike[str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise HostRuntimeClientError("profile_name must be non-empty")
        self.profile_name = profile_name
        self.socket_path = Path(
            socket_path
            or os.environ.get("GPU_MANAGER_HOST_SUPERVISOR_SOCKET")
            or DEFAULT_SOCKET_PATH
        )
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 300.0))
        self._transition_owner: str | None = None
        self._transition_fence: int | None = None

    def bind_transition(self, owner: str, fence: int) -> None:
        """Bind subsequent calls to the controller's current transition lease."""
        if not isinstance(owner, str) or not owner.strip():
            raise HostRuntimeClientError("transition owner must be non-empty")
        if isinstance(fence, bool) or not isinstance(fence, int) or fence < 1:
            raise HostRuntimeClientError("transition fence must be a positive integer")
        self._transition_owner = owner
        self._transition_fence = fence

    def clear_transition(self) -> None:
        self._transition_owner = None
        self._transition_fence = None

    def _request(
        self, action: str, instance_id: str, profile: Mapping[str, Any]
    ) -> dict[str, Any]:
        if action not in _ACTIONS:
            raise HostRuntimeClientError(f"unsupported runtime action: {action!r}")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise HostRuntimeClientError("instance_id must be non-empty")
        if self._transition_owner is None or self._transition_fence is None:
            raise HostRuntimeClientError("runtime adapter has no bound transition fence")
        errors = validate_runtime_profile(self.profile_name, profile)
        if errors:
            raise HostRuntimeClientError("invalid runtime profile: " + "; ".join(errors))
        return {
            "schema_version": ACTION_SCHEMA,
            "action": action,
            "instance_id": instance_id,
            "profile_name": self.profile_name,
            "profile_fingerprint": runtime_profile_fingerprint(profile),
            "engine": profile.get("engine"),
            "adapter": profile.get("adapter"),
            "model_set": profile.get("model_set"),
            "transition_owner": self._transition_owner,
            "transition_fence": self._transition_fence,
        }

    async def _call(
        self, action: str, instance_id: str, profile: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        payload = self._request(action, instance_id, profile)
        connector = UnixConnector(path=str(self.socket_path))
        timeout = ClientTimeout(total=self.timeout_seconds)
        try:
            async with ClientSession(connector=connector, timeout=timeout) as session:
                async with session.post(
                    f"http://localhost/v1/runtime/{action}", json=payload
                ) as response:
                    raw = await response.content.read(_MAX_RESPONSE_BYTES + 1)
                    if len(raw) > _MAX_RESPONSE_BYTES:
                        raise HostRuntimeClientError("host supervisor response exceeds 1 MiB")
                    try:
                        result = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise HostRuntimeClientError(
                            f"host supervisor returned invalid JSON (HTTP {response.status})"
                        ) from exc
                    if not isinstance(result, Mapping):
                        raise HostRuntimeClientError("host supervisor result must be an object")
                    if response.status != 200:
                        reason = str(result.get("error") or "request rejected")[:500]
                        raise HostRuntimeClientError(
                            f"host supervisor {action} failed (HTTP {response.status}): {reason}"
                        )
        except HostRuntimeClientError:
            raise
        except Exception as exc:
            raise HostRuntimeClientError(
                f"host supervisor {action} transport failed: {exc}"
            ) from exc

        if result.get("schema_version") != RESULT_SCHEMA:
            raise HostRuntimeClientError("unsupported host supervisor result schema")
        for field in (
            "action",
            "instance_id",
            "profile_fingerprint",
            "transition_owner",
            "transition_fence",
        ):
            if result.get(field) != payload[field]:
                raise HostRuntimeClientError(
                    f"host supervisor result {field} does not match request"
                )
        if result.get("status") != "completed":
            raise HostRuntimeClientError("host supervisor action was not completed")
        if action in _OBSERVATION_ACTIONS:
            observation = result.get("observation")
            if not isinstance(observation, Mapping):
                raise HostRuntimeClientError(
                    f"host supervisor {action} result has no observation"
                )
            return dict(observation)
        return result

    async def inspect(self, instance_id, profile):
        return await self._call("inspect", instance_id, profile)

    async def prepare(self, instance_id, profile):
        await self._call("prepare", instance_id, profile)

    async def load(self, instance_id, profile):
        await self._call("load", instance_id, profile)

    async def drain(self, instance_id, profile):
        await self._call("drain", instance_id, profile)

    async def unload(self, instance_id, profile):
        await self._call("unload", instance_id, profile)

    async def stop(self, instance_id, profile):
        await self._call("stop", instance_id, profile)

    async def health(self, instance_id, profile):
        return await self._call("health", instance_id, profile)

    async def reconcile(self, instance_id, profile):
        return await self._call("reconcile", instance_id, profile)


__all__ = [
    "ACTION_SCHEMA",
    "DEFAULT_SOCKET_PATH",
    "HostRuntimeClientError",
    "HostSupervisorRuntimeAdapter",
    "RESULT_SCHEMA",
]
