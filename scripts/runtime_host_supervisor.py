#!/usr/bin/env python3
"""Minimal allowlisted host supervisor for GPU Manager runtime transitions.

Run this on the host, not in the controller container.  The private overlay
binds immutable profile identities to exact systemd units and loopback probes.
Requests can select only a known instance and action; they cannot supply a
command, unit, URL, path, environment, mount, or device.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web

from runtime_host_client import ACTION_SCHEMA, RESULT_SCHEMA
from runtime_contracts import (
    model_set_fingerprint,
    runtime_profile_fingerprint,
    validate_model_set,
    validate_runtime_profile,
)


OVERLAY_SCHEMA = "runtime-host-overlay.v1"
FENCE_SCHEMA = "runtime-host-fences.v1"
_ACTIONS = frozenset(
    {"inspect", "prepare", "load", "drain", "unload", "stop", "health", "reconcile"}
)
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MAX_PROBE_BYTES = 1024 * 1024


class HostSupervisorError(RuntimeError):
    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostSupervisorError(f"cannot load {path.name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise HostSupervisorError(f"{path.name} must contain an object")
    return value


def _validate_probe(name: str, probe: object) -> list[str]:
    if probe is None:
        return []
    if not isinstance(probe, Mapping):
        return [f"{name} must be an object"]
    errors: list[str] = []
    if set(probe) - {"method", "url", "body", "expect", "timeout_seconds"}:
        errors.append(f"{name} contains unsupported fields")
    timeout = probe.get("timeout_seconds", 15)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 120:
        errors.append(f"{name}.timeout_seconds must be between 1 and 120")
    method = probe.get("method", "GET")
    if method not in {"GET", "POST"}:
        errors.append(f"{name}.method must be GET or POST")
    url = probe.get("url")
    if not isinstance(url, str):
        errors.append(f"{name}.url must be a string")
    else:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in _LOCAL_HOSTS:
            errors.append(f"{name}.url must be an explicit loopback HTTP URL")
        try:
            port = parsed.port
        except ValueError:
            port = None
            errors.append(f"{name}.url has an invalid port")
        if port is not None and not 1 <= port <= 65535:
            errors.append(f"{name}.url has an invalid port")
        if parsed.username or parsed.password or parsed.fragment:
            errors.append(f"{name}.url may not contain credentials or a fragment")
    if "body" in probe and not isinstance(probe["body"], Mapping):
        errors.append(f"{name}.body must be an object")
    if "expect" in probe and not isinstance(probe["expect"], Mapping):
        errors.append(f"{name}.expect must be an object")
    return errors


def _validate_resolved_binding(prefix: str, binding: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_fields = {
        "profile_name", "profile_fingerprint", "engine", "adapter", "model_set",
        "unit", "load_mode", "actions", "health_probe", "model_probe", "drain_probe", "unload_probe",
    }
    if set(binding) - allowed_fields:
        errors.append(f"{prefix} contains unsupported fields")
    for field in ("profile_name", "engine", "adapter", "model_set"):
        if not isinstance(binding.get(field), str) or not binding[field]:
            errors.append(f"{prefix}.{field} must be non-empty")
    if not _HEX64_RE.fullmatch(str(binding.get("profile_fingerprint") or "")):
        errors.append(f"{prefix}.profile_fingerprint must be a lowercase SHA-256")
    if not _UNIT_RE.fullmatch(str(binding.get("unit") or "")):
        errors.append(f"{prefix}.unit must be an explicit .service name")
    if binding.get("load_mode", "start") not in {"start", "restart"}:
        errors.append(f"{prefix}.load_mode must be start or restart")
    actions = binding.get("actions")
    if (
        not isinstance(actions, list)
        or not actions
        or not all(isinstance(item, str) and item in _ACTIONS for item in actions)
    ):
        errors.append(f"{prefix}.actions contains an unsupported action")
    for probe_name in ("health_probe", "model_probe", "drain_probe", "unload_probe"):
        errors.extend(
            f"{prefix}.{message}" for message in _validate_probe(probe_name, binding.get(probe_name))
        )
    if "health" in (actions or []) and binding.get("health_probe") is None:
        errors.append(f"{prefix}.health_probe is required for health")
    if "drain" in (actions or []) and binding.get("drain_probe") is None:
        errors.append(f"{prefix}.drain_probe is required for drain")
    model_probe = binding.get("model_probe")
    if (
        "health" in (actions or [])
        and (
            not isinstance(model_probe, Mapping)
            or not isinstance(model_probe.get("expect"), Mapping)
            or not model_probe["expect"]
        )
    ):
        errors.append(f"{prefix}.model_probe.expect must identify the loaded model")
    if "unload" in (actions or []) and binding.get("unload_probe") is None:
        errors.append(f"{prefix}.unload_probe is required for unload")
    return errors


def validate_overlay(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("schema_version") != OVERLAY_SCHEMA:
        errors.append(f"schema_version must be {OVERLAY_SCHEMA!r}")
    instances = value.get("instances")
    if not isinstance(instances, Mapping) or not instances:
        return errors + ["instances must be a non-empty object"]
    unit_owners: dict[str, str] = {}
    endpoint_owners: dict[str, str] = {}

    def record_physical_ownership(
        prefix: str, instance_id: str, binding: Mapping[str, Any]
    ) -> None:
        unit = binding.get("unit")
        if isinstance(unit, str) and unit:
            prior = unit_owners.get(unit)
            if prior is not None:
                errors.append(
                    f"{prefix}.unit duplicates {prior!r}; each profile requires "
                    "a distinct launch unit"
                )
            else:
                unit_owners[unit] = prefix

        for probe_name in (
            "health_probe", "model_probe", "drain_probe", "unload_probe"
        ):
            probe = binding.get(probe_name)
            url = probe.get("url") if isinstance(probe, Mapping) else None
            if not isinstance(url, str):
                continue
            parsed = urlsplit(url)
            try:
                port = parsed.port
            except ValueError:
                continue
            endpoint = f"{parsed.hostname}:{port or 80}"
            prior_instance = endpoint_owners.get(endpoint)
            if prior_instance is not None and prior_instance != instance_id:
                errors.append(
                    f"{prefix}.{probe_name} shares {endpoint!r} with runtime "
                    f"instance {prior_instance!r}; shared endpoints require one instance lock"
                )
            else:
                endpoint_owners[endpoint] = instance_id

    for instance_id, binding in instances.items():
        prefix = f"instances.{instance_id}"
        if not isinstance(instance_id, str) or not _ID_RE.fullmatch(instance_id):
            errors.append(f"{prefix}: invalid instance ID")
            continue
        if not isinstance(binding, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        profiles = binding.get("profiles")
        if profiles is None:
            errors.extend(_validate_resolved_binding(prefix, binding))
            record_physical_ownership(prefix, str(instance_id), binding)
            continue
        common_allowed = {
            "unit", "load_mode", "actions", "health_probe", "model_probe",
            "drain_probe", "unload_probe", "profiles",
        }
        if set(binding) - common_allowed:
            errors.append(f"{prefix} shared-profile binding contains unsupported fields")
        if not isinstance(profiles, Mapping) or not profiles:
            errors.append(f"{prefix}.profiles must be a non-empty object")
            continue
        profile_allowed = {
            "profile_fingerprint", "engine", "adapter", "model_set", "load_mode",
            "unit", "health_probe", "model_probe", "drain_probe", "unload_probe",
        }
        common = {key: val for key, val in binding.items() if key != "profiles"}
        for profile_name, profile in profiles.items():
            profile_prefix = f"{prefix}.profiles.{profile_name}"
            if not isinstance(profile_name, str) or not _ID_RE.fullmatch(profile_name):
                errors.append(f"{profile_prefix}: invalid profile name")
                continue
            if not isinstance(profile, Mapping):
                errors.append(f"{profile_prefix} must be an object")
                continue
            if set(profile) - profile_allowed:
                errors.append(f"{profile_prefix} contains unsupported fields")
            resolved = {**common, **profile, "profile_name": profile_name}
            errors.extend(_validate_resolved_binding(profile_prefix, resolved))
            record_physical_ownership(profile_prefix, str(instance_id), resolved)
    return sorted(set(errors))


def validate_overlay_profiles(
    value: Mapping[str, Any],
    profiles_root: Path,
    model_sets_root: Path | None = None,
) -> list[str]:
    """Check private bindings against the public immutable profile files."""

    errors: list[str] = []
    instances = value.get("instances")
    if not isinstance(instances, Mapping):
        return ["instances must be a non-empty object"]
    for instance_id, instance in instances.items():
        if not isinstance(instance, Mapping):
            continue
        profiles = instance.get("profiles")
        if isinstance(profiles, Mapping):
            bindings = [
                (
                    str(profile_name),
                    {
                        **{key: item for key, item in instance.items() if key != "profiles"},
                        **profile,
                        "profile_name": profile_name,
                    },
                )
                for profile_name, profile in profiles.items()
                if isinstance(profile, Mapping)
            ]
        else:
            bindings = [(str(instance.get("profile_name") or ""), instance)]
        for profile_name, binding in bindings:
            prefix = f"instances.{instance_id}.profiles.{profile_name}"
            path = profiles_root / f"{profile_name}.runtime-profile.json"
            try:
                profile = _load_json(path)
            except HostSupervisorError:
                errors.append(f"{prefix}: public runtime profile is missing or invalid")
                continue
            profile_errors = validate_runtime_profile(profile_name, profile)
            errors.extend(f"{prefix}: {error}" for error in profile_errors)
            if model_sets_root is not None:
                model_set_name = profile.get("model_set")
                if not isinstance(model_set_name, str) or not model_set_name:
                    errors.append(f"{prefix}: model_set is required")
                else:
                    model_set_path = model_sets_root / f"{model_set_name}.model-set.json"
                    try:
                        model_set = _load_json(model_set_path)
                    except HostSupervisorError:
                        errors.append(f"{prefix}: public model set is missing or invalid")
                    else:
                        errors.extend(
                            f"{prefix}: model_set: {error}"
                            for error in validate_model_set(model_set_name, model_set)
                        )
                        if profile.get("model_set_revision") != model_set.get("revision"):
                            errors.append(f"{prefix}: model_set_revision does not match public model set")
                        if profile.get("model_set_fingerprint") != model_set_fingerprint(model_set):
                            errors.append(f"{prefix}: model_set_fingerprint does not match public model set")
            expected = {
                "profile_fingerprint": runtime_profile_fingerprint(profile),
                "engine": profile.get("engine"),
                "adapter": profile.get("adapter"),
                "model_set": profile.get("model_set"),
            }
            for field, expected_value in expected.items():
                if binding.get(field) != expected_value:
                    errors.append(f"{prefix}: {field} does not match public profile")
            profile_actions = set(profile.get("capabilities") or [])
            binding_actions = set(binding.get("actions") or [])
            if not binding_actions <= profile_actions:
                errors.append(f"{prefix}: overlay actions exceed public profile capabilities")
            requirements = profile.get("resource_requirements")
            if (
                isinstance(requirements, Mapping)
                and requirements.get("fresh_process") is True
                and binding.get("load_mode") != "restart"
            ):
                errors.append(f"{prefix}: fresh_process profile requires load_mode=restart")
    return sorted(set(errors))


class FenceLedger:
    """Durably retain the highest accepted transition fence per instance."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._records: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._records is not None:
            return self._records
        if not self.path.exists():
            self._records = {}
            return self._records
        value = _load_json(self.path)
        if value.get("schema_version") != FENCE_SCHEMA or not isinstance(value.get("instances"), Mapping):
            raise HostSupervisorError("runtime fence ledger is invalid", status=503)
        records: dict[str, dict[str, Any]] = {}
        for key, record in value["instances"].items():
            if (
                not isinstance(key, str)
                or not _ID_RE.fullmatch(key)
                or not isinstance(record, Mapping)
                or not isinstance(record.get("owner"), str)
                or not record["owner"]
                or isinstance(record.get("fence"), bool)
                or not isinstance(record.get("fence"), int)
                or record["fence"] < 1
            ):
                raise HostSupervisorError("runtime fence ledger is invalid", status=503)
            records[key] = {"owner": record["owner"], "fence": record["fence"]}
        self._records = records
        return self._records

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": FENCE_SCHEMA, "instances": self._records}, handle, sort_keys=True)
                handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    async def accept(self, instance_id: str, owner: str, fence: int) -> None:
        async with self._lock:
            records = self._load()
            current = records.get(instance_id)
            if current is not None:
                old_fence = int(current.get("fence", 0))
                old_owner = current.get("owner")
                if fence < old_fence or (fence == old_fence and owner != old_owner):
                    raise HostSupervisorError("stale runtime transition fence", status=409)
            if current != {"owner": owner, "fence": fence}:
                records[instance_id] = {"owner": owner, "fence": fence}
                try:
                    self._persist()
                except OSError as exc:
                    raise HostSupervisorError("cannot persist runtime transition fence", status=503) from exc


def _field(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


class HostRuntimeSupervisor:
    def __init__(self, overlay: Mapping[str, Any], ledger: FenceLedger) -> None:
        errors = validate_overlay(overlay)
        if errors:
            raise HostSupervisorError("invalid host overlay: " + "; ".join(errors))
        self.instances = overlay["instances"]
        self.ledger = ledger
        self._locks: dict[str, asyncio.Lock] = {}
        self._drained: set[str] = set(self.instances)  # fail closed after daemon restart

    def _binding(self, instance_id: str, profile_name: object) -> Mapping[str, Any]:
        instance = self.instances.get(instance_id)
        if not isinstance(instance, Mapping):
            raise HostSupervisorError("unknown runtime instance", status=404)
        profiles = instance.get("profiles")
        if profiles is None:
            return instance
        profile = profiles.get(profile_name) if isinstance(profiles, Mapping) else None
        if not isinstance(profile, Mapping):
            raise HostSupervisorError("unknown runtime profile for instance", status=409)
        return {
            **{key: value for key, value in instance.items() if key != "profiles"},
            **profile,
            "profile_name": profile_name,
        }

    def _profile_bindings(self, instance_id: str) -> list[Mapping[str, Any]]:
        instance = self.instances.get(instance_id)
        if not isinstance(instance, Mapping):
            raise HostSupervisorError("unknown runtime instance", status=404)
        profiles = instance.get("profiles")
        if not isinstance(profiles, Mapping):
            return [instance]
        return [self._binding(instance_id, name) for name in sorted(profiles)]

    async def _systemctl(self, *args: str) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            "systemctl", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except BaseException:
            # Cancelling communicate() does not stop the subprocess.  Never
            # release the instance lock while an unowned systemctl client is
            # still running, otherwise a later fence can race the old action.
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
            raise
        return int(process.returncode or 0), (stdout or stderr).decode("utf-8", errors="replace")[:4096]

    async def _unit_state(self, unit: str) -> tuple[str, int]:
        code, output = await self._systemctl(
            "show", unit, "--property=ActiveState", "--property=MainPID", "--no-page"
        )
        if code != 0:
            raise HostSupervisorError("systemd unit inspection failed", status=503)
        values = {
            key: value
            for line in output.splitlines()
            if "=" in line
            for key, value in (line.strip().split("=", 1),)
        }
        active = values.get("ActiveState", "unknown")
        try:
            pid = int(values.get("MainPID", "0"))
        except ValueError:
            pid = 0
        return active, pid

    async def _probe(self, probe: Mapping[str, Any] | None) -> bool:
        if not isinstance(probe, Mapping):
            return False
        method = probe.get("method", "GET")
        kwargs = {"json": probe.get("body")} if "body" in probe else {}
        try:
            async with ClientSession(timeout=ClientTimeout(total=probe.get("timeout_seconds", 15))) as session:
                async with session.request(method, probe["url"], **kwargs) as response:
                    raw = await response.content.read(_MAX_PROBE_BYTES + 1)
                    if response.status != 200 or len(raw) > _MAX_PROBE_BYTES:
                        return False
            value = json.loads(raw.decode("utf-8")) if raw else {}
            return all(_field(value, str(path)) == expected for path, expected in (probe.get("expect") or {}).items())
        except Exception:
            return False

    @staticmethod
    def _generation(pid: int, fence: int) -> int:
        if pid > 0:
            try:
                fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
                return max(1, int(fields[21]))
            except (OSError, ValueError, IndexError):
                pass
        return max(1, fence)

    async def _observation(self, instance_id: str, binding: Mapping[str, Any], fence: int) -> dict[str, Any]:
        active_state, pid = await self._unit_state(binding["unit"])
        process_ready = active_state == "active" and pid > 0
        health_ready = process_ready and await self._probe(binding.get("health_probe"))
        model_ready = health_ready and await self._probe(binding.get("model_probe"))
        accepting = model_ready and instance_id not in self._drained
        state = "ready" if accepting else ("draining" if process_ready and instance_id in self._drained else ("loading" if process_ready else "stopped"))
        return {
            "profile_name": binding["profile_name"],
            "profile_fingerprint": binding["profile_fingerprint"],
            "state": state,
            "process_generation": self._generation(pid, fence),
            "process_ready": process_ready,
            "model_ready": model_ready,
            "accepting": accepting,
            "model_set": binding["model_set"],
        }

    async def _stop_active_profiles(
        self,
        instance_id: str,
        *,
        except_unit: str | None = None,
    ) -> None:
        """Drain and stop active sibling units before a profile handoff."""

        self._drained.add(instance_id)
        for sibling in self._profile_bindings(instance_id):
            unit = str(sibling["unit"])
            if except_unit is not None and unit == except_unit:
                continue
            active_state, pid = await self._unit_state(unit)
            if active_state in {"inactive", "failed"} and pid <= 0:
                continue
            if active_state != "active" or pid <= 0:
                raise HostSupervisorError(
                    f"sibling runtime unit is not conclusively stopped: {active_state}",
                    status=503,
                )
            if not await self._probe(sibling.get("drain_probe")):
                raise HostSupervisorError(
                    "active sibling runtime did not drain", status=503
                )
            if not await self._probe(sibling.get("unload_probe")):
                raise HostSupervisorError(
                    "active sibling runtime did not unload", status=503
                )
            code, _ = await self._systemctl("--job-mode=fail", "stop", unit)
            if code != 0:
                raise HostSupervisorError(
                    "active sibling runtime did not stop", status=503
                )
            final_state, final_pid = await self._unit_state(unit)
            if final_state not in {"inactive", "failed"} or final_pid > 0:
                raise HostSupervisorError(
                    "active sibling runtime did not reach a stopped state", status=503
                )

    async def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("schema_version") != ACTION_SCHEMA:
            raise HostSupervisorError("unsupported action schema")
        action = request.get("action"); instance_id = request.get("instance_id")
        owner = request.get("transition_owner"); fence = request.get("transition_fence")
        if action not in _ACTIONS or not isinstance(instance_id, str):
            raise HostSupervisorError("invalid runtime action or instance")
        binding = self._binding(instance_id, request.get("profile_name"))
        for field in ("profile_name", "profile_fingerprint", "engine", "adapter", "model_set"):
            if request.get(field) != binding.get(field):
                raise HostSupervisorError(f"runtime identity mismatch: {field}", status=409)
        if action not in binding["actions"]:
            raise HostSupervisorError("runtime action is not allowed for this instance", status=403)
        if not isinstance(owner, str) or not owner or isinstance(fence, bool) or not isinstance(fence, int) or fence < 1:
            raise HostSupervisorError("transition owner/fence is invalid")
        lock = self._locks.setdefault(instance_id, asyncio.Lock())
        async with lock:
            # Fence acceptance and the physical action are one serialized
            # operation.  A newer owner cannot supersede this request while
            # its systemd/probe work is still in flight.
            await self.ledger.accept(instance_id, owner, fence)
            if action == "prepare":
                await self._unit_state(binding["unit"])
            elif action == "load":
                # Profiles in one runtime instance may share a port/device but
                # never a launch unit.  Quiesce the active sibling before the
                # selected profile is allowed to start.
                await self._stop_active_profiles(
                    instance_id, except_unit=str(binding["unit"])
                )
                code, _ = await self._systemctl(
                    "--job-mode=fail",
                    binding.get("load_mode", "start"),
                    binding["unit"],
                )
                if code != 0:
                    raise HostSupervisorError("systemd start failed", status=503)
                self._drained.discard(instance_id)
            elif action == "drain":
                self._drained.add(instance_id)
                if binding.get("drain_probe") and not await self._probe(binding["drain_probe"]):
                    raise HostSupervisorError("runtime drain endpoint failed", status=503)
            elif action == "unload":
                self._drained.add(instance_id)
                if not await self._probe(binding.get("unload_probe")):
                    raise HostSupervisorError("runtime unload endpoint failed", status=503)
            elif action == "stop":
                # Desired stopped applies to the physical runtime instance,
                # not only to whichever profile named this request.
                await self._stop_active_profiles(instance_id)

            result = {
                "schema_version": RESULT_SCHEMA,
                "status": "completed",
                **{key: request[key] for key in ("action", "instance_id", "profile_fingerprint", "transition_owner", "transition_fence")},
            }
            if action in {"inspect", "health", "reconcile"}:
                result["observation"] = await self._observation(instance_id, binding, fence)
            return result


def create_app(supervisor: HostRuntimeSupervisor) -> web.Application:
    async def handle(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, Mapping):
                raise HostSupervisorError("request body must be an object")
            if request.match_info["action"] != body.get("action"):
                raise HostSupervisorError("route action does not match request")
            return web.json_response(await supervisor.execute(body))
        except HostSupervisorError as exc:
            return web.json_response({"error": str(exc)}, status=exc.status)
        except Exception:
            return web.json_response({"error": "host supervisor internal failure"}, status=500)

    app = web.Application(client_max_size=128 * 1024)
    app.router.add_post("/v1/runtime/{action}", handle)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--socket", type=Path, default=Path("/run/gpu-manager-host/supervisor.sock"))
    parser.add_argument("--fence-ledger", type=Path, default=Path("/var/lib/gpu-manager-host/fences.json"))
    parser.add_argument("--profiles-root", type=Path)
    parser.add_argument("--model-sets-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    overlay = _load_json(args.overlay)
    errors = validate_overlay(overlay)
    if args.profiles_root is not None:
        errors.extend(
            validate_overlay_profiles(
                overlay, args.profiles_root, args.model_sets_root
            )
        )
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, sort_keys=True))
        return 2
    if args.check:
        print(json.dumps({"valid": True, "instance_count": len(overlay["instances"])}, sort_keys=True))
        return 0
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists() or args.socket.is_symlink():
        mode = args.socket.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise HostSupervisorError("refusing to replace a non-socket path")
        args.socket.unlink()
    supervisor = HostRuntimeSupervisor(overlay, FenceLedger(args.fence_ledger))
    old_umask = os.umask(0o007)
    try:
        web.run_app(create_app(supervisor), path=str(args.socket), print=None)
    finally:
        os.umask(old_umask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FENCE_SCHEMA", "FenceLedger", "HostRuntimeSupervisor", "HostSupervisorError",
    "OVERLAY_SCHEMA", "create_app", "validate_overlay", "validate_overlay_profiles",
]
