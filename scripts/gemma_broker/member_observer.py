"""Direct, fail-closed observation of Combined Gemma members.

The standalone broker owns this observer.  It only reads systemd state and
member HTTP endpoints; it never starts, stops, restarts, or otherwise mutates a
member.  The injected callables make the evidence collector deterministic in
unit tests and keep the subprocess/network edges isolated.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import aiohttp

from .probe_adapter import _classify_probe_error, member_probes_from_payloads


SystemdProbe = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
HttpProbe = Callable[[dict[str, Any], str], Awaitable[tuple[int, Any]]]
DispatcherState = Callable[..., tuple[bool, int]]


async def read_systemd_facts(config: dict[str, Any], *, timeout: float = 3.0) -> dict[str, Any]:
    """Read ActiveState/MainPID without performing a lifecycle mutation."""
    unit = config.get("systemd_unit")
    if not isinstance(unit, str) or not unit:
        return {"systemd_active": False, "main_pid": None}
    try:
        process = await asyncio.create_subprocess_exec(
            "systemctl",
            "show",
            "--property=ActiveState",
            "--property=MainPID",
            "--value",
            unit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        if process.returncode != 0:
            return {"systemd_active": False, "main_pid": None, "probe_error": "systemd_error"}
        lines = stdout.decode("utf-8", "replace").splitlines()
        active = bool(lines and lines[0].strip() == "active")
        try:
            pid = int(lines[1].strip()) if len(lines) > 1 else 0
        except ValueError:
            pid = 0
        return {"systemd_active": active, "main_pid": pid if pid > 0 else None}
    except Exception as exc:  # fail closed; diagnostics retain the class
        return {
            "systemd_active": False,
            "main_pid": None,
            "probe_error": _classify_probe_error(exc) or "unknown:systemd",
        }


async def read_http_probe(
    session: aiohttp.ClientSession,
    config: dict[str, Any],
    path: str,
    *,
    timeout: float = 3.0,
) -> tuple[int, Any]:
    """Fetch and JSON-decode one member evidence endpoint."""
    port = config.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
        raise ValueError("MEMBER_PORT_INVALID")
    # Backend launch configs commonly bind to 0.0.0.0.  That is a bind
    # address, not the destination the broker should use for local probes.
    base = str(config.get("probe_host") or "127.0.0.1")
    async with session.get(f"http://{base}:{port}{path}", timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        status = int(response.status)
        if status != 200:
            return status, None
        try:
            payload = await response.json(content_type=None)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("bad_json") from exc
        return status, payload


class MemberObserver:
    """Collect one complete readiness snapshot per configured member."""

    PROBE_PATHS = (("health", "/health"), ("slots", "/slots"), ("models", "/v1/models"))

    def __init__(
        self,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        session: aiohttp.ClientSession | None = None,
        systemd_probe: SystemdProbe | None = None,
        http_probe: HttpProbe | None = None,
        dispatcher_state: DispatcherState | None = None,
        timeout: float = 3.0,
    ) -> None:
        self.configs = {name: dict(config) for name, config in configs.items()}
        self.session = session
        self.systemd_probe = systemd_probe
        self.http_probe = http_probe
        self.dispatcher_state: DispatcherState = dispatcher_state or (
            lambda *_args: (False, 0)
        )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)

    async def _systemd(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.systemd_probe is not None:
            return dict(await self.systemd_probe(config))
        return await read_systemd_facts(config, timeout=self.timeout)

    async def _http(self, config: dict[str, Any], path: str) -> tuple[int, Any]:
        if self.http_probe is not None:
            return await self.http_probe(config, path)
        if self.session is None:
            raise RuntimeError("HTTP_SESSION_REQUIRED")
        return await read_http_probe(self.session, config, path, timeout=self.timeout)

    def _dispatcher(self, member_name: str) -> tuple[bool, int]:
        """Read per-member lease state, preserving legacy no-arg probes."""
        try:
            return self.dispatcher_state(member_name)
        except TypeError:
            return self.dispatcher_state()

    async def observe(self, name: str, *, now: float) -> dict[str, Any]:
        config = self.configs.get(name)
        if config is None:
            raise KeyError(f"UNKNOWN_MEMBER:{name}")
        if config.get("member_type") == "openai_compatible":
            return await self._observe_openai_compatible(name, config, now=now)
        payloads: dict[str, Any] = {}
        probe_errors: dict[str, str] = {}
        facts = await self._systemd(config)
        systemd_error = facts.pop("probe_error", None)
        if systemd_error is not None:
            probe_errors["systemd"] = str(systemd_error)

        for probe_name, path in self.PROBE_PATHS:
            try:
                status, payload = await self._http(config, path)
                if status != 200:
                    probe_errors[probe_name] = f"http_{status}"
                else:
                    payloads[probe_name] = payload
            except Exception as exc:
                probe_errors[probe_name] = _classify_probe_error(exc) or f"unknown:{type(exc).__name__}"

        try:
            registered, leases = self._dispatcher(name)
        except Exception as exc:
            registered, leases = False, 0
            probe_errors["dispatcher"] = f"unknown:{type(exc).__name__}"

        facts.update(
            {
                "idle_service_effective": (
                    True
                    if config.get("cpu_only")
                    else config.get("idle_service_configured") == name
                ),
                "dispatcher_registered": registered is True,
                "dispatcher_leases": leases if isinstance(leases, int) and leases >= 0 else 0,
                "probe_error": probe_errors or None,
            }
        )
        return member_probes_from_payloads(
            config=config,
            payloads=payloads,
            manager_facts=facts,
            observed_at=float(now),
        )

    async def _observe_openai_compatible(
        self, name: str, config: dict[str, Any], *, now: float
    ) -> dict[str, Any]:
        """Prove one externally hosted OpenAI-compatible singular worker."""
        facts = await self._systemd(config)
        probe_errors: dict[str, str] = {}
        model = config.get("model")
        health_url = config.get("health_url")
        models_url = config.get("models_url")
        health_ok = False
        model_ok = False
        if self.session is None:
            probe_errors["health"] = "http_session_required"
        else:
            for probe_name, url in (("health", health_url), ("models", models_url)):
                if not isinstance(url, str) or not url:
                    probe_errors[probe_name] = "url_missing"
                    continue
                try:
                    async with self.session.get(
                        url, timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        payload = await response.json(content_type=None)
                        if response.status != 200:
                            probe_errors[probe_name] = f"http_{response.status}"
                        elif probe_name == "health":
                            health_ok = payload.get("status") == "ok"
                        else:
                            model_ok = any(
                                isinstance(item, dict) and item.get("id") == model
                                for item in payload.get("data", [])
                            )
                except Exception as exc:
                    probe_errors[probe_name] = (
                        _classify_probe_error(exc) or f"unknown:{type(exc).__name__}"
                    )
        try:
            registered, leases = self._dispatcher(name)
        except Exception:
            registered, leases = False, 0
        semantic_health = health_ok and model_ok
        slots = int(config.get("parallel", 1))
        return {
            "semantic_health": semantic_health,
            "health_ok": health_ok,
            "probe_error": next(iter(probe_errors.values()), None),
            "probe_errors": probe_errors or None,
            "model_resident": None,
            "backend_slots_total": slots if semantic_health else None,
            "backend_slots_busy": 0 if semantic_health else None,
            "systemd_active": facts.get("systemd_active") is True,
            "main_pid": facts.get("main_pid"),
            "idle_service_effective": None,
            "dispatcher_registered": registered is True,
            "dispatcher_leases": leases if isinstance(leases, int) and leases >= 0 else 0,
            "observed_at": float(now),
        }

    async def observe_all(self, *, now: float) -> dict[str, dict[str, Any]]:
        results = await asyncio.gather(
            *(self.observe(name, now=now) for name in self.configs),
            return_exceptions=True,
        )
        output: dict[str, dict[str, Any]] = {}
        for name, result in zip(self.configs, results):
            if isinstance(result, Exception):
                output[name] = member_probes_from_payloads(
                    config=self.configs[name],
                    payloads={},
                    manager_facts={"probe_error": {"observer": type(result).__name__}},
                    observed_at=float(now),
                )
            else:
                output[name] = result
        return output


__all__ = ["MemberObserver", "read_http_probe", "read_systemd_facts"]
