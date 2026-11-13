"""Bounded node leases and authenticated GPU Manager peer routing."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, UnixConnector, web
from api_auth import bearer_matches

HEARTBEAT_SECONDS = 10.0
LEASE_SECONDS = 30.0
MAX_FORWARD_SECONDS = 120.0


@dataclass(frozen=True)
class NodeCredential:
    node_id: str
    token: str
    scopes: frozenset[str]
    tailnet_names: frozenset[str] = frozenset()


@dataclass
class NodeLease:
    node_id: str
    lease_id: str
    hardware: dict[str, Any]
    caps: dict[str, int]
    endpoint: str
    last_heartbeat: float
    expires_at: float
    sequence: int = 0
    draining: bool = False


class NodeRegistry:
    """In-memory, monotonic-clock-backed registry for short node leases."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 lease_seconds: float = LEASE_SECONDS):
        if not 0 < float(lease_seconds) <= LEASE_SECONDS:
            raise ValueError("node lease must be at most 30 seconds")
        self._clock = clock
        self.lease_seconds = float(lease_seconds)
        self._nodes: dict[str, NodeLease] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _validated_hardware(hardware: object) -> dict[str, Any]:
        if not isinstance(hardware, Mapping) or set(hardware) - {"models", "gpus", "platform"}:
            raise ValueError("invalid node hardware")
        models = hardware.get("models", [])
        gpus = hardware.get("gpus", [])
        platform = hardware.get("platform", "")
        if (not isinstance(models, list) or len(models) > 32
                or not all(isinstance(m, str) and 0 < len(m) <= 128 for m in models)
                or not isinstance(gpus, list) or len(gpus) > 32
                or not isinstance(platform, str) or len(platform) > 128):
            raise ValueError("invalid node hardware")
        clean_gpus: list[dict[str, Any]] = []
        for item in gpus:
            if not isinstance(item, Mapping) or set(item) - {"id", "name", "vram_total_mib"}:
                raise ValueError("invalid GPU record")
            vram = item.get("vram_total_mib", 0)
            if type(vram) is not int or vram < 0:
                raise ValueError("invalid GPU VRAM")
            clean_gpus.append({"id": str(item.get("id", ""))[:64],
                               "name": str(item.get("name", ""))[:128],
                               "vram_total_mib": vram})
        return {"models": list(dict.fromkeys(models)), "gpus": clean_gpus, "platform": platform}

    @staticmethod
    def _validated_caps(caps: object) -> dict[str, int]:
        if not isinstance(caps, Mapping) or set(caps) - {"available_slots", "free_vram_mib", "active_requests"}:
            raise ValueError("invalid node capacity")
        result: dict[str, int] = {}
        for name in ("available_slots", "free_vram_mib", "active_requests"):
            value = caps.get(name, 0)
            if type(value) is not int or not 0 <= value <= 2**31:
                raise ValueError(f"invalid node capacity: {name}")
            result[name] = value
        return result

    async def register(self, node_id: str, hardware: Mapping[str, Any],
                       caps: Mapping[str, int], *, endpoint: str = "") -> NodeLease:
        if not isinstance(node_id, str) or not node_id or len(node_id) > 32:
            raise ValueError("invalid node id")
        parsed_endpoint = endpoint.rstrip("/")
        if parsed_endpoint and not parsed_endpoint.startswith(("http://", "https://")):
            raise ValueError("invalid peer endpoint")
        clean_hardware = self._validated_hardware(hardware)
        clean_caps = self._validated_caps(caps)
        now = self._clock()
        async with self._lock:
            current = self._nodes.get(node_id)
            if (current and current.expires_at > now and current.endpoint == parsed_endpoint
                    and current.hardware == clean_hardware and current.caps == clean_caps):
                current.last_heartbeat = now
                current.expires_at = now + self.lease_seconds
                return current
            lease = NodeLease(node_id, uuid.uuid4().hex, clean_hardware, clean_caps,
                              parsed_endpoint, now, now + self.lease_seconds)
            self._nodes[node_id] = lease
            return lease

    async def heartbeat(self, node_id: str, *, lease_id: str | None = None,
                        sequence: int | None = None, caps: Mapping[str, int] | None = None) -> NodeLease | None:
        now = self._clock()
        async with self._lock:
            lease = self._nodes.get(node_id)
            if lease is None or lease.expires_at <= now or lease.draining:
                return None
            if lease_id is not None and not hmac.compare_digest(lease.lease_id, lease_id):
                return None
            if sequence is not None:
                if type(sequence) is not int or sequence <= lease.sequence:
                    return None
                lease.sequence = sequence
            if caps is not None:
                lease.caps = self._validated_caps(caps)
            lease.last_heartbeat = now
            lease.expires_at = now + self.lease_seconds
            return lease

    async def drain(self, node_id: str, *, lease_id: str | None = None,
                    draining: bool = True) -> NodeLease | None:
        async with self._lock:
            lease = self._nodes.get(node_id)
            if lease is None or lease.expires_at <= self._clock():
                return None
            if lease_id is not None and not hmac.compare_digest(lease.lease_id, lease_id):
                return None
            lease.draining = bool(draining)
            return lease

    async def pick_node(self, request: Mapping[str, Any]) -> NodeLease | None:
        model = request.get("model") if isinstance(request, Mapping) else None
        needed = request.get("required_vram_mib", 0) if isinstance(request, Mapping) else 0
        if not isinstance(model, str) or not model or type(needed) is not int or needed < 0:
            return None
        now = self._clock()
        async with self._lock:
            eligible = [node for node in self._nodes.values()
                        if node.expires_at > now and not node.draining
                        and node.caps["available_slots"] > 0
                        and node.caps["free_vram_mib"] >= needed
                        and model in node.hardware["models"] and node.endpoint]
            return max(eligible, key=lambda n: (n.caps["available_slots"], n.expires_at), default=None)

    def snapshot(self, *, include_expired: bool = False) -> list[dict[str, Any]]:
        now = self._clock()
        return [{"node_id": n.node_id, "lease_id": n.lease_id, "hardware": n.hardware,
                 "caps": n.caps, "expires_in_seconds": max(0.0, n.expires_at - now),
                 "draining": n.draining, "sequence": n.sequence}
                for n in self._nodes.values()
                if include_expired or n.expires_at > now]


def load_node_credentials(authority_file: str | Path, credentials_dir: str | Path | None) -> dict[str, NodeCredential]:
    """Load existing node principals from authority.json and systemd credentials."""
    if credentials_dir is None:
        raise RuntimeError("node credentials are not provided by systemd")
    document = json.loads(Path(authority_file).read_text(encoding="utf-8"))
    result: dict[str, NodeCredential] = {}
    for principal in document.get("principals", []):
        if not isinstance(principal, Mapping):
            continue
        principal_id = principal.get("principal_id")
        token_name = principal.get("token_file")
        if (not isinstance(principal_id, str) or not principal_id.startswith("node:")
                or not isinstance(token_name, str) or Path(token_name).name != token_name):
            continue
        node_id = principal_id.removeprefix("node:")
        credential = Path(credentials_dir) / token_name
        info = credential.stat()
        if info.st_mode & 0o077:
            raise RuntimeError("node credential permissions are too broad")
        token = credential.read_text(encoding="utf-8").strip()
        if not token or len(token) > 4096:
            raise RuntimeError("node credential is invalid")
        scopes = principal.get("scopes", [])
        if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
            raise RuntimeError("node scopes are invalid")
        names = principal.get("tailnet_names", [])
        if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
            names = []
        result[node_id] = NodeCredential(node_id, token, frozenset(scopes),
                                         frozenset(n.lower().rstrip(".") for n in names))
    if not result:
        raise RuntimeError("node principals are unavailable")
    return result


async def tailscale_peer_node(remote: str | None, credentials: Mapping[str, NodeCredential],
                              socket_path: str = "/run/tailscale/tailscaled.sock") -> str | None:
    """Resolve remote IP to an authority-bound Tailscale identity via LocalAPI."""
    if not remote:
        return None
    try:
        ipaddress.ip_address(remote)
        from aiohttp import UnixConnector
        async with ClientSession(connector=UnixConnector(path=socket_path), timeout=ClientTimeout(total=2)) as session:
            async with session.get("http://local-tailscaled.sock/localapi/v0/whois", params={"addr": remote}) as response:
                if response.status != 200:
                    return None
                document = await response.json()
        node = document.get("Node") if isinstance(document, Mapping) else None
        if not isinstance(node, Mapping) or node.get("Online") is False:
            return None
        name = node.get("Name")
        if not isinstance(name, str):
            return None
        normalized = name.lower().rstrip(".")
        matches = [cred.node_id for cred in credentials.values() if normalized in cred.tailnet_names]
        return matches[0] if len(matches) == 1 else None
    except Exception:
        return None


@dataclass(frozen=True)
class _ForwardResult:
    status: int
    body: bytes
    content_type: str


class NodeAPI:
    def __init__(self, registry: NodeRegistry, *, credential_provider: Callable[[], Mapping[str, NodeCredential]],
                 peer_node: Callable[[str | None, Mapping[str, NodeCredential]], Any],
                 endpoint_for_peer: Callable[[str, str], str],
                 local_node_id: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 event_sink: Callable[[str, str, str | None, str | None], None] | None = None):
        self.registry = registry
        self.credential_provider = credential_provider
        self.peer_node = peer_node
        self.endpoint_for_peer = endpoint_for_peer
        self.local_node_id = local_node_id
        self.clock = clock
        self.event_sink = event_sink
        self._forwards: dict[tuple[str, str], tuple[str, float, asyncio.Future[_ForwardResult | None] | _ForwardResult | None]] = {}
        self._forward_lock = asyncio.Lock()

    def _emit_transition(self, event_type: str, node_id: str,
                         from_state: str | None, to_state: str | None) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(event_type, node_id, from_state, to_state)
        except Exception:
            # Observability must never change lease/authentication outcomes.
            return

    async def _principal(self, request: web.Request, scope: str) -> NodeCredential:
        credentials = self.credential_provider()
        authorization = request.headers.get("Authorization", "")
        principal = next((c for c in credentials.values() if bearer_matches(request, c.token)), None)
        if principal is None:
            raise web.HTTPUnauthorized(text='{"error":"authentication required"}', content_type="application/json")
        if scope not in principal.scopes:
            raise web.HTTPForbidden(text='{"error":"forbidden_scope"}', content_type="application/json")
        peer_id = await self.peer_node(request.remote, credentials)
        if peer_id != principal.node_id:
            raise web.HTTPForbidden(text='{"error":"peer_identity_mismatch"}', content_type="application/json")
        return principal

    async def authorize_forwarded(self, request: web.Request) -> bool:
        """Require a scoped node token and trusted peer identity on forwarded LLM calls."""
        if (request.headers.get("X-Node-Forwarded") != "1"
                or not request.headers.get("X-Node-Lease-ID")
                or not request.headers.get("Idempotency-Key")):
            return False
        try:
            await self._principal(request, "agent.llm_forward")
            return True
        except web.HTTPException:
            return False

    async def register(self, request: web.Request) -> web.Response:
        try:
            principal = await self._principal(request, "agent.heartbeat")
            payload = await request.json()
            if not isinstance(payload, Mapping) or set(payload) != {"hardware", "caps"}:
                raise web.HTTPBadRequest(text='{"error":"invalid_payload"}', content_type="application/json")
            endpoint = self.endpoint_for_peer(principal.node_id, str(request.remote or ""))
            existing = next((node for node in self.registry.snapshot()
                             if node.get("node_id") == principal.node_id), None)
            previous_state = (
                "draining" if existing.get("draining") else "active"
            ) if existing else "absent"
            lease = await self.registry.register(principal.node_id, payload["hardware"], payload["caps"], endpoint=endpoint)
            next_state = "draining" if lease.draining else "active"
            if previous_state != next_state:
                self._emit_transition("node_lease", principal.node_id, previous_state, next_state)
            return web.json_response(self._lease_response(lease), status=201)
        except ValueError:
            raise web.HTTPBadRequest(text='{"error":"invalid_payload"}', content_type="application/json")
        except (OSError, RuntimeError, json.JSONDecodeError):
            raise web.HTTPServiceUnavailable(text='{"error":"node_authority_unavailable"}', content_type="application/json")

    async def heartbeat(self, request: web.Request) -> web.Response:
        principal = await self._principal(request, "agent.heartbeat")
        try:
            payload = await request.json()
            if (not isinstance(payload, Mapping) or set(payload) - {"lease_id", "sequence", "caps"}
                    or not {"lease_id", "sequence"} <= set(payload)):
                raise ValueError("invalid heartbeat")
            lease = await self.registry.heartbeat(principal.node_id, lease_id=payload["lease_id"],
                                                  sequence=payload["sequence"], caps=payload.get("caps"))
            if lease is None:
                raise web.HTTPConflict(text='{"error":"lease_expired_or_replayed"}', content_type="application/json")
            return web.json_response(self._lease_response(lease))
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text='{"error":"invalid_payload"}', content_type="application/json")

    async def drain(self, request: web.Request) -> web.Response:
        principal = await self._principal(request, "agent.desired_state")
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping) or set(payload) != {"lease_id", "draining"} or type(payload["draining"]) is not bool:
                raise ValueError("invalid drain")
            existing = next((node for node in self.registry.snapshot()
                             if node.get("node_id") == principal.node_id), None)
            previous_state = (
                "draining" if existing.get("draining") else "active"
            ) if existing else "unknown"
            lease = await self.registry.drain(principal.node_id, lease_id=payload["lease_id"], draining=payload["draining"])
            if lease is None:
                raise web.HTTPConflict(text='{"error":"lease_expired"}', content_type="application/json")
            next_state = "draining" if lease.draining else "active"
            if previous_state != next_state:
                self._emit_transition("node_lease", principal.node_id, previous_state, next_state)
            return web.json_response({"node_id": lease.node_id, "lease_id": lease.lease_id, "draining": lease.draining})
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text='{"error":"invalid_payload"}', content_type="application/json")

    def add_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/nodes/register", self.register)
        app.router.add_post("/v1/nodes/heartbeat", self.heartbeat)
        app.router.add_post("/v1/nodes/drain", self.drain)

    def _lease_response(self, lease: NodeLease) -> dict[str, Any]:
        return {"node_id": lease.node_id, "lease_id": lease.lease_id, "sequence": lease.sequence,
                "expires_in_seconds": max(0.0, lease.expires_at - self.clock()), "draining": lease.draining}

    async def forward_request(self, request: web.Request, body: bytes, *, model: str,
                              required_vram_mib: int = 0) -> web.Response | None:
        lease = await self.registry.pick_node({"model": model, "required_vram_mib": required_vram_mib})
        return await self.forward(request, lease, body) if lease is not None else None

    async def forward(self, request: web.Request, lease: NodeLease, body: bytes) -> web.Response | None:
        key = request.headers.get("Idempotency-Key", "")
        if not key or len(key) > 256:
            return None
        source = self.credential_provider().get(self.local_node_id) if self.local_node_id else None
        if source is None or "agent.llm_forward" not in source.scopes:
            return None
        digest = hashlib.sha256(request.method.encode() + b"\0" + request.raw_path.encode() + b"\0" + body).hexdigest()
        cache_key = (lease.node_id, key)
        async with self._forward_lock:
            for old_key, (_, expiry, outcome) in list(self._forwards.items()):
                if expiry <= self.clock() and not isinstance(outcome, asyncio.Future):
                    self._forwards.pop(old_key, None)
            existing = self._forwards.get(cache_key)
            if existing:
                old_digest, _, outcome = existing
                if not hmac.compare_digest(old_digest, digest):
                    return web.json_response({"error": "idempotency_conflict"}, status=409)
                if isinstance(outcome, _ForwardResult):
                    return web.Response(status=outcome.status, body=outcome.body, content_type=outcome.content_type)
                if outcome is None:
                    return web.json_response({"error": "outcome_unknown"}, status=409)
                future = outcome
                owner = False
            else:
                future: asyncio.Future[_ForwardResult | None] = asyncio.get_running_loop().create_future()
                self._forwards[cache_key] = (digest, self.clock() + 300, future)
                owner = True
        if not owner:
            try:
                outcome = await asyncio.shield(future)
                if outcome is None:
                    return web.json_response({"error": "outcome_unknown"}, status=409)
                return web.Response(status=outcome.status, body=outcome.body, content_type=outcome.content_type)
            except Exception:
                return web.json_response({"error": "outcome_unknown"}, status=409)
        try:
            deadline_raw = request.headers.get("X-Request-Deadline-Seconds", "30")
            deadline = min(MAX_FORWARD_SECONDS, max(0.1, float(deadline_raw)))
        except (TypeError, ValueError):
            deadline = 30.0
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in {"host", "content-length", "transfer-encoding", "authorization"}}
        headers.update({"Authorization": f"Bearer {source.token}",
                        "Idempotency-Key": key, "X-Node-Lease-ID": lease.lease_id,
                        "X-Node-Forwarded": "1",
                        "X-Request-Deadline-Seconds": str(deadline)})
        url = lease.endpoint + request.path
        try:
            async with ClientSession(timeout=ClientTimeout(total=deadline)) as session:
                async with session.request(request.method, url, params=request.query, data=body, headers=headers) as response:
                    result = _ForwardResult(response.status, await response.read(), response.content_type or "application/octet-stream")
            async with self._forward_lock:
                self._forwards[cache_key] = (digest, self.clock() + 300, result)
            if not future.done():
                future.set_result(result)
            return web.Response(status=result.status, body=result.body, content_type=result.content_type)
        except ClientConnectorError:
            async with self._forward_lock:
                self._forwards.pop(cache_key, None)
            if not future.done():
                future.set_result(None)
            return None
        except Exception:
            async with self._forward_lock:
                self._forwards[cache_key] = (digest, self.clock() + 300, None)
            if not future.done():
                future.set_result(None)
            return web.json_response({"error": "outcome_unknown"}, status=504)


def default_endpoint_for_peer(node_id: str, remote: str) -> str:
    """Build the peer URL only from the socket peer and fixed service port."""
    address = ipaddress.ip_address(remote)
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{host}:8091"


def create_runtime_node_api() -> NodeAPI:
    authority_file = Path(os.environ.get("GPU_MANAGER_AUTHORITY_FILE", Path(__file__).resolve().parents[1] / "config" / "authority.json"))

    def credentials() -> Mapping[str, NodeCredential]:
        return load_node_credentials(authority_file, os.environ.get("CREDENTIALS_DIRECTORY"))

    async def peer_node(remote: str | None, _credentials: Mapping[str, NodeCredential]) -> str | None:
        return await tailscale_peer_node(remote, credentials())

    return NodeAPI(NodeRegistry(), credential_provider=credentials, peer_node=peer_node,
                   endpoint_for_peer=default_endpoint_for_peer,
                   local_node_id=os.environ.get("GPU_MANAGER_NODE_ID"))


def add_node_routes(app: web.Application, api: NodeAPI | None = None) -> NodeAPI:
    api = api or create_runtime_node_api()
    api.add_routes(app)
    app["gpu_manager_node_api"] = api
    return api


__all__ = ["HEARTBEAT_SECONDS", "LEASE_SECONDS", "NodeAPI", "NodeCredential", "NodeLease",
           "NodeRegistry", "add_node_routes", "create_runtime_node_api", "default_endpoint_for_peer",
           "load_node_credentials", "tailscale_peer_node"]
