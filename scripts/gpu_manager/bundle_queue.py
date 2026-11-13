"""Facade-bound bundle queues, priority tiers, and port proxy templates.

The bind step creates facade-local classes/functions; mutable registries and
PRIORITY_TIER_MAP remain owned by the facade.
"""
from __future__ import annotations

import enum
import types
from enum import Enum

NAMES = ('BundleRequestQueue', '_get_bundle_request_queue', 'PriorityTier', 'PortProxy', '_start_all_port_proxies', '_stop_all_port_proxies', '_BundleRequestQueue', '_get_bundle_queue', '_check_bundle_services_healthy', 'priority_to_tier')
BUNDLE_PROXY_WAIT_TIMEOUT = 300  # template-only value; bind snapshots the facade value

class BundleRequestQueue:
    """Pending HTTP request queue for a single bundle.

    When a request arrives for a bundle that isn't loaded, the request
    is held here as a Future.  When the bundle loads successfully all
    futures are resolved with True; on failure they resolve with False
    and the caller returns 503.

    Each pending entry carries:
      - ``future``: asyncio.Future[bool] that the handler awaits
      - ``enqueued_at``: timestamp for timeout checks
      - ``request_info``: dict with method/path/remote for logging

    Configurable per-bundle timeout (``bundle_config.get('timeout', 60)``).
    """

    def __init__(self, bundle_name: str, timeout: float = 60.0):
        self.bundle_name = bundle_name
        self.timeout = timeout
        # List of pending Future[bool] — True=loaded, False=failed
        self._pending: list[dict] = []
        self._lock = asyncio.Lock()

    # ── Public API ──────────────────────────────────────────────────

    async def wait_for_load(self, request_info: dict | None = None) -> bool:
        """Block until the bundle is ready, fails, or times out.

        Returns True when the bundle loaded successfully, False on
        failure or timeout.  *request_info* is an optional dict
        (method, path, remote) used for logging.
        """
        fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        entry = {
            "future": fut,
            "enqueued_at": time.time(),
            "request_info": request_info or {},
        }
        async with self._lock:
            self._pending.append(entry)

        tag = request_info.get("path", "") if request_info else ""
        logger.info(
            f"BundleQueue[{self.bundle_name}]: request queued "
            f"(pending={len(self._pending)}) {tag}"
        )

        try:
            result = await asyncio.wait_for(fut, timeout=self.timeout)
            return result
        except asyncio.TimeoutError:
            # Remove ourselves from the pending list
            async with self._lock:
                self._pending = [e for e in self._pending if e["future"] is not fut]
            logger.warning(
                f"BundleQueue[{self.bundle_name}]: request timed out "
                f"after {self.timeout}s"
            )
            return False

    async def resolve_all(self, success: bool) -> int:
        """Resolve every pending future and return the count resolved."""
        async with self._lock:
            entries = self._pending[:]
            self._pending.clear()

        for entry in entries:
            fut = entry["future"]
            if not fut.done():
                fut.set_result(success)

        status = "loaded" if success else "failed"
        logger.info(
            f"BundleQueue[{self.bundle_name}]: resolved {len(entries)} "
            f"pending request(s) as {status}"
        )
        return len(entries)

    def pending_count(self) -> int:
        return len(self._pending)

    def to_dict(self) -> dict:
        return {
            "bundle_name": self.bundle_name,
            "timeout": self.timeout,
            "pending_count": len(self._pending),
            "pending_requests": [
                {
                    "enqueued_at": e["enqueued_at"],
                    "elapsed": round(time.time() - e["enqueued_at"], 1),
                    "info": e.get("request_info", {}),
                }
                for e in self._pending
            ],
        }


def _get_bundle_request_queue(bundle_name: str, timeout: float = 60.0) -> BundleRequestQueue:
    """Get or create the request queue for a bundle."""
    if bundle_name not in _bundle_request_queues:
        _bundle_request_queues[bundle_name] = BundleRequestQueue(bundle_name, timeout)
    return _bundle_request_queues[bundle_name]


class PriorityTier(Enum):
    HIGH = 1       # Interactive requests - guaranteed 1 slot
    MEDIUM = 2     # Normal batch work
    LOW = 3        # Background/preemptible work


class PortProxy:
    """TCP port forwarding proxy that bridges external connections to internal services."""

    def __init__(self, name: str, listen_port: int, target_host: str, target_port: int, logger: logging.Logger):
        self.name = name
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self.logger = logger
        self._server = None

    async def start(self):
        """Start the TCP proxy server."""
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            peername = writer.get_extra_info('peername')
            self.logger.info(f"PortProxy[{self.name}]: connection from {peername}")
            backend_writer = None
            try:
                backend_reader, backend_writer = await asyncio.wait_for(
                    asyncio.open_connection(self.target_host, self.target_port),
                    timeout=10
                )
            except Exception as e:
                self.logger.warning(f"PortProxy[{self.name}]: backend connect failed: {e}")
                try:
                    writer.close()
                except Exception:
                    pass
                return

            async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter, direction: str):
                try:
                    while True:
                        data = await src.read(65536)
                        if not data:
                            break
                        dst.write(data)
                        await dst.drain()
                except Exception as e:
                    self.logger.debug(f"PortProxy[{self.name}]: pipe {direction} closed: {e}")
                finally:
                    try:
                        if direction == "upload":
                            backend_writer.close()
                        else:
                            writer.close()
                    except Exception:
                        pass

            try:
                await asyncio.gather(
                    pipe(reader, backend_writer, "upload"),
                    pipe(backend_reader, writer, "download"),
                )
            except Exception:
                pass
            self.logger.info(f"PortProxy[{self.name}]: connection closed {peername}")

        self._server = await asyncio.start_server(
            handle_client, '0.0.0.0', self.listen_port
        )
        self.logger.info(f"PortProxy[{self.name}]: listening on 0.0.0.0:{self.listen_port} → {self.target_host}:{self.target_port}")

    async def stop(self):
        """Stop the TCP proxy server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self.logger.info(f"PortProxy[{self.name}]: stopped")
            self._server = None

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._server.is_serving()


async def _start_all_port_proxies(config: dict):
    """Port proxies disabled — nginx handles all proxy routing.
    See nginx-manager.sh and _regenerate_nginx_config()."""
    pass


async def _stop_all_port_proxies():
    """Stop all running port proxies."""
    for name in list(_port_proxies.keys()):
        try:
            await _port_proxies[name].stop()
        except Exception as e:
            logger.error(f"PortProxy[{name}]: stop failed: {e}")
    _port_proxies.clear()


class _BundleRequestQueue:
    """Manages queued HTTP requests for a single bundle.

    When the bundle is not loaded, incoming requests are held here.
    ``signal_ready()`` unblocks all waiters; ``signal_failed(err)``
    returns 503 to all queued requests.
    """

    def __init__(self, bundle_name: str):
        self.bundle_name = bundle_name
        self._ready_event = asyncio.Event()
        self._failed: bool = False
        self._fail_reason: str = ""
        self._queued_count: int = 0
        self._total_queued: int = 0
        self._total_served: int = 0

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set() and not self._failed

    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def queued_count(self) -> int:
        return self._queued_count

    def mark_loading(self) -> None:
        """Reset state for a new load attempt."""
        self._ready_event.clear()
        self._failed = False
        self._fail_reason = ""

    def signal_ready(self) -> None:
        """Bundle is healthy — unblock all queued requests."""
        self._failed = False
        self._ready_event.set()

    def signal_failed(self, reason: str = "bundle load failed") -> None:
        """Bundle failed to load — queued requests get 503."""
        self._failed = True
        self._fail_reason = reason
        self._ready_event.set()

    async def wait_for_ready(self, timeout: float = BUNDLE_PROXY_WAIT_TIMEOUT) -> bool:
        """Block until bundle is ready or timeout/failure. Returns True if ready."""
        self._queued_count += 1
        self._total_queued += 1
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return self.is_ready
        except asyncio.TimeoutError:
            return False
        finally:
            self._queued_count -= 1

    def mark_served(self) -> None:
        self._total_served += 1

    def to_dict(self) -> dict:
        return {
            "bundle_name": self.bundle_name,
            "ready": self.is_ready,
            "failed": self._failed,
            "fail_reason": self._fail_reason,
            "queued": self._queued_count,
            "total_queued": self._total_queued,
            "total_served": self._total_served,
        }

    def reset(self) -> None:
        """Reset counters (e.g. after bundle reload)."""
        self._ready_event.clear()
        self._failed = False
        self._fail_reason = ""
        self._queued_count = 0
        self._total_queued = 0
        self._total_served = 0


def _get_bundle_queue(bundle_name: str) -> _BundleRequestQueue:
    """Get or create the request queue for a bundle."""
    if bundle_name not in _bundle_proxy_queues:
        _bundle_proxy_queues[bundle_name] = _BundleRequestQueue(bundle_name)
    return _bundle_proxy_queues[bundle_name]


def _check_bundle_services_healthy(bundle: dict) -> bool:
    """Check if ALL services in a bundle are running and port-responding.

    Uses a quick TCP connect to the service port to verify liveness.
    """
    services_cfg = _services_config.get("services", {})
    for svc_name in bundle.get("services", []):
        svc = services_cfg.get(svc_name, {})
        port = svc.get("port")
        if not port:
            continue
        try:
            import socket as _sock
            with _sock.create_connection(("localhost", int(port)), timeout=2) as s:
                pass  # port is open
        except (OSError, TimeoutError):
            return False
    return True


def priority_to_tier(priority: int) -> PriorityTier:
    """Map generation priority through the shared named-priority contract."""
    priority_class = priority_envelope(
        priority,
        convention=PriorityConvention.GENERATION_HIGHER_IS_URGENT,
    ).priority_class.value
    return {
        "interactive": PriorityTier.HIGH,
        "normal": PriorityTier.MEDIUM,
        "background": PriorityTier.LOW,
    }[priority_class]


class _Probe(Enum):
    MEMBER = 1


_ENUM_INTERNAL = frozenset(
    set(vars(_Probe))
    - {"MEMBER", "__module__", "__doc__", "__firstlineno__", "__static_attributes__",
       "_generate_next_value_"}
)


def bind(facade_globals):
    """Bind queue definitions to a facade, preserving its timeout snapshot."""
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    module_name = facade_globals.get("__name__", __name__)

    def bind_definition(template):
        class_cell = types.CellType()

        def rebuild(function, *, snapshot_timeout=False):
            closure = function.__closure__
            if closure and "__class__" in function.__code__.co_freevars:
                closure = tuple(
                    class_cell if name == "__class__" else cell
                    for name, cell in zip(function.__code__.co_freevars, closure)
                )
            defaults = function.__defaults__
            if snapshot_timeout:
                defaults = (facade_globals["BUNDLE_PROXY_WAIT_TIMEOUT"],)
            rebuilt = types.FunctionType(
                function.__code__, facade_globals, function.__name__, defaults, closure
            )
            rebuilt.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
            rebuilt.__annotations__ = function.__annotations__.copy()
            rebuilt.__doc__ = function.__doc__
            rebuilt.__qualname__ = function.__qualname__
            rebuilt.__module__ = module_name
            rebuilt.__dict__.update(function.__dict__)
            return rebuilt

        def convert(key, value):
            if isinstance(value, types.FunctionType):
                return rebuild(value, snapshot_timeout=(
                    template.__name__ == "_BundleRequestQueue" and key == "wait_for_ready"
                ))
            if isinstance(value, (staticmethod, classmethod)):
                return type(value)(rebuild(value.__func__))
            if type(value) is property:
                return property(
                    rebuild(value.fget) if value.fget else None,
                    rebuild(value.fset) if value.fset else None,
                    rebuild(value.fdel) if value.fdel else None, value.__doc__,
                )
            return value

        meta = type(template)
        if isinstance(template, enum.EnumMeta):
            namespace = meta.__prepare__(template.__name__, template.__bases__)
            members = template._member_map_
            default_next = enum.Enum._generate_next_value_.__code__
            for key, value in vars(template).items():
                if key in _ENUM_INTERNAL or key in members:
                    continue
                if key == "_generate_next_value_" and getattr(value, "__func__", value).__code__ is default_next:
                    continue
                namespace[key] = convert(key, value)
            namespace["__qualname__"] = template.__qualname__
            namespace["__module__"] = module_name
            for name, member in members.items():
                namespace[name] = member._value_
        else:
            namespace = {}
            for key, value in vars(template).items():
                if key in ("__dict__", "__weakref__"):
                    continue
                namespace[key] = convert(key, value)
            namespace["__module__"] = module_name
        bound = meta(template.__name__, template.__bases__, namespace)
        class_cell.cell_contents = bound
        return bound

    return tuple(
        bind_definition(globals()[name]) if isinstance(globals()[name], type)
        else _bind_function(globals()[name], facade_globals, module_name)
        for name in NAMES
    )


def _bind_function(function, facade_globals, module_name):
    bound = types.FunctionType(function.__code__, facade_globals, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    bound.__annotations__ = function.__annotations__.copy()
    bound.__doc__ = function.__doc__
    bound.__qualname__ = function.__qualname__
    bound.__module__ = module_name
    bound.__dict__.update(function.__dict__)
    return bound
