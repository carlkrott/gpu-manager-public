"""gpu-guard: observe-only GPU client attribution (collector and bounded state).

Pure functions over injected ``/proc`` and ``/sys`` roots plus an in-memory
``GuardState``.  Nothing here opens a device node: file descriptors that point at
a render node or ``/dev/kfd`` are only ``readlink``ed, and their ``fdinfo`` text
is read from the proc tree.  No ROCm/OpenCL import, no GPU call, no process
action of any kind.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import tempfile
import time
import threading
from aiohttp import web
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger("gpu_guard")

KFD_PATH = "/dev/kfd"
DEFAULT_WINDOW_SECONDS = 120.0
DEFAULT_MAX_PIDS = 4096
DEFAULT_MAX_SAMPLES = 240
DEFAULT_MAX_CMDLINE = 256
DEFAULT_MAX_UNREADABLE = 64
DEFAULT_MAX_WARNINGS = 1024
DEFAULT_HISTORY_SECONDS = 10.0
DEFAULT_MAX_HISTORY = 256
FAULT_BURST_SECONDS = 30.0
FAULT_MAX_LINE_BYTES = 64 * 1024
FAULT_MAX_RECEIPT_BYTES = 8 *  1024 * 1024
FAULT_MAX_LINES = 256

# "ring gfx_" is deliberately absent: it matches the benign init line
# "ring gfx_0.0.0 uses VM inv eng 0 on hub 0"; ring timeouts are covered by
# "timeout, signaled seq=". bad_op_irq and queue preemption are parts of the
# 2026-10-07 crash signature.
_FAULT_MARKERS = (
    "illegal opcode", "bad_op_irq", "queue preemption failed", "mode1 reset",
    "timeout, signaled seq=", "gpu hang detected", "gpu reset begin",
)

_ENGINES = ("engine_gfx_ns", "engine_compute_ns")


def matches_fault(line: str) -> bool:
    """Recognize bounded GPU fault signatures without case sensitivity."""
    lowered = str(line).lower()
    return any(marker in lowered for marker in _FAULT_MARKERS)


# --------------------------------------------------------------------------- #
# Parsing and discovery
# --------------------------------------------------------------------------- #

def discover_render_nodes(sys_root: Path | str, pci_addresses: Iterable[str]) -> dict[str, str]:
    """Map ``/dev/dri/renderDN`` to its PCI address for the wanted addresses.

    Node numbers are never configuration: the address comes from
    ``PCI_SLOT_NAME`` in ``<sys_root>/class/drm/renderD*/device/uevent``.
    """
    wanted = {a.lower() for a in pci_addresses}
    found: dict[str, str] = {}
    drm = Path(sys_root) / "class" / "drm"
    try:
        entries = sorted(drm.glob("renderD*"))
    except OSError:
        return found
    for entry in entries:
        try:
            text = (entry / "device" / "uevent").read_text()
        except OSError:
            continue
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key == "PCI_SLOT_NAME" and value.strip().lower() in wanted:
                found[f"/dev/dri/{entry.name}"] = value.strip()
                break
    return found


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_fdinfo(text: str) -> dict[str, Any]:
    """Extract the drm client id and cumulative counters; missing -> ``None``."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip()] = value.strip()

    def counter(key: str) -> int | None:
        raw = values.get(key)
        if raw is None:
            return None
        return _int_or_none(raw.split()[0]) if raw.split() else None

    client = values.get("drm-client-id")
    return {
        "drm_client_id": str(int(client)) if client is not None and _int_or_none(client) is not None else None,
        "engine_gfx_ns": counter("drm-engine-gfx"),
        "engine_compute_ns": counter("drm-engine-compute"),
        "total_vram_kib": counter("drm-total-vram"),
    }


def parse_proc_stat_starttime(text: str) -> int:
    """Return field 22 (starttime) of ``/proc/<pid>/stat``.

    ``comm`` is parenthesized and may hold spaces or ``)``, so split after the
    last ``)``; the first remaining token is field 3 (state).
    """
    _, _, rest = text.rpartition(")")
    return int(rest.split()[19])


def _unit_from_cgroup(text: str) -> str:
    """Deepest ``*.service``/``*.scope`` component of the cgroup path, else ``""``."""
    unit = ""
    for line in text.splitlines():
        path = line.rpartition(":")[2]
        for part in path.split("/"):
            if part.endswith((".service", ".scope")):
                unit = part
    return unit


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def _read_comm(pid_dir: Path) -> str:
    try:
        return pid_dir.joinpath("comm").read_text(errors="replace").strip()
    except OSError:
        return ""


def _holder_metadata(pid_dir: Path, pid: int, max_cmdline: int) -> dict[str, Any]:
    """Identity of one process incarnation; raises OSError/ValueError if gone."""
    starttime = parse_proc_stat_starttime(pid_dir.joinpath("stat").read_text(errors="replace"))
    try:
        raw = pid_dir.joinpath("cmdline").read_bytes()
    except OSError:
        raw = b""
    cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()[:max_cmdline]
    try:
        unit = _unit_from_cgroup(pid_dir.joinpath("cgroup").read_text(errors="replace"))
    except OSError:
        unit = ""
    return {
        "pid": pid,
        "starttime": starttime,
        "comm": _read_comm(pid_dir),
        "cmdline": cmdline,
        "unit": unit,
    }


def collect_snapshot(
    proc_root: Path | str,
    render_nodes: Mapping[str, str],
    sampled_at: float,
    *,
    managed_units: Iterable[str] = (),
    desktop_allowlist: Iterable[str] = (),
    max_cmdline: int = DEFAULT_MAX_CMDLINE,
    max_unreadable: int = DEFAULT_MAX_UNREADABLE,
) -> dict[str, Any]:
    """One pass over ``proc_root``.

    Returns ``{"sampled_at", "clients", "kfd_holders", "unreadable"}``.  Clients
    are deduplicated by ``(device, drm_client_id)`` and keep every pid/starttime
    holder.  KFD-only holders are listed separately: they carry no engine
    counters and are never ranked as engine-active.  Vanished pids are skipped
    silently; permission failures are reported in ``unreadable``.
    ``managed_units``/``desktop_allowlist`` are accepted for call symmetry with
    ``GuardState``; classification itself happens in the state.
    """
    root = Path(proc_root)
    clients: dict[tuple[str, str], dict[str, Any]] = {}
    kfd_holders: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    unreadable_total = 0
    try:
        pids = sorted(int(n) for n in os.listdir(root) if n.isdigit())
    except OSError:
        pids = []

    for pid in pids:
        pid_dir = root / str(pid)
        try:
            fds = os.listdir(pid_dir / "fd")
        except PermissionError:
            unreadable_total += 1
            if len(unreadable) < max_unreadable:
                unreadable.append({"pid": pid, "comm": _read_comm(pid_dir), "reason": "fd_list_denied"})
            continue
        except OSError:
            continue  # vanished or not a process directory

        holder: dict[str, Any] | None = None
        holds_kfd = False
        for fd in fds:
            try:
                target = os.readlink(pid_dir / "fd" / fd)
            except OSError:
                continue
            device = render_nodes.get(target)
            if device is None and target != KFD_PATH:
                continue
            if holder is None:
                try:
                    holder = _holder_metadata(pid_dir, pid, max_cmdline)
                except (OSError, ValueError, IndexError):
                    break  # vanished mid-read or unparsable: skip silently
            if target == KFD_PATH:
                holds_kfd = True
                continue
            try:
                info = parse_fdinfo((pid_dir / "fdinfo" / fd).read_text(errors="replace"))
            except PermissionError:
                unreadable_total += 1
                if len(unreadable) < max_unreadable:
                    unreadable.append({"pid": pid, "comm": holder["comm"], "device": device,
                                       "node": target, "reason": "fdinfo_denied"})
                continue
            except OSError:
                continue
            client_id = info["drm_client_id"]
            key = (device, client_id if client_id is not None else f"unknown:{pid}:{holder['starttime']}")
            client = clients.get(key)
            if client is None:
                client = clients[key] = {
                    "device": device,
                    "node": target,
                    "drm_client_id": client_id,
                    "engine_gfx_ns": None,
                    "engine_compute_ns": None,
                    "total_vram_kib": None,
                    "holders": [],
                }
            for field in (*_ENGINES, "total_vram_kib"):
                if client[field] is None:
                    client[field] = info[field]
            if not any(h["pid"] == pid and h["starttime"] == holder["starttime"] for h in client["holders"]):
                client["holders"].append(dict(holder))
        if holds_kfd and holder is not None:
            kfd_holders.append(dict(holder))

    return {
        "sampled_at": sampled_at,
        "clients": list(clients.values()),
        "kfd_holders": kfd_holders,
        "unreadable": unreadable,
        "unreadable_total": unreadable_total,
    }


# --------------------------------------------------------------------------- #
# Bounded attribution state
# --------------------------------------------------------------------------- #

def _incarnations(client: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    return frozenset((h["pid"], h["starttime"]) for h in client.get("holders", ()))


def _engine_delta(prev: tuple[int | None, int | None], cur: tuple[int | None, int | None]) -> int | None:
    """Sum of per-engine increases, or ``None`` when continuity cannot be proven.

    An engine reported on neither side is ignored; reported on one side only, or
    regressing, makes the whole delta unknown.
    """
    total = 0
    usable = False
    for before, after in zip(prev, cur):
        if before is None and after is None:
            continue
        if before is None or after is None or after < before:
            return None
        total += after - before
        usable = True
    return total if usable else None


class _Obs:
    __slots__ = ("ts", "engines", "incarnations", "delta")

    def __init__(self, ts, engines, incarnations, delta):
        self.ts = ts
        self.engines = engines
        self.incarnations = incarnations
        self.delta = delta


class GuardState:
    """Rolling per-client history with bounded pids, samples and warnings."""

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_pids: int = DEFAULT_MAX_PIDS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        managed_units: Iterable[str] = (),
        desktop_allowlist: Iterable[str] = (),
        max_warnings: int = DEFAULT_MAX_WARNINGS,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.max_pids = max(1, int(max_pids))
        self.max_samples = max(1, int(max_samples))
        self.managed_units = frozenset(managed_units)
        self.desktop_allowlist = tuple(desktop_allowlist)
        self.max_warnings = max(1, int(max_warnings))
        self.max_history = max(1, int(max_history))
        self._series: dict[tuple[str, str], deque[_Obs]] = {}
        self._clients: dict[tuple[str, str], dict[str, Any]] = {}
        self._refs: Counter = Counter()
        self._warned: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._foreign: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._prev_ts: float | None = None
        self._latest_ts: float | None = None
        self.last_snapshot: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=self.max_history)
        self.rejected_snapshots = 0

    # -- sizes ------------------------------------------------------------- #
    @property
    def sample_count(self) -> int:
        return max((len(s) for s in self._series.values()), default=0)

    @property
    def pid_count(self) -> int:
        return len(self._refs)

    # -- classification ---------------------------------------------------- #
    def _classify_holder(self, holder: Mapping[str, Any]) -> str:
        if holder.get("unit") and holder["unit"] in self.managed_units:
            return "managed"
        comm = holder.get("comm", "")
        if comm and any(fnmatch.fnmatchcase(comm, pat) for pat in self.desktop_allowlist):
            return "desktop"
        return "foreign"

    def classify(self, client: Mapping[str, Any]) -> str:
        """``managed``/``desktop``/``foreign`` for a client (or a bare holder).

        Exact managed unit names win over the desktop glob allowlist; a client
        with any foreign holder is foreign so a shared descriptor cannot hide it.
        """
        holders = client["holders"] if "holders" in client else [client]
        kinds = [self._classify_holder(h) for h in holders]
        if not kinds or "foreign" in kinds:
            return "foreign"
        return "managed" if "managed" in kinds else "desktop"

    # -- recording --------------------------------------------------------- #
    def record(self, snapshot: Mapping[str, Any]) -> None:
        ts = snapshot["sampled_at"]
        if self._latest_ts is not None and ts <= self._latest_ts:
            self.rejected_snapshots += 1
            # Keep the bounded observational history complete, without letting
            # an out-of-order/equal-time sample affect continuity or eviction.
            self._history.append(dict(snapshot))
            return
        continuous_prev = self._prev_ts
        self.last_snapshot = dict(snapshot)
        self._history.append(dict(snapshot))
        seen: set[tuple[str, str]] = set()
        for client in snapshot.get("clients", ()):
            key = (client["device"], client["drm_client_id"] if client.get("drm_client_id") is not None
                   else "unknown:" + ",".join(f"{p}.{s}" for p, s in sorted(_incarnations(client))))
            if key in seen:
                continue
            seen.add(key)
            engines = (client.get("engine_gfx_ns"), client.get("engine_compute_ns"))
            incs = _incarnations(client)
            series = self._series.setdefault(key, deque())
            delta = None
            if series and continuous_prev is not None and series[-1].ts == continuous_prev \
                    and series[-1].incarnations & incs:
                delta = _engine_delta(series[-1].engines, engines)
            series.append(_Obs(ts, engines, incs, delta))
            self._refs.update(incs)
            self._clients[key] = dict(client)
            self._warn_foreign(client)
        for holder in snapshot.get("kfd_holders", ()):
            self._warn_foreign(holder)
        self._prev_ts = ts
        self._latest_ts = ts
        self._evict(ts)

    def _warn_foreign(self, subject: Mapping[str, Any]) -> None:
        holders = subject["holders"] if "holders" in subject else [subject]
        for holder in holders:
            if self._classify_holder(holder) != "foreign":
                continue
            ident = (holder["pid"], holder["starttime"])
            self._foreign[ident] = dict(holder)
            self._foreign.move_to_end(ident)
            while len(self._foreign) > self.max_warnings:
                self._foreign.popitem(last=False)
            if ident in self._warned:
                continue
            self._warned[ident] = None
            while len(self._warned) > self.max_warnings:
                self._warned.popitem(last=False)
            logger.warning(
                "gpu-guard: foreign GPU client pid=%s comm=%s cmdline=%s unit=%s",
                holder["pid"], holder.get("comm", ""), holder.get("cmdline", ""), holder.get("unit", ""),
            )

    def _drop_head(self, key: tuple[str, str]) -> None:
        series = self._series[key]
        obs = series.popleft()
        self._refs.subtract(obs.incarnations)
        for inc in obs.incarnations:
            if self._refs[inc] <= 0:
                del self._refs[inc]
        if not series:
            del self._series[key]
            self._clients.pop(key, None)

    def _evict(self, now: float) -> None:
        horizon = now - self.window_seconds
        for key in list(self._series):
            series = self._series[key]
            while key in self._series and series and (series[0].ts <= horizon or len(series) > self.max_samples):
                self._drop_head(key)
        while len(self._refs) > self.max_pids and self._series:
            oldest = min(self._series, key=lambda k: self._series[k][0].ts)
            self._drop_head(oldest)

    # -- queries ----------------------------------------------------------- #
    def active_clients(self, since: float = 2.0) -> list[dict[str, Any]]:
        """Clients with a positive, uninterrupted engine delta newer than ``now - since``."""
        if self._latest_ts is None:
            return []
        cutoff = self._latest_ts - since
        active = []
        for key, series in self._series.items():
            total = sum(o.delta for o in series if o.ts > cutoff and o.delta)
            if total <= 0:
                continue
            client = self._clients[key]
            active.append({
                "device": client["device"],
                "drm_client_id": client.get("drm_client_id"),
                "engine_delta_ns": total,
                "last_seen": series[-1].ts,
                "holders": [dict(h) for h in client["holders"]],
                "classification": self.classify(client),
            })
        active.sort(key=lambda c: (-c["engine_delta_ns"], c["device"],
                                   c.get("drm_client_id") or "",
                                   min(((h.get("pid", 0), h.get("starttime", 0))
                                        for h in c["holders"]), default=(0, 0))))
        return active

    def foreign_holders(self) -> list[dict[str, Any]]:
        return [dict(h) for h in self._foreign.values()]

    def history(self, seconds: float = DEFAULT_HISTORY_SECONDS) -> list[dict[str, Any]]:
        """Return full snapshots within the rolling horizon, oldest first."""
        if self._latest_ts is None:
            return []
        cutoff = self._latest_ts - max(0.0, float(seconds))
        return [dict(snapshot) for snapshot in self._history
                if snapshot.get("sampled_at", float("-inf")) >= cutoff]


def _wall_time_string(wall_clock: Callable[[], Any]) -> str:
    value = wall_clock()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def build_receipt(
    *, matched_lines: Iterable[str], state: GuardState,
    snapshot: Mapping[str, Any], loadavg: Any = None, wall_time: Any = None,
    loadavg_provider: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Build a non-causal fault receipt from recent state and a full snapshot."""
    latest = state._latest_ts if state._latest_ts is not None else float(snapshot.get("sampled_at", 0.0))
    history = state.history(DEFAULT_HISTORY_SECONDS)
    ranked = state.active_clients(since=2.0)
    # Include independently observed clients from full snapshots as well. The
    # attribution state intentionally rejects equal-time samples for continuity;
    # the receipt can still report their positive counter movement without
    # treating it as causal evidence.
    observations: dict[tuple[str, str], list[tuple[float, Mapping[str, Any]]]] = {}
    for sample in history:
        ts = float(sample.get("sampled_at", 0.0))
        for client in sample.get("clients", ()):
            cid = client.get("drm_client_id")
            key = (str(client.get("device", "")), str(cid) if cid is not None else "")
            observations.setdefault(key, []).append((ts, client))
    for (device, client_id), rows in observations.items():
        rows.sort(key=lambda row: row[0])
        if len(rows) < 2 or rows[-1][0] <= latest - 2.0:
            continue
        before, after = rows[-2][1], rows[-1][1]
        deltas = [_engine_delta((before.get("engine_gfx_ns"), before.get("engine_compute_ns")),
                                (after.get("engine_gfx_ns"), after.get("engine_compute_ns")))]
        delta = deltas[0]
        if delta is None or delta <= 0:
            continue
        item = {"device": device, "drm_client_id": client_id or None,
                "engine_delta_ns": delta, "last_seen": rows[-1][0],
                "holders": [dict(h) for h in after.get("holders", ())],
                "classification": state.classify(after)}
        if not any(x.get("device") == device and x.get("drm_client_id") == item["drm_client_id"] for x in ranked):
            ranked.append(item)
    ranked.sort(key=lambda c: (-c["engine_delta_ns"], c["device"],
                               c.get("drm_client_id") or "",
                               min(((h.get("pid", 0), h.get("starttime", 0))
                                    for h in c.get("holders", ())), default=(0, 0))))
    # Resolve active engine identities to every available holder; never imply
    # that temporal proximity establishes causation.
    active = []
    for item in ranked:
        holders = sorted(item.get("holders", ()), key=lambda h: (h.get("pid", 0), h.get("starttime", 0)))
        for holder in holders or [{}]:
            active.append({**item, **holder})
    holders_by_node: dict[str, list[dict[str, Any]]] = {}
    for client in snapshot.get("clients", ()):
        device = str(client.get("device", "unknown"))
        holders_by_node.setdefault(device, []).extend(dict(h) for h in client.get("holders", ()))
    if loadavg is None and loadavg_provider is not None:
        try:
            loadavg = loadavg_provider()
        except Exception:
            loadavg = None
    if wall_time is None:
        wall_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "wall_time": str(wall_time), "matched_lines": list(matched_lines),
        "history": history, "active_clients": active,
        "holders_by_node": {k: sorted(v, key=lambda h: (h.get("pid", 0), h.get("starttime", 0)))
                             for k, v in sorted(holders_by_node.items())},
        "kfd_holders": [dict(h) for h in snapshot.get("kfd_holders", ())],
        "unreadable": [dict(x) for x in snapshot.get("unreadable", ())],
        "unreadable_total": snapshot.get("unreadable_total", len(snapshot.get("unreadable", ()))),
        "loadavg": loadavg, "sampled_at": latest,
        "attribution": "temporal_association_not_causation",
    }


class FaultRecorder:
    """Atomically maintain one bounded JSON receipt per fixed 30-second burst."""

    def __init__(self, root: Path | str, *, monotonic_clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], Any] | None = None,
                 loadavg_provider: Callable[[], Any] | None = None) -> None:
        self.root = Path(root)
        self.monotonic_clock = monotonic_clock
        self.wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self.loadavg_provider = loadavg_provider
        self.error_count = 0
        self._burst_started: float | None = None
        self._receipt: Path | None = None

    def _safe_root(self) -> None:
        if self.root.is_symlink():
            raise ValueError("receipt root must not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("receipt root must be a real directory")
        os.chmod(self.root, 0o700)

    def record(self, line: str, event: Mapping[str, Any] | None = None,
               *, state: GuardState | None = None, snapshot: Mapping[str, Any] | None = None) -> Path | None:
        line = str(line)
        event = dict(event or {})
        if len(line.encode("utf-8")) > FAULT_MAX_LINE_BYTES or len(json.dumps(event, ensure_ascii=False).encode("utf-8")) > FAULT_MAX_LINE_BYTES:
            raise ValueError("fault receipt input exceeds line cap")
        now = float(self.monotonic_clock())
        if self._burst_started is None or now >= self._burst_started + FAULT_BURST_SECONDS:
            self._burst_started = now
            stamp = _wall_time_string(self.wall_clock).replace(":", "").replace("-", "")
            self._receipt = self.root / f"fault-{stamp}-{int(now * 1000000)}.json"
        assert self._receipt is not None
        if not matches_fault(line):
            return self._receipt
        try:
            self._safe_root()
            target = self._receipt
            if target.is_symlink():
                raise ValueError("receipt target must not be a symlink")
            if target.exists():
                current = json.loads(target.read_text(encoding="utf-8"))
                lines = current.get("matched_lines", [])
                events = current.get("events", [])
                if len(lines) >= FAULT_MAX_LINES:
                    raise ValueError("fault receipt burst line cap exceeded")
            else:
                lines, events = [], []
                current = {"wall_time": _wall_time_string(self.wall_clock), "matched_lines": lines, "events": events}
            lines = [*lines, line]
            events = [*events, event]
            current.update(matched_lines=lines, events=events)
            if state is not None and snapshot is not None:
                current["context"] = build_receipt(matched_lines=lines, state=state, snapshot=snapshot,
                                                    loadavg_provider=self.loadavg_provider)
            payload = json.dumps(current, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(payload) > FAULT_MAX_RECEIPT_BYTES:
                raise ValueError("fault receipt exceeds file cap")
            fd, temp_name = tempfile.mkstemp(prefix=".fault-", suffix=".tmp", dir=self.root)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, target)
                os.chmod(target, 0o600)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
            return target
        except ValueError:
            raise
        except Exception as exc:
            self.error_count += 1
            logger.warning("gpu-guard: fault receipt write failed: %s", exc)
            return None


class JournalFollower:
    """Own a bounded journalctl follower and consume only JSON MESSAGE values."""

    def __init__(self, *, process_factory=None, receipt_recorder, sleep=asyncio.sleep,
                 monotonic_clock=time.monotonic, max_record_bytes=FAULT_MAX_LINE_BYTES):
        self.process_factory = process_factory or self._spawn
        self.receipt_recorder = receipt_recorder
        self.sleep = sleep
        self.monotonic_clock = monotonic_clock
        self.max_record_bytes = max(1, int(max_record_bytes))
        self.process = None
        self.retry_delays = []
        self.malformed_records = self.oversized_records = self.accepted_records = 0
        self._retry_delay = 1
        self._closed = False

    @staticmethod
    async def _spawn(argv):
        return await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.DEVNULL)

    async def _new_process(self):
        argv = ["journalctl", "-k", "-f", "--output=json", "--no-pager", "-n", "0"]
        proc = self.process_factory(argv)
        self.process = await proc if hasattr(proc, "__await__") else proc
        return self.process

    async def start(self):
        if self._closed or self.process is not None:
            return
        await self._new_process()

    async def read_available(self):
        proc = self.process
        if proc is None or self._closed:
            return
        while not self._closed:
            raw = await proc.stdout.readline()
            if not raw:
                return
            if len(raw) > self.max_record_bytes:
                self.oversized_records += 1
                continue
            try:
                record = json.loads(raw)
                line = record.get("MESSAGE") if isinstance(record, dict) else None
                if not isinstance(line, str):
                    raise ValueError("missing MESSAGE")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                self.malformed_records += 1
                continue
            self.accepted_records += 1
            if matches_fault(line):
                try:
                    result = self.receipt_recorder(line, record)
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    logger.exception("gpu-guard: journal receipt callback failed")

    async def restart_after_exit(self):
        if self._closed:
            return
        proc = self.process
        if proc is not None:
            await proc.wait()
            self.process = None
        delay = self._retry_delay
        self.retry_delays.append(delay)
        await self.sleep(delay)
        self._retry_delay = min(30, delay * 2)
        if not self._closed:
            proc = await self._new_process()
            # This path is called after EOF; reap a process that exited during
            # the retry handoff before the next attempt.
            if getattr(proc, "returncode", None) is not None:
                await proc.wait()
                self.process = None

    async def run(self):
        while not self._closed:
            if self.process is None:
                await self._new_process()
            await self.read_available()
            if self._closed:
                break
            await self.restart_after_exit()
            # A test or injected clock may implement sleep without yielding.
            await asyncio.sleep(0)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        proc, self.process = self.process, None
        if proc is not None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


class GuardRuntime:
    """Async owner for the sampler; slow proc inspection never blocks the loop."""

    def __init__(self, *, config, proc_root, sys_root, receipt_root, collector=None,
                 monotonic_clock=time.monotonic, sleep=asyncio.sleep, process_factory=None,
                 auth_decision=None, managed_units=(), desktop_allowlist=()):
        self.enabled = bool(config.get("enabled", False))
        self.interval_seconds = max(0.5, float(config.get("interval_seconds", 1)))
        self.window_seconds = float(config.get("window_seconds", DEFAULT_WINDOW_SECONDS))
        self.max_pids = int(config.get("max_pids", DEFAULT_MAX_PIDS))
        self.max_samples = int(config.get("max_samples", DEFAULT_MAX_SAMPLES))
        self.max_cmdline = int(config.get("max_cmdline", DEFAULT_MAX_CMDLINE))
        self.debounce_seconds = float(config.get("debounce_seconds", FAULT_BURST_SECONDS))
        self.pci_addresses = tuple(config.get("pci_addresses", ("0000:03:00.0", "0000:0e:00.0", "0000:75:00.0")))
        self.proc_root, self.sys_root, self.receipt_root = map(Path, (proc_root, sys_root, receipt_root))
        self.monotonic_clock, self.sleep = monotonic_clock, sleep
        self.collector = collector or collect_snapshot
        self.managed_units = tuple(managed_units)
        self.desktop_allowlist = tuple(desktop_allowlist)
        self.state = GuardState(window_seconds=self.window_seconds, max_pids=self.max_pids,
                                max_samples=self.max_samples, managed_units=self.managed_units,
                                desktop_allowlist=self.desktop_allowlist)
        self.recorder = FaultRecorder(self.receipt_root, monotonic_clock=monotonic_clock)
        self.follower = JournalFollower(process_factory=process_factory,
                                        receipt_recorder=self._record_fault, sleep=sleep,
                                        monotonic_clock=monotonic_clock)
        self.auth_decision = auth_decision or (lambda _request: True)
        self.task = None
        self.follower_task = None
        self.last_error = None
        self.error_count = 0
        self.last_pass_seconds = None
        self._closed = False
        self._state_lock = threading.Lock()
        self.cleanup_count = 0

    def _collect(self):
        nodes = discover_render_nodes(self.sys_root, self.pci_addresses)
        return self.collector(self.proc_root, nodes, self.monotonic_clock(),
                              managed_units=self.managed_units,
                              desktop_allowlist=self.desktop_allowlist,
                              max_cmdline=self.max_cmdline)

    async def run_one_pass(self):
        if not self.enabled or self._closed:
            return None
        started = self.monotonic_clock()
        try:
            snapshot = await asyncio.to_thread(self._collect)
            with self._state_lock:
                if self._closed:
                    return None
                self.state.record(snapshot)
                self.last_error = None
            self.last_pass_seconds = max(0.0, self.monotonic_clock() - started)
            # Repeated slow passes back off; ordinary passes converge on configured cadence.
            if self.last_pass_seconds > self.interval_seconds:
                self.interval_seconds = min(30.0, max(0.5, self.interval_seconds * 2))
            return snapshot
        except asyncio.CancelledError:
            raise
        except Exception:
            self.error_count += 1
            self.last_error = "collector_error"
            self.interval_seconds = min(30.0, max(0.5, self.interval_seconds * 2))
            logger.exception("gpu-guard: collector pass failed")
            return None

    async def _sample_loop(self):
        while not self._closed:
            await self.run_one_pass()
            await self.sleep(self.interval_seconds)
            # Injected clocks may complete without yielding; preserve cancellation fairness.
            await asyncio.sleep(0)

    async def _record_fault(self, line, event):
        if self._closed:
            return
        snapshot = self.state.last_snapshot or {"sampled_at": self.monotonic_clock(), "clients": [], "kfd_holders": []}
        await asyncio.to_thread(self.recorder.record, line, event, state=self.state, snapshot=snapshot)

    async def start(self):
        if not self.enabled or self.task is not None or self._closed:
            return
        self.task = asyncio.create_task(self._sample_loop(), name="gpu-guard-sampler")
        self.follower_task = asyncio.create_task(self.follower.run(), name="gpu-guard-journal")

    async def close(self):
        if self.cleanup_count:
            return
        self.cleanup_count += 1
        with self._state_lock:
            self._closed = True
        tasks = [task for task in (self.task, self.follower_task) if task is not None]
        for task in tasks:
            task.cancel()
        await self.follower.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def status(self):
        with self._state_lock:
            snapshot = self.state.last_snapshot or {}
            return {"enabled": self.enabled, "status": "on" if self.enabled else "off",
                    "sampled_at": snapshot.get("sampled_at"),
                    "holders": [h for c in snapshot.get("clients", ()) for h in c.get("holders", ())],
                    "kfd_holders": list(snapshot.get("kfd_holders", ())),
                    "unreadable": list(snapshot.get("unreadable", ())),
                    "unreadable_total": snapshot.get("unreadable_total", 0),
                    "errors": self.error_count + self.recorder.error_count,
                    "last_error": self.last_error,
                    "last_pass_seconds": self.last_pass_seconds,
                    "latest_receipt": str(self.recorder._receipt) if self.recorder._receipt and self.recorder._receipt.exists() else None}


async def _gpu_guard_cleanup_context(app):
    service = app["gpu_guard"]
    await service.start()
    try:
        yield
    finally:
        await service.close()


def register_gpu_guard(app, *, config, proc_root="/proc", sys_root="/sys",
                       receipt_root=None, collector=None,
                       monotonic_clock=time.monotonic, sleep=asyncio.sleep,
                       process_factory=None, auth_decision=None, managed_units=(),
                       desktop_allowlist=()):
    """Install an authenticated observational status endpoint and owned lifecycle."""
    if receipt_root is None:
        receipt_root = Path("~/.local/state/gpu-manager/gpu-guard").expanduser()
    service = GuardRuntime(config=config, proc_root=proc_root, sys_root=sys_root,
                           receipt_root=receipt_root, collector=collector,
                           monotonic_clock=monotonic_clock, sleep=sleep,
                           process_factory=process_factory, auth_decision=auth_decision,
                           managed_units=managed_units, desktop_allowlist=desktop_allowlist)
    app["gpu_guard"] = service

    async def status(request):
        try:
            authorized = service.auth_decision(request)
            if hasattr(authorized, "__await__"):
                authorized = await authorized
        except Exception:
            authorized = False
        if not authorized:
            raise web.HTTPUnauthorized(text="authentication required")
        return web.json_response(service.status())

    app.router.add_get("/v1/gpu-guard/status", status)
    app.cleanup_ctx.append(_gpu_guard_cleanup_context)
    return service
