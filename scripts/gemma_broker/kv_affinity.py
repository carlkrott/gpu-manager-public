"""KV slot affinity and cold storage for a llama-server broker member (Phase 4d).

Enabled per member by the services.json field (default off, absent = off):

    "kv_affinity": {"enabled": true,
                    "slot_save_dir": "<the member's --slot-save-path>",
                    "forbidden_roots": ["<host NAS mount, host config only>"],
                    "engine": "llama",
                    "min_tokens": 8192, "io_timeout": 120,
                    "max_gb": 100, "max_age_days": 7, "max_files": 32}

Affinity applies only to requests with an authenticated principal (the verified
token role gpu-manager stores on the request); the session key is
sha256(principal + ":" + X-Session-Id, or + system + first user message), so
two principals never share a slot file. The broker keeps a session -> slot map and sends ``id_slot`` so a session's
follow-up turns land on the slot that still holds its prompt cache. When the
chosen slot holds another session with more than ``min_tokens`` cached tokens,
that session is saved first (``POST /slots/N?action=save``) and the newcomer's
saved file, if any, is restored. Before a planned member stop every large slot
is saved; the next request of that session restores it lazily. Every save
first prunes the directory to max_gb / max_age_days / max_files and is skipped
(or its file dropped) when the budget cannot be met. Every failure is logged
and the request is dispatched normally (fail-open).

Engine-neutral by design (Breccia tie-in, 2026-10-27): session routing, the session key, retention
and the flat ``<fingerprint>-<session>.bin`` layout do not depend on the engine. Only the cold-tier
transport does, and only ``engine: "llama"`` (the /slots HTTP API) has one. Any other engine (Breccia
today: no slot concept, no save/restore API) gets no pinning and no save/restore; it fails open to a
plain dispatch until it declares ``kv_identity`` and gets a transport here. The fingerprint takes the
engine's own declared layout (``kv_identity``: engine, arch, k_layout, v_layout, state_version), never
argv, because an engine can accept -ctk/-ctv and run another layout. ``slot_save_dir`` is this module's
own directory (``kv-slots/``): prune() deletes any *.bin in it, so an engine's own cold tier (Breccia
T3) must never share it. Both budgets draw on one disk allowance, so ``max_gb`` here plus the engine
tier's cap must sum to that allowance, not each claim it whole.

Saved files are ``<fingerprint>-<session>.bin`` inside slot_save_dir. The flat
name is forced: llama-server rejects sub-directories in slot filenames
(fs_validate_filename, allow_subdirs=false), so the fingerprint is a prefix.
The fingerprint covers the model shards and mmproj (path, size, mtime), the KV
cache types, the image token bounds, and the build: llama-server plus every
lib*.so* beside it (size, mtime) and the sha256 of common/build-info.cpp, which
holds the commit. libllama carries LLAMA_STATE_SEQ_VERSION, so a rebuilt library
changes the key. Not covered: a same-size library rewrite that keeps its mtime
(UNVERIFIED — requires the GPU qualification card). A file from another model or
build must never be restored: llama-server reuses the common token prefix of the
restored slot, so a mismatched file can corrupt output.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("gemma_broker.kv_affinity")

SESSION_FIELD = "_kv_session_id"  # server-only: DurableBrokerService.submit drops client copies, sets it from compat's kv_scope; stripped before any member
DEFAULT_MIN_TOKENS = 8192
DEFAULT_MAX_BYTES = 100 * 1024**3
DEFAULT_MAX_AGE_S = 7 * 86400
SLOT_ENGINES = ("llama",)  # engines with a save/restore transport in KvAffinity
DEFAULT_MAX_FILES = 32  # distinct saved sessions per directory
_NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse", "fuse.sshfs", "9p", "ceph", "glusterfs")


def session_key(body: dict[str, Any], header_value: Any = None, principal: str = "") -> str:
    """Filename-safe session id scoped by principal: hash of the caller id, else of system + first user."""
    if header_value:
        material = "hdr:" + str(header_value)
    else:
        system, first_user = "", ""
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            if message.get("role") == "system" and not system:
                system = text
            elif message.get("role") == "user":
                first_user = text
                break
        material = "msg:" + system + "\x00" + first_user
    # Two principals never share a key, so one can never restore the other's saved state.
    return hashlib.sha256((principal + ":" + material).encode("utf-8", "replace")).hexdigest()[:24]


def _stat_entry(path: str) -> list[Any]:
    try:
        st = os.stat(path)
        return [path, st.st_size, int(st.st_mtime)]
    except OSError:
        return [path, None, None]


def _build_identity(bin_path: str) -> list[Any]:
    """llama-server stat, every lib*.so* beside it (libllama holds the state format), and the commit."""
    if not bin_path:
        return []
    bin_dir = os.path.dirname(bin_path)
    try:
        with open(os.path.join(os.path.dirname(bin_dir), "common", "build-info.cpp"), "rb") as info:
            commit = hashlib.sha256(info.read()).hexdigest()
    except OSError:
        commit = None
    libs = [_stat_entry(p) for p in sorted(glob.glob(os.path.join(bin_dir, "lib*.so*")))]
    return [_stat_entry(bin_path), libs, commit]


def _arg(args: list[Any], *names: str) -> Any:
    for i, item in enumerate(args[:-1]):
        if item in names:
            return args[i + 1]
    return None


def fingerprint(service: dict[str, Any]) -> str:
    """Hash of everything that makes a saved KV state incompatible (see module docstring)."""
    args = [str(a) for a in service.get("launch_args") or []]
    model = str(service.get("model_path") or _arg(args, "-m", "--model") or "")
    shards = sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model))) or [model]
    mmproj = _arg(args, "--mmproj", "-mm")
    identity = service.get("kv_identity")
    declared = isinstance(identity, dict) and identity
    material = {
        "kv_identity": {k: identity.get(k) for k in ("engine", "arch", "k_layout", "v_layout", "state_version")}
        if declared else None,
        "model": [_stat_entry(p) for p in shards],
        "mmproj": _stat_entry(mmproj) if mmproj else None,
        # a declared layout replaces argv: the engine may accept -ctk/-ctv and run another layout
        "ctk": None if declared else service.get("cache_type_k") or _arg(args, "-ctk", "--cache-type-k"),
        "ctv": None if declared else service.get("cache_type_v") or _arg(args, "-ctv", "--cache-type-v"),
        "image_tokens": [_arg(args, "--image-min-tokens"), _arg(args, "--image-max-tokens")],
        "build": _build_identity(str(service.get("build_path") or "")),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:12]


def storage_rejection(path: str, forbidden_roots: Any = ()) -> str | None:
    """Reason the directory must not hold KV files (forbidden root, backup, network or syncthing), or None.

    forbidden_roots comes from the member's host config (kv_affinity.forbidden_roots, e.g. the NAS mount);
    host paths are not compiled into this published module.
    """
    real = os.path.realpath(path)
    roots = [os.path.normpath(str(r)) for r in forbidden_roots if r]
    # Checked on the resolved and the literal path, so a symlink into a forbidden root or a backup tree is caught too.
    for candidate in {real, os.path.abspath(path)}:
        if any(candidate == root or candidate.startswith(root.rstrip("/") + "/") for root in roots):
            return "forbidden root"
        if any(part.lower().startswith("backup") for part in Path(candidate).parts):
            return "backup path"
    if any(part.lower().startswith("syncthing") or part == "Sync" for part in Path(real).parts):
        return "syncthing path"
    probe = Path(real)
    while True:
        if (probe / ".stfolder").exists():
            return "syncthing folder"
        if probe.parent == probe:
            break
        probe = probe.parent
    best, fstype = "", ""
    try:
        with open("/proc/mounts", encoding="utf-8") as mounts:
            for line in mounts:
                fields = line.split()
                target = fields[1].replace("\\040", " ")
                # Later entries win (autofs is listed before the real ext4 mount).
                if (real == target or real.startswith(target.rstrip("/") + "/")) and len(target) >= len(best):
                    best, fstype = target, fields[2]
    except OSError:
        return "cannot read /proc/mounts"
    if fstype in _NETWORK_FS or fstype.startswith("fuse"):
        return f"network filesystem {fstype}"
    return None


class KvAffinity:
    """Per-member slot affinity, save and restore. All methods are fail-open."""

    def __init__(self, *, base_url: str, slot_save_dir: str, fp: str, session: Any,
                 min_tokens: int = DEFAULT_MIN_TOKENS, io_timeout: float = 120.0,
                 max_bytes: int = DEFAULT_MAX_BYTES, max_age_s: float = DEFAULT_MAX_AGE_S,
                 max_files: int = DEFAULT_MAX_FILES, engine: str = "llama") -> None:
        self.engine = engine
        self.base_url = base_url.rstrip("/")
        self.slot_save_dir = slot_save_dir
        self.fp = fp
        self.session = session
        self.min_tokens = int(min_tokens)
        self.io_timeout = float(io_timeout)
        self.max_bytes, self.max_age_s, self.max_files = int(max_bytes), float(max_age_s), int(max_files)
        self.session_slot: dict[str, int] = {}
        self.occupant: dict[int, str] = {}
        self.in_flight: set[int] = set()
        # shortcut: one lock serialises slot choice + save/restore per member, so a long save delays
        # the other slot's dispatch; upgrade to per-slot locks if saves show up in queue latency.
        self.lock = asyncio.Lock()

    def filename(self, session: str) -> str:
        return f"{self.fp}-{session}.bin"

    def _within_budget(self, keep: str) -> bool:
        """Prune age/LRU/count around ``keep``; False when the budget still cannot be met."""
        try:
            prune(self.slot_save_dir, max_bytes=self.max_bytes, max_age_s=self.max_age_s,
                  max_files=self.max_files, dry_run=False, keep=keep)
            files = _bin_files(self.slot_save_dir)
        except OSError as exc:
            logger.warning("kv budget check failed in %s: %s", self.slot_save_dir, exc)
            return False
        return sum(f[1] for f in files) <= self.max_bytes and len(files) <= self.max_files

    async def _get_slots(self) -> list[dict[str, Any]]:
        from gemma_broker.runtime import aiohttp_total_timeout

        async with self.session.get(f"{self.base_url}/slots", timeout=aiohttp_total_timeout(10)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"GET /slots HTTP {resp.status}")
            return [s for s in await resp.json() if isinstance(s, dict)]

    async def _slot_action(self, slot: int, action: str, session: str) -> bool:
        from gemma_broker.runtime import aiohttp_total_timeout

        name = self.filename(session)
        path = os.path.join(self.slot_save_dir, name)
        if self.engine not in SLOT_ENGINES or not self.slot_save_dir:  # storage rejected or unset: never ask llama to write
            return False
        if action == "save" and not self._within_budget(keep=path):
            logger.warning("kv slot save skipped slot=%s session=%s: storage budget", slot, session)
            return False
        try:
            async with self.session.post(
                f"{self.base_url}/slots/{slot}?action={action}",
                json={"filename": name},
                timeout=aiohttp_total_timeout(self.io_timeout),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                await resp.read()
        except Exception as exc:
            logger.warning("kv slot %s failed slot=%s session=%s: %s", action, slot, session, exc)
            return False
        try:
            if action == "save":
                os.chmod(path, 0o600)  # llama-server's umask is out of scope; tighten after the fact
                if not self._within_budget(keep=path):  # the new file alone is over budget
                    os.unlink(path)
                    logger.warning("kv slot save dropped slot=%s session=%s: over budget", slot, session)
                    return False
            else:
                os.utime(path)  # restore counts as use for the LRU pruner
        except OSError as exc:
            logger.warning("kv slot %s post-step failed for %s: %s", action, name, exc)
        logger.info("kv slot %s ok slot=%s session=%s", action, slot, session)
        return True

    async def prepare(self, payload: dict[str, Any], session: str) -> int | None:
        """Pick a slot for this session, save/restore as needed, set id_slot. Returns the held slot."""
        if self.engine not in SLOT_ENGINES:
            return None  # no slot concept to pin or save: plain dispatch
        async with self.lock:
            try:
                slots = await self._get_slots()
            except Exception as exc:
                logger.warning("kv affinity: /slots unavailable, dispatching unpinned: %s", exc)
                return None
            free = {int(s["id"]): int(s.get("n_prompt_tokens") or 0)
                    for s in slots if "id" in s and not s.get("is_processing")
                    and int(s["id"]) not in self.in_flight}
            if not free:
                # shortcut: no free slot -> llama picks one and the occupant map may go stale; a stale
                # entry only costs speed (see module docstring). Upgrade: pin to the preferred slot.
                return None
            slot = self.session_slot.get(session)
            if slot not in free:
                slot = min(free, key=lambda sid: (sid in self.occupant, free[sid]))
            occupant, tokens = self.occupant.get(slot), free[slot]
            if occupant and occupant != session and tokens > self.min_tokens:
                await self._slot_action(slot, "save", occupant)
            resident = occupant == session and tokens > 0
            if not resident and self.slot_save_dir and os.path.isfile(
                os.path.join(self.slot_save_dir, self.filename(session))
            ):
                await self._slot_action(slot, "restore", session)
            if self.session_slot.get(occupant) == slot and occupant != session:
                self.session_slot.pop(occupant, None)
            old = self.session_slot.get(session)
            if old is not None and old != slot and self.occupant.get(old) == session:
                del self.occupant[old]  # the older copy must never be saved over the session's file
            self.session_slot[session] = slot
            self.occupant[slot] = session
            self.in_flight.add(slot)
            payload["id_slot"] = slot
            return slot

    def release(self, slot: int | None) -> None:
        if slot is not None:
            self.in_flight.discard(slot)

    async def save_all(self) -> int:
        """Save every idle slot over min_tokens before a planned stop; returns files saved."""
        saved = 0
        if self.engine not in SLOT_ENGINES:
            return 0
        async with self.lock:
            try:
                slots = await self._get_slots()
            except Exception as exc:
                logger.warning("kv affinity: save-before-evict skipped, /slots unavailable: %s", exc)
                slots = []
            for s in slots:
                sid = int(s.get("id", -1))
                occupant = self.occupant.get(sid)
                if occupant and not s.get("is_processing") and int(s.get("n_prompt_tokens") or 0) > self.min_tokens:
                    saved += await self._slot_action(sid, "save", occupant)
            # The process is about to stop: slots come back empty and are restored lazily.
            self.session_slot.clear()
            self.occupant.clear()
        return saved

    async def close_session(self, session: str) -> int:
        """Forget a closed session and delete its saved files under any fingerprint; returns files removed."""
        # Caller trigger (kanban/T3 session close) is NOT WIRED: nothing calls this yet.
        # UNVERIFIED — requires the GPU qualification card.
        removed = 0
        async with self.lock:
            if self.slot_save_dir:
                suffix = f"-{session}.bin"  # fingerprint and session are dash-free, so the match is exact
                try:
                    entries = list(os.scandir(self.slot_save_dir))
                except OSError as exc:
                    logger.warning("kv close_session cannot list %s: %s", self.slot_save_dir, exc)
                    entries = []
                for entry in entries:
                    if entry.name.endswith(suffix) and entry.is_file(follow_symlinks=False):
                        try:
                            os.unlink(entry.path)
                            removed += 1
                        except OSError as exc:
                            logger.warning("kv close_session could not delete %s: %s", entry.path, exc)
            for slot in [sid for sid, owner in self.occupant.items() if owner == session]:
                del self.occupant[slot]
            self.session_slot.pop(session, None)
        return removed


class KvAffinityTransport:
    """Wraps the broker transport: strips the session field and applies affinity for enabled members."""

    def __init__(self, inner: Any, kv_affinity: dict[str, KvAffinity]) -> None:
        self.inner = inner
        self.kv_affinity = kv_affinity

    async def send(self, member: Any, body: dict[str, Any], on_accepted: Any) -> dict[str, Any]:
        affinity = self.kv_affinity.get(getattr(member, "name", None))
        if affinity is None:
            return await self.inner.send(member, body, on_accepted)
        payload = dict(body)
        scope = payload.pop(SESSION_FIELD, None)
        if not isinstance(scope, dict) or not scope.get("principal"):
            return await self.inner.send(member, payload, on_accepted)  # no authenticated principal
        slot = None
        try:
            slot = await affinity.prepare(
                payload, session_key(payload, scope.get("session"), str(scope["principal"])))
        except Exception as exc:  # never block dispatch on affinity
            logger.warning("kv affinity prepare failed for %s: %s", getattr(member, "name", "?"), exc)
        try:
            return await self.inner.send(member, payload, on_accepted)
        finally:
            affinity.release(slot)


def build_kv_affinity(services: dict[str, Any], member_names: Any, session: Any) -> dict[str, KvAffinity]:
    """KvAffinity for each member whose kv_affinity.enabled is true; empty when the flag is off."""
    result: dict[str, KvAffinity] = {}
    for name in member_names:
        service = services.get(name) or {}
        cfg = service.get("kv_affinity") or {}
        if cfg.get("enabled") is not True:
            continue
        if name.lower().startswith("d1"):  # the d1 decision model never takes the KV path
            logger.warning("kv affinity for %s ignored: d1 members are excluded", name)
            continue
        save_dir = str(cfg.get("slot_save_dir") or "")
        forbidden = cfg.get("forbidden_roots") or []
        forbidden = [forbidden] if isinstance(forbidden, str) else list(forbidden)
        reason = storage_rejection(save_dir, forbidden) if save_dir else "slot_save_dir not set"
        if reason:
            logger.error("kv affinity for %s: save/restore disabled (%s); slot affinity only", name, reason)
            save_dir = ""
        if save_dir:
            try:
                os.makedirs(save_dir, mode=0o700, exist_ok=True)
                # Llama writes each file before the broker's chmod 0600; a 0700 directory keeps other users out of that window.
                # Real fix is UMask=0077 on the Qwen-Dual unit, which lives outside this repo: UNVERIFIED — requires the GPU qualification card.
                os.chmod(save_dir, 0o700)
            except OSError as exc:
                logger.error("kv affinity for %s: save dir %s unusable (%s); slot affinity only", name, save_dir, exc)
                save_dir = ""
        result[name] = KvAffinity(
            base_url=f"http://127.0.0.1:{int(service['port'])}",
            slot_save_dir=save_dir,
            fp=fingerprint(service),
            engine=str(cfg.get("engine") or "llama"),
            session=session,
            min_tokens=int(cfg.get("min_tokens", DEFAULT_MIN_TOKENS)),
            io_timeout=float(cfg.get("io_timeout", 120.0)),
            max_bytes=int(float(cfg.get("max_gb", 100)) * 1024**3),
            max_age_s=float(cfg.get("max_age_days", 7)) * 86400,
            max_files=int(cfg.get("max_files", DEFAULT_MAX_FILES)),
        )
    return result


def _bin_files(directory: str) -> list[tuple[float, int, str]]:
    """(mtime, size, path) of regular *.bin files, oldest first; symlinks are never touched."""
    files = []
    for entry in os.scandir(directory):
        if entry.name.endswith(".bin") and entry.is_file(follow_symlinks=False):
            st = entry.stat(follow_symlinks=False)
            files.append((st.st_mtime, st.st_size, entry.path))
    return sorted(files)


def prune(directory: str, *, max_bytes: int = DEFAULT_MAX_BYTES, max_age_s: float = DEFAULT_MAX_AGE_S,
          max_files: int | None = None, dry_run: bool = True, now: float | None = None,
          keep: str | None = None) -> list[str]:
    """Delete *.bin older than max_age_s, then oldest-first while over max_bytes or max_files."""
    now = time.time() if now is None else now
    files = _bin_files(directory)
    total, count = sum(size for _, size, _ in files), len(files)
    doomed = []
    for mtime, size, path in files:
        if path == keep:
            continue
        if now - mtime > max_age_s or total > max_bytes or (max_files is not None and count > max_files):
            doomed.append(path)
            total -= size
            count -= 1
    for path in doomed:
        if not dry_run:
            try:
                os.unlink(path)
            except OSError as exc:
                logger.warning("kv prune could not delete %s: %s", path, exc)
    return doomed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KV slot file retention (dry-run unless --apply)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--dir", required=True)
    p.add_argument("--max-gb", type=float, default=100.0)
    p.add_argument("--max-age-days", type=float, default=7.0)
    p.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--forbid", action="append", default=[], help="root to refuse (repeatable, e.g. the NAS mount)")
    args = parser.parse_args(argv)
    reason = storage_rejection(args.dir, args.forbid)
    if reason:
        parser.error(f"refusing {args.dir}: {reason}")
    doomed = prune(args.dir, max_bytes=int(args.max_gb * 1024**3), max_age_s=args.max_age_days * 86400,
                   max_files=args.max_files, dry_run=not args.apply)
    for path in doomed:
        print(("deleted " if args.apply else "would delete ") + path)
    print(f"{len(doomed)} file(s) {'deleted' if args.apply else 'selected (dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
