#!/usr/bin/env python3
"""Select and verify a GPU Manager snapshot/restore-receipt source bundle.

Source-selection stage (offline, no SSH) selects the bundle; the copy stage
transfers it to a pre-validated remote destination with atomic publish.

CLI:
  --source-directory      Local backup source directory (contains snapshot+receipt)
  --ssh-alias            SSH alias for remote host (from ~/.ssh/config)
  --remote-directory     Absolute remote destination directory
  --timeout-seconds      Max seconds per SSH operation (1-120, default 120)

Exit codes:
  0  success (bundle published)
  1  copy/transfer/delivery failure (local backup untouched)
  2  usage error (invalid args)
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import stat
import subprocess
import sys
from typing import Any, Callable


# ── Schema identifiers ────────────────────────────────────────────────────────

SNAPSHOT_SCHEMA = "gpu-manager-private-snapshot.v1"
RECEIPT_ISOLATION = "private-valkey-unix-socket-no-workers"
DESTINATION_SCHEMA = "gpu-manager-backup-destination.v1"
BUNDLE_MANIFEST_SCHEMA = "gpu-manager-backup-bundle.v1"

SNAPSHOT_NAME = re.compile(r"^(\d{8}T\d{6}Z)-(\d+)\.json\.gz$")

# ── Errors ─────────────────────────────────────────────────────────────────────

class BackupError(Exception):
    """Base for non-usage errors."""
    pass


class NoVerifiedSnapshotError(BackupError):
    """No regular snapshot/receipt pair passes the source contract."""
    pass


class MountMissingError(BackupError):
    """Remote destination mount is absent or mismatched."""
    pass


class TransferError(BackupError):
    """SSH or file-operation failure during copy or readback."""
    pass


class HashMismatchError(BackupError):
    """Readback hash does not match local source hash."""
    pass


class PublishError(BackupError):
    """Atomic publish (renameat2) failed."""
    pass


# ── Destination manifest ───────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Destination:
    directory: Path
    mountpoint: str
    fstype: str
    source: str
    ssh_alias: str
    timeout: float


def read_destination_manifest(path: Path) -> Destination:
    """Load and validate a gpu-manager-backup-destination.v1 manifest."""
    try:
        raw = path.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"cannot read destination manifest {path}: {exc}") from exc

    if doc.get("schema") != DESTINATION_SCHEMA:
        raise ValueError(f"destination manifest schema mismatch: {doc.get('schema')!r}")

    directory = Path(doc["directory"])
    if not directory.is_absolute():
        raise ValueError(f"destination directory must be absolute: {directory}")
    mountpoint = doc.get("mountpoint", "/")
    fstype = doc.get("fstype", "unknown")
    source = doc.get("source", "unknown")
    ssh_alias = doc.get("ssh_alias", "")
    timeout = float(doc.get("timeout_seconds", 120))
    if not (0 < timeout <= 120):
        raise ValueError(f"timeout must be 0 < timeout <= 120, got {timeout}")
    return Destination(
        directory=directory,
        mountpoint=mountpoint,
        fstype=fstype,
        source=source,
        ssh_alias=ssh_alias,
        timeout=timeout,
    )


# ── Bundle manifest ────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class BundleManifest:
    schema: str
    timestamp: str
    snapshot_name: str
    receipt_name: str
    snapshot_sha256: str
    receipt_sha256: str
    published: bool = False


# ── Source selection ───────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class VerifiedBundle:
    snapshot: Path
    receipt: Path
    snapshot_bytes: bytes
    receipt_bytes: bytes
    snapshot_sha256: str
    receipt_sha256: str
    timestamp: str


def _read_regular(path: Path) -> tuple[bytes, tuple[int, int, int, int]]:
    """Read an unchanged regular file without following a symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path.name}")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"source changed while opening: {path.name}")
        payload = stream.read()
    after = path.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    final = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    opened_identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    if identity != final or opened_identity != identity or len(payload) != before.st_size:
        raise ValueError(f"source changed while reading: {path.name}")
    return payload, identity


def _load_json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def _valid_pair(snapshot: Path, receipt: Path, timestamp: str) -> VerifiedBundle | None:
    try:
        snapshot_bytes, _ = _read_regular(snapshot)
        receipt_bytes, _ = _read_regular(receipt)
        document = _load_json(gzip.decompress(snapshot_bytes))
        metadata = _load_json(receipt_bytes)
        if not isinstance(document, dict) or document.get("schema") != SNAPSHOT_SCHEMA:
            return None
        redis_records = document.get("redis")
        files = document.get("files")
        if not isinstance(redis_records, list) or not isinstance(files, dict):
            return None
        if not isinstance(metadata, dict):
            return None
        named_snapshot = metadata.get("snapshot")
        if not isinstance(named_snapshot, str):
            return None
        if Path(named_snapshot).resolve(strict=True) != snapshot.resolve(strict=True):
            return None
        for field, expected in (("restored_records", len(redis_records)),
                                ("verified_files", len(files))):
            count = metadata.get(field)
            if type(count) is not int or count < 0 or count != expected:
                return None
        replay_count = metadata.get("replay_count")
        if type(replay_count) is not int or replay_count != 0:
            return None
        if metadata.get("isolation") != RECEIPT_ISOLATION:
            return None
        return VerifiedBundle(
            snapshot=snapshot,
            receipt=receipt,
            snapshot_bytes=snapshot_bytes,
            receipt_bytes=receipt_bytes,
            snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
            receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            timestamp=timestamp,
        )
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError,
            gzip.BadGzipFile, EOFError):
        return None


def select_newest_verified(source_directory: Path) -> VerifiedBundle:
    """Return newest eligible snapshot/receipt pair, or fail explicitly."""
    try:
        directory_info = source_directory.lstat()
    except OSError as exc:
        raise NoVerifiedSnapshotError("no_verified_snapshot") from exc
    if not stat.S_ISDIR(directory_info.st_mode):
        raise NoVerifiedSnapshotError("no_verified_snapshot")

    candidates: list[tuple[str, str, Path, Path]] = []
    try:
        entries = tuple(source_directory.iterdir())
    except OSError as exc:
        raise NoVerifiedSnapshotError("no_verified_snapshot") from exc
    for snapshot in entries:
        match = SNAPSHOT_NAME.fullmatch(snapshot.name)
        if not match:
            continue
        timestamp, _pid = match.groups()
        try:
            from datetime import datetime, timezone
            parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        receipt = snapshot.with_suffix(".restore.json")
        candidates.append((parsed.strftime("%Y%m%dT%H%M%SZ"), snapshot.name, snapshot, receipt))

    for timestamp, _name, snapshot, receipt in sorted(candidates, reverse=True):
        bundle = _valid_pair(snapshot, receipt, timestamp)
        if bundle is not None:
            return bundle
    raise NoVerifiedSnapshotError("no_verified_snapshot")


# ── SSH transport ──────────────────────────────────────────────────────────────

def ssh_base_args(
    ssh_config: str | None = None,
    ssh_identity_file: str | None = None,
    ssh_known_hosts: str | None = None,
) -> list[str]:
    """Build hardened SSH arguments using runtime account paths or overrides."""
    if ssh_config is None or ssh_identity_file is None or ssh_known_hosts is None:
        uid = 1000 if os.geteuid() == 0 else os.geteuid()
        try:
            account_home = Path(pwd.getpwuid(uid).pw_dir)
        except KeyError as exc:
            raise ValueError(f"cannot resolve SSH compatibility account UID {uid}") from exc
        if not account_home.is_absolute():
            raise ValueError(f"SSH compatibility account UID {uid} has no absolute home")
        ssh_dir = account_home / ".ssh"
        ssh_config = ssh_config or str(ssh_dir / "config")
        ssh_identity_file = ssh_identity_file or str(ssh_dir / "id_ed25519")
        ssh_known_hosts = ssh_known_hosts or str(ssh_dir / "known_hosts")
    return [
        "-F", ssh_config,
        "-i", ssh_identity_file,
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none",
        "-o", "ForwardAgent=no",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={ssh_known_hosts}",
    ]


# Kept as an exported compatibility constant for existing importers.
SSH_BASE_ARGS = ssh_base_args()

# Type for the SSH runner: (ssh_alias, timeout, *cmd, stdin_bytes) -> (code, stdout, stderr)
SSHRunner = Callable[[str, float, str, tuple[str, ...], bytes], tuple[int, bytes, bytes]]


def _default_ssh(ssh_alias: str, timeout: float, script: str,
                 _cmd: tuple[str, ...] = (), stdin_bytes: bytes = b"",
                 *, ssh_config: str | None = None,
                 ssh_identity_file: str | None = None,
                 ssh_known_hosts: str | None = None) -> tuple[int, bytes, bytes]:
    """Default SSH runner: runs a remote Python script via ssh ... python3 -c ..."""
    full_cmd = ["ssh"] + ssh_base_args(ssh_config, ssh_identity_file, ssh_known_hosts) + [
        ssh_alias, "--", f"python3 -c {shlex.quote(script)}"
    ]
    proc = subprocess.run(
        full_cmd,
        input=stdin_bytes if stdin_bytes else None,
        capture_output=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ── Lock acquisition (flock over SSH) ─────────────────────────────────────────

REMOTE_LOCK_SUFFIX = ".gpu-manager-backup.lock"


def _acquire_remote_lock(runner: SSHRunner, ssh_alias: str, remote_dir: str,
                         timeout: float) -> tuple[int, int]:
    """Acquire exclusive flock on remote lockfile. Returns (fd, exit_code)."""
    script = f"""
import fcntl, os, sys, time, errno
parent = {remote_dir!r}
os.makedirs(parent, exist_ok=True)
lock_path = parent + '/.gpu-manager-backup.lock'
timeout = {timeout!r}
start = time.monotonic()
lf = open(lock_path, 'w')
while True:
    try:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lf.write(str(os.getpid()))
        lf.flush()
        break
    except OSError as e:
        if e.errno != errno.EWOULDBLOCK:
            sys.exit(e.errno)
        if time.monotonic() - start >= timeout:
            sys.exit(124)
        time.sleep(0.1)
print(lf.fileno(), flush=True)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), b"")
    if code != 0:
        return -1, code
    try:
        fd = int(stdout.decode().strip())
    except (ValueError, IndexError):
        return -1, 1
    return fd, 0


def _release_remote_lock(runner: SSHRunner, ssh_alias: str, timeout: float, fd: int) -> None:
    """Release the remote flock given a file descriptor number."""
    script = f"""
import fcntl, os
fcntl.flock({fd}, fcntl.LOCK_UN)
os.close({fd})
"""
    runner(ssh_alias, timeout, script, (), b"")


# ── Mount identity validation ──────────────────────────────────────────────────

def _validate_mount(runner: SSHRunner, ssh_alias: str, remote_dir: str,
                     timeout: float, expected_mount: str, expected_fstype: str) -> None:
    """Verify the remote directory is on the expected mount; reject symlinks."""
    script = f"""
import os, sys

path = {remote_dir!r}
expected_mount = {expected_mount!r}
parts = path.split('/')
# Walk path components checking for symlinks
for i in range(1, len(parts) + 1):
    component = '/'.join([''] + parts[:i])
    if os.path.islink(component):
        sys.exit(1)
# Verify mount info
found = False
try:
    with open('/proc/self/mountinfo') as f:
        for line in f:
            fields = line.split()
            if len(fields) < 5:
                continue
            mp = fields[4]
            if mp == expected_mount or mp == path:
                print(line, end='', flush=True)
                found = True
                break
except OSError:
    pass
if not found:
    sys.exit(2)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), b"")
    if code == 1:
        raise MountMissingError(f"symlink component in remote path: {remote_dir}")
    if code == 2 or not stdout:
        raise MountMissingError(
            f"mountpoint {expected_mount!r} not found in /proc/self/mountinfo for {remote_dir}"
        )
    # Filesystem type is compared HOST-SIDE against the mountinfo line the
    # remote script already printed: /proc/self/mountinfo puts the fstype in
    # the field AFTER the ' - ' separator, not at a fixed index. The remote
    # script stays unchanged (its exit codes are 1=symlink, 2=not-found).
    # An unset fstype (manifest has no 'fstype' key -> 'unknown', or empty)
    # keeps the previous behaviour: no comparison.
    if expected_fstype and expected_fstype != "unknown":
        line = stdout.decode(errors="replace").strip()
        _pre, separator, _post = line.partition(" - ")
        _post_fields = _post.split()
        observed = _post_fields[0] if separator and _post_fields else ""
        if observed != expected_fstype:
            raise MountMissingError(
                f"filesystem type mismatch for {remote_dir}: expected fstype "
                f"{expected_fstype!r}, observed {observed!r} in mountinfo line {line!r}"
            )


# ── Staging directory creation ────────────────────────────────────────────────

def _create_staging(runner: SSHRunner, ssh_alias: str, remote_dir: str,
                    timeout: float, bundle_id: str) -> str:
    """Create a unique private staging child directory. Returns staging path."""
    staging = f"{remote_dir}/.staging-{bundle_id}"
    script = f"""
import os
staging = {staging!r}
os.makedirs(staging, mode=0o700, exist_ok=True)
print(staging, flush=True)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), b"")
    if code != 0:
        raise TransferError(f"failed to create staging dir: {stderr.decode()}")
    return stdout.decode().strip()


# ── File write (Python write over SSH) ────────────────────────────────────────

def _remote_write(runner: SSHRunner, ssh_alias: str, timeout: float,
                  local_bytes: bytes, remote_path: str) -> None:
    """Write bytes to a remote path via Python over SSH, streaming stdin."""
    # Stream bytes via stdin + cat to avoid ARG_MAX limits.
    # The remote script reads stdin and writes the file (not via argv).
    script = f"""
import os, sys
path = {remote_path!r}
dir = os.path.dirname(path) or '/'
os.makedirs(dir, exist_ok=True)
with open(path, 'wb') as f:
    f.write(sys.stdin.buffer.read())
os.chmod(path, 0o644)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), local_bytes)
    if code != 0:
        raise TransferError(f"remote write failed for {remote_path}: {stderr.decode()}")


# ── Readback hash ─────────────────────────────────────────────────────────────

def _readback_hash(runner: SSHRunner, ssh_alias: str, timeout: float,
                   remote_path: str) -> str:
    """Read a remote file and return its SHA-256 hex digest."""
    script = f"""
import hashlib, sys
path = {remote_path!r}
with open(path, 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest()
print(h, flush=True)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), b"")
    if code != 0:
        raise TransferError(f"readback failed for {remote_path}: {stderr.decode()}")
    return stdout.decode().strip()


# ── Atomic publish (renameat2 RENAME_NOREPLACE) ────────────────────────────────

def _publish_bundle(runner: SSHRunner, ssh_alias: str, timeout: float,
                    staging_dir: str, final_dir: str) -> None:
    """Atomically publish staging to final via renameat2 RENAME_NOREPLACE."""
    script = f"""
import os, sys, ctypes, errno as _errno

staging = {staging_dir!r}
final = {final_dir!r}

if os.path.exists(final):
    sys.exit(1)  # collision

libc = ctypes.CDLL('libc.so.6', use_errno=True)
ret = libc.renameat2(0, staging.encode(), 0, final.encode(), 1)
if ret != 0:
    e = ctypes.get_errno()
    if e == _errno.EXDEV:
        sys.exit(30)
    if e in (_errno.ENOTEMPTY, _errno.EEXIST):
        sys.exit(31)
    sys.exit(e)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), b"")
    if code == 1:
        raise PublishError(f"final bundle directory already exists: {final_dir}")
    if code == 30:
        raise PublishError("RENAME_NOREPLACE not supported (EXDEV)")
    if code == 31:
        raise PublishError("collision during atomic publish")
    if code != 0:
        raise PublishError(f"renameat2 failed with code {code}: {stderr.decode()}")


# ── Retention pruning ─────────────────────────────────────────────────────────

MAX_BUNDLES = 30

# Compiled once at module load — used in the remote prune script.
_REMOTE_BUNDLE_RE = __import__("re").compile(r"^(\d{8}T\d{6}Z)-(\d+)$")


def _prune_remote_bundles(
    runner: SSHRunner,
    ssh_alias: str,
    remote_dir: str,
    timeout: float,
    published_snapshot_sha256: str | None = None,
    published_receipt_sha256: str | None = None,
) -> int:
    """Remove oldest non-compliant/unrecognised surplus bundles beyond MAX_BUNDLES.

    Only direct children that are:
      1. directories (not symlinks) matching ``\\d{8}T\\d{6}Z-\\d+``;
      2. contain a parseable ``.bundle-manifest.json``;
      3. contain both the recorded ``snapshot_name`` and ``receipt_name`` files;
      4. have matching SHA-256 hashes for both files;
    are considered retention candidates.

    The current (newly published) bundle is identified by its hashes so it is
    never considered for pruning even if it would rank beyond the retention
    window.

    Sort order: parsed UTC timestamp descending, then stable bundle name
    descending (so identical timestamps are deterministically ordered by the
    numeric PID suffix).

    Returns the number of directories actually removed, or 0 on any error.
    The function never removes the current publication or any local file.
    """
    published_snapshot_hex = published_snapshot_sha256 or ""
    published_receipt_hex = published_receipt_sha256 or ""

    script = rf"""
import json, os, re, hashlib, shutil

remote_dir = {remote_dir!r}
max_bundles = {MAX_BUNDLES}
current_snapshot_hex = {published_snapshot_hex!r}
current_receipt_hex  = {published_receipt_hex!r}
BUNDLE_RE = re.compile(r"^(\d{{8}}T\d{{6}}Z)-(\d+)$")
STAGING_RE = re.compile(r"^\.staging-(\d{{8}}T\d{{6}}Z)-(\d+)$")

# ── 1. Enumerate strict-name non-symlink direct-child bundle directories ─────
candidates = []
for entry in os.listdir(remote_dir):
    full = os.path.join(remote_dir, entry)
    if not os.path.isdir(full):
        continue
    if os.path.islink(full):          # never traverse a symlink
        continue
    m = BUNDLE_RE.match(entry)
    if not m:
        continue
    candidates.append(entry)

# ── 2. Validate each candidate bundle ─────────────────────────────────────────
validated = []
for entry in candidates:
    bundle_dir = os.path.join(remote_dir, entry)
    manifest_path = os.path.join(bundle_dir, ".bundle-manifest.json")

    # 2a. parse manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            mfst = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue                       # malformed / absent manifest → skip

    if mfst.get("schema") != {BUNDLE_MANIFEST_SCHEMA!r}:
        continue

    snap_name = mfst.get("snapshot_name", "")
    recv_name = mfst.get("receipt_name",  "")
    snap_hex  = mfst.get("snapshot_sha256", "")
    recv_hex  = mfst.get("receipt_sha256",  "")

    # 2b. require exact expected files
    snap_path = os.path.join(bundle_dir, snap_name)
    recv_path = os.path.join(bundle_dir, recv_name)
    if not (os.path.isfile(snap_path) and os.path.isfile(recv_path)):
        continue

    # 2c. verify hashes
    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        if sha256(snap_path) != snap_hex:
            continue
        if sha256(recv_path) != recv_hex:
            continue
    except OSError:
        continue

    # ── 3. Identify the newly-published bundle (exclude from pruning) ──────────
    if snap_hex == current_snapshot_hex and recv_hex == current_receipt_hex:
        continue                       # current publication — skip

    # ── 4. Parse sortable timestamp from bundle name ───────────────────────────
    m = BUNDLE_RE.match(entry)
    ts_str, pid_str = m.group(1), m.group(2)
    # timestamp descending, then bundle name (timestamp-pid) descending
    sort_key = (ts_str, entry)
    validated.append((sort_key, entry))

# ── 5. Sort by timestamp desc, bundle name desc (stable tie-break) ─────────────
validated.sort(key=lambda x: x[0], reverse=True)

# ── 6. Delete surplus validated candidates beyond MAX_BUNDLES ─────────────────
pruned = 0
for _sort_key, entry in validated[max_bundles:]:
    bundle_path = os.path.join(remote_dir, entry)
    for root, dirs, files in os.walk(bundle_path, topdown=False):
        for fname in files:
            os.unlink(os.path.join(root, fname))
        for dname in dirs:
            os.rmdir(os.path.join(root, dname))
    os.rmdir(bundle_path)
    pruned += 1

# ── 7. Reclaim stale .staging-<timestamp>-<pid> orphans ───────────────────────
# A copy that fails mid-transfer leaves its staging directory behind. Those are
# named .staging-<ts>-<pid>, which BUNDLE_RE never matches, so without this sweep
# they accumulate forever. A staging dir is reclaimed only when its timestamp is
# STRICTLY OLDER than the oldest bundle still retained: anything at or after that
# edge is a copy still in progress, or a fresh orphan the next prune will take.
retained = validated[:max_bundles]
if retained:
    oldest_retained_ts = min(sort_key[0] for sort_key, _entry in retained)
else:
    oldest_retained_ts = None

if oldest_retained_ts is not None:
    for entry in os.listdir(remote_dir):
        if BUNDLE_RE.match(entry):            # never touch a published bundle name
            continue
        full = os.path.join(remote_dir, entry)
        if os.path.islink(full):              # never traverse a symlink
            continue
        m = STAGING_RE.match(entry)
        if not m:
            continue
        if not os.path.isdir(full):           # a plain file named like a staging dir
            continue
        if m.group(1) >= oldest_retained_ts:
            continue                         # in progress, not stale
        try:
            shutil.rmtree(full)
        except OSError:
            continue                         # raced with the copy; next prune retries
        pruned += 1

print(pruned, flush=True)
"""
    code, stdout, stderr = runner(ssh_alias, timeout, script, (), b"")
    if code != 0:
        return 0
    try:
        return int(stdout.decode().strip())
    except (ValueError, IndexError):
        return 0


# ── Staging cleanup ───────────────────────────────────────────────────────────

def _cleanup_staging(runner: SSHRunner, ssh_alias: str, timeout: float,
                     staging_dir: str, remote_dir: str) -> None:
    """Remove staging directory; confined to owned staging children only."""
    script = f"""
import shutil, os
staging = {staging_dir!r}
parent = {remote_dir!r}
if staging.startswith(parent + '/') and os.path.exists(staging):
    shutil.rmtree(staging, ignore_errors=True)
"""
    runner(ssh_alias, timeout, script, (), b"")


# ── Main copy operation ────────────────────────────────────────────────────────

def copy_bundle(
    source_directory: Path,
    ssh_alias: str,
    remote_directory: Path | None,
    timeout: float,
    destination_manifest: Path | None = None,
    _runner: SSHRunner | None = None,
    ssh_config: str | None = None,
    ssh_identity_file: str | None = None,
    ssh_known_hosts: str | None = None,
) -> dict[str, Any]:
    """Select, transfer, verify, publish, and prune a GPU Manager backup bundle.

    _runner is exposed for fake-transport testing: replaces all SSH subprocess
    calls. Signature: (ssh_alias, timeout, script, cmd_tuple, stdin_bytes)
    -> (returncode, stdout_bytes, stderr_bytes).
    """
    if _runner is not None:
        runner = _runner
    else:
        def runner(alias, timeout_value, script, cmd=(), stdin_bytes=b""):
            if ssh_config is None and ssh_identity_file is None and ssh_known_hosts is None:
                return _default_ssh(alias, timeout_value, script, cmd, stdin_bytes)
            return _default_ssh(
                alias, timeout_value, script, cmd, stdin_bytes,
                ssh_config=ssh_config,
                ssh_identity_file=ssh_identity_file,
                ssh_known_hosts=ssh_known_hosts,
            )

    # ── Resolve destination contract ──────────────────────────────────────────
    if destination_manifest is None:
        dest_manifest_path = source_directory / ".gpu-manager-backup-destination.json"
    else:
        dest_manifest_path = destination_manifest

    if dest_manifest_path.exists():
        dest = read_destination_manifest(dest_manifest_path)
    else:
        raise MountMissingError(
            f"destination manifest {dest_manifest_path} not found; "
            "explicit provisioning required"
        )

    # CLI overrides
    if ssh_alias:
        dest = dataclasses.replace(dest, ssh_alias=ssh_alias)
    if remote_directory:
        dest = dataclasses.replace(dest, directory=remote_directory)
    if timeout:
        dest = dataclasses.replace(dest, timeout=timeout)

    remote_dir_str = str(dest.directory)

    # ── Select source bundle ──────────────────────────────────────────────────
    bundle = select_newest_verified(source_directory)
    bundle_id = f"{bundle.timestamp}-{os.getpid()}"
    staging_dir = f"{remote_dir_str}/.staging-{bundle_id}"
    final_dir = f"{remote_dir_str}/{bundle_id}"

    # ── Validate mount identity BEFORE anything creates the remote dir ────────
    _validate_mount(runner, dest.ssh_alias, remote_dir_str, dest.timeout,
                    dest.mountpoint, dest.fstype)

    # ── Acquire remote lock ────────────────────────────────────────────────────
    lock_fd, lock_code = _acquire_remote_lock(runner, dest.ssh_alias, remote_dir_str, dest.timeout)
    if lock_code == 124:
        raise TransferError("failed to acquire remote lock: timeout")
    if lock_code != 0:
        raise TransferError(f"failed to acquire remote lock: code {lock_code}")

    def _release_lock():
        _release_remote_lock(runner, dest.ssh_alias, dest.timeout, lock_fd)

    try:
        # ── Create staging ──────────────────────────────────────────────────
        staging_dir = _create_staging(runner, dest.ssh_alias, remote_dir_str,
                                      dest.timeout, bundle_id)

        # ── Transfer snapshot and receipt ──────────────────────────────────
        remote_snapshot = f"{staging_dir}/{bundle.snapshot.name}"
        remote_receipt = f"{staging_dir}/{bundle.receipt.name}"

        try:
            _remote_write(runner, dest.ssh_alias, dest.timeout,
                          bundle.snapshot_bytes, remote_snapshot)
            _remote_write(runner, dest.ssh_alias, dest.timeout,
                          bundle.receipt_bytes, remote_receipt)
        except TransferError:
            _cleanup_staging(runner, dest.ssh_alias, dest.timeout, staging_dir, remote_dir_str)
            raise

        # ── Readback and verify hashes ──────────────────────────────────────
        # A readback TRANSFER failure must not orphan the staging dir: the
        # enclosing try/finally only releases the lock, so clean up here.
        try:
            snap_hash = _readback_hash(runner, dest.ssh_alias, dest.timeout, remote_snapshot)
            recv_hash = _readback_hash(runner, dest.ssh_alias, dest.timeout, remote_receipt)
        except TransferError:
            _cleanup_staging(runner, dest.ssh_alias, dest.timeout, staging_dir, remote_dir_str)
            raise

        if snap_hash != bundle.snapshot_sha256:
            _cleanup_staging(runner, dest.ssh_alias, dest.timeout, staging_dir, remote_dir_str)
            raise HashMismatchError(
                f"snapshot hash mismatch: remote={snap_hash}, expected={bundle.snapshot_sha256}"
            )
        if recv_hash != bundle.receipt_sha256:
            _cleanup_staging(runner, dest.ssh_alias, dest.timeout, staging_dir, remote_dir_str)
            raise HashMismatchError(
                f"receipt hash mismatch: remote={recv_hash}, expected={bundle.receipt_sha256}"
            )

        # ── Write bundle manifest ────────────────────────────────────────────
        manifest = BundleManifest(
            schema=BUNDLE_MANIFEST_SCHEMA,
            timestamp=bundle.timestamp,
            snapshot_name=bundle.snapshot.name,
            receipt_name=bundle.receipt.name,
            snapshot_sha256=bundle.snapshot_sha256,
            receipt_sha256=bundle.receipt_sha256,
            published=False,
        )
        manifest_bytes = json.dumps(dataclasses.asdict(manifest), sort_keys=True).encode()
        manifest_remote = f"{staging_dir}/.bundle-manifest.json"
        _remote_write(runner, dest.ssh_alias, dest.timeout, manifest_bytes, manifest_remote)

        # ── Atomic publish (staging -> final; manifest moves with it) ──────
        _publish_bundle(runner, dest.ssh_alias, dest.timeout, staging_dir, final_dir)

        # Now update the manifest to published=True (the dir is at final_dir)
        manifest = dataclasses.replace(manifest, published=True)
        manifest_bytes = json.dumps(dataclasses.asdict(manifest), sort_keys=True).encode()
        manifest_remote = f"{final_dir}/.bundle-manifest.json"
        _remote_write(runner, dest.ssh_alias, dest.timeout, manifest_bytes, manifest_remote)

        # ── Retention pruning (only after successful publish) ───────────────
        pruned = _prune_remote_bundles(
            runner, dest.ssh_alias, remote_dir_str, dest.timeout,
            published_snapshot_sha256=bundle.snapshot_sha256,
            published_receipt_sha256=bundle.receipt_sha256,
        )

    finally:
        _release_lock()

    return {
        "status": "published",
        "bundle": final_dir,
        "timestamp": bundle.timestamp,
        "snapshot_sha256": bundle.snapshot_sha256,
        "receipt_sha256": bundle.receipt_sha256,
        "pruned": pruned,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--ssh-alias", required=True)
    parser.add_argument("--remote-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--destination-manifest", type=Path, default=None)
    parser.add_argument("--ssh-config", default=None)
    parser.add_argument("--ssh-identity-file", default=None)
    parser.add_argument("--ssh-known-hosts-file", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 < args.timeout_seconds <= 120:
        # Argparse-style error: emit JSON error and return 1 (not 2)
        # so callers can distinguish from unknown flag.
        print(json.dumps({
            "status": "error",
            "detail": "--timeout-seconds must be greater than 0 and at most 120",
        }))
        return 1

    try:
        result = copy_bundle(
            source_directory=args.source_directory,
            ssh_alias=args.ssh_alias,
            remote_directory=args.remote_directory,
            timeout=args.timeout_seconds,
            destination_manifest=args.destination_manifest,
            ssh_config=args.ssh_config,
            ssh_identity_file=args.ssh_identity_file,
            ssh_known_hosts=args.ssh_known_hosts_file,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except NoVerifiedSnapshotError as exc:
        print(json.dumps({"status": "no_verified_snapshot", "detail": str(exc)}))
        return 1
    except MountMissingError as exc:
        print(json.dumps({"status": "mount_missing", "detail": str(exc)}))
        return 1
    except TransferError as exc:
        print(json.dumps({"status": "transfer_error", "detail": str(exc)}))
        return 1
    except HashMismatchError as exc:
        print(json.dumps({"status": "hash_mismatch", "detail": str(exc)}))
        return 1
    except PublishError as exc:
        print(json.dumps({"status": "publish_error", "detail": str(exc)}))
        return 1
    except BackupError as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
