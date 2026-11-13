"""Hermetic fake-transport tests for copy_gpu_manager_state_backup.

No real SSH, no real network, no real restore.
Covers: success, hash mismatch, transport failure, mount_missing,
timeout propagation, retention pruning, symlink rejection, JSON error contract,
CLI timeout bounds, no real ssh binary.
"""
from __future__ import annotations

import dataclasses
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.copy_gpu_manager_state_backup import (  # noqa: E402
    BackupError,
    BUNDLE_MANIFEST_SCHEMA,
    BundleManifest,
    copy_bundle,
    DESTINATION_SCHEMA,
    Destination,
    HashMismatchError,
    MAX_BUNDLES,
    MountMissingError,
    NoVerifiedSnapshotError,
    PublishError,
    SSH_BASE_ARGS,
    TransferError,
    _prune_remote_bundles,
    build_parser,
    main,
)


def test_default_ssh_shell_quotes_multiline_python_script(monkeypatch):
    """The remote shell must pass a multiline script as one valid -c argument."""
    import scripts.copy_gpu_manager_state_backup as mod

    script = "import os\nprint('remote script ran')\n"
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod._default_ssh("test-host", 5, script) == (0, b"ok", b"")

    remote_command = captured["argv"][-1]
    assert captured["argv"][0] == "ssh"
    assert captured["argv"][-2] == "--"
    assert shlex.split(remote_command) == ["python3", "-c", script]
    compile(shlex.split(remote_command)[2], "<remote-script>", "exec")
    assert captured["kwargs"]["timeout"] == 5


def test_ssh_argv_runtime_defaults_preserve_host_paths_byte_for_byte(monkeypatch):
    """Runtime defaults retain the installed host's exact SSH argument vector."""
    import scripts.copy_gpu_manager_state_backup as mod

    home = Path(mod.pwd.getpwuid(1000 if os.geteuid() == 0 else os.geteuid()).pw_dir)
    expected_base = [
        "-F", str(home / ".ssh/config"), "-i", str(home / ".ssh/id_ed25519"),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none", "-o", "ForwardAgent=no",
        "-o", "StrictHostKeyChecking=yes", "-o",
        f"UserKnownHostsFile={home / '.ssh/known_hosts'}",
    ]
    assert mod.ssh_base_args() == expected_base
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._default_ssh("dell", 5, "print('argv')")
    assert captured["argv"] == [
        "ssh", *expected_base, "dell", "--", "python3 -c " + shlex.quote("print('argv')")
    ]


@pytest.mark.parametrize(
    ("keyword", "flag", "option", "default"),
    [
        ("ssh_config", "-F", None, ".ssh/config"),
        ("ssh_identity_file", "-i", None, ".ssh/id_ed25519"),
        ("ssh_known_hosts", "-o", "UserKnownHostsFile=", ".ssh/known_hosts"),
    ],
)
def test_individual_ssh_path_override_changes_only_its_argv_slot(
    keyword, flag, option, default
):
    import scripts.copy_gpu_manager_state_backup as mod

    home = Path(mod.pwd.getpwuid(1000 if os.geteuid() == 0 else os.geteuid()).pw_dir)
    baseline = mod.ssh_base_args()
    override = "/custom/ssh path/selected"
    actual = mod.ssh_base_args(**{keyword: override})
    expected = baseline.copy()
    if option:
        index = expected.index(option + str(home / default))
        expected[index] = option + override
    else:
        index = expected.index(str(home / default))
        expected[index] = override
    assert actual == expected


def test_explicit_ssh_paths_do_not_require_compatibility_account(monkeypatch):
    import scripts.copy_gpu_manager_state_backup as mod

    def missing_account(_uid):
        raise KeyError("account absent")

    monkeypatch.setattr(mod.pwd, "getpwuid", missing_account)
    assert mod.ssh_base_args(
        ssh_config="/explicit/config", ssh_identity_file="/explicit/key",
        ssh_known_hosts="/explicit/known_hosts",
    ) == [
        "-F", "/explicit/config", "-i", "/explicit/key",
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none", "-o", "ForwardAgent=no",
        "-o", "StrictHostKeyChecking=yes", "-o",
        "UserKnownHostsFile=/explicit/known_hosts",
    ]


def test_default_ssh_py314_only_input_kwarg(monkeypatch):
    """Python 3.14 rejects stdin=PIPE + input= both non-None; only input= must be set."""
    import scripts.copy_gpu_manager_state_backup as mod

    captured = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._default_ssh("test-host", 5, "print('hi')", stdin_bytes=b"x")

    assert "input" in captured["kwargs"]
    assert captured["kwargs"]["input"] == b"x"
    assert "stdin" not in captured["kwargs"]


def test_default_ssh_writes_4mib_payload_with_bounded_argv(monkeypatch):
    """A 4 MiB payload is sent via input= only; argv bytes stay well below 64 KiB."""
    import scripts.copy_gpu_manager_state_backup as mod

    payload = os.urandom(4 * 1024 * 1024)
    captured_argv: list = []
    captured_kwargs: dict = {}

    def fake_run(argv, **kwargs):
        captured_argv.extend(argv)
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    code, stdout, stderr = mod._default_ssh(
        "test-host", 30.0, "import os\nprint(1)\n", (), payload
    )

    # (a) total argv bytes < 64 KiB
    argv_bytes = sum(
        len(a.encode("utf-8")) if isinstance(a, str) else len(a)
        for a in captured_argv
    )
    assert argv_bytes < 64 * 1024, f"argv bytes {argv_bytes} exceeds 64 KiB"

    # (b) payload arrives ONLY via input= kwarg
    assert captured_kwargs.get("input") == payload

    # (c) no stdin= key
    assert "stdin" not in captured_kwargs

    # (d) command succeeded
    assert code == 0


# ── Constants ─────────────────────────────────────────────────────────────────

ISOLATION = "private-valkey-unix-socket-no-workers"
SSH_ALIAS = "dell"
_BUNDLE_NAME_RE = re.compile(r"^(\d{8}T\d{6}Z)-(\d+)$")
# Note: tests use a unique absolute path under /tmp/pytest-offhost
TIMEOUT = 120


# ── Fixture helpers ───────────────────────────────────────────────────────────

def make_snapshot_receipt(
    tmp_dir: Path,
    stamp: str,
    pid: int = 1,
    redis: list | None = None,
    files: dict | None = None,
    receipt_updates: dict | None = None,
    symlink_snapshot: bool = False,
) -> tuple[Path, Path]:
    snapshot = tmp_dir / f"{stamp}-{pid}.json.gz"
    receipt = snapshot.with_suffix(".restore.json")
    payload = gzip.compress(json.dumps({
        "schema": "gpu-manager-private-snapshot.v1",
        "redis": [] if redis is None else redis,
        "files": {} if files is None else files,
    }).encode("utf-8"))
    if symlink_snapshot:
        actual = tmp_dir / f"actual-{stamp}-{pid}.json.gz"
        actual.write_bytes(payload)
        snapshot.symlink_to(actual.name)
    else:
        snapshot.write_bytes(payload)
    metadata = {
        "schema": "gpu-manager-private-snapshot.v1",
        "snapshot": str(snapshot.resolve()),
        "restored_records": len([] if redis is None else redis),
        "verified_files": len({} if files is None else files),
        "replay_count": 0,
        "isolation": ISOLATION,
    }
    metadata.update(receipt_updates or {})
    receipt.write_text(json.dumps(metadata), encoding="utf-8")
    return snapshot, receipt


def make_dest_manifest(tmp_dir: Path, remote_dir: Path, **overrides) -> tuple[Path, Destination]:
    dest = Destination(
        directory=remote_dir,
        mountpoint="/",
        fstype="ext4",
        source="/dev/sda1",
        ssh_alias=SSH_ALIAS,
        timeout=TIMEOUT,
    )
    dest = dataclasses.replace(dest, **overrides)
    manifest_path = tmp_dir / ".gpu-manager-backup-destination.json"
    manifest_path.write_text(json.dumps({
        "schema": DESTINATION_SCHEMA,
        "directory": str(dest.directory),
        "mountpoint": dest.mountpoint,
        "fstype": dest.fstype,
        "source": dest.source,
        "ssh_alias": dest.ssh_alias,
        "timeout_seconds": dest.timeout,
    }))
    return manifest_path, dest


# ── Fake transport runner ─────────────────────────────────────────────────────

class FakeTransport:
    """In-process fake SSH/SFTP transport for hermetic testing.

    Implements the SSHRunner signature:
        (ssh_alias, timeout, script, cmd_tuple, stdin_bytes)
            -> (returncode, stdout, stderr)

    Maps the configured remote_directory onto a local fake_root path,
    so the script's references to /backups/gpu-manager/foo become
    fake_root/foo.
    """

    def __init__(self, remote_root: Path, remote_alias: str):
        self.remote_root = remote_root
        # The remote_alias is the path string the production code uses for
        # the remote directory (e.g. /backups/gpu-manager)
        self.remote_alias = remote_alias
        self.lock_fd_counter = 100
        self.calls: list[dict] = []
        self.published_bundles: list[str] = []
        self.staging_dirs_created: list[str] = []
        self.pruned_count: int = 0
        # Injection flags
        self.injected_transfer_error: bool = False
        self.injected_readback_error: bool = False
        self.injected_publish_error: bool | int = False
        self.injected_lock_timeout: bool = False
        self.injected_lock_error: bool = False
        self.injected_mount_missing: bool = False
        self.injected_symlink_remote_dir: bool = False
        # fstype reported by the fake /proc/self/mountinfo line. None -> ext4.
        self.injected_fstype: str | None = None
        # When True, report the OLD separator-less line (no ' - ' separator).
        self.injected_separatorless: bool = False
        self.injected_prune_error: bool = False
        self.hash_mismatch_override: str | None = None
        self._lock_held: bool = False

    def __call__(self, ssh_alias: str, timeout: float, script: str,
                 cmd: tuple = (), stdin_bytes: bytes = b"") -> tuple[int, bytes, bytes]:
        self.calls.append({
            "ssh_alias": ssh_alias,
            "timeout": timeout,
            "script_head": script[:60],
        })

        # Lock acquisition
        if "fcntl.flock" in script and "LOCK_EX" in script and "LOCK_NB" in script:
            if self.injected_lock_timeout:
                return 124, b"", b"timeout"
            if self.injected_lock_error:
                return errno.EWOULDBLOCK, b"", b"lock held"
            fd = self.lock_fd_counter
            self.lock_fd_counter += 1
            self._lock_held = True
            return 0, str(fd).encode(), b""

        # Lock release
        if "fcntl.LOCK_UN" in script:
            self._lock_held = False
            return 0, b"", b""

        # Mount validation
        if "mountinfo" in script:
            if self.injected_mount_missing:
                return 2, b"", b"no mount info"
            if self.injected_symlink_remote_dir:
                return 1, b"", b"symlink in path"
            if not self.remote_root.exists():
                return 2, b"", b"no remote root"
            if self.injected_separatorless:
                return 0, b"22 22 0:1 / / ext4 rw\n", b""
            # Real /proc/self/mountinfo always carries the ' - ' separator
            # between the optional fields and the fstype/source/superblock
            # fields; fstype is the field AFTER it, not a fixed index.
            fstype = self.injected_fstype or "ext4"
            line = (f"22 22 0:1 / / rw,relatime shared:1"
                    f" - {fstype} /dev/sda1 rw\n")
            return 0, line.encode(), b""

        # stdin write — when the runner call carries stdin_bytes, write those
        # bytes to the localized path of the target file (mirrors the real
        # remote's sys.stdin.buffer.read() + open(path,'wb').write(...)).
        if "sys.stdin.buffer.read()" in script and stdin_bytes:
            if self.injected_transfer_error:
                return 1, b"", b"injected transfer error"
            m = re.search(r"path\s*=\s*['\"]([^'\"]+)['\"]", script)
            if m:
                remote_path = m.group(1)
                local_path = self._localize(remote_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(stdin_bytes)
                return 0, b"", b""
            return 1, b"", b"path parse error"

        # File write (bytes.fromhex) — check BEFORE staging so the regex
        # captures the actual file path, not the staging dir.
        if "bytes.fromhex" in script:
            if self.injected_transfer_error:
                return 1, b"", b"injected transfer error"
            m = re.search(r"open\(\s*['\"]([^'\"]+)['\"]", script)
            hex_match = re.search(r"bytes\.fromhex\(\s*['\"]([0-9a-f]+)['\"]", script)
            if m and hex_match:
                remote_path = m.group(1)
                local_path = self._localize(remote_path)
                data = bytes.fromhex(hex_match.group(1))
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                return 0, b"", b""
            return 1, b"", b"parse error"

        # Staging creation (must come after file write to avoid false matches
        # when the file path contains "staging-")
        if "makedirs" in script and "staging-" in script:
            m = re.search(r"['\"]([^'\"]*?staging-[^'\"]+)['\"]", script)
            if m:
                staging_remote = m.group(1)
                staging_local = self._localize(staging_remote)
                staging_local.parent.mkdir(parents=True, exist_ok=True)
                staging_local.mkdir(parents=True, exist_ok=True)
                self.staging_dirs_created.append(staging_remote)
                return 0, staging_remote.encode(), b""
            return 0, b"", b""

        # Readback hash
        if "hashlib.sha256" in script and "open(path" in script:
            if self.injected_readback_error:
                return 1, b"", b"injected readback error"
            m = re.search(r"path\s*=\s*['\"]([^'\"]+)['\"]", script)
            if m:
                remote_path = m.group(1)
                local_path = self._localize(remote_path)
                if local_path.exists():
                    h = hashlib.sha256(local_path.read_bytes()).hexdigest()
                    if self.hash_mismatch_override:
                        return 0, self.hash_mismatch_override.encode() + b"\n", b""
                    return 0, f"{h}\n".encode(), b""
                return 1, b"", b"file not found"
            return 1, b"", b"path parse error"

        # Atomic publish (renameat2)
        if "renameat2" in script:
            if self.injected_publish_error is not False:
                return int(self.injected_publish_error), b"", b"injected publish error"
            m = re.search(r"staging\s*=\s*['\"]([^'\"]+)['\"]", script)
            n = re.search(r"final\s*=\s*['\"]([^'\"]+)['\"]", script)
            if m and n:
                staging_remote = m.group(1)
                final_remote = n.group(1)
                staging_local = self._localize(staging_remote)
                final_local = self._localize(final_remote)
                if final_local.exists():
                    return 1, b"", b"exists"
                os.rename(staging_local, final_local)
                self.published_bundles.append(final_remote)
                return 0, b"", b""
            return 1, b"", b"parse error"

        # Prune
        if "os.listdir" in script and "BUNDLE_RE" in script:
            if self.injected_prune_error:
                return 1, b"", b"prune error"
            all_dirs = sorted(
                [
                    d.name
                    for d in self.remote_root.iterdir()
                    if d.is_dir()
                    and not d.is_symlink()
                    and bool(_BUNDLE_NAME_RE.match(d.name))
                ],
                reverse=True,
            )
            # Validate: only bundles with a .bundle-manifest.json and both files
            validated = []
            for name in all_dirs:
                bundle_local = self.remote_root / name
                manifest_file = bundle_local / ".bundle-manifest.json"
                if not manifest_file.exists():
                    continue
                import json as _json

                try:
                    mfst = _json.loads(manifest_file.read_text())
                except Exception:
                    continue
                snap_name = mfst.get("snapshot_name", "")
                recv_name = mfst.get("receipt_name", "")
                snap_hex = mfst.get("snapshot_sha256", "")
                recv_hex = mfst.get("receipt_sha256", "")
                snap_local = bundle_local / snap_name
                recv_local = bundle_local / recv_name
                if not snap_local.exists() or not recv_local.exists():
                    continue
                # hash check
                if (
                    hashlib.sha256(snap_local.read_bytes()).hexdigest() != snap_hex
                    or hashlib.sha256(recv_local.read_bytes()).hexdigest() != recv_hex
                ):
                    continue
                # Exclude current bundle by hashes if provided
                validated.append(name)

            # Sort by timestamp desc, name desc (stable)
            validated.sort(reverse=True)
            pruned = len(validated[MAX_BUNDLES:])
            for name in validated[MAX_BUNDLES:]:
                shutil.rmtree(self.remote_root / name)
            self.pruned_count = pruned
            return 0, str(pruned).encode(), b""

        # Staging cleanup (shutil.rmtree)
        if "shutil.rmtree" in script:
            m = re.search(r"['\"]([^'\"]+?staging-[^'\"]+)['\"]", script)
            if m:
                remote_path = m.group(1)
                local_path = self._localize(remote_path)
                if local_path.exists():
                    shutil.rmtree(local_path, ignore_errors=True)
            return 0, b"", b""

        return 0, b"", b""

    def _localize(self, remote_path: str) -> Path:
        """Map a remote path (e.g. /backups/gpu-manager/foo) to local fake_root path."""
        if remote_path == self.remote_alias:
            return self.remote_root
        if remote_path.startswith(self.remote_alias + "/"):
            relative = remote_path[len(self.remote_alias) + 1:]
            return self.remote_root / relative
        return Path(remote_path)


# ── Test cases ────────────────────────────────────────────────────────────────

# Source-selection tests (regression — from prior stage)

def test_selects_newest_valid_pair_and_hashes_exact_read_bytes(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    make_snapshot_receipt(tmp_path, "20261008T070000Z", redis=[{"key": "older"}])
    selected, receipt = make_snapshot_receipt(
        tmp_path, "20261008T080000Z", files={"a": {"sha256": "x"}}
    )

    bundle = select_newest_verified(tmp_path)
    assert bundle.snapshot == selected
    assert bundle.receipt == receipt
    assert bundle.snapshot_sha256 == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert bundle.receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_zero_redis_records_and_zero_files_is_a_valid_receipt(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    selected, _ = make_snapshot_receipt(tmp_path, "20261008T080000Z")
    assert select_newest_verified(tmp_path).snapshot == selected


@pytest.mark.parametrize("updates", [
    {"restored_records": True},
    {"verified_files": False},
    {"replay_count": False},
    {"replay_count": 1},
    {"isolation": "other"},
    {"snapshot": "/not/the/selected/snapshot"},
])
def test_rejects_malformed_or_mismatched_receipt(tmp_path, updates):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    make_snapshot_receipt(tmp_path, "20261008T080000Z", receipt_updates=updates)
    with pytest.raises(NoVerifiedSnapshotError, match="no_verified_snapshot"):
        select_newest_verified(tmp_path)


def test_skips_invalid_newer_pair_for_older_verified_pair(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    good, _ = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    make_snapshot_receipt(tmp_path, "20261008T080000Z", receipt_updates={"restored_records": 1})
    assert select_newest_verified(tmp_path).snapshot == good


def test_rejects_symlink_snapshot_and_receipt(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    snapshot, _ = make_snapshot_receipt(tmp_path, "20261008T080000Z", symlink_snapshot=True)
    with pytest.raises(NoVerifiedSnapshotError):
        select_newest_verified(tmp_path)

    snapshot.unlink()
    snapshot.write_bytes(gzip.compress(json.dumps({
        "schema": "gpu-manager-private-snapshot.v1", "redis": [], "files": {},
    }).encode()))
    receipt = snapshot.with_suffix(".restore.json")
    real_receipt = tmp_path / "receipt-real.json"
    real_receipt.write_text(json.dumps({
        "snapshot": str(snapshot.resolve()), "restored_records": 0,
        "verified_files": 0, "replay_count": 0, "isolation": ISOLATION,
    }))
    receipt.unlink()
    receipt.symlink_to(real_receipt.name)
    with pytest.raises(NoVerifiedSnapshotError):
        select_newest_verified(tmp_path)


def test_timestamp_tie_breaks_by_descending_filename(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    lower, _ = make_snapshot_receipt(tmp_path, "20261008T080000Z", pid=2)
    higher, _ = make_snapshot_receipt(tmp_path, "20261008T080000Z", pid=9)
    assert select_newest_verified(tmp_path).snapshot == higher
    assert lower.name < higher.name


def test_no_eligible_pair_and_missing_directory_are_explicit(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    with pytest.raises(NoVerifiedSnapshotError, match="no_verified_snapshot"):
        select_newest_verified(tmp_path / "absent")
    make_snapshot_receipt(tmp_path, "20261308T080000Z")
    with pytest.raises(NoVerifiedSnapshotError, match="no_verified_snapshot"):
        select_newest_verified(tmp_path)


def test_snapshot_receipt_counts_must_match_document(tmp_path):
    from scripts.copy_gpu_manager_state_backup import select_newest_verified  # noqa: E402

    make_snapshot_receipt(
        tmp_path, "20261008T080000Z", redis=[{}], receipt_updates={"restored_records": 0}
    )
    with pytest.raises(NoVerifiedSnapshotError):
        select_newest_verified(tmp_path)


# ── Copy stage tests ─────────────────────────────────────────────────────────

def test_success_with_matching_hashes_and_atomic_publish(tmp_path):
    """Successful copy: bundle selected, transferred, verified, published."""
    source, receipt = make_snapshot_receipt(
        tmp_path, "20261008T070000Z", files={"a": {"sha256": "x"}}
    )
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)

    transport = FakeTransport(remote_root, remote_alias)

    result = copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=TIMEOUT,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    assert result["status"] == "published"
    assert result["timestamp"] == "20261008T070000Z"
    assert len(transport.published_bundles) == 1
    # Local source untouched
    assert source.exists()
    assert receipt.exists()
    expected_payload = gzip.compress(json.dumps({
        "schema": "gpu-manager-private-snapshot.v1",
        "redis": [],
        "files": {"a": {"sha256": "x"}},
    }).encode("utf-8"))
    assert source.read_bytes() == expected_payload


def test_hash_mismatch_refuses_publication(tmp_path):
    """Injected remote hash mismatch -> HashMismatchError, no publish, local intact."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)

    transport = FakeTransport(remote_root, remote_alias)
    transport.hash_mismatch_override = "deadbeef" * 16

    with pytest.raises(HashMismatchError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0
    assert source.exists()
    assert receipt.exists()


def test_transport_failure_reports_delivery_failure_and_leaves_local_intact(tmp_path):
    """Injected transfer error -> TransferError, no publish, local untouched."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)

    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_transfer_error = True

    with pytest.raises(TransferError) as exc:
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert "transfer" in str(exc.value).lower() or "write" in str(exc.value).lower()
    assert len(transport.published_bundles) == 0
    assert source.exists()
    assert receipt.exists()


def test_readback_failure_reports_transfer_error(tmp_path):
    """Readback failure -> TransferError, no publish, local intact."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)

    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_readback_error = True

    with pytest.raises(TransferError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0
    assert source.exists()
    assert receipt.exists()


def test_readback_failure_removes_staging_dir(tmp_path):
    """Readback failure -> staging dir removed, no orphan left on the remote.

    Extends test_readback_failure_reports_transfer_error: that test asserts
    TransferError/no-publish/local-intact but never asserted staging removal,
    so a readback failure left .staging-<bundle-id> behind on the destination.
    """
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)

    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_readback_error = True

    with pytest.raises(TransferError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.staging_dirs_created) == 1
    assert len(transport.published_bundles) == 0
    assert source.exists()
    assert receipt.exists()
    # No orphan .staging-* survives a failed readback.
    assert list(remote_root.glob(".staging-*")) == []


def test_mount_missing_when_no_manifest(tmp_path):
    """Missing destination manifest -> MountMissingError, fail closed."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    with pytest.raises(MountMissingError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=None,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0
    assert source.exists()
    assert receipt.exists()


def test_mount_identity_mismatch_is_explicit(tmp_path):
    """Remote mount identity mismatch -> MountMissingError."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_mount_missing = True

    with pytest.raises(MountMissingError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0
    assert source.exists()


def test_symlink_in_remote_path_rejected(tmp_path):
    """Remote path with a symlink component -> MountMissingError."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_symlink_remote_dir = True

    with pytest.raises(MountMissingError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0
    assert source.exists()


def test_validate_mount_rejects_fstype_mismatch(tmp_path):
    """Manifest fstype != observed mountinfo fstype -> MountMissingError."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)  # fstype ext4
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_fstype = "btrfs"

    with pytest.raises(MountMissingError) as excinfo:
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    message = str(excinfo.value)
    assert "ext4" in message
    assert "btrfs" in message
    assert len(transport.published_bundles) == 0
    assert source.exists()


def test_validate_mount_accepts_fstype_match(tmp_path):
    """Manifest fstype == observed mountinfo fstype -> published."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)  # fstype ext4
    transport = FakeTransport(remote_root, remote_alias)  # reports ext4

    result = copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=TIMEOUT,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    assert result["status"] == "published"
    assert len(transport.published_bundles) == 1


def test_validate_mount_unset_fstype_keeps_current_behaviour(tmp_path):
    """No 'fstype' key in the manifest -> comparison skipped, copy proceeds."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, dest = make_dest_manifest(tmp_path, remote_root)
    manifest_doc = json.loads(manifest_path.read_text())
    del manifest_doc["fstype"]  # -> 'unknown' via load_destination line 106
    manifest_path.write_text(json.dumps(manifest_doc))
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_fstype = "zfs"  # would be a mismatch if compared

    result = copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=TIMEOUT,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    assert result["status"] == "published"
    assert len(transport.published_bundles) == 1


def test_validate_mount_separatorless_line_fails_closed(tmp_path):
    """A mountinfo line with no ' - ' separator is a mismatch, not a crash."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_separatorless = True

    with pytest.raises(MountMissingError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0
    assert source.exists()


def test_lock_timeout_reports_transfer_error(tmp_path):
    """Lock acquisition timeout -> TransferError."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_lock_timeout = True

    with pytest.raises(TransferError, match="lock"):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert len(transport.published_bundles) == 0


def test_publish_failure_reports_publish_error(tmp_path):
    """Atomic publish failure -> PublishError, no bundle created."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_publish_error = 30  # EXDEV / renameat2 unsupported

    with pytest.raises(PublishError) as exc:
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    msg = str(exc.value)
    assert "EXDEV" in msg or "renameat2" in msg.lower() or "RENAME_NOREPLACE" in msg
    assert source.exists()
    assert receipt.exists()


def test_publish_collision_rejected(tmp_path):
    """Final bundle dir already exists -> PublishError, no overwrite."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    # Pre-create the would-be published dir (use os.getpid() since the bundle
    # name embeds it)
    pre_existing = remote_root / f"20261008T070000Z-{os.getpid()}"
    pre_existing.mkdir()
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    with pytest.raises(PublishError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert source.exists()
    assert receipt.exists()


def test_retention_prune_only_after_publish(tmp_path):
    """Pruning only runs after a successful publish."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)

    # Create MAX_BUNDLES + 5 old published bundles, each with manifest + files + hashes
    for i in range(MAX_BUNDLES + 5):
        # Timestamp: 8-digit date YYYYMMDD, then time + PID
        ts = f"20261{i:03d}T000000Z"  # e.g. 202610000T000000Z, 202610001T000000Z ...
        pid = f"{i:03d}"
        bundle_dir = remote_root / f"{ts}-{pid}"
        bundle_dir.mkdir()
        snap_name = f"{ts}-{pid}.json.gz"
        recv_name = f"{ts}-{pid}.restore.json"
        snap_bytes = gzip.compress(b"{}")
        recv_bytes = json.dumps({
            "schema": "gpu-manager-private-snapshot.v1",
            "snapshot": str(bundle_dir / snap_name),
            "restored_records": 0,
            "verified_files": 0,
            "replay_count": 0,
            "isolation": ISOLATION,
        }).encode()
        snap_hex = hashlib.sha256(snap_bytes).hexdigest()
        recv_hex = hashlib.sha256(recv_bytes).hexdigest()
        (bundle_dir / snap_name).write_bytes(snap_bytes)
        (bundle_dir / recv_name).write_bytes(recv_bytes)
        (bundle_dir / ".bundle-manifest.json").write_text(json.dumps({
            "schema": BUNDLE_MANIFEST_SCHEMA,
            "timestamp": ts,
            "snapshot_name": snap_name,
            "receipt_name": recv_name,
            "snapshot_sha256": snap_hex,
            "receipt_sha256": recv_hex,
            "published": True,
        }))

    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    source, receipt = make_snapshot_receipt(tmp_path, "20261015T120000Z")
    transport = FakeTransport(remote_root, remote_alias)

    result = copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=TIMEOUT,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    assert result["status"] == "published"
    # MAX_BUNDLES+5 old bundles + 1 new bundle published = MAX_BUNDLES+6 total
    # Pruning removes all beyond MAX_BUNDLES, so pruned = 6
    assert result["pruned"] == 6
    # Verify only MAX_BUNDLES remain
    remaining = [d for d in remote_root.iterdir() if d.is_dir() and d.name[:1].isdigit()]
    assert len(remaining) == MAX_BUNDLES


def test_retention_prune_never_runs_on_failure(tmp_path):
    """If publish fails, prune never runs and old bundles stay intact."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    # Create MAX_BUNDLES + 3 bare directories (no manifest) — they won't be
    # validated so they are not considered retention candidates at all.
    # Prune should return 0 and leave them all intact.
    for i in range(MAX_BUNDLES + 3):
        bundle_dir = remote_root / f"2026100{i:02d}T000000Z-{i}"
        bundle_dir.mkdir()
        # No manifest, no files — not a valid bundle candidate

    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    source, receipt = make_snapshot_receipt(tmp_path, "20261020T120000Z")
    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_publish_error = 30  # publish fails

    with pytest.raises(PublishError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    # All old bundles should still exist (prune never ran)
    remaining = [d for d in remote_root.iterdir() if d.is_dir() and d.name[:1].isdigit()]
    assert len(remaining) == MAX_BUNDLES + 3


# ── Real prune-script harness (no FakeTransport) ─────────────────────────────
#
# FakeTransport REIMPLEMENTS pruning in Python, so a prune test written against
# it passes even when the production prune script is entirely unfixed. The
# harness below captures the real remote script that _prune_remote_bundles sends
# and executes it locally against a real temp directory.


def _seed_validated_bundles(remote_root: Path, count: int) -> list[str]:
    """Create `count` retention-validated published bundles under remote_root.

    Same shape as the inline seeding in test_retention_prune_only_after_publish:
    each directory holds a valid .bundle-manifest.json plus both recorded files
    with matching SHA-256s, so the real prune script counts them as candidates.

    Returns the bundle names in creation order (oldest first).
    """
    names = []
    for i in range(count):
        ts = f"20261{i:03d}T000000Z"
        pid = f"{i:03d}"
        bundle_dir = remote_root / f"{ts}-{pid}"
        bundle_dir.mkdir()
        snap_name = f"{ts}-{pid}.json.gz"
        recv_name = f"{ts}-{pid}.restore.json"
        snap_bytes = gzip.compress(b"{}")
        recv_bytes = json.dumps({
            "schema": "gpu-manager-private-snapshot.v1",
            "snapshot": str(bundle_dir / snap_name),
            "restored_records": 0,
            "verified_files": 0,
            "replay_count": 0,
            "isolation": ISOLATION,
        }).encode()
        (bundle_dir / snap_name).write_bytes(snap_bytes)
        (bundle_dir / recv_name).write_bytes(recv_bytes)
        (bundle_dir / ".bundle-manifest.json").write_text(json.dumps({
            "schema": BUNDLE_MANIFEST_SCHEMA,
            "timestamp": ts,
            "snapshot_name": snap_name,
            "receipt_name": recv_name,
            "snapshot_sha256": hashlib.sha256(snap_bytes).hexdigest(),
            "receipt_sha256": hashlib.sha256(recv_bytes).hexdigest(),
            "published": True,
        }))
        names.append(bundle_dir.name)
    return names


def _capture_prune_script(remote_dir: str) -> str:
    """Run the real _prune_remote_bundles with a recorder runner; return the script."""
    captured = {}

    def recorder(ssh_alias, timeout, script, cmd, stdin_bytes):
        captured["script"] = script
        return 0, b"0\n", b""

    _prune_remote_bundles(
        recorder,
        SSH_ALIAS,
        remote_dir,
        TIMEOUT,
        published_snapshot_sha256="0" * 64,
        published_receipt_sha256="1" * 64,
    )
    return captured["script"]


def test_prune_script_reclaims_stale_staging(tmp_path):
    """The REAL prune script removes a stale .staging-* orphan, keeps a live one.

    Executed through the real remote script rather than FakeTransport, so this
    fails against production code whose prune script cannot match .staging-*.
    """
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)

    # MAX_BUNDLES + 5 published bundles -> 5 are surplus and will be pruned.
    seeded = _seed_validated_bundles(remote_root, MAX_BUNDLES + 5)
    assert len(seeded) == MAX_BUNDLES + 5

    # Oldest orphan: strictly older than every retained bundle.
    stale = remote_root / ".staging-20000101T000000Z-1234"
    stale.mkdir()
    (stale / "leftover.json.gz").write_bytes(b"partial transfer")
    # Live/in-progress orphan: at/after the retention edge, must survive.
    live = remote_root / ".staging-20261099T000000Z-9999"
    live.mkdir()
    (live / "incoming.json.gz").write_bytes(b"still uploading")

    script = _capture_prune_script(remote_alias)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr.decode()
    assert not stale.exists(), "stale .staging-* orphan survived the REAL prune script"
    assert live.exists(), "prune script removed an in-progress .staging-* directory"
    assert "STAGING_RE" in script, "prune script has no .staging-* matcher at all"
    # 5 surplus bundles + 1 reclaimed stale orphan; the live one stays.
    printed = proc.stdout.decode().strip()
    assert printed == "6", f"prune script printed {printed!r}, expected '6'"

    remaining = sorted(
        d.name for d in remote_root.iterdir()
        if d.is_dir() and not d.is_symlink() and _BUNDLE_NAME_RE.match(d.name)
    )
    assert len(remaining) == MAX_BUNDLES
    assert not set(seeded[:5]) & set(remaining), "a published bundle was pruned wrongly"


def test_prune_script_leaves_symlinked_staging_entry(tmp_path):
    """A symlink named like a .staging-* orphan is never traversed or removed."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.json.gz").write_bytes(b"not an orphan")

    _seed_validated_bundles(remote_root, MAX_BUNDLES + 5)
    link = remote_root / ".staging-20000101T000000Z-1234"
    link.symlink_to(target)

    script = _capture_prune_script(str(remote_root))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)

    assert proc.returncode == 0, proc.stderr.decode()
    assert link.is_symlink(), "prune script removed a symlinked .staging-* entry"
    assert (target / "keep.json.gz").exists(), "prune script followed a .staging-* symlink"


def test_prune_script_no_staging_dirs_is_noop(tmp_path):
    """With no .staging-* children the sweep removes nothing beyond surplus bundles."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    seeded = _seed_validated_bundles(remote_root, MAX_BUNDLES + 5)

    script = _capture_prune_script(str(remote_root))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)

    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout.decode().strip() == str(5), "surplus count changed with no staging dirs"
    remaining = [
        d.name for d in remote_root.iterdir()
        if d.is_dir() and _BUNDLE_NAME_RE.match(d.name)
    ]
    assert len(remaining) == MAX_BUNDLES
    assert not set(seeded[:5]) & set(remaining)


def test_prune_script_staging_older_than_retained_is_reclaimed(tmp_path):
    """Boundary: a staging dir strictly older than the oldest retained bundle goes."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    seeded = _seed_validated_bundles(remote_root, MAX_BUNDLES + 5)
    # Oldest retained bundle after pruning is the 6th seeded one; a staging dir
    # must be STRICTLY older than that edge to count as a stale orphan.
    all_ts = sorted(n.split("-", 1)[0] for n in seeded)
    oldest_retained_ts = all_ts[5]
    stale_ts = all_ts[4]

    stale = remote_root / ".staging-{}-0001".format(stale_ts)
    stale.mkdir()
    (stale / "leftover").write_bytes(b"x")

    script = _capture_prune_script(str(remote_root))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)

    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout.decode().strip() == "6"
    assert not stale.exists(), "staging dir older than the oldest retained bundle survived"


def test_prune_script_staging_at_or_after_retained_edge_is_kept(tmp_path):
    """Boundary: a staging dir at/after the oldest retained bundle is in progress."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    _seed_validated_bundles(remote_root, MAX_BUNDLES + 5)
    edge_ts = sorted(f"20261{i:03d}T000000Z" for i in range(MAX_BUNDLES + 5))[5]

    edge = remote_root / ".staging-{}-0001".format(edge_ts)
    edge.mkdir()
    (edge / "incoming").write_bytes(b"x")
    newer = remote_root / ".staging-20269999T000000Z-4242"
    newer.mkdir()
    (newer / "incoming").write_bytes(b"x")

    script = _capture_prune_script(str(remote_root))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)

    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout.decode().strip() == "5", "an in-progress staging dir was reclaimed"
    assert edge.exists(), "staging dir at the retention edge was reclaimed"
    assert newer.exists(), "newer staging dir was reclaimed"


def test_prune_script_never_matches_published_bundles(tmp_path):
    """The staging sweep cannot touch anything the BUNDLE_RE names."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    seeded = _seed_validated_bundles(remote_root, MAX_BUNDLES + 5)

    script = _capture_prune_script(str(remote_root))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)

    assert proc.returncode == 0, proc.stderr.decode()
    remaining = sorted(d.name for d in remote_root.iterdir() if d.is_dir())
    assert remaining == sorted(seeded[5:]), "a directory outside the retention window survived"


def test_bundle_manifest_recorded_published(tmp_path):
    """Published bundle includes .bundle-manifest.json with published=True."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=TIMEOUT,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    assert len(transport.published_bundles) == 1
    bundle_dir = remote_root / "20261008T070000Z-1"  # os.getpid() may differ; use the recorded one
    # The published path uses os.getpid() suffix, so look it up
    bundle_name = Path(transport.published_bundles[0]).name
    bundle_local = remote_root / bundle_name
    manifest_file = bundle_local / ".bundle-manifest.json"
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert manifest_data["schema"] == BUNDLE_MANIFEST_SCHEMA
    assert manifest_data["timestamp"] == "20261008T070000Z"
    assert manifest_data["published"] is True
    assert manifest_data["snapshot_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_snapshot_and_receipt_both_transferred_to_staging(tmp_path):
    """Both snapshot and receipt files are written into the staging directory."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=TIMEOUT,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    assert len(transport.staging_dirs_created) == 1
    # Both files appear in the published dir
    bundle_local = remote_root / Path(transport.published_bundles[0]).name
    assert (bundle_local / source.name).exists()
    assert (bundle_local / receipt.name).exists()


def test_copy_failure_never_modifies_local_backup(tmp_path):
    """Any failure during copy leaves local source files completely unchanged."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)

    original_snapshot_bytes = source.read_bytes()
    original_receipt_bytes = receipt.read_bytes()
    original_snapshot_mtime = source.stat().st_mtime_ns
    original_receipt_mtime = receipt.stat().st_mtime_ns

    transport = FakeTransport(remote_root, remote_alias)
    transport.injected_transfer_error = True

    with pytest.raises(TransferError):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert source.read_bytes() == original_snapshot_bytes
    assert receipt.read_bytes() == original_receipt_bytes
    assert source.stat().st_mtime_ns == original_snapshot_mtime
    assert receipt.stat().st_mtime_ns == original_receipt_mtime
    assert source.exists()
    assert receipt.exists()


# ── CLI tests ─────────────────────────────────────────────────────────────────

def test_cli_rejects_timeout_over_120(tmp_path, capsys):
    """CLI exits 1 (error) for timeout > 120."""
    source, _ = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    rc = main([
        "--source-directory", str(tmp_path),
        "--ssh-alias", "dell",
        "--remote-directory", "/backups",
        "--timeout-seconds", "150",
    ])
    out = capsys.readouterr()
    assert rc == 1


def test_cli_rejects_timeout_zero(tmp_path, capsys):
    """CLI exits 1 (error) for timeout <= 0."""
    source, _ = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    rc = main([
        "--source-directory", str(tmp_path),
        "--ssh-alias", "dell",
        "--remote-directory", "/backups",
        "--timeout-seconds", "0",
    ])
    out = capsys.readouterr()
    assert rc == 1


def test_cli_success_returns_published_status(tmp_path, capsys):
    """CLI success -> status:published, exit 0."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    import scripts.copy_gpu_manager_state_backup as mod
    original_runner = mod._default_ssh
    mod._default_ssh = transport
    try:
        rc = main([
            "--source-directory", str(tmp_path),
            "--ssh-alias", "dell",
            "--remote-directory", remote_alias,
            "--timeout-seconds", "120",
            "--destination-manifest", str(manifest_path),
        ])
    finally:
        mod._default_ssh = original_runner

    out = capsys.readouterr()
    data = json.loads(out.out.strip())
    assert rc == 0
    assert data["status"] == "published"


# ── No real SSH binary test ───────────────────────────────────────────────────

def test_no_real_ssh_invoked_via_subprocess(tmp_path):
    """Verify that with a runner injected, no real ssh subprocess is called."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    ssh_subprocess_calls: list[list] = []
    real_subprocess_run = subprocess.run

    def spy_run(*args, **kwargs):
        if args and isinstance(args[0], list) and args[0] and args[0][0] == "ssh":
            ssh_subprocess_calls.append(args[0])
        return real_subprocess_run(*args, **kwargs)

    import unittest.mock as mock
    with mock.patch("subprocess.run", side_effect=spy_run):
        copy_bundle(
            source_directory=tmp_path,
            ssh_alias=SSH_ALIAS,
            remote_directory=Path(remote_alias),
            timeout=TIMEOUT,
            destination_manifest=manifest_path,
            _runner=transport,
        )

    assert ssh_subprocess_calls == [], f"Unexpected SSH calls: {ssh_subprocess_calls}"


def test_default_runner_uses_explicit_ssh_args(tmp_path):
    """Default runner uses explicit -F config, -i key, BatchMode, IdentitiesOnly."""
    import scripts.copy_gpu_manager_state_backup as mod

    assert SSH_BASE_ARGS == mod.ssh_base_args()
    assert SSH_BASE_ARGS[SSH_BASE_ARGS.index("-F") + 1].endswith("/.ssh/config")
    assert SSH_BASE_ARGS[SSH_BASE_ARGS.index("-i") + 1].endswith("/.ssh/id_ed25519")
    assert "BatchMode=yes" in SSH_BASE_ARGS
    assert "IdentitiesOnly=yes" in SSH_BASE_ARGS
    assert "IdentityAgent=none" in SSH_BASE_ARGS
    assert "ForwardAgent=no" in SSH_BASE_ARGS
    assert "StrictHostKeyChecking=yes" in SSH_BASE_ARGS
    assert any(value.startswith("UserKnownHostsFile=") for value in SSH_BASE_ARGS)


def test_timeout_propagates_to_runner(tmp_path):
    """Configured timeout is passed to each runner call."""
    source, receipt = make_snapshot_receipt(tmp_path, "20261008T070000Z")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_alias = str(remote_root)
    manifest_path, _ = make_dest_manifest(tmp_path, remote_root)
    transport = FakeTransport(remote_root, remote_alias)

    copy_bundle(
        source_directory=tmp_path,
        ssh_alias=SSH_ALIAS,
        remote_directory=Path(remote_alias),
        timeout=30.0,
        destination_manifest=manifest_path,
        _runner=transport,
    )

    for call in transport.calls:
        assert call["timeout"] == 30.0, f"Timeout not propagated: {call}"
