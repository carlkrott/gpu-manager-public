"""Hermetic collection and temporal identity contract for gpu_guard.

The module API is intentionally exercised only with injected fake proc/sys roots:

- discover_render_nodes(sys_root, pci_addresses) -> {node_path: pci_bdf}
- parse_fdinfo(text) -> client_id + engine_gfx_ns/engine_compute_ns + vram_kib
- collect_snapshot(proc_root, render_nodes, sampled_at, *, managed_units,
  desktop_allowlist, max_cmdline) -> {"clients": [...], "unreadable": [...]}
- GuardState(*, window_seconds, max_pids, max_samples,
  managed_units, desktop_allowlist).record(snapshot) updates history;
  .active_clients(since=seconds) returns identities with positive uninterrupted
  engine deltas; .classify(client) returns managed/desktop/foreign.

No target device is opened: fake fd symlinks contain device path strings only.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpu_manager import gpu_guard


GPU0 = "0000:03:00.0"
GPU1 = "0000:0e:00.0"
RENDER0 = "/dev/dri/renderD128"
KFD = "/dev/kfd"


def _fdinfo(client_id: int, gfx: int = 0, compute: int = 0, vram: int = 0) -> str:
    return (
        "drm-driver: amdgpu\n"
        f"drm-client-id:\t{client_id}\n"
        f"drm-engine-gfx:\t{gfx} ns\n"
        f"drm-engine-compute:\t{compute} ns\n"
        f"drm-total-vram:\t{vram} KiB\n"
    )


def _proc(root: Path, pid: int, *, comm: str, cmdline: str, starttime: int,
          fds: dict[str, str], fdinfo: dict[str, str] | None = None) -> None:
    p = root / str(pid)
    (p / "fd").mkdir(parents=True)
    (p / "fdinfo").mkdir()
    (p / "comm").write_text(comm + "\n")
    (p / "cmdline").write_bytes(cmdline.encode() + b"\0")
    # Linux stat's comm field is parenthesized and may contain spaces or ')'.
    fields = ["S"] + ["0"] * 18 + [str(starttime)]
    (p / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
    for fd, target in fds.items():
        (p / "fd" / fd).symlink_to(target)
        if fdinfo and fd in fdinfo:
            (p / "fdinfo" / fd).write_text(fdinfo[fd])


def _collect(proc_root: Path, *, sampled_at: float = 1.0, max_cmdline: int = 24):
    return gpu_guard.collect_snapshot(
        proc_root,
        {RENDER0: GPU0},
        sampled_at,
        managed_units={"gpu-manager.service"},
        desktop_allowlist={"kwin_x11", "plasmashell"},
        max_cmdline=max_cmdline,
    )


def test_discover_render_nodes_uses_pci_slot_not_render_number(tmp_path):
    sysroot = tmp_path / "sys"
    for name, bdf in (("renderD128", GPU0), ("renderD129", GPU1)):
        device = sysroot / "class/drm" / name / "device"
        device.mkdir(parents=True)
        (device / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={bdf}\n")

    assert gpu_guard.discover_render_nodes(sysroot, {GPU0}) == {RENDER0: GPU0}
    assert gpu_guard.discover_render_nodes(sysroot, {GPU1}) == {
        "/dev/dri/renderD129": GPU1
    }


def test_parse_fdinfo_counters_and_missing_or_malformed_values():
    parsed = gpu_guard.parse_fdinfo(_fdinfo(71, gfx=100, compute=250, vram=4096))
    assert parsed == {
        "drm_client_id": "71",
        "engine_gfx_ns": 100,
        "engine_compute_ns": 250,
        "total_vram_kib": 4096,
    }
    assert gpu_guard.parse_fdinfo("drm-driver: amdgpu\n") == {
        "drm_client_id": None,
        "engine_gfx_ns": None,
        "engine_compute_ns": None,
        "total_vram_kib": None,
    }
    malformed = gpu_guard.parse_fdinfo(
        "drm-client-id: bad\ndrm-engine-gfx: nope ns\n"
        "drm-engine-compute: 3 ns\ndrm-total-vram: 2 KiB\n"
    )
    assert malformed["drm_client_id"] is None
    assert malformed["engine_gfx_ns"] is None
    assert malformed["engine_compute_ns"] == 3
    assert malformed["total_vram_kib"] == 2


def test_collect_deduplicates_shared_fds_by_device_client_but_keeps_pid_holders(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid, comm, fd in ((100, "renderer-a", "3"), (200, "renderer-b", "9")):
        _proc(proc, pid, comm=comm, cmdline=f"{comm} --gpu", starttime=pid + 1000,
              fds={fd: RENDER0}, fdinfo={fd: _fdinfo(55, gfx=600, vram=128)})

    snapshot = _collect(proc)
    clients = snapshot["clients"]
    identities = {(c["device"], c["drm_client_id"]) for c in clients}
    assert identities == {(GPU0, "55")}
    assert len(clients) == 1
    assert {holder["pid"] for holder in clients[0]["holders"]} == {100, 200}
    assert clients[0]["holders"][0]["starttime"] in {1100, 1200}


def test_collect_tracks_distinct_pid_incarnations_and_truncates_metadata(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc(proc, 123, comm="worker (gpu)", cmdline="worker (gpu) --a-very-long-argument",
          starttime=987654, fds={"4": RENDER0}, fdinfo={"4": _fdinfo(9, gfx=1)})

    client = _collect(proc, max_cmdline=12)["clients"][0]
    holder = client["holders"][0]
    assert holder["pid"] == 123
    assert holder["starttime"] == 987654
    assert holder["comm"] == "worker (gpu)"
    assert len(holder["cmdline"]) <= 12
    assert holder["cmdline"].startswith("worker (gpu)")


def test_collect_reports_unreadable_and_vanished_entries_without_live_fallback(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc(proc, 301, comm="denied", cmdline="denied", starttime=1,
          fds={"3": RENDER0}, fdinfo={"3": _fdinfo(1)})
    # Inject an fdinfo permission failure explicitly; root's filesystem privileges
    # must not make this test pass by reading a real or fixture file accidentally.
    real_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if str(path) == str(proc / "301/fdinfo/3"):
            raise PermissionError("synthetic unreadable fdinfo")
        if not str(path).startswith(str(proc)):
            raise AssertionError(f"unexpected filesystem read: {path}")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = _collect(proc)
    assert result["clients"] == []
    assert result["unreadable"]
    assert any(item["pid"] == 301 for item in result["unreadable"])


def test_collect_includes_kfd_only_holder_without_fdinfo_and_skips_vanished_pid(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc(proc, 401, comm="compute", cmdline="compute --run", starttime=20,
          fds={"7": KFD})
    result = _collect(proc)
    assert result["clients"] == []
    assert len(result["kfd_holders"]) == 1
    assert result["kfd_holders"][0]["pid"] == 401
    # A second collection after the fake process disappears simply omits it.
    import shutil
    shutil.rmtree(proc / "401")
    assert _collect(proc)["kfd_holders"] == []


def _state(**kwargs):
    return gpu_guard.GuardState(
        window_seconds=kwargs.pop("window_seconds", 120),
        max_pids=kwargs.pop("max_pids", 10),
        max_samples=kwargs.pop("max_samples", 3),
        managed_units={"gpu-manager.service"},
        desktop_allowlist={"kwin_x11"},
        **kwargs,
    )


def _sample(ts, *, pid=10, starttime=100, client_id="7", gfx=0, compute=0,
            comm="renderer", unit="", device=GPU0):
    return {
        "sampled_at": ts,
        "clients": [{
            "device": device,
            "drm_client_id": client_id,
            "engine_gfx_ns": gfx,
            "engine_compute_ns": compute,
            "holders": [{"pid": pid, "starttime": starttime, "comm": comm,
                         "cmdline": comm, "unit": unit}],
        }],
        "kfd_holders": [],
        "unreadable": [],
    }


def test_engine_delta_needs_consecutive_known_observations_and_positive_increase():
    state = _state()
    state.record(_sample(1, gfx=100))
    assert state.active_clients(since=2) == []  # no baseline delta
    state.record(_sample(2, gfx=140))
    active = state.active_clients(since=2)
    assert len(active) == 1
    assert active[0]["engine_delta_ns"] == 40
    state.record(_sample(3, gfx=None))
    state.record(_sample(4, gfx=180))
    assert state.active_clients(since=2) == []  # missing counter breaks continuity
    state.record(_sample(5, gfx=170))
    assert state.active_clients(since=2) == []  # regressed counter is unknown


def test_reappeared_client_requires_a_new_counter_baseline():
    state = _state()
    state.record(_sample(1, gfx=100, compute=200))
    state.record({"sampled_at": 2, "clients": [], "kfd_holders": [], "unreadable": []})
    state.record(_sample(3, gfx=140, compute=260))
    assert state.active_clients(since=2) == []  # absence breaks continuity
    state.record(_sample(4, gfx=150, compute=280))
    active = state.active_clients(since=2)
    assert len(active) == 1
    assert active[0]["engine_delta_ns"] == 30


def test_ring_eviction_bounds_samples_pids_and_process_incarnations():
    state = _state(max_pids=2, max_samples=2)
    state.record(_sample(1, pid=1, starttime=10, gfx=1))
    state.record(_sample(2, pid=1, starttime=10, gfx=2))
    state.record(_sample(3, pid=1, starttime=10, gfx=3))
    assert state.sample_count <= 2
    state.record(_sample(4, pid=2, starttime=20, gfx=1))
    state.record(_sample(5, pid=3, starttime=30, gfx=1))
    assert state.pid_count <= 2
    # PID reuse must not bridge engine counters across process starttimes.
    state = _state()
    state.record(_sample(1, pid=9, starttime=100, gfx=500))
    state.record(_sample(2, pid=9, starttime=200, gfx=900))
    assert state.active_clients(since=2) == []


def test_classification_is_exact_and_foreign_warning_once_per_incarnation(caplog):
    state = _state()
    managed = _sample(1, comm="managed", unit="gpu-manager.service")["clients"][0]
    desktop = _sample(1, comm="kwin_x11")["clients"][0]
    foreign = _sample(1, pid=44, starttime=440, comm="unknown-app")["clients"][0]
    assert state.classify(managed) == "managed"
    assert state.classify(desktop) == "desktop"
    assert state.classify(foreign) == "foreign"
    state.record({"sampled_at": 1, "clients": [foreign], "kfd_holders": [], "unreadable": []})
    state.record({"sampled_at": 2, "clients": [foreign], "kfd_holders": [], "unreadable": []})
    assert sum("unknown-app" in r.message for r in caplog.records) == 1
    new_incarnation = _sample(3, pid=44, starttime=441, comm="unknown-app")["clients"][0]
    state.record({"sampled_at": 3, "clients": [new_incarnation], "kfd_holders": [], "unreadable": []})
    assert sum("unknown-app" in r.message for r in caplog.records) == 2


def test_desktop_membership_requires_allowlisted_process_and_foreign_is_not_managed():
    state = _state()
    desktop = _sample(1, comm="kwin_x11")["clients"][0]
    wrong_unit = _sample(1, comm="other", unit="gpu-managerish.service")["clients"][0]
    assert state.classify(desktop) == "desktop"
    assert state.classify(wrong_unit) == "foreign"


def test_proc_stat_starttime_parser_handles_spaces_and_right_parenthesis():
    # After comm, field 3 is state; fields 4-21 precede starttime (field 22).
    stat = "812 (renderer with ) paren) S " + " ".join(str(i) for i in range(1, 19)) + " 987654 22 23\n"
    assert gpu_guard.parse_proc_stat_starttime(stat) == 987654
    with pytest.raises((ValueError, IndexError)):
        gpu_guard.parse_proc_stat_starttime("812 (broken) S 1 2")


# Fault/receipt contract. These fixtures are copied from the observed kernel
# messages in gpu-guard-spec.md; clients are correlated in time, never blamed.
FAULT_LINES = (
    "[drm:gfx_v10_0_bad_op_irq [amdgpu]] *ERROR* Illegal opcode in command stream ",
    "amdgpu 0000:03:00.0: amdgpu: MODE1 reset",
    "GPU mode1 reset",
    "amdgpu: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=2",
    "amdgpu: GPU Hang detected",
    "amdgpu: gpu reset begin",
    # Observed crash signature 2026-10-07 (journalctl -k -b -1): each marker alone must trigger.
    "[drm:gfx_v10_0_bad_op_irq [amdgpu]] *ERROR* unexpected bad op",
    "amdgpu 0000:03:00.0: amdgpu: Queue preemption failed for queue with doorbell_id: 80006000",
)


@pytest.mark.parametrize("line", FAULT_LINES)
def test_matches_fault_recognizes_fault_signatures_case_insensitively(line):
    assert gpu_guard.matches_fault(line)
    assert gpu_guard.matches_fault(line.swapcase())


@pytest.mark.parametrize("line", [
    "amdgpu: initialized successfully",
    "amdgpu 0000:03:00.0: ring test completed",
    "GPU temperature is normal",
    # Benign init lines seen at every boot and driver re-init (journalctl -k -b 0).
    "amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 uses VM inv eng 0 on hub 0",
    "amdgpu 0000:75:00.0: amdgpu: ring gfx_0.1.0 uses VM inv eng 1 on hub 0",
])
def test_matches_fault_rejects_normal_amdgpu_messages(line):
    assert not gpu_guard.matches_fault(line)


def _receipt_snapshot():
    return {
        "sampled_at": 110.0,
        "clients": [
            {"device": GPU0, "drm_client_id": "1", "engine_gfx_ns": 140,
             "engine_compute_ns": 20, "total_vram_kib": 512,
             "holders": [{"pid": 101, "starttime": 1001, "comm": "kwin_x11",
                          "cmdline": "kwin_x11", "unit": ""}]},
            {"device": GPU1, "drm_client_id": "2", "engine_gfx_ns": 50,
             "engine_compute_ns": 400, "total_vram_kib": 2048,
             "holders": [{"pid": 202, "starttime": 2002, "comm": "compute",
                          "cmdline": "compute", "unit": ""}]},
        ],
        "kfd_holders": [{"pid": 202, "starttime": 2002, "comm": "compute",
                         "cmdline": "compute", "unit": ""}],
        "unreadable": [{"pid": 303, "comm": "lact", "reason": "permission_denied"}],
    }


def test_build_receipt_captures_10_seconds_ranks_last_2_seconds_deterministically():
    state = _state(max_samples=20)
    # Out of-window entries must not leak into the receipt; active clients in
    # the final two seconds precede inactive holders. Ties have stable ordering.
    for sample in (
        _sample(99.9, pid=900, client_id="old", gfx=1),
        _sample(100.0, pid=22, client_id="tie-b", gfx=10),
        _sample(100.0, pid=11, client_id="tie-a", gfx=10),
        _sample(109.0, pid=22, client_id="tie-b", gfx=30),
        _sample(109.0, pid=11, client_id="tie-a", gfx=30),
        _sample(109.0, pid=33, client_id="not-active", gfx=50),
        _sample(110.0, pid=33, client_id="not-active", gfx=50),
    ):
        state.record(sample)

    receipt = gpu_guard.build_receipt(
        matched_lines=[FAULT_LINES[0]], state=state,
        snapshot=_receipt_snapshot(), loadavg=[0.25, 0.5, 0.75],
        wall_time="2030-01-02T03:04:05Z",
    )
    assert receipt["matched_lines"] == [FAULT_LINES[0]]
    assert receipt["wall_time"] == "2030-01-02T03:04:05Z"
    assert all(100.0 <= entry["sampled_at"] <= 110.0
               for entry in receipt["history"])
    ranked = receipt["active_clients"]
    assert [(item["drm_client_id"], item["pid"]) for item in ranked[:2]] == [
        ("tie-a", 11), ("tie-b", 22)
    ]
    assert "not-active" not in {item["drm_client_id"] for item in ranked}
    assert receipt["holders_by_node"][GPU0][0]["pid"] == 101
    assert receipt["holders_by_node"][GPU1][0]["pid"] == 202
    assert receipt["kfd_holders"][0]["pid"] == 202
    assert receipt["unreadable"][0]["pid"] == 303
    assert receipt["loadavg"] == [0.25, 0.5, 0.75]
    assert receipt["attribution"] == "temporal_association_not_causation"


def test_fault_recorder_uses_fixed_30_second_burst_and_appends_later_lines(tmp_path):
    now = [100.0]
    wall = ["2030-01-02T03:04:05Z"]
    recorder = gpu_guard.FaultRecorder(
        tmp_path / "guard", monotonic_clock=lambda: now[0],
        wall_clock=lambda: wall[0],
    )
    first = recorder.record(FAULT_LINES[0], {"sequence": 1})
    now[0] = 129.999
    wall[0] = "2030-01-02T03:04:34Z"
    same_burst = recorder.record(FAULT_LINES[1], {"sequence": 2})
    assert same_burst == first
    now[0] = 130.0
    wall[0] = "2030-01-02T03:04:35Z"
    second_burst = recorder.record(FAULT_LINES[2], {"sequence": 3})
    assert second_burst != first

    import json
    first_doc = json.loads(first.read_text())
    second_doc = json.loads(second_burst.read_text())
    assert first_doc["matched_lines"] == [FAULT_LINES[0], FAULT_LINES[1]]
    assert first_doc["events"] == [{"sequence": 1}, {"sequence": 2}]
    assert second_doc["matched_lines"] == [FAULT_LINES[2]]


def test_fault_recorder_permissions_symlink_rejection_and_size_caps(tmp_path):
    import json
    import stat

    root = tmp_path / "guard"
    recorder = gpu_guard.FaultRecorder(
        root, monotonic_clock=lambda: 10.0,
        wall_clock=lambda: "2030-01-02T03:04:05Z",
    )
    receipt = recorder.record(FAULT_LINES[0], {"ok": True})
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert receipt.stat().st_size <= 8 * 1024 * 1024

    with pytest.raises(ValueError):
        recorder.record("x" * (64 * 1024 + 1), {})
    with pytest.raises(ValueError):
        recorder.record(FAULT_LINES[0], {"payload": "x" * (8 * 1024 * 1024)})
    # The fixed burst cap is measured in lines, not in elapsed time.
    bounded = gpu_guard.FaultRecorder(
        tmp_path / "bounded", monotonic_clock=lambda: 10.0,
        wall_clock=lambda: "2030-01-02T03:04:05Z",
    )
    for index in range(256):
        bounded.record(FAULT_LINES[0], {"sequence": index})
    with pytest.raises(ValueError):
        bounded.record(FAULT_LINES[0], {"sequence": 256})

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"untouched": True}))
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(outside)
    unsafe = gpu_guard.FaultRecorder(
        linked_root, monotonic_clock=lambda: 10.0,
        wall_clock=lambda: "2030-01-02T03:04:05Z",
    )
    with pytest.raises((OSError, ValueError)):
        unsafe.record(FAULT_LINES[0], {})
    assert json.loads(outside.read_text()) == {"untouched": True}


def test_fault_recorder_contains_atomic_write_failure(tmp_path, monkeypatch, caplog):
    recorder = gpu_guard.FaultRecorder(
        tmp_path / "guard", monotonic_clock=lambda: 10.0,
        wall_clock=lambda: "2030-01-02T03:04:05Z",
    )

    def fail_replace(*args, **kwargs):
        raise OSError("synthetic atomic replace failure")

    monkeypatch.setattr(gpu_guard.os, "replace", fail_replace)
    result = recorder.record(FAULT_LINES[0], {"sequence": 1})
    assert result is None
    assert recorder.error_count == 1
    assert any("synthetic atomic replace failure" in record.message
               for record in caplog.records)
