#!/usr/bin/env python3
"""Hermetic fixture tests for ops/r139_deploy.sh; no live service calls."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import pytest
import sqlite3
import subprocess
import tempfile
import textwrap
import threading

WORKSPACE = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = WORKSPACE / "deployment" / "oom_inflight_deploy.sh"
COMMIT = "6ce2400bb90dea20d820c10f279b62c4bf827544"
STALE_COMMIT = "6baae36aac905e325b6296b6960738a11813bd4b"
PREVIOUS = "gpu-manager-unified-6baae36-r139-20261009T212514Z"
API_LINE = "LoadCredentialEncrypted=gpu-manager-api-token:/etc/credstore.encrypted/gpu-manager-api-token.cred"


def put(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def make_fixture(root: Path) -> dict[str, str | Path]:
    ctrl = root / "controller"
    previous = ctrl / "releases" / PREVIOUS
    (previous / "scripts").mkdir(parents=True)
    put(previous / "scripts" / "gpu-manager.py", "# previous release fixture\n")
    (ctrl / "current").symlink_to(f"releases/{PREVIOUS}")

    export = root / "export"
    release_entry = "scripts/gpu-manager.py"
    release_payload = "# candidate release fixture\n"
    put(export / release_entry, release_payload, executable=True)
    manifest = {
        "files": [
            {
                "path": release_entry,
                "sha256": hashlib.sha256(release_payload.encode()).hexdigest(),
                "size": len(release_payload.encode()),
            }
        ],
        "private_commit_sha": COMMIT,
        "public_commit_sha": STALE_COMMIT,
        "schema_version": "gpu-manager.assembled-release.v1",
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    put(export / "provenance" / "assembled-manifest.json", manifest_bytes.decode())
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    candidate = root / "candidate-repository"
    candidate.mkdir(parents=True)
    rollback_unit = textwrap.dedent(
        """\
        [Unit]
        Description=GPU Manager rollback fixture

        [Service]
        WorkingDirectory=/fixture/previous/scripts
        LoadCredentialEncrypted=redis-password:/etc/credstore.encrypted/redis-password.cred
        ExecStart=/usr/bin/python3 /fixture/previous/scripts/gpu-manager.py

        [Install]
        WantedBy=multi-user.target
        """
    )
    put(candidate / "rollback" / "systemd" / "gpu-manager.service", rollback_unit)
    candidate_unit_sha = hashlib.sha256(rollback_unit.encode()).hexdigest()

    live_unit = textwrap.dedent(
        f"""\
        [Unit]
        Description=GPU Manager live fixture

        [Service]
        WorkingDirectory=/fixture/current/scripts
        LoadCredentialEncrypted=redis-password:/etc/credstore.encrypted/redis-password.cred
        {API_LINE}
        ExecStart=/usr/bin/python3 /fixture/current/scripts/gpu-manager.py

        [Install]
        WantedBy=multi-user.target
        """
    )
    live_unit_path = root / "systemd" / "gpu-manager.service"
    put(live_unit_path, live_unit)
    dropins = root / "systemd" / "gpu-manager.service.d"
    put(dropins / "10-fixture.conf", "[Service]\nEnvironment=SAFE=1\n")
    services = root / "services.json"
    put(services, json.dumps({"services": {"comfyui": {"systemd_unit": "comfyui.service"}}}) + "\n")
    services_sha = hashlib.sha256(services.read_bytes()).hexdigest()
    live_unit_sha = hashlib.sha256(live_unit_path.read_bytes()).hexdigest()
    dropin_sha = hashlib.sha256((dropins / "10-fixture.conf").read_bytes()).hexdigest()

    leases = root / "leases.db"
    with sqlite3.connect(leases) as connection:
        connection.execute("CREATE TABLE leases (key TEXT, expires REAL)")
    preflight = {
        "/health": {"active_requests": 0, "queued_requests": 0},
        "/v1/queue": {"services": {"comfyui": {"queue_depth": 0, "slots_active": 0}}, "routing_groups": {}},
        "/v1/workers": {"pools": [{"slots_active": 0}], "total_active": 0},
        "/v1/job/38ca767a": {"service_name": "comfyui", "status": "cancelled"},
        "/v1/metrics/broker": {
            "schema": "combined-gemma.broker-metric.v1",
            "queue": {"depth": 0, "fresh": True},
            "jobs": {"depth": 0, "total": 0, "by_state": {}},
            "members": [{"fresh": True, "live_lease_count": 0, "dispatcher_leases": 0}],
        },
    }
    preflight_path = root / "preflight.json"
    put(preflight_path, json.dumps(preflight) + "\n")

    stubs = root / "stubs"
    put(
        stubs / "sudo",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'sudo invoked\\n' >> "$OOM_INFLIGHT_TEST_ROOT/calls.log"
if [[ ${1:-} == -n ]]; then shift; fi
exec "$@"
""",
        executable=True,
    )
    put(
        stubs / "systemctl",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'systemctl %s\\n' "$*" >> "$OOM_INFLIGHT_TEST_ROOT/calls.log"
case "${1:-}" in
  restart)
    pidfile="$OOM_INFLIGHT_TEST_ROOT/service.pid"
    if [[ -f "$pidfile" ]]; then old=$(<"$pidfile"); kill "$old" 2>/dev/null || true; fi
    scripts=$(readlink -f "$OOM_INFLIGHT_TEST_CTRL_DIR/current/scripts")
    (cd "$scripts" && exec sleep 90) >/dev/null 2>&1 &
    printf '%s\\n' "$!" > "$pidfile"
    ;;
  show)
    if [[ " $* " == *" -p ActiveState "* ]]; then
      printf 'ActiveState=%s\\nSubState=dead\\nResult=success\\nExecMainStatus=0\\n' "${OOM_INFLIGHT_TEST_ACTIVE_STATE:-inactive}"
      exit 0
    fi
    [[ -f "$OOM_INFLIGHT_TEST_ROOT/service.pid" ]] || exit 1
    cat "$OOM_INFLIGHT_TEST_ROOT/service.pid"
    ;;
  *) exit 2 ;;
esac
""",
        executable=True,
    )
    put(
        stubs / "curl",
        f"""#!/usr/bin/env bash
set -euo pipefail
url=${{!#}}
current=$(readlink -f "$OOM_INFLIGHT_TEST_CTRL_DIR/current")
code=200
if [[ ${{OOM_INFLIGHT_TEST_MODE_SCENARIO:-success}} == health-failure && "$url" == http://127.0.0.1:8091/health/live && "$current" != *{PREVIOUS} ]]; then
  code=503
fi
printf 'curl %s -> %s\\n' "$url" "$code" >> "$OOM_INFLIGHT_TEST_ROOT/calls.log"
printf '%s' "$code"
""",
        executable=True,
    )
    put(
        stubs / "git",
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ ${{1:-}} == -C ]]; then shift 2; fi
if [[ ${{1:-}} == rev-parse && ${{2:-}} == HEAD ]]; then
  printf '%s\\n' '{COMMIT}'
elif [[ ${{1:-}} == status && ${{2:-}} == --porcelain ]]; then
  exit 0
else
  exit 2
fi
""",
        executable=True,
    )
    put(
        stubs / "systemd-analyze",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'systemd-analyze %s\\n' \"$*\" >> \"$OOM_INFLIGHT_TEST_ROOT/calls.log\"\nexit 0\n",
        executable=True,
    )
    put(
        stubs / "journalctl",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'journalctl %s\\n' \"$*\" >> \"$OOM_INFLIGHT_TEST_ROOT/calls.log\"\nif [[ -f \"$OOM_INFLIGHT_TEST_ROOT/kernel.txt\" ]]; then cat \"$OOM_INFLIGHT_TEST_ROOT/kernel.txt\"; fi\nexit 0\n",
        executable=True,
    )
    return {
        "ctrl": ctrl,
        "export": export,
        "candidate": candidate,
        "live_unit": live_unit_path,
        "dropins": dropins,
        "services": services,
        "leases": leases,
        "preflight": preflight_path,
        "manifest_sha": manifest_sha,
        "services_sha": services_sha,
        "live_unit_sha": live_unit_sha,
        "dropin_sha": dropin_sha,
        "candidate_unit_sha": candidate_unit_sha,
    }


def run_deploy(root: Path, fixture: dict[str, str | Path], scenario: str, cwd: Path = WORKSPACE, arguments: tuple[str, ...] = (), extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / 'stubs'}:/usr/bin:/bin",
            "OOM_INFLIGHT_TEST_MODE": "1",
            "OOM_INFLIGHT_TEST_ROOT": str(root),
            "OOM_INFLIGHT_TEST_CTRL_DIR": str(fixture["ctrl"]),
            "OOM_INFLIGHT_TEST_RELEASE_PATH": str(fixture["export"]),
            "OOM_INFLIGHT_TEST_CANDIDATE_REPOSITORY": str(fixture["candidate"]),
            "OOM_INFLIGHT_TEST_RELEASE_PATH": str(fixture["export"]),
            "OOM_INFLIGHT_TEST_LIVE_UNIT": str(fixture["live_unit"]),
            "OOM_INFLIGHT_TEST_DROPINS_DIR": str(fixture["dropins"]),
            "OOM_INFLIGHT_TEST_SERVICES_JSON": str(fixture["services"]),
            "OOM_INFLIGHT_TEST_LEASE_DB": str(fixture["leases"]),
            "OOM_INFLIGHT_TEST_PREFLIGHT_FIXTURE": str(fixture["preflight"]),
            "OOM_INFLIGHT_TEST_EXPECTED_COMMIT": COMMIT,
            "OOM_INFLIGHT_TEST_EXPECTED_PUBLIC_COMMIT": STALE_COMMIT,
            "OOM_INFLIGHT_TEST_EXPECTED_MANIFEST_SHA": str(fixture["manifest_sha"]),
            "OOM_INFLIGHT_TEST_C3_SERVICES_SHA": str(fixture["services_sha"]),
            "OOM_INFLIGHT_TEST_MODE_SCENARIO": scenario,
            "OOM_INFLIGHT_TEST_NOW": "2026-10-11T12:00:00",
        }
    )
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), *arguments],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def cleanup_service(root: Path) -> None:
    pidfile = root / "service.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text().strip()), 15)
        except (ProcessLookupError, ValueError):
            pass


def assert_common_safe(fixture: dict[str, str | Path], root: Path) -> None:
    assert hashlib.sha256(Path(fixture["services"]).read_bytes()).hexdigest() == fixture["services_sha"]
    assert hashlib.sha256(Path(fixture["live_unit"]).read_bytes()).hexdigest() == fixture["live_unit_sha"]
    assert hashlib.sha256((Path(fixture["dropins"]) / "10-fixture.conf").read_bytes()).hexdigest() == fixture["dropin_sha"]
    calls = (root / "calls.log").read_text(encoding="utf-8")
    assert "/v1/jobs" not in calls and "catch-all" not in calls
    assert "LoadCredentialEncrypted" not in calls
    assert (Path(fixture["candidate"]) / "rollback/systemd/gpu-manager.service").exists()


def test_success_case() -> None:
    with tempfile.TemporaryDirectory(prefix="r139-deploy-success-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        try:
            result = run_deploy(root, fixture, "success")
            assert result.returncode == 0, result.stdout + result.stderr
            current = os.readlink(Path(fixture["ctrl"]) / "current")
            assert current.startswith(f"releases/gpu-manager-unified-{COMMIT[:7]}-r139-")
            deployed = Path(fixture["ctrl"]) / current
            repaired = (deployed / "rollback/systemd/gpu-manager.service").read_text(encoding="utf-8")
            assert repaired.count(API_LINE) == 1
            calls = (root / "calls.log").read_text(encoding="utf-8")
            assert calls.count("systemctl restart gpu-manager.service") == 1
            assert calls.count("sudo invoked") == 2
            assert "http://127.0.0.1:8091/health/live -> 200" in calls
            assert "http://127.0.0.1:8094/health -> 200" in calls
            snapshot = next((Path(fixture["ctrl"]) / "r139-snapshots").iterdir())
            rollback_unit = (deployed / "rollback/systemd/gpu-manager.service").read_text(encoding="utf-8")
            assert API_LINE in rollback_unit
            assert "secret-value" not in rollback_unit.lower()
            assert not (snapshot / "services.json").exists()
            assert_common_safe(fixture, root)
        finally:
            cleanup_service(root)


def test_health_failure_rolls_back_case() -> None:
    with tempfile.TemporaryDirectory(prefix="r139-deploy-rollback-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        try:
            result = run_deploy(root, fixture, "health-failure")
            assert result.returncode != 0, "candidate health failure unexpectedly succeeded"
            assert "automatic rollback verified" in result.stderr, result.stderr
            current = os.readlink(Path(fixture["ctrl"]) / "current")
            assert current == f"releases/{PREVIOUS}", current
            calls = (root / "calls.log").read_text(encoding="utf-8")
            assert calls.count("systemctl restart gpu-manager.service") == 2
            assert calls.count("sudo invoked") == 3
            assert f"http://127.0.0.1:8091/health/live -> 503" in calls
            assert "http://127.0.0.1:8091/health/live -> 200" in calls
            assert_common_safe(fixture, root)
        finally:
            cleanup_service(root)


def test_wrong_cwd_case() -> None:
    with tempfile.TemporaryDirectory(prefix="r139-deploy-wrong-cwd-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        try:
            result = run_deploy(root, fixture, "success", cwd=root)
            assert result.returncode != 0
            assert "wrong cwd" in result.stderr
            assert not (root / "calls.log").exists(), "wrong-cwd guard invoked a stubbed operation"
            assert os.readlink(Path(fixture["ctrl"]) / "current") == f"releases/{PREVIOUS}"
        finally:
            cleanup_service(root)


def test_self_consistent_stale_payload_case() -> None:
    with tempfile.TemporaryDirectory(prefix="r139-deploy-stale-payload-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        export = Path(fixture["export"])
        manifest_path = export / "provenance" / "assembled-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["private_commit_sha"] = STALE_COMMIT

        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        manifest_path.write_bytes(manifest_bytes)
        fixture["manifest_sha"] = hashlib.sha256(manifest_bytes).hexdigest()
        # Payload entries still hash correctly; only the commit binding is stale.
        for entry in manifest["files"]:
            payload = export / entry["path"]
            assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry["sha256"]
            assert payload.stat().st_size == entry["size"]
        try:
            result = run_deploy(root, fixture, "success")
            assert result.returncode != 0, "self-consistent stale payload unexpectedly passed"
            assert "assembled release commit identity mismatch" in result.stderr, result.stderr
            assert not (root / "calls.log").exists(), "stale payload reached host commands"
            assert os.readlink(Path(fixture["ctrl"]) / "current") == f"releases/{PREVIOUS}"
        finally:
            cleanup_service(root)


def test_read_only_preflight_case() -> None:
    with tempfile.TemporaryDirectory(prefix="oom-inflight-preflight-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        ctrl = Path(fixture["ctrl"])
        original_link = os.readlink(ctrl / "current")
        result = run_deploy(root, fixture, "success", arguments=("--preflight",))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "read-only authenticated preflight passed" in result.stdout
        assert os.readlink(ctrl / "current") == original_link
        assert not (ctrl / "r139-snapshots").exists()
        calls = (root / "calls.log").read_text(encoding="utf-8")
        assert "systemctl restart" not in calls
        assert hashlib.sha256(Path(fixture["services"]).read_bytes()).hexdigest() == fixture["services_sha"]


def test_live_execution_gate_case() -> None:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("OOM_INFLIGHT_TEST_"):
            env.pop(name)
    env.pop("OOM_INFLIGHT_SUPERVISOR_REVIEW", None)
    env["OOM_INFLIGHT_TEST_MODE"] = "0"
    for args, message in (([], "use --preflight or --execute explicitly"), (["--execute"], "separate operator and Claude Code RELEASE GO evidence")):
        result = subprocess.run(
            ["bash", str(DEPLOY_SCRIPT), *args],
            cwd=WORKSPACE,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert result.returncode != 0
        assert message in result.stderr, result.stderr


def test_kernel_marker_case() -> None:
    # Benign kernel lines seen live since boot (116 matched the old 'ring' pattern) must not roll back;
    # a real gfx fault must.
    benign = "fbcon: Deferring console take-over\nvethf863a0e (unregistering): left promiscuous mode\n"
    fault = "amdgpu 0000:03:00.0: amdgpu: gfx_v10_0_bad_op_irq: Illegal opcode in command stream\n"
    ring = "amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=2\n"
    for lines, should_deploy in ((benign, True), (benign + fault, False), (benign + ring, False)):
        with tempfile.TemporaryDirectory(prefix="r139-deploy-kernel-") as tmp:
            root = Path(tmp)
            fixture = make_fixture(root)
            put(root / "kernel.txt", lines)
            try:
                result = run_deploy(root, fixture, "success")
                current = os.readlink(Path(fixture["ctrl"]) / "current")
                if should_deploy:
                    assert result.returncode == 0, result.stdout + result.stderr
                    assert current.startswith(f"releases/gpu-manager-unified-{COMMIT[:7]}-r139-"), current
                else:
                    assert result.returncode != 0
                    assert "kernel GPU fault" in result.stderr, result.stderr
                    assert current == f"releases/{PREVIOUS}", current
                assert_common_safe(fixture, root)
            finally:
                cleanup_service(root)


def test_dynamic_rollback_target_case() -> None:
    """The runner resolves the prior release at execution time, not from a stale pin."""
    with tempfile.TemporaryDirectory(prefix="oom-inflight-dynamic-rollback-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        ctrl = Path(fixture["ctrl"])
        alternate = ctrl / "releases" / "gpu-manager-unified-prior-r140-20261011T000000Z"
        (alternate / "scripts").mkdir(parents=True)
        put(alternate / "scripts" / "gpu-manager.py", "# runtime prior release fixture\n")
        (ctrl / "current").unlink()
        (ctrl / "current").symlink_to("releases/gpu-manager-unified-prior-r140-20261011T000000Z")
        try:
            result = run_deploy(root, fixture, "success")
            assert result.returncode == 0, result.stdout + result.stderr
            current = os.readlink(ctrl / "current")
            assert current.startswith(f"releases/gpu-manager-unified-{COMMIT[:7]}-r139-")
            snapshot = next((ctrl / "r139-snapshots").iterdir())
            before = (snapshot / "preflight.txt").read_text(encoding="utf-8")
            assert "current_link=releases/gpu-manager-unified-prior-r140-20261011T000000Z" in before
        finally:
            cleanup_service(root)


def test_authenticated_preflight_case() -> None:
    with tempfile.TemporaryDirectory(prefix="oom-inflight-auth-preflight-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        responses = json.loads(Path(fixture["preflight"]).read_text(encoding="utf-8"))
        responses.pop("/v1/job/38ca767a")
        auth_seen: list[tuple[str, str | None]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                auth_seen.append((self.path, self.headers.get("Authorization")))
                payload = json.dumps(responses[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        credential = root / "api-token"
        put(credential, "synthetic-fixture-token\n")
        helper = WORKSPACE / "deployment" / "oom_inflight_preflight.py"
        env = os.environ.copy()
        env["PATH"] = f"{root / 'stubs'}:/usr/bin:/bin"
        env["OOM_INFLIGHT_TEST_ROOT"] = str(root)
        try:
            result = subprocess.run(
                [
                    "python3", str(helper),
                    "--auth-source-dir", str(WORKSPACE / "scripts"),
                    "--services-json", str(fixture["services"]),
                    "--lease-db", str(fixture["leases"]),
                    "--credential-file", str(credential),
                    "--base-url", f"http://127.0.0.1:{server.server_port}",
                ],
                cwd=WORKSPACE,
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert len(auth_seen) == 4, auth_seen
            assert {route for route, _ in auth_seen} == {
                "/health", "/v1/queue", "/v1/workers", "/v1/metrics/broker"
            }
            assert all(header == "Bearer synthetic-fixture-token" for _, header in auth_seen)
            assert "synthetic-fixture-token" not in result.stdout + result.stderr
            assert "local active leases=0" in result.stdout
            assert "image job=" not in result.stdout
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payloads, leases: payloads["/health"].__setitem__("queued_requests", 1), "queued_requests is nonzero"),
        (lambda payloads, leases: payloads["/v1/queue"]["services"]["comfyui"].__setitem__("queue_depth", 1), "queue_depth is nonzero"),
        (lambda payloads, leases: payloads["/v1/workers"]["pools"][0].__setitem__("slots_active", 1), "slots_active is nonzero"),
        (lambda payloads, leases: payloads["/v1/metrics/broker"]["members"][0].__setitem__("live_lease_count", 1), "live_lease_count is nonzero"),
        (lambda payloads, leases: leases.execute("INSERT INTO leases VALUES ('fixture', 9999999999)"), "local active lease count is nonzero"),
    ],
)
def test_preflight_nonzero_state_refusal_case(mutate, message: str) -> None:
    with tempfile.TemporaryDirectory(prefix="oom-inflight-preflight-busy-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        responses = json.loads(Path(fixture["preflight"]).read_text(encoding="utf-8"))
        with sqlite3.connect(fixture["leases"]) as leases:
            mutate(responses, leases)
        Path(fixture["preflight"]).write_text(json.dumps(responses) + "\n", encoding="utf-8")
        original_link = os.readlink(Path(fixture["ctrl"]) / "current")
        result = run_deploy(root, fixture, "success", arguments=("--preflight",))
        assert result.returncode != 0
        assert message in result.stderr
        assert os.readlink(Path(fixture["ctrl"]) / "current") == original_link
        assert not (Path(fixture["ctrl"]) / "r139-snapshots").exists()
        calls = (root / "calls.log").read_text(encoding="utf-8")
        assert "systemctl restart" not in calls


@pytest.mark.parametrize("active_state", ["active", "activating"])
def test_preflight_active_backend_refusal_case(active_state: str) -> None:
    with tempfile.TemporaryDirectory(prefix="oom-inflight-preflight-active-backend-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        result = run_deploy(
            root, fixture, "success", arguments=("--preflight",),
            extra_env={"OOM_INFLIGHT_TEST_ACTIVE_STATE": active_state},
        )
        assert result.returncode != 0
        assert "backend unit comfyui.service is not inactive" in result.stderr
        calls = (root / "calls.log").read_text(encoding="utf-8")
        assert "systemctl restart" not in calls


def test_preflight_wrong_time_refusal_case() -> None:
    with tempfile.TemporaryDirectory(prefix="oom-inflight-preflight-time-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        original_link = os.readlink(Path(fixture["ctrl"]) / "current")
        result = run_deploy(root, fixture, "success", arguments=("--preflight",), extra_env={"OOM_INFLIGHT_TEST_NOW": "2026-10-11T03:30:00"})
        assert result.returncode != 0
        assert "03:15-03:55" in result.stderr
        assert os.readlink(Path(fixture["ctrl"]) / "current") == original_link
        calls = (root / "calls.log").read_text(encoding="utf-8")
        assert "systemctl restart" not in calls


def test_optional_job_id_validation_is_fail_closed() -> None:
    with tempfile.TemporaryDirectory(prefix="oom-inflight-optional-job-") as tmp:
        root = Path(tmp)
        fixture = make_fixture(root)
        payloads = json.loads(Path(fixture["preflight"]).read_text(encoding="utf-8"))
        payloads.pop("/v1/job/38ca767a")
        helper = WORKSPACE / "deployment" / "oom_inflight_preflight.py"
        spec = importlib.util.spec_from_file_location("oom_inflight_preflight_optional", helper)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.validate_observations(payloads)["job_status"] == "not_requested"
        payloads["/v1/job/custom-42"] = {"service_name": "comfyui", "status": "completed"}
        assert module.validate_observations(payloads, "custom-42")["job_status"] == "completed"
        for job in (
            None,
            {"service_name": "comfyui", "status": "running"},
            {"error": "missing"},
        ):
            if job is None:
                payloads.pop("/v1/job/custom-42")
            else:
                payloads["/v1/job/custom-42"] = job
            with pytest.raises(module.PreflightError):
                module.validate_observations(payloads, "custom-42")


def test_preflight_window_bounds_case() -> None:
    module_path = WORKSPACE / "deployment" / "oom_inflight_preflight.py"
    spec = importlib.util.spec_from_file_location("oom_inflight_preflight", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.in_disallowed_window(datetime(2026, 10, 11, 3, 14))
    assert module.in_disallowed_window(datetime(2026, 10, 11, 3, 15))
    assert module.in_disallowed_window(datetime(2026, 10, 11, 3, 55))
    assert not module.in_disallowed_window(datetime(2026, 10, 11, 3, 56))


def main() -> int:
    # Delegate to pytest so this standalone entrypoint exercises exactly the same
    # collection as `pytest tests/publication/test_oom_inflight_deploy.py`. A
    # hand-maintained call list silently skipped parametrized cases and printed a
    # hardcoded count that did not match the tests it actually ran.
    return int(pytest.main([__file__, "-p", "no:cacheprovider", "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
