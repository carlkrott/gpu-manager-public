#!/usr/bin/env bash
# r139 S1/S3/S4 one-deploy runner. Live execution requires a later supervisor release.
set -Eeuo pipefail
umask 077

fail() {
  printf 'r139-deploy: %s\n' "$*" >&2
  exit 1
}

SCRIPT_PATH=$(realpath -e -- "${BASH_SOURCE[0]}") || fail 'cannot resolve script path'
WORKSPACE=$(dirname -- "$(dirname -- "$SCRIPT_PATH")")
ACTUAL_CWD=$(pwd -P)
[[ "$ACTUAL_CWD" == "$WORKSPACE" ]] || fail "wrong cwd: expected $WORKSPACE, got $ACTUAL_CWD"

EXPECTED_COMMIT=${OOM_INFLIGHT_CANDIDATE_COMMIT:-}
EXPECTED_PUBLIC_COMMIT=${OOM_INFLIGHT_PUBLIC_COMMIT:-${OOM_INFLIGHT_TEST_EXPECTED_PUBLIC_COMMIT:-}}
EXPECTED_MANIFEST_SHA=${OOM_INFLIGHT_MANIFEST_SHA:-${OOM_INFLIGHT_TEST_EXPECTED_MANIFEST_SHA:-}}
EXPECTED_C3_SERVICES_SHA=${OOM_INFLIGHT_C3_SERVICES_SHA:-${OOM_INFLIGHT_TEST_C3_SERVICES_SHA:-}}
MANAGER_HEALTH_URL=http://127.0.0.1:8091/health/live
GEMMA_HEALTH_URL=http://127.0.0.1:8094/health
TEST_MODE=${OOM_INFLIGHT_TEST_MODE:-0}
if [[ "$TEST_MODE" == 0 ]]; then
  [[ $# == 1 && ( "$1" == --preflight || "$1" == --execute ) ]] || fail 'use --preflight or --execute explicitly'
  if [[ "$1" == --execute ]]; then
    [[ "${OOM_INFLIGHT_OPERATOR_GO:-}" == present && "${OOM_INFLIGHT_CLAUDE_GO:-}" == present ]] || fail 'deployment requires separate operator and Claude Code RELEASE GO evidence'
    [[ "${OOM_INFLIGHT_PIBOOK_CHECK:-}" == clear ]] || fail 'deployment requires a fresh pibook lease check and operator message'
  fi
elif [[ "$TEST_MODE" == 1 ]]; then
  [[ $# == 0 || ( $# == 1 && "$1" == --preflight ) ]] || fail 'fixture mode accepts only --preflight'
else
  fail 'OOM_INFLIGHT_TEST_MODE must be 0 or 1'
fi

if [[ "$TEST_MODE" == 1 ]]; then
  [[ -n "${OOM_INFLIGHT_TEST_ROOT:-}" ]] || fail 'test mode requires OOM_INFLIGHT_TEST_ROOT'
  TEST_ROOT=$(realpath -e -- "$OOM_INFLIGHT_TEST_ROOT") || fail 'cannot resolve test root'
  [[ -d "$TEST_ROOT" ]] || fail 'test root is not a directory'
  EXPECTED_COMMIT=${OOM_INFLIGHT_TEST_EXPECTED_COMMIT:-$EXPECTED_COMMIT}
  EXPECTED_PUBLIC_COMMIT=${OOM_INFLIGHT_TEST_EXPECTED_PUBLIC_COMMIT:-$EXPECTED_PUBLIC_COMMIT}
  EXPECTED_MANIFEST_SHA=${OOM_INFLIGHT_TEST_EXPECTED_MANIFEST_SHA:-$EXPECTED_MANIFEST_SHA}
  EXPECTED_C3_SERVICES_SHA=${OOM_INFLIGHT_TEST_C3_SERVICES_SHA:-$EXPECTED_C3_SERVICES_SHA}
  CTRL_DIR=${OOM_INFLIGHT_TEST_CTRL_DIR:?test mode requires OOM_INFLIGHT_TEST_CTRL_DIR}
  EXPORT_DIR=${OOM_INFLIGHT_TEST_RELEASE_PATH:?test mode requires OOM_INFLIGHT_TEST_RELEASE_PATH}
  CANDIDATE_REPO=${OOM_INFLIGHT_TEST_CANDIDATE_REPOSITORY:?test mode requires OOM_INFLIGHT_TEST_CANDIDATE_REPOSITORY}
  LIVE_UNIT=${OOM_INFLIGHT_TEST_LIVE_UNIT:?test mode requires OOM_INFLIGHT_TEST_LIVE_UNIT}
  DROPINS_DIR=${OOM_INFLIGHT_TEST_DROPINS_DIR:?test mode requires OOM_INFLIGHT_TEST_DROPINS_DIR}
  SERVICES_JSON=${OOM_INFLIGHT_TEST_SERVICES_JSON:?test mode requires OOM_INFLIGHT_TEST_SERVICES_JSON}
  LEASE_DB=${OOM_INFLIGHT_TEST_LEASE_DB:?test mode requires OOM_INFLIGHT_TEST_LEASE_DB}
  PREFLIGHT_FIXTURE=${OOM_INFLIGHT_TEST_PREFLIGHT_FIXTURE:?test mode requires OOM_INFLIGHT_TEST_PREFLIGHT_FIXTURE}
  for path in "$CTRL_DIR" "$EXPORT_DIR" "$CANDIDATE_REPO" "$LIVE_UNIT" "$DROPINS_DIR" "$SERVICES_JSON" "$LEASE_DB" "$PREFLIGHT_FIXTURE"; do
    resolved=$(realpath -m -- "$path") || fail "cannot resolve test path: $path"
    case "$resolved/" in
      "$TEST_ROOT"/*) ;;
      *) fail "test path escapes OOM_INFLIGHT_TEST_ROOT: $path" ;;
    esac
  done
  for command_name in sudo systemctl curl git systemd-analyze journalctl; do
    resolved=$(type -P "$command_name" || true)
    case "$resolved" in
      "$TEST_ROOT"/stubs/*) ;;
      *) fail "test mode requires a $command_name stub under $TEST_ROOT/stubs" ;;
    esac
  done
  MANAGER_WAIT_SECONDS=${OOM_INFLIGHT_TEST_MANAGER_WAIT_SECONDS:-1}
  GEMMA_WAIT_SECONDS=${OOM_INFLIGHT_TEST_GEMMA_WAIT_SECONDS:-1}
else
  [[ "$TEST_MODE" == 0 ]] || fail 'OOM_INFLIGHT_TEST_MODE must be 0 or 1'
  CTRL_DIR=${HOME:?HOME must be set}/.local/share/gpu-manager/controller
  EXPORT_DIR=${OOM_INFLIGHT_RELEASE_PATH:?set OOM_INFLIGHT_RELEASE_PATH to the assembled runtime release}
  CANDIDATE_REPO=${OOM_INFLIGHT_CANDIDATE_REPOSITORY:?set OOM_INFLIGHT_CANDIDATE_REPOSITORY to the candidate Git repository}
  LIVE_UNIT=/etc/systemd/system/gpu-manager.service
  DROPINS_DIR=/etc/systemd/system/gpu-manager.service.d
  SERVICES_JSON=${HOME:?HOME must be set}/.config/gpu-manager/services.json
  LEASE_DB=${HOME:?HOME must be set}/.hermes/locks/leases.db
  MANAGER_WAIT_SECONDS=120
  GEMMA_WAIT_SECONDS=300
fi

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 'invalid candidate commit pin'
[[ "$EXPECTED_PUBLIC_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 'invalid public provenance commit pin'
[[ "$EXPECTED_MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid manifest hash pin'
[[ -z "$EXPECTED_C3_SERVICES_SHA" || "$EXPECTED_C3_SERVICES_SHA" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid services hash pin'
[[ "$MANAGER_WAIT_SECONDS" =~ ^[1-9][0-9]*$ && "$GEMMA_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail 'health wait bounds must be positive integer seconds'

[[ -d "$CTRL_DIR" && -d "$CTRL_DIR/releases" ]] || fail 'controller release directories are missing'
[[ -d "$EXPORT_DIR" && -d "$CANDIDATE_REPO" ]] || fail 'candidate repository or assembled runtime release is missing'
[[ -f "$LIVE_UNIT" && ! -L "$LIVE_UNIT" ]] || fail 'live systemd unit is not a regular file'
[[ -f "$SERVICES_JSON" && ! -L "$SERVICES_JSON" ]] || fail 'C3 services.json is not a regular file'
actual_head=$(git -C "$CANDIDATE_REPO" rev-parse HEAD) || fail 'cannot read candidate HEAD'
[[ "$actual_head" == "$EXPECTED_COMMIT" ]] || fail "candidate HEAD mismatch: $actual_head"
candidate_status=$(git -C "$CANDIDATE_REPO" status --porcelain --untracked-files=all) || fail 'cannot read candidate status'
[[ -z "$candidate_status" ]] || fail 'candidate tree is dirty; refusing deployment'

python3 - "$EXPORT_DIR" "$EXPECTED_MANIFEST_SHA" "$EXPECTED_COMMIT" "$EXPECTED_PUBLIC_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected_sha, expected_commit, expected_public_commit = sys.argv[2:]
manifest_path = root / "provenance/assembled-manifest.json"
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise SystemExit("assembled manifest is missing or unsafe")
manifest_bytes = manifest_path.read_bytes()
actual_sha = hashlib.sha256(manifest_bytes).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(f"assembled manifest hash mismatch: {actual_sha}")
manifest = json.loads(manifest_bytes)
if manifest.get("schema_version") != "gpu-manager.assembled-release.v1":
    raise SystemExit("assembled release schema mismatch")
if manifest.get("private_commit_sha") != expected_commit or manifest.get("public_commit_sha") != expected_public_commit:
    raise SystemExit("assembled release commit identity mismatch")
entries = manifest.get("files")
if not isinstance(entries, list) or not entries:
    raise SystemExit("assembled release manifest has no file list")
for entry in entries:
    rel = entry.get("path") if isinstance(entry, dict) else None
    digest = entry.get("sha256") if isinstance(entry, dict) else None
    size = entry.get("size") if isinstance(entry, dict) else None
    if not isinstance(rel, str) or not rel or pathlib.PurePosixPath(rel).is_absolute() or ".." in pathlib.PurePosixPath(rel).parts:
        raise SystemExit("assembled release contains an unsafe manifest path")
    path = root / rel
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"assembled release file missing or unsafe: {rel}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SystemExit(f"assembled release path escapes root: {rel}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest or len(payload) != size:
        raise SystemExit(f"assembled release file integrity mismatch: {rel}")
print(f"assembled manifest verified: {len(entries)} files, sha256={actual_sha}")
PY

# C3 is read-only: the services registry is hashed but never copied or modified.
read -r SERVICES_SHA_BEFORE _ < <(sha256sum -- "$SERVICES_JSON")
[[ -z "$EXPECTED_C3_SERVICES_SHA" || "$SERVICES_SHA_BEFORE" == "$EXPECTED_C3_SERVICES_SHA" ]] || fail "C3 services.json hash mismatch: $SERVICES_SHA_BEFORE"
EXPECTED_C3_SERVICES_SHA=$SERVICES_SHA_BEFORE

CTRL_DIR=$(realpath -e -- "$CTRL_DIR") || fail 'cannot resolve controller path'
case "$CTRL_DIR" in /*) ;;
  *) fail 'controller path is not absolute' ;;
esac
cd -- "$CTRL_DIR" || fail 'cannot enter controller directory'
[[ "$(pwd -P)" == "$CTRL_DIR" ]] || fail 'controller cwd assertion failed'

PRE_LINK=$(readlink -- "$CTRL_DIR/current") || fail 'current release pointer is missing'
PRE_RELEASE=$(realpath -e -- "$CTRL_DIR/current") || fail 'cannot resolve current release'
RELEASES_REAL=$(realpath -e -- "$CTRL_DIR/releases") || fail 'cannot resolve releases directory'
case "$PRE_RELEASE/" in "$RELEASES_REAL"/*) ;;
  *) fail 'current release resolves outside releases directory' ;;
esac
[[ -d "$PRE_RELEASE/scripts" ]] || fail 'rollback release has no scripts directory'

probe_once() {
  local url=$1
  local code
  code=$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 2 --max-time 5 "$url" 2>/dev/null) || code=000
  [[ "$code" == 200 ]]
}
wait_http() {
  local url=$1 limit=$2 end_at code
  end_at=$((SECONDS + limit))
  while (( SECONDS <= end_at )); do
    code=$(curl --silent --output /dev/null --write-out '%{http_code}' --connect-timeout 2 --max-time 5 "$url" 2>/dev/null) || code=000
    [[ "$code" == 200 ]] && return 0
    sleep 1
  done
  return 1
}
verify_mainpid_cwd() {
  local expected_scripts=$1 pid live_cwd
  expected_scripts=$(realpath -e -- "$expected_scripts") || return 1
  pid=$(systemctl show -p MainPID --value gpu-manager.service 2>/dev/null) || return 1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  live_cwd=$(readlink -e -- "/proc/$pid/cwd") || return 1
  case "$live_cwd" in
    "$expected_scripts"|"$expected_scripts"/*) return 0 ;;
    *) printf 'r139-deploy: process cwd is not under expected scripts directory\n' >&2; return 1 ;;
  esac
}

probe_once "$MANAGER_HEALTH_URL" || fail 'preflight /health/live is not 200'
probe_once "$GEMMA_HEALTH_URL" || fail 'preflight Gemma :8094 /health is not 200'
preflight_args=(--auth-source-dir "$PRE_RELEASE/scripts" --services-json "$SERVICES_JSON" --lease-db "$LEASE_DB")
if [[ "$TEST_MODE" == 1 ]]; then
  preflight_args+=(--responses-fixture "$PREFLIGHT_FIXTURE")
  if [[ -n "${OOM_INFLIGHT_TEST_NOW:-}" ]]; then preflight_args+=(--now "$OOM_INFLIGHT_TEST_NOW"); fi
fi
python3 "$WORKSPACE/deployment/oom_inflight_preflight.py" "${preflight_args[@]}" || fail 'authenticated zero-state, lease, time, or unit preflight failed'
if [[ "${1:-}" == --preflight ]]; then
  printf 'oom-inflight-deploy: read-only authenticated preflight passed; no files, service state, or pointer changed\n'
  exit 0
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SNAPSHOT_DIR="$CTRL_DIR/r139-snapshots/$STAMP"
[[ ! -e "$SNAPSHOT_DIR" && ! -L "$SNAPSHOT_DIR" ]] || fail "snapshot already exists: $SNAPSHOT_DIR"
mkdir -p -- "$SNAPSHOT_DIR"
cp -a -- "$LIVE_UNIT" "$SNAPSHOT_DIR/live-gpu-manager.service"
if [[ -d "$DROPINS_DIR" ]]; then
  cp -a -- "$DROPINS_DIR" "$SNAPSHOT_DIR/gpu-manager.service.d"
  DROPINS_STATE=present
else
  DROPINS_STATE=absent
fi
read -r LIVE_UNIT_SHA _ < <(sha256sum -- "$LIVE_UNIT")
{
  printf 'current_link=%s\n' "$PRE_LINK"
  printf 'live_unit_sha256=%s\n' "$LIVE_UNIT_SHA"
  printf 'dropins=%s\n' "$DROPINS_STATE"
  printf 'services_json_sha256=%s\n' "$SERVICES_SHA_BEFORE"
  printf 'candidate_commit=%s\n' "$actual_head"
  printf 'assembled_manifest_sha256=%s\n' "$EXPECTED_MANIFEST_SHA"
} > "$SNAPSHOT_DIR/preflight.txt"

SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"
NEW_NAME="gpu-manager-unified-$SHORT_COMMIT-r139-$STAMP"
NEW="$CTRL_DIR/releases/$NEW_NAME"
STAGE="$CTRL_DIR/releases/.$NEW_NAME.staging"
EXPECTED_TARGET="releases/$NEW_NAME"
[[ "$(dirname -- "$NEW")" == "$CTRL_DIR/releases" && "$NEW_NAME" != . && "$NEW_NAME" != .. ]] || fail 'new release is not an immediate releases child'
[[ ! -e "$NEW" && ! -L "$NEW" && ! -e "$STAGE" && ! -L "$STAGE" ]] || fail 'new release or staging path already exists'
STAGE_CREATED=0
FLIP_ATTEMPTED=0
ROLLBACK_ATTEMPTED=0

rollback_to_snapshot() {
  local current_link
  current_link=$(readlink -- "$CTRL_DIR/current") || return 1
  if [[ "$current_link" == "$EXPECTED_TARGET" ]]; then
    ln -s -- "$PRE_LINK" "$CTRL_DIR/current.r139-rollback.new" || return 1
    mv -Tf -- "$CTRL_DIR/current.r139-rollback.new" "$CTRL_DIR/current" || return 1
    current_link=$(readlink -- "$CTRL_DIR/current") || return 1
    [[ "$current_link" == "$PRE_LINK" ]] || return 1
    printf 'r139-deploy: rollback pointer restored to %s\n' "$PRE_LINK" >&2
    sudo -n systemctl restart gpu-manager.service || return 1
    wait_http "$MANAGER_HEALTH_URL" "$MANAGER_WAIT_SECONDS" || return 1
    wait_http "$GEMMA_HEALTH_URL" "$GEMMA_WAIT_SECONDS" || return 1
    verify_mainpid_cwd "$PRE_RELEASE/scripts" || return 1
    return 0
  fi
  if [[ "$current_link" == "$PRE_LINK" ]]; then
    return 0
  fi
  printf 'r139-deploy: refusing to overwrite unexpected current pointer: %s\n' "$current_link" >&2
  return 1
}
cleanup() {
  local result=$?
  trap - EXIT
  if [[ "$STAGE_CREATED" == 1 && -d "${STAGE:-/nonexistent}" ]]; then
    rm -rf -- "$STAGE"
  fi
  if (( result != 0 && FLIP_ATTEMPTED == 1 && ROLLBACK_ATTEMPTED == 0 )); then
    ROLLBACK_ATTEMPTED=1
    set +e
    rollback_to_snapshot
    local rollback_result=$?
    set -e
    if (( rollback_result == 0 )); then
      printf 'r139-deploy: automatic rollback verified; original failure remains; no deployment retry\n' >&2
    else
      printf 'r139-deploy: automatic rollback did not verify; operator intervention required; no retry\n' >&2
    fi
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

mkdir -- "$STAGE"
STAGE_CREATED=1
timeout 300 cp -a -- "$EXPORT_DIR/." "$STAGE/"
mkdir -p -- "$STAGE/rollback/systemd"
# Preserve the unit bound to the execution-time previous release for rollback provenance.
cp -a -- "$LIVE_UNIT" "$STAGE/rollback/systemd/gpu-manager.service"
ROLLBACK_UNIT="$STAGE/rollback/systemd/gpu-manager.service"
[[ -f "$ROLLBACK_UNIT" && ! -L "$ROLLBACK_UNIT" ]] || fail 'assembled rollback unit is missing or unsafe'

# Take only the credential-reference directive from the active forward unit; never read credential bytes.
api_token_line=$(python3 - "$LIVE_UNIT" <<'PY'
import pathlib
import re
import sys
lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
found = [line for line in lines if line.startswith("LoadCredentialEncrypted=gpu-manager-api-token:")]
if len(found) != 1 or not re.fullmatch(r"LoadCredentialEncrypted=gpu-manager-api-token:[A-Za-z0-9_./-]+", found[0]):
    raise SystemExit("live forward unit must have exactly one safe api-token credential reference")
print(found[0])
PY
) || fail 'cannot resolve the live api-token credential reference'
cp -a -- "$ROLLBACK_UNIT" "$SNAPSHOT_DIR/rollback-unit.before"
python3 - "$ROLLBACK_UNIT" "$api_token_line" <<'PY'
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
credential_line = sys.argv[2]
text = path.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)
prefix = "LoadCredentialEncrypted=gpu-manager-api-token:"
existing = [line.rstrip("\n") for line in lines if line.startswith(prefix)]
if existing:
    if existing != [credential_line]:
        raise SystemExit("assembled rollback unit has a conflicting or duplicate api-token reference")
    raise SystemExit(0)
service_start = next((i for i, line in enumerate(lines) if line.strip() == "[Service]"), None)
if service_start is None:
    raise SystemExit("assembled rollback unit has no [Service] section")
service_end = next((i for i in range(service_start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
load_lines = [i for i in range(service_start + 1, service_end) if lines[i].startswith("LoadCredentialEncrypted=")]
if not load_lines:
    raise SystemExit("assembled rollback unit has no existing credential declaration anchor")
insert_at = load_lines[-1] + 1
lines.insert(insert_at, credential_line + "\n")
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY
cp -a -- "$ROLLBACK_UNIT" "$SNAPSHOT_DIR/rollback-unit.after"
diff_status=0
diff -u --label rollback-unit.before --label rollback-unit.after "$SNAPSHOT_DIR/rollback-unit.before" "$SNAPSHOT_DIR/rollback-unit.after" > "$SNAPSHOT_DIR/c6-rollback-unit.diff" || diff_status=$?
[[ "$diff_status" == 0 || "$diff_status" == 1 ]] || fail 'cannot produce the secret-free C6 diff'
python3 - "$SNAPSHOT_DIR/rollback-unit.before" "$SNAPSHOT_DIR/rollback-unit.after" "$api_token_line" "$SNAPSHOT_DIR/c6-rollback-unit.diff" <<'PY'
import pathlib
import sys

before = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
after = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
expected = sys.argv[3]
diff_lines = pathlib.Path(sys.argv[4]).read_text(encoding="utf-8").splitlines()
if expected in before:
    if before != after:
        raise SystemExit("C6 changed rollback unit even though api-token reference was already present")
else:
    if len(after) != len(before) + 1:
        raise SystemExit("C6 rollback diff is not exactly one added line")
    if after.count(expected) != 1:
        raise SystemExit("C6 api-token reference is not present exactly once")
    removed = [line for line in diff_lines if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff_lines if line.startswith("+") and not line.startswith("+++")]
    if removed or added != [expected]:
        raise SystemExit("C6 diff contains a change beyond the single credential-reference line")
print("C6 diff verified: secret-free, credential reference only")
PY

timeout 30 systemd-analyze verify "$ROLLBACK_UNIT" > "$SNAPSHOT_DIR/systemd-analyze.log" 2>&1 || fail 'assembled rollback unit failed systemd-analyze verify'
mv -- "$STAGE" "$NEW"
STAGE_CREATED=0
[[ -x "$NEW/scripts/gpu-manager.py" || -f "$NEW/scripts/gpu-manager.py" ]] || fail 'new release has no controller entry point'

# Repeat release-sensitive preflight immediately before the pointer change.
read -r SERVICES_SHA_NOW _ < <(sha256sum -- "$SERVICES_JSON")
[[ "$SERVICES_SHA_NOW" == "$SERVICES_SHA_BEFORE" ]] || fail 'services.json changed during S1; refusing flip'
[[ "$(readlink -- "$CTRL_DIR/current")" == "$PRE_LINK" ]] || fail 'current release changed during S1; refusing flip'
cd -- "$CTRL_DIR" || fail 'cannot return to controller directory'
[[ "$(pwd -P)" == "$CTRL_DIR" ]] || fail 'flip cwd assertion failed'
[[ "${NEW%/*}" == "$CTRL_DIR/releases" && -n "${NEW##*/}" && "${NEW##*/}" != . && "${NEW##*/}" != .. ]] || fail 'new release is not an immediate child of releases'

ln -s -- "$EXPECTED_TARGET" "$CTRL_DIR/current.r139.new"
FLIP_ATTEMPTED=1
mv -Tf -- "$CTRL_DIR/current.r139.new" "$CTRL_DIR/current" || fail 'atomic current pointer flip failed'
[[ "$(readlink -- "$CTRL_DIR/current")" == "$EXPECTED_TARGET" ]] || fail 'post-swap current pointer mismatch; refusing restart'
export OOM_INFLIGHT_EXPECTED_NEW_RELEASE="$NEW"
DEPLOY_START=$(date --iso-8601=seconds)
sudo -n systemctl restart gpu-manager.service || fail 'controller restart failed'
wait_http "$MANAGER_HEALTH_URL" "$MANAGER_WAIT_SECONDS" || fail 'gpu-manager /health/live did not reach 200 within 120 seconds'
wait_http "$GEMMA_HEALTH_URL" "$GEMMA_WAIT_SECONDS" || fail 'Gemma :8094 /health did not reach 200 within five minutes'
verify_mainpid_cwd "$NEW/scripts" || fail 'post-restart /proc cwd does not sit in the new scripts directory'
# Supervisor GPU-safety markers; a bare 'ring' also matches benign lines (unregistering, Deferring).
if journalctl -k --since="$DEPLOY_START" --no-pager -o cat 2>/dev/null | grep -Eiq 'bad_op|ring.*(timeout|timed out)|MODE1|GPU Hang|GPU reset'; then
  fail 'new kernel GPU fault marker observed after restart'
fi
read -r SERVICES_SHA_AFTER _ < <(sha256sum -- "$SERVICES_JSON")
[[ "$SERVICES_SHA_AFTER" == "$SERVICES_SHA_BEFORE" ]] || fail 'services.json changed during deployment; rollback required'
[[ "$(readlink -- "$CTRL_DIR/current")" == "$EXPECTED_TARGET" ]] || fail 'current release pointer drifted after health checks'
printf 'r139-deploy: deployed %s; rollback target %s; manifest %s; C3 services hash unchanged; /health/live=200; Gemma /health=200; process cwd verified\n' "$EXPECTED_TARGET" "$PRE_LINK" "$EXPECTED_MANIFEST_SHA"
printf 'r139-deploy: snapshot=%s; C6 diff=%s\n' "$SNAPSHOT_DIR" "$SNAPSHOT_DIR/c6-rollback-unit.diff"
