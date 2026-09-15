#!/usr/bin/env python3
"""Private, scoped GPU Manager snapshot and isolated no-replay restore.

Redis keys are captured in one read-only server-side operation. Files are
independently atomic documents captured afterwards; their capture window is
recorded. Restore is inspection-only: it never starts workers or contacts a
model. Native tasks that were accepted must be reconciled, never replayed.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import redis
from native_task_host import atomic_write
from queue_engine import redis_connection_kwargs


PATTERNS = ["queue:*", "job:*", "job-idempotency:*", "pipeline_coordinator:*",
            "gpu:bundle:lifecycle:*"]
CAPTURE = """
local result = {}
local seen = {}
local bytes = 0
for _, pattern in ipairs(cjson.decode(ARGV[1])) do
    local cursor = '0'
    repeat
        local page = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', 500)
        cursor = page[1]
        for _, key in ipairs(page[2]) do
            if not seen[key] then
                seen[key] = true
                local dump = redis.call('DUMP', key)
                if dump then
                    bytes = bytes + string.len(dump)
                    if #result >= 10000 or bytes > 268435456 then
                        return redis.error_reply('scoped snapshot exceeds safe size bound')
                    end
                    table.insert(result, {key, dump, redis.call('PTTL', key)})
                end
            end
        end
    until cursor == '0'
end
return result
"""


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode('ascii')


def capture(args) -> Path:
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.time()
    options = redis_connection_kwargs()
    options.update(decode_responses=False, protocol=2, socket_timeout=30)
    client = redis.Redis(**options)
    rows = client.eval(CAPTURE, 0, json.dumps(PATTERNS))
    redis_captured = time.time()
    record = {
        'schema': 'gpu-manager-private-snapshot.v1', 'started_at': started,
        'redis_captured_at': redis_captured, 'patterns': PATTERNS,
        'source_release': str(args.release.resolve()),
        'redis': [{'key': encoded(key), 'dump': encoded(raw), 'ttl_ms': ttl}
                  for key, raw, ttl in rows],
        'files': {}, 'restore_mode': 'isolated-inspection-no-workers',
        'off_host': 'not_configured',
    }
    paths = {
        'registry.json': args.release.resolve() / 'config/services.json',
        'runtime-state.json': args.runtime_state,
        'runtime-host-fences.json': args.host_fences,
    }
    for path in args.native_tasks.glob('*/*.json'):
        if path.name in {'state.json', 'request.json', 'response.json'}:
            paths['native-tasks/' + str(path.relative_to(args.native_tasks))] = path
    for label, path in paths.items():
        raw = path.read_bytes()
        record['files'][label] = {'data': encoded(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    record['finished_at'] = time.time()
    content = gzip.compress(json.dumps(record, separators=(',', ':')).encode())
    target = root / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'-{os.getpid()}.json.gz')
    atomic_write(target, content)
    os.chmod(target, 0o600)
    print(json.dumps({'snapshot': str(target), 'sha256': hashlib.sha256(content).hexdigest(),
                      'redis_records': len(rows), 'files': len(paths),
                      'redis_capture_seconds': redis_captured - started,
                      'off_host': 'not_configured'}))
    return target


def restore_isolated(snapshot: Path) -> None:
    document = json.loads(gzip.decompress(snapshot.read_bytes()))
    if document.get('schema') != 'gpu-manager-private-snapshot.v1':
        raise ValueError('unsupported snapshot schema')
    for item in document['files'].values():
        if hashlib.sha256(base64.b64decode(item['data'])).hexdigest() != item['sha256']:
            raise ValueError('snapshot file hash mismatch')
    # No caller-controlled restore address: only this newly owned Unix socket.
    with tempfile.TemporaryDirectory(prefix='gpu-manager-backup-restore-') as directory:
        socket = str(Path(directory) / 'store.sock')
        process = subprocess.Popen([
            'valkey-server', '--port', '0', '--unixsocket', socket,
            '--unixsocketperm', '600', '--dir', directory, '--save', '', '--appendonly', 'no',
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            store = redis.Redis(unix_socket_path=socket, protocol=2)
            for _ in range(100):
                try:
                    store.ping()
                    break
                except redis.ConnectionError:
                    if process.poll() is not None:
                        raise RuntimeError('isolated restore store exited')
                    time.sleep(.05)
            counts = Counter()
            handles = artifacts = 0
            for item in document['redis']:
                key, raw = base64.b64decode(item['key']), base64.b64decode(item['dump'])
                # Inspection retains the exact snapshot even if a historical
                # terminal TTL would have elapsed during recovery.
                store.restore(key, 0, raw)
                # RESTORE validates the RDB payload checksum. Re-DUMP is not
                # byte-stable across Redis/Valkey versions or hash encodings;
                # inspect the restored records, not their serialization.
                if store.type(key) == b'none':
                    raise RuntimeError('restored Redis record is absent')
                if key.startswith(b'job:') and store.type(key) == b'hash':
                    job = store.hgetall(key)
                    counts[job.get(b'status', b'unknown').decode()] += 1
                    handles += bool(job.get(b'native_task_handle') or job.get(b'comfyui_prompt_id'))
                    artifacts += b'path_or_url' in job.get(b'result', b'')
            if store.dbsize() != len(document['redis']):
                raise RuntimeError('restore record count differs')
            receipt = {'snapshot': str(snapshot), 'source_release': document['source_release'],
                       'restored_records': store.dbsize(), 'job_statuses': dict(counts),
                       'provider_handles': handles, 'jobs_with_artifact_references': artifacts,
                       'verified_files': len(document['files']), 'replay_count': 0,
                       'isolation': 'private-valkey-unix-socket-no-workers',
                       'off_host': document['off_host']}
            atomic_write(snapshot.with_suffix('.restore.json'), json.dumps(receipt, indent=2).encode())
            print(json.dumps(receipt))
        finally:
            process.terminate()
            process.wait(timeout=10)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--runtime-state', type=Path)
    parser.add_argument('--host-fences', type=Path)
    parser.add_argument('--native-tasks', type=Path)
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--verify-restore', action='store_true')
    args = parser.parse_args()
    if args.restore:
        restore_isolated(args.restore)
    else:
        if not all((args.destination, args.release, args.runtime_state, args.host_fences, args.native_tasks)):
            parser.error('capture requires destination, release and all three state paths')
        snapshot = capture(args)
        if args.verify_restore:
            restore_isolated(snapshot)


if __name__ == '__main__':
    main()
