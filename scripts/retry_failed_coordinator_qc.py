#!/usr/bin/env python3
"""Operator-only, one-shot QC repair preserving generation and failed attempts.

Only a terminal provider failure at a read-only QC stage can be retried.
Never reopens unknown outcomes, cancellation, generation or delivery. Uses
the existing systemd Redis credential contract; does not print credentials.
"""
import argparse
import hashlib
import json
from pathlib import Path

import redis
from pipeline_coordinator import RedisCoordinatorStore, _now_iso
from queue_engine import redis_connection_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-error', required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    client = redis.Redis(**{**redis_connection_kwargs(), 'decode_responses': True, 'protocol': 2})
    key = RedisCoordinatorStore.HASH_PREFIX + args.run_id
    fence = 'pipeline_coordinator:qc-retry:' + args.run_id
    with client.pipeline() as pipe:
        pipe.watch(key, fence)
        raw = pipe.hgetall(key)
        state = {name: json.loads(value) for name, value in raw.items()}
        if not state or pipe.exists(fence):
            raise RuntimeError('missing run or operator QC retry already admitted')
        if state['status'] != 'failed' or state.get('terminal_code') != 'provider_error':
            raise RuntimeError('only known terminal provider failures are eligible')
        if state.get('cancelled') or state.get('cancellation_intent'):
            raise RuntimeError('cancelled work cannot be reopened')
        stage_id = state.get('current_stage_id')
        stage = next((s for s in state['compiled_pipeline']['stages'] if s['id'] == stage_id), {})
        attempts = state.get('stage_attempts', {}).get(stage_id, [])
        last = attempts[-1] if attempts else {}
        if stage.get('kind') != 'qc' or last.get('status') != 'failed':
            raise RuntimeError('cursor is not a failed QC observation')
        if args.expected_error not in str(last.get('error') or '') or 'unsafe_external_resume' in str(last):
            raise RuntimeError('failure identity mismatch or ambiguous external outcome')
        if state.get('pending_stage_ids'):
            raise RuntimeError('pending correction stages require separate reconciliation')
        if any(c.get('status') not in {'completed', 'failed', 'cancelled'} for c in state.get('child_jobs', [])):
            raise RuntimeError('child outcome is not terminal')
        artifact = state.get('artifact') or {}
        path = Path(artifact.get('path_or_url', ''))
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != artifact.get('sha256'):
            raise RuntimeError('saved artifact identity mismatch')
        job_key = 'job:' + state['parent_job_id']
        pipe.watch(job_key)
        if pipe.hget(job_key, 'status') != 'failed':
            raise RuntimeError('normal parent receipt is no longer failed')
        previous_result = pipe.hget(job_key, 'result') or ''
        now = _now_iso()
        evidence = {'stage_id': stage_id, 'reason': args.reason,
                    'previous_terminal_reason': state.get('terminal_reason'),
                    'previous_terminal_at': state.get('terminal_at'),
                    'retained_artifact_sha256': digest, 'at': now}
        state.update(status='running', terminal_code=None, terminal_reason=None,
                     terminal_at=None, updated_at=now,
                     cursor_note='operator repair: retry QC only; generation retained')
        pipe.multi()
        pipe.set(fence, json.dumps(evidence))
        pipe.hset(key, mapping={name: json.dumps(value) for name, value in state.items()})
        pipe.persist(key)
        pipe.hset(job_key, mapping={'status': 'in_flight', 'error': '', 'completed_at': '',
                                   'previous_terminal_result': previous_result, 'result': '',
                                   'operator_repair': json.dumps(evidence)})
        pipe.persist(job_key)
        pipe.xadd(RedisCoordinatorStore.EVENT_PREFIX + args.run_id,
                  {'type': 'operator_qc_retry', 'timestamp': now, 'payload': json.dumps(evidence)})
        pipe.xadd(RedisCoordinatorStore.STREAM_KEY,
                  {name: state[name] for name in ('run_id', 'parent_job_id', 'pipeline_id')})
        pipe.execute()
    print(json.dumps({'run_id': args.run_id, 'stage_id': stage_id,
                      'status': 'qc_retry_admitted', 'retained_artifact_sha256': digest}))


if __name__ == '__main__':
    main()
