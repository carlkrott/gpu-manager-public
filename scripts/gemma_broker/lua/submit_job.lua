-- combined-gemma.submit.v2
-- KEYS: idempotency hash, job hash, priority queue zset
-- ARGV: request sha256, job id, record json, enqueue score, state_version
if redis.call('EXISTS', KEYS[1]) == 1 then
  local prior_hash = redis.call('HGET', KEYS[1], 'request_sha256')
  local prior_job = redis.call('HGET', KEYS[1], 'job_id')
  if prior_hash ~= ARGV[1] then
    return {err='IDEMPOTENCY_KEY_REUSED'}
  end
  return {'existing', prior_job}
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return {err='DUPLICATE_JOB_ID'}
end
redis.call('HSET', KEYS[1], 'request_sha256', ARGV[1], 'job_id', ARGV[2])
redis.call('HSET', KEYS[2],
  'record', ARGV[3], 'state', 'queued', 'state_version', ARGV[5])
redis.call('ZADD', KEYS[3], ARGV[4], ARGV[2])
return {'created', ARGV[2]}
