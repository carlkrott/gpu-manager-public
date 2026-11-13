-- combined-gemma.submit.v3
-- KEYS: idempotency hash, job hash, priority queue zset
-- ARGV: request sha256, job id, record json, enqueue score, state_version,
--       prior job key pattern (prefix .. 'job:')
if redis.call('EXISTS', KEYS[1]) == 1 then
  local prior_hash = redis.call('HGET', KEYS[1], 'request_sha256')
  local prior_job = redis.call('HGET', KEYS[1], 'job_id')
  if prior_hash ~= ARGV[1] then
    return {err='IDEMPOTENCY_KEY_REUSED'}
  end
  -- A prior job CANCELLED before it crossed the accepted boundary never ran,
  -- so the retry that the 504 (compatibility_timeout, retry_safe=true)
  -- promised can still be taken: drop the key mapping and fall through so the
  -- new job below owns it. Anything accepted, in flight or completed -- and a
  -- job cancelled AFTER acceptance -- keeps replaying. The whole check and
  -- the DEL happen inside this single EVAL, so two concurrent same-key
  -- retries produce exactly one new job and the loser replays the winner.
  local prior_key = ARGV[6] .. prior_job
  -- The prior job expired (terminal retention): the mapping is stale.
  if redis.call('EXISTS', prior_key) == 0 then
    redis.call('DEL', KEYS[1])
  else
    local raw = redis.call('HGET', prior_key, 'record')
    if raw then
      local decoded_ok, decoded = pcall(cjson.decode, raw)
      if decoded_ok and decoded.state == 'cancelled'
         and decoded.accepted_boundary_crossed == false then
        redis.call('DEL', KEYS[1])
      end
    end
  end
  if redis.call('EXISTS', KEYS[1]) == 1 then
    return {'existing', prior_job}
  end
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return {err='DUPLICATE_JOB_ID'}
end
redis.call('HSET', KEYS[1], 'request_sha256', ARGV[1], 'job_id', ARGV[2])
-- Bounded: a mapping outlives its job only until this expiry (jobs end < 1 h).
redis.call('EXPIRE', KEYS[1], 86400)
redis.call('HSET', KEYS[2],
  'record', ARGV[3], 'state', 'queued', 'state_version', ARGV[5])
redis.call('ZADD', KEYS[3], ARGV[4], ARGV[2])
return {'created', ARGV[2]}