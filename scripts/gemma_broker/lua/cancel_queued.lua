-- combined-gemma.cancel-queued.v1
-- KEYS: job hash, priority queue
-- ARGV: expected_version, new_version, record_json, job_id
if redis.call('EXISTS', KEYS[1]) ~= 1 then
  return {err='UNKNOWN_JOB_ID'}
end
local state = redis.call('HGET', KEYS[1], 'state')
if state ~= 'queued' then
  return {'already_not_queued'}
end
if redis.call('HGET', KEYS[1], 'state_version') ~= ARGV[1] then
  return {err='JOB_STATE_VERSION_STALE'}
end
if redis.call('ZREM', KEYS[2], ARGV[4]) ~= 1 then
  return {err='REDIS_STATE_UNKNOWN'}
end
redis.call('HSET', KEYS[1],
  'record', ARGV[3], 'state', 'cancelled', 'state_version', ARGV[2])
return {'cancelled'}
