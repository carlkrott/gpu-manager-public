-- combined-gemma.repair-dead-job.v1
-- KEYS: job hash, queue zset, repair hash, member lease set, attempt hash
-- ARGV: owner, fence, expected_job_version, new_job_version, job_record,
--       new_job_state, queue_score, attempt_id, attempt_record, attempt_state,
--       job_id, terminal_attempt_retention_seconds
if redis.call('EXISTS', KEYS[1]) ~= 1 then
  return {err='UNKNOWN_JOB_ID'}
end
local prior_status = redis.call('HGET', KEYS[3], 'status')
local prior_fence = tonumber(redis.call('HGET', KEYS[3], 'fence') or '0')
if prior_status == 'completed' or prior_fence >= tonumber(ARGV[2]) then
  return {'already_repaired'}
end
if redis.call('HGET', KEYS[1], 'state_version') ~= ARGV[3] then
  return {err='JOB_STATE_VERSION_STALE'}
end
local state = redis.call('HGET', KEYS[1], 'state')
if state ~= 'claimed' and state ~= 'accepted' and state ~= 'in_flight' then
  return {'already_terminal'}
end
redis.call('HSET', KEYS[3], 'owner', ARGV[1], 'fence', ARGV[2], 'status', 'pending')
redis.call('HSET', KEYS[1],
  'record', ARGV[5], 'state', ARGV[6], 'state_version', ARGV[4])
if ARGV[6] == 'queued' then
  redis.call('ZADD', KEYS[2], ARGV[7], ARGV[11])
end
redis.call('SREM', KEYS[4], ARGV[8])
if redis.call('EXISTS', KEYS[5]) == 1 then
  redis.call('HSET', KEYS[5], 'record', ARGV[9], 'state', ARGV[10])
  redis.call('EXPIRE', KEYS[5], tonumber(ARGV[12]))
end
redis.call('HSET', KEYS[3], 'status', 'completed')
return {'repaired', ARGV[6]}
