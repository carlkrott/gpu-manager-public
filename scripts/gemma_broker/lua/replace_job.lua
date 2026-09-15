-- combined-gemma.replace-job.v2
-- KEYS: job hash, current attempt hash (or job hash when absent)
-- ARGV: expected_state_version, new_state_version, job_json, job_state,
--       update_attempt (0|1), attempt_json, attempt_state,
--       terminal_attempt_retention_seconds
if redis.call('EXISTS', KEYS[1]) ~= 1 then
  return {err='UNKNOWN_JOB_ID'}
end
local current_version = redis.call('HGET', KEYS[1], 'state_version')
if current_version ~= ARGV[1] then
  return {err='JOB_STATE_VERSION_STALE'}
end
if tonumber(ARGV[2]) ~= tonumber(ARGV[1]) + 1 then
  return {err='JOB_STATE_VERSION_INVALID'}
end
if ARGV[5] == '1' and redis.call('EXISTS', KEYS[2]) ~= 1 then
  return {err='UNKNOWN_ATTEMPT_ID'}
end
redis.call('HSET', KEYS[1],
  'record', ARGV[3], 'state', ARGV[4], 'state_version', ARGV[2])
if ARGV[5] == '1' then
  redis.call('HSET', KEYS[2], 'record', ARGV[6], 'state', ARGV[7])
  if ARGV[7] == 'completed' or ARGV[7] == 'failed'
     or ARGV[7] == 'cancelled' or ARGV[7] == 'outcome_unknown' then
    redis.call('EXPIRE', KEYS[2], tonumber(ARGV[8]))
  end
end
if ARGV[4] == 'queued' then
  redis.call('HDEL', KEYS[1],
    'selected_member', 'member_state_version', 'current_attempt_id',
    'queue_claimed_at')
end
return {'updated', ARGV[2]}
