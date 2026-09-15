-- combined-gemma.requeue-claim.v1
-- Atomically revert a claimed/accepted/in_flight job to the head of its priority
-- queue while releasing the member dispatcher lease and free slot. Pre-acceptance
-- only: refused once the boundary is crossed.
-- KEYS: job hash, priority queue zset, member hash, member lease set,
--       current attempt hash
-- ARGV: expected_job_version, new_job_version, job_record_json, queue_score,
--       job_id, expected_member_version, new_member_version, member_record_json,
--       member_state, member_accepting, compatible_free_slots, generation_fence,
--       attempt_id, failed_attempt_json, attempt_terminal_at, error_category,
--       terminal_attempt_retention_seconds
if redis.call('EXISTS', KEYS[1]) ~= 1 then
  return {err='UNKNOWN_JOB_ID'}
end
if redis.call('HGET', KEYS[1], 'state_version') ~= ARGV[1] then
  return {err='JOB_STATE_VERSION_STALE'}
end
local state = redis.call('HGET', KEYS[1], 'state')
if state ~= 'claimed' and state ~= 'accepted' and state ~= 'in_flight' then
  return {err='REQUEUE_INVALID_STATE'}
end
if redis.call('HGET', KEYS[1], 'accepted_boundary_crossed') == '1' then
  return {err='CANCEL_UNSAFE_AFTER_ACCEPTANCE'}
end
if redis.call('EXISTS', KEYS[3]) == 1 then
  if redis.call('HGET', KEYS[3], 'state_version') ~= ARGV[6] then
    return {err='RESERVATION_STATE_STALE'}
  end
end
if redis.call('EXISTS', KEYS[5]) ~= 1 then
  return {err='UNKNOWN_ATTEMPT_ID'}
end
if redis.call('HGET', KEYS[5], 'state') ~= 'reserved' then
  return {err='ATTEMPT_STATE_STALE'}
end
if redis.call('HGET', KEYS[5], 'job_id') ~= ARGV[5] then
  return {err='ATTEMPT_STATE_STALE'}
end
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[5])
redis.call('HSET', KEYS[1],
  'record', ARGV[3], 'state', 'queued', 'state_version', ARGV[2])
redis.call('HDEL', KEYS[1],
  'selected_member', 'member_state_version', 'current_attempt_id',
  'queue_claimed_at')
if redis.call('EXISTS', KEYS[3]) == 1 then
  redis.call('SREM', KEYS[4], ARGV[13])
  redis.call('HSET', KEYS[3],
    'record', ARGV[8], 'state', ARGV[9], 'accepting', ARGV[10],
    'compatible_free_slots', ARGV[11], 'state_version', ARGV[7],
    'generation_fence', ARGV[12])
end
redis.call('HSET', KEYS[5],
  'record', ARGV[14], 'state', 'failed',
  'terminal_at', ARGV[15], 'error_category', ARGV[16])
redis.call('EXPIRE', KEYS[5], tonumber(ARGV[17]))
return {'requeued', ARGV[5]}
