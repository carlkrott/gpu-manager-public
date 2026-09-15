-- combined-gemma.reserve.v3
-- KEYS: job hash, priority queue zset, member hash, member lease set,
--       dispatcher leader hash, attempt hash
-- ARGV: job_id, member_name, expected member state_version, now, attempt_id,
--       claimed_job_json, reserved_member_json, leader_owner, fencing_token,
--       expected_job_state_version, attempt_json
local leader_owner = redis.call('HGET', KEYS[5], 'owner')
local leader_token = redis.call('HGET', KEYS[5], 'token')
local leader_expiry = tonumber(redis.call('HGET', KEYS[5], 'expires_at') or '-1')
if leader_owner ~= ARGV[8] or leader_token ~= ARGV[9] or leader_expiry <= tonumber(ARGV[4]) then
  return {err='DISPATCHER_FENCE_STALE'}
end
local state = redis.call('HGET', KEYS[3], 'state')
local accepting = redis.call('HGET', KEYS[3], 'accepting')
local state_version = redis.call('HGET', KEYS[3], 'state_version')
local free = tonumber(redis.call('HGET', KEYS[3], 'compatible_free_slots') or '-1')
local live_leases = redis.call('SCARD', KEYS[4])
local configured_slots = tonumber(redis.call('HGET', KEYS[3], 'configured_slots') or '-1')
if configured_slots <= 0 then
  local current_record = redis.call('HGET', KEYS[3], 'record')
  if current_record then
    local decoded = cjson.decode(current_record)
    configured_slots = tonumber(decoded.configured_slots or '-1')
  end
end
if state_version ~= ARGV[3] then
  return {err='RESERVATION_MEMBER_VERSION_STALE'}
end
if state ~= 'ready_accepting' or accepting ~= '1' or free <= 0
   or configured_slots <= 0 or live_leases >= configured_slots then
  return {err='RESERVATION_MEMBER_UNAVAILABLE'}
end
if redis.call('HGET', KEYS[1], 'state') ~= 'queued' then
  return {err='JOB_NOT_QUEUED'}
end
if redis.call('HGET', KEYS[1], 'state_version') ~= ARGV[10] then
  return {err='JOB_STATE_VERSION_STALE'}
end
local head = redis.call('ZRANGE', KEYS[2], 0, 0)
if #head ~= 1 or head[1] ~= ARGV[1] then
  return {err='QUEUE_HEAD_STALE'}
end
if redis.call('EXISTS', KEYS[6]) == 1 then
  return {err='ATTEMPT_EXISTS'}
end
if redis.call('ZREM', KEYS[2], ARGV[1]) ~= 1 then
  return {err='QUEUE_CLAIM_CONFLICT'}
end
redis.call('HSET', KEYS[1],
  'record', ARGV[6], 'state', 'claimed',
  'state_version', tostring(tonumber(ARGV[10]) + 1),
  'selected_member', ARGV[2], 'member_state_version', ARGV[3],
  'queue_claimed_at', ARGV[4], 'current_attempt_id', ARGV[5])
redis.call('HSET', KEYS[6],
  'record', ARGV[11], 'state', 'reserved', 'job_id', ARGV[1],
  'member_name', ARGV[2], 'reservation_fence', ARGV[9],
  'reserved_at', ARGV[4])
redis.call('SADD', KEYS[4], ARGV[5])
live_leases = redis.call('SCARD', KEYS[4])
local current_state = redis.call('HGET', KEYS[3], 'state') or 'ready_accepting'
local current_accepting = redis.call('HGET', KEYS[3], 'accepting') or '0'
local current_version = tonumber(redis.call('HGET', KEYS[3], 'state_version') or ARGV[3])
local current_generation = tonumber(redis.call('HGET', KEYS[3], 'generation_fence') or ARGV[9])
local record = cjson.decode(ARGV[7])
local new_free = math.max(0, math.min(free - 1, configured_slots - live_leases))
record.state = current_state
record.accepting = current_accepting == '1'
record.state_version = current_version
record.generation_fence = current_generation
record.dispatcher_leases = live_leases
record.compatible_free_slots = new_free
redis.call('HSET', KEYS[3],
  'record', cjson.encode(record),
  'compatible_free_slots', new_free,
  'configured_slots', configured_slots)
return {'claimed', ARGV[1], ARGV[2], ARGV[5], ARGV[9]}
