-- combined-gemma.member-drain.v2
-- KEYS: member hash, member lease set
-- ARGV: mode, expected_version, claimed_version, attempt_id, new_version,
--       new_record_json, new_state, accepting, compatible_free_slots,
--       generation_fence, configured_slots, backend_slots_busy
local mode = ARGV[1]
if redis.call('EXISTS', KEYS[1]) ~= 1 then
  return {err='UNKNOWN_MEMBER'}
end
local version = redis.call('HGET', KEYS[1], 'state_version')
local state = redis.call('HGET', KEYS[1], 'state')
if mode ~= 'release_v2' and version ~= ARGV[2] then
  return {err='RESERVATION_STATE_STALE'}
end
if tonumber(ARGV[5]) ~= tonumber(ARGV[2]) + 1 then
  return {err='MEMBER_STATE_VERSION_INVALID'}
end
if mode == 'begin' then
  if state ~= 'ready_accepting' and state ~= 'ready_busy' and state ~= 'joining' and state ~= 'unhealthy' then
    return {err='DRAIN_INVALID_STATE'}
  end
elseif mode == 'release' then
  if tonumber(ARGV[3]) ~= tonumber(ARGV[2]) then
    return {err='RESERVATION_STATE_STALE'}
  end
  if redis.call('SISMEMBER', KEYS[2], ARGV[4]) ~= 1 then
    return {err='LEASE_NOT_OWNED'}
  end
  redis.call('SREM', KEYS[2], ARGV[4])
elseif mode == 'release_v2' then
  -- Happy-path terminal release. Independent of readiness-refresh state_version
  -- so that a refresh between Python's get_member() and Lua commit does NOT
  -- pin a stale lease. Only fail-closed checks: lease ownership and the
  -- (current+1) invariant on the new state_version.
  if redis.call('SISMEMBER', KEYS[2], ARGV[4]) ~= 1 then
    return {err='LEASE_NOT_OWNED'}
  end
  redis.call('SREM', KEYS[2], ARGV[4])
elseif mode == 'complete' then
  if state ~= 'draining' then
    return {err='DRAIN_INVALID_STATE'}
  end
  if redis.call('SCARD', KEYS[2]) ~= 0 then
    return {err='MEMBER_LEASES_REMAIN'}
  end
elseif mode == 'begin_rejoin' then
  if state ~= 'offline' then
    return {err='REJOIN_INVALID_STATE'}
  end
  local current_generation = tonumber(redis.call('HGET', KEYS[1], 'generation_fence') or '0')
  if tonumber(ARGV[10]) <= current_generation then
    return {err='GENERATION_FENCE_STALE'}
  end
elseif mode == 'complete_rejoin' then
  if state ~= 'joining' then
    return {err='REJOIN_INVALID_STATE'}
  end
  if redis.call('SCARD', KEYS[2]) ~= 0 then
    return {err='MEMBER_LEASES_REMAIN'}
  end
else
  return {err='DRAIN_MODE_INVALID'}
end
if mode == 'release' or mode == 'release_v2' then
  -- SREM and mirror reconciliation happen in this same script. Python-side
  -- "cached leases - 1" arithmetic is not authoritative.
  local live_leases = redis.call('SCARD', KEYS[2])
  local free = tonumber(ARGV[9]) or 0
  if ARGV[7] == 'draining' then
    free = 0
  else
    local configured = tonumber(ARGV[11]) or 0
    local backend_busy = tonumber(ARGV[12]) or 0
    if configured > 0 then
      local backend_free = math.max(0, configured - backend_busy)
      local dispatcher_free = math.max(0, configured - live_leases)
      free = math.min(configured, backend_free, dispatcher_free)
    end
  end
  local record = cjson.decode(ARGV[6])
  record.state = ARGV[7]
  record.accepting = ARGV[8] == '1'
  record.state_version = tonumber(ARGV[5])
  record.generation_fence = tonumber(ARGV[10])
  record.dispatcher_leases = live_leases
  record.compatible_free_slots = free
  redis.call('HSET', KEYS[1],
    'record', cjson.encode(record), 'state', ARGV[7], 'accepting', ARGV[8],
    'compatible_free_slots', tostring(free), 'state_version', ARGV[5],
    'generation_fence', ARGV[10])
  return {mode, ARGV[5], tostring(live_leases), tostring(free)}
end
local record = cjson.decode(ARGV[6])
record.state = ARGV[7]
record.accepting = ARGV[8] == '1'
record.state_version = tonumber(ARGV[5])
record.generation_fence = tonumber(ARGV[10])
redis.call('HSET', KEYS[1],
  'record', cjson.encode(record), 'state', ARGV[7], 'accepting', ARGV[8],
  'compatible_free_slots', ARGV[9], 'state_version', ARGV[5],
  'generation_fence', ARGV[10])
return {mode, ARGV[5]}
