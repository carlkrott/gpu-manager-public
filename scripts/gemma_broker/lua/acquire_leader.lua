-- combined-gemma.leader-acquire.v1
-- KEYS: leader hash, monotonic fence counter
-- ARGV: owner, now, ttl_seconds
local current_owner = redis.call('HGET', KEYS[1], 'owner')
local current_token = redis.call('HGET', KEYS[1], 'token')
local current_expiry = tonumber(redis.call('HGET', KEYS[1], 'expires_at') or '-1')
if current_owner and current_expiry > tonumber(ARGV[2]) then
  if current_owner ~= ARGV[1] then
    return {err='DISPATCHER_LEASE_HELD'}
  end
  return {current_token, tostring(current_expiry)}
end
local token = redis.call('INCR', KEYS[2])
local expires_at = tonumber(ARGV[2]) + tonumber(ARGV[3])
redis.call('HSET', KEYS[1],
  'owner', ARGV[1], 'token', tostring(token), 'expires_at', tostring(expires_at))
return {tostring(token), tostring(expires_at)}
