-- combined-gemma.leader-renew.v1
-- KEYS: leader hash
-- ARGV: owner, fencing_token, now, ttl_seconds
local current_owner = redis.call('HGET', KEYS[1], 'owner')
local current_token = redis.call('HGET', KEYS[1], 'token')
local current_expiry = tonumber(redis.call('HGET', KEYS[1], 'expires_at') or '-1')
if current_owner ~= ARGV[1] or current_token ~= ARGV[2] or current_expiry <= tonumber(ARGV[3]) then
  return {err='DISPATCHER_FENCE_STALE'}
end
local expires_at = tonumber(ARGV[3]) + tonumber(ARGV[4])
redis.call('HSET', KEYS[1], 'expires_at', tostring(expires_at))
return {current_token, tostring(expires_at)}
