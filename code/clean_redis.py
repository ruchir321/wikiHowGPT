import redis

# 1. connect to redis server
client = redis.Redis(decode_responses=True)

client.flushdb()