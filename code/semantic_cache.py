import redis
from redisvl.extensions.llmcache import SemanticCache

def get_model_output(query):
    pass

client = redis.Redis(decode_responses=True, db=1)

llmcache = SemanticCache(
    name="llmcache",
    distance_threshold=0.4,
    redis_client=client,
)

from langchain_redis.cache import RedisSemanticCache # simply a wrapper for the original redis SemanticCache package