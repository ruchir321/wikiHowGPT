import redis
import json
import time
from pprint import pprint

############
## DOCUMENTS

# 1. connect to redis server
client = redis.Redis(decode_responses=True)

# 2. fetch data
with open("data/wikihow_10.json", "r", encoding="utf-8") as f:
    start = time.time()
    data = json.load(f)
    end = time.time()
    elapsed = end - start
    print(f"data loaded in {elapsed:2f} seconds")

# print(data[0].keys())
# pprint(data[0])
# print(type(data[0]))

# 3. store data in redis
pipeline = client.pipeline() # lazy loading 

for i, article in enumerate(data, start=1):
    redis_key = f"article:{i:02}"
    pipeline.json().set(name=redis_key, path="$", obj=article)

pipeline.execute()
