import redis
import json
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from pprint import pprint

with open("data/wikihowAll.json", "r", encoding="utf-8") as f:
    start = time.time()
    data = json.load(f)
    end = time.time()
    elapsed = end - start
    print(f"data loaded in {elapsed:2f} seconds")

client = redis.Redis(decode_responses=True)
pipeline = client.pipeline()

for i, article in enumerate(data[0]):
    redis_key = f"article:{i:03}"
    pipeline.json().set(name=redis_key, path="$", obj=article)

pipeline.execute()

embedding_model = SentenceTransformer(
    model_name_or_path="msmarco-distilbert-base-v4"
)

keys = sorted(client.keys(pattern="article:*"))

text = client.json().mget(keys=keys, path="$.text")
text = [item for sublist in text for item in sublist]

text_embeddings = embedding_model.encode(
    sentences=text
).astype(dtype=np.float16).tolist()

pipeline = client.pipeline()

for key, embedding in zip(keys, text_embeddings):
    pipeline.json().set(name=key, path="$.text_embeddings", obj=embedding)

pipeline.execute()

res = client.json().get(name="article:007")
pprint(res)


# text embeddings should be sufficient for retrieval

# headline = client.json().mget(keys=keys, path="$.headline")
# headline = [item for sublist in headline for item in sublist]

# headline_embeddings = embedding_model.encode(
#     sentences=headline
# ).astype(dtype=np.float16).tolist()

# title = client.json().mget(keys=keys, path="$.title")
# title = [item for sublist in title for item in sublist]

# title_embeddings = embedding_model.encode(
#     sentences=title
# ).astype(dtype=np.float16).tolist()
# for key, embedding in zip(keys, headline_embeddings):
#     pipeline.json().set(name=key, path="$.headline_embeddings", obj=embedding)

# for key, embedding in zip(keys, title_embeddings):
#     pipeline.json().set(name=key, path="$.title_embeddings", obj=embedding)