import redis
import json
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from pprint import pprint

from redis.commands.search.field import (
    TextField,
    NumericField,
    TagField,
    VectorField
)

from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

from redis.commands.search.query import Query

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

# 3. store data in redis
pipeline = client.pipeline() # lazy loading 

for i, article in enumerate(data[0]):
    redis_key = f"article:{i:03}"
    pipeline.json().set(name=redis_key, path="$", obj=article)

pipeline.execute()

# 4. select text embedding model
embedding_model = SentenceTransformer(
    model_name_or_path="msmarco-distilbert-base-v4"
)

# 5. generate and store text embeddings
keys = sorted(client.keys(pattern="article:*"))

text = client.json().mget(keys=keys, path="$.text")
text = [item for sublist in text for item in sublist]

text_embeddings = embedding_model.encode(
    sentences=text
).astype(dtype=np.float16).tolist()

VECTOR_DIM = len(text_embeddings[0])
print("embeddings ready")

pipeline = client.pipeline()

for key, embedding in zip(keys, text_embeddings):
    pipeline.json().set(name=key, path="$.text_embeddings", obj=embedding)

pipeline.execute()

res = client.json().get(name="article:007")
pprint(res)

############
## QUERY

# 1. create index

schema = (
    TextField(name="$.title", no_stem=True, as_name="title"),
    TextField(name="$.headline", no_stem=True, as_name="headline"),
    TextField(name="$.text", no_stem=True, as_name="text"),
    VectorField(
        name="$.text_embeddings",
        algorithm="FLAT",
        attributes={
            "TYPE": np.float16,
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE"
        },
        as_name="vector"
        )
)

definition = IndexDefinition(prefix="article:", index_type=IndexType.JSON)

res = client.ft(index_name="idx:article_vs").create_index(fields=schema,definition=definition)

info = client.ft("idx:article_vs").info()
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]

print(f"{num_docs} documents indexed with {indexing_failures} failures")