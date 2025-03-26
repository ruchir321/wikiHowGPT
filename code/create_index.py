import redis
import time
import numpy as np
from pprint import pprint

from redis.commands.search.field import (
    TextField,
    VectorField
)

from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

client = redis.Redis(decode_responses=True)

############
## QUERY

# 1. create index

VECTOR_DIM = len(client.json().get(name="article:01")["text_embeddings"])

schema = (
    TextField(name="$.title", as_name="title"),
    TextField(name="$.headline", as_name="headline"),
    TextField(name="$.text", as_name="text"),
    VectorField(
        name="$.text_embeddings",
        algorithm="FLAT",
        attributes={
            "TYPE": "FLOAT16",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE"
            },
        as_name="vector"
        )
)

definition = IndexDefinition(prefix="article:", index_type=IndexType.JSON)

res = client.ft(index_name="idx:article_vss").create_index(fields=schema,definition=definition)

info = client.ft("idx:article_vss").info()
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]

print(f"{num_docs} documents indexed with {indexing_failures} failures")