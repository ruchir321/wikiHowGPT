import redis
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


# EMBEDDING_MODEL = "msmarco-distilbert-base-v4" # 66.4M params
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2" # 17.4M params

client = redis.Redis(decode_responses=True)

# 4. select text embedding model
embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL
)

# 5. generate and store text embeddings
keys = sorted(client.keys(pattern="article:*"))

text = client.json().mget(keys=keys, path="$.text")
text = [item for sublist in text for item in sublist]

# generate
text_embeddings = embedding_model.encode(
    sentences=text
).astype(dtype=np.float16).tolist()

VECTOR_DIM = len(text_embeddings[0])
print("embeddings ready")

# store
pipeline = client.pipeline()

for key, embedding in zip(keys, text_embeddings):
    pipeline.json().set(name=key, path="$.text_embeddings", obj=embedding)

pipeline.execute()
