import redis
import time
import numpy as np
from pprint import pprint
from sentence_transformers import SentenceTransformer

from redis.commands.search.query import Query


# EMBEDDING_MODEL = "msmarco-distilbert-base-v4" # 66.4M params
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2" # 17.4M params

# 4. select text embedding model
embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL
)

client = redis.Redis(decode_responses=True)

queries = "How do I get into arts coming from an investment management background"

encoded_queries = embedding_model.encode(
    sentences=queries
)

query = (
    Query('(*)=>[KNN 2 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'title', 'headline')
     .dialect(2)
)

res = client.ft('idx:article_vss').search(
        query,
        {
        'query_vector': np.array(encoded_queries, dtype=np.float16).tobytes()
        }
    ).docs

print(res)