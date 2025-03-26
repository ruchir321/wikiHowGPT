import redis
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from redis.commands.search.query import Query


# EMBEDDING_MODEL = "msmarco-distilbert-base-v4" # 66.4M params
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2" # 17.4M params

# 4. select text embedding model
embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL
)

client = redis.Redis(decode_responses=True)


def create_query_results_table(query, queries, encoded_queries, extra_params=None):
    """Create a df to present similarity search results"""
    results_list = []

    for i, encoded_query in enumerate(encoded_queries):

        result_doc = client.ft('idx:article_vss').search(
            query,
            {
            'query_vector': np.array(encoded_query, dtype=np.float16).tobytes()
            } | 
            (extra_params if extra_params else {})
        ).docs

        for doc in result_doc:
            vector_score = round(1 - float(doc.vector_score), 2)
            results_list.append(
                {
                    "query": query,
                    "score": vector_score,
                    "id": doc.id,
                    "title": doc.title,
                    "headline": doc.headline
                }
            )
    
    query_table = pd.DataFrame(data=results_list)

    query_table.sort_values(
        by=['query', 'score'], ascending=[True, False], inplace=True
    )

    return query_table.to_markdown(index=False)
    

queries = [
    "How do I get into arts coming from an investment management background",
    "I want to learn to make digital art"
    ]

encoded_queries = embedding_model.encode(
    queries
)

# print(encoded_queries[0].shape)

# print(len(client.json().get(name="article:01")["text_embeddings"]))

query = (
    Query('(*)=>[KNN 2 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'title', 'headline')
     .dialect(2)
)

res = create_query_results_table(query, queries, encoded_queries)
print(res)