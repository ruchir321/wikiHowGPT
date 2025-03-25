# wikiHow to rag

RAG practice repo

I want to build a RAG powered chatbot for wikiHow type questions

WikiHow articles need a TLDR section. My chatbot will attempt to prepare TLDR summaries (`text-summarization`) for these articles, alongwith answering questions (`question-answering`)

RAG techniques have become more sophisticated than a simple similarity search

I aim to implement [hybrid search](https://www.pinecone.io/learn/hybrid-search-intro/), rerankers [[1](https://www.pinecone.io/learn/series/rag/rerankers/), [2](https://www.pinecone.io/learn/refine-with-rerank/)]

I will be methodical in choosing an appropriate [embedding-model](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) and [vector database](https://www.pinecone.io/learn/vector-database/)

## vector db

redis [[1](https://redis.io/docs/latest/develop/get-started/vector-database/#4-generate-text-embeddings), [2](https://redis.io/docs/latest/develop/get-started/rag/)]

## embedding model

[`msmarco-distilbert-base-v4`](https://microsoft.github.io/msmarco/) is widely used for embedding process for chatbots, semantic search.
