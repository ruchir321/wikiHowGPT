# wikiAgent

Agent framework implementation for wikiHow chatbot.

A chatbot to answer wikiHow type questions

WikiHow articles need a TLDR section. My chatbot will attempt to prepare TLDR summaries (`text-summarization`) for these articles, alongwith answering questions (`question-answering`)

## Architecture

Redis vector DB: LLM Memory and document retrival

Agent tools:

1. DuckDuckGo Search
2. Any relevant tool

Agent framework provider: `langchain`

## TODO

* build RAG powered chatbot
* Do i need local data for RAG?
* try out langraph

## notes about the branch

wikiAgent is an upgrade to a RAG powered chatbot implemented earlier.

## ref

1. [Using Redis for real-time RAG goes beyond a Vector Database](https://redis.io/blog/using-redis-for-real-time-rag-goes-beyond-a-vector-database/)
2. [Build a chatbot](https://python.langchain.com/docs/tutorials/chatbot/)
