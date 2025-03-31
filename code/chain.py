import os
import dotenv

from langchain_redis import RedisConfig, RedisVectorStore

#prompt template
from langchain import hub

# semantic cache
from redisvl.extensions.llmcache import SemanticCache

# LLM memory
from redisvl.extensions.session_manager import SemanticSessionManager
import uuid

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

dotenv.load_dotenv()

# eval
os.environ["LANGSMITH_API_KEY"] = dotenv.dotenv_values()["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = dotenv.dotenv_values()["LANGSMITH_TRACING"]
os.environ["LANGSMITH_PROJECT"] = dotenv.dotenv_values()["LANGSMITH_PROJECT"]

REDIS_URL = dotenv.dotenv_values()["redis_vector_db"]
REDIS_SEMANTIC_CACHE_URL = dotenv.dotenv_values()["redis_semantic_cache"]
REDIS_LLM_MEMORY_URL = dotenv.dotenv_values()["redis_llm_memory"]

PROMPT_TEMPLATE = "rlm/rag-prompt"
# EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = "llama3.2" # TODO: change to nomic
LLM_MODEL = "gemma3"

import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_resources():
    # query embedding model
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # vector db connection
    config = RedisConfig(
        index_name="article",
        redis_url=REDIS_URL
    )

    vector_store = RedisVectorStore(embeddings=embedding_model, config=config)

    # generative LLM
    llm = ChatOllama(
        model=LLM_MODEL
    )
    
    # chat prompt template
    prompt = hub.pull(PROMPT_TEMPLATE)

    # doc retrieval tool
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

@st.cache_data
def get_model_output(query, session_id):

    results = llmcache.check(
        prompt=query,
        num_results=1,
        distance_threshold=0.1,
        return_fields=['response']
    )

    if results:
        print("Cache hit!")
        return results[0]["response"]

    else:
        print("Cache miss")
        response = rag_chain.invoke(query)
        llmcache.store(
            prompt=query,
            response=response
        )
        print("Cache set!")

        return response


# streamlit app
st.set_page_config(
    page_title="wikiHowGPT",
    page_icon=":bulb:"
)

# Initialize resources once
rag_chain = load_resources()

llmmemory = SemanticSessionManager(
    name="llm_memory",
    redis_url=REDIS_LLM_MEMORY_URL,
    vectorizer=EMBEDDING_MODEL
)

llmcache = SemanticCache(
            name="wikiGPTcache",
            vectorizer=OllamaEmbeddings(model=EMBEDDING_MODEL), # the default is sentence-transformers/all-mpnet-base-v2
            distance_threshold=0.1,
            redis_url=REDIS_SEMANTIC_CACHE_URL,
            ttl=1024
            )

st.header("Know the How")
form_input = st.text_input("Enter query")
submit = st.button("Generate")

# session id
session_id = st.session_state.get('session_id', str(uuid.uuid4()))
st.session_state['session_id'] = session_id 

if submit:
    st.write(get_model_output(form_input, session_id))
