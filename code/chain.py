import os
import dotenv

from langchain_redis import RedisConfig, RedisVectorStore

#prompt template
from langchain import hub

# semantic cache
# from langchain.globals import set_llm_cache
# from langchain_community.cache import RedisSemanticCache
from redisvl.extensions.llmcache import SemanticCache

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.outputs import Generation

import streamlit as st

dotenv.load_dotenv()

# eval
os.environ["LANGSMITH_API_KEY"] = dotenv.dotenv_values()["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = dotenv.dotenv_values()["LANGSMITH_TRACING"]
os.environ["LANGSMITH_PROJECT"] = dotenv.dotenv_values()["LANGSMITH_PROJECT"]

REDIS_URL = dotenv.dotenv_values()["redis_vector_db"]
REDIS_SEMANTIC_CACHE_URL = dotenv.dotenv_values()["redis_semantic_cache"]
PROMPT_TEMPLATE = "rlm/rag-prompt"
LLM_MODEL = "llama3.2"

import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_resources():
    # query embedding model
    embedding_model = OllamaEmbeddings(model=LLM_MODEL)
    
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
def get_model_output(query):

    # semantic cache check
    # result = cache.lookup(prompt=query, llm_string=LLM_MODEL)
    # print(f"\n\nRESULT: \n{result}\n")

    # if result:
    #     print("Semantic cache hit:", result[0].text)
    #     return result[0].text
    
    # else:
    #     print("Semantic cache miss")
    #     cache.update(prompt=query, llm_string=LLM_MODEL, return_val=[Generation(text=response)])
    

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

# semantic cache init
# cache = RedisSemanticCache(
#         redis_url=REDIS_SEMANTIC_CACHE_URL,
#         embedding=OllamaEmbeddings(model=LLM_MODEL),
#         score_threshold=0.1
#         )
# set_llm_cache(value=cache)

llmcache = SemanticCache(
            name="wikiGPTcache",
            distance_threshold=0.1,
            redis_url=REDIS_SEMANTIC_CACHE_URL,
            ttl=1024
            )

st.header("Know the How")
form_input = st.text_input("Enter query")
submit = st.button("Generate")

if submit:
    st.write(get_model_output(form_input))
