import os
import dotenv

from langchain_redis import RedisConfig, RedisVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub

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
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2" # 17.4M params
PROMPT_TEMPLATE = "rlm/rag-prompt"
LLM_MODEL = "llama3.2"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_resources():
    embedding_model = OllamaEmbeddings(model=LLM_MODEL)

    config = RedisConfig(
        index_name="article",
        redis_url=REDIS_URL
    )

    vector_store = RedisVectorStore(embeddings=embedding_model, config=config)

    llm = ChatOllama(
        model=LLM_MODEL
    )

    prompt = hub.pull(PROMPT_TEMPLATE)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()

    )

    return rag_chain

@st.cache_data
def get_model_output(query):
    response = rag_chain.invoke(query)
    return response

# streamlit app
st.set_page_config(
    page_title="wikiHowGPT",
    page_icon=":bulb:"
)

# Initialize resources once
rag_chain = load_resources()


st.header("Know the How")
form_input = st.text_input("Enter query")
submit = st.button("Generate")

if submit:
    st.write(get_model_output(form_input))
