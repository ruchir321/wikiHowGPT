
import dotenv

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from langchain_redis import RedisConfig, RedisVectorStore
from langchain_ollama import OllamaEmbeddings


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")

    if "source" in metadata:
        source = metadata["source"].split("/")
        source = source[source.index("data"):]
        metadata["source"] = "/".join(source) # store relative source path

    return metadata

dotenv.load_dotenv()

# Text splitter
CHUNK_SIZE = 500
CHUNK_OVERLAP=300
REDIS_URL = dotenv.dotenv_values()["redis_vector_db"]
FILE_PATH = dotenv.dotenv_values()["wikiHow_100_articles"] # input data

loader = JSONLoader(
    file_path=FILE_PATH,
    jq_schema='.[]',
    content_key="text",
    metadata_func=metadata_func
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

chunked_docs = text_splitter.split_documents(
    documents=docs
)

# ## Initialize vector db
config = RedisConfig(
    index_name="article",
    redis_url=REDIS_URL
)

EMBEDDING_MODEL = "llama3.2" # 3B params
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = RedisVectorStore.from_documents(documents=chunked_docs, embedding=embedding_model, config=config)
