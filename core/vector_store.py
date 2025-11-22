# The purpose of this file is to initialize and return a vector store using ChromaDB as well as a retriever
# This file will import in the embeddings function created in embeddings.py to use in the vector store

import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from core.embeddings import embedding_model
from app.settings import Settings

settings = Settings()

def get_vector_store() -> Chroma:
    print(f"Initializing Vector Store at {settings.VECTOR_DB_PATH}")

    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    except Exception as e:
        print(f"Error initializing chromaDB vector store: {e}")
        raise

    vector_store = Chroma(client=client, collection_name=settings.VECTOR_DB_COLLECTION_NAME, embedding_function=embedding_model)
    print(f"Success initializing vector store: {settings.VECTOR_DB_COLLECTION_NAME}")
    return vector_store

vector_store = get_vector_store()


def get_retriever() -> VectorStoreRetriever:
    print(f"Initializing retriever with k={settings.RETRIEVER_K_VALUE}")

    return vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVER_K_VALUE})


base_retriever = get_retriever()