# The purpose of this file is to create a function that returns the embedding model that we'll be using for this application
# We import in settings from app.settings that contains embedding information, and we use HuggingFacEmbeddings to create an embedding model

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from app.settings import Settings

settings = Settings()

def get_embedding_model() -> Embeddings:

    model_kwargs = {"device": settings.EMBEDDING_DEVICE}

    encode_kwargs = {"normalize_embeddings": settings.EMBEDDING_NORMALIZE,
                     "batch_size": settings.EMBEDDING_BATCH_SIZE}

    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embedding_model

embedding_model: Embeddings = get_embedding_model()