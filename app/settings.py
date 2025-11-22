# The purpose of this file is to house all of the different settings of the application and to export them out.

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    LLM_PROVIDER: str = Field("groq", env="LLM_PROVIDER")
    LLM_TEMPERATURE: float = Field(0.0, env = "LLM_TEMPERATURE")
    #LLM_MODEL: str = Field("llama-3.1-8B-instant", env = "LLM_MODEL")
    LLM_MODEL: str = Field("llama-3.3-70b-versatile", env = "LLM_MODEL")
    GROQ_API_KEY: SecretStr = Field(..., env = "GROQ_API_KEY")

    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", env = "EMBEDDING_MODEL_NAME")
    EMBEDDING_DEVICE: str = Field("cpu", env="EMBEDDING_DEVICE")
    EMBEDDING_NORMALIZE: bool = Field(False, env="EMBEDDING_NORMALIZE")
    EMBEDDING_BATCH_SIZE: int = Field(1, env="EMBEDDING_BATCH_SIZE")
    GITHUB_USERNAME: str = Field(..., env = "GITHUB_USERNAME")

    VECTOR_DB_PATH: str = Field("./vector_db_store", env = "VECTOR_DB_PATH")
    VECTOR_DB_COLLECTION_NAME: str = Field("resume_rag", env = "VECTOR_DB_NAME")
    RETRIEVER_K_VALUE: int = Field(7, env="RETRIEVER_K_VALUE") 
    
    # 2. The "Strict Filter": How many docs to send to the LLM after reranking
    # RERANKER_TOP_N: int = Field(3, env="RERANKER_TOP_N")

    CHUNK_SIZE: int = Field(1000, env = "CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env = "CHUNK_OVERLAP")


    class Config():
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# GITHUB_USERNAME : str = Settings.GITHUB_USERNAME