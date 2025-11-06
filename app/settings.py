from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    LLM_PROVIDER: str = Field("groq", env="LLM_PROVIDER")
    LLM_TEMPERATURE: float = Field(0.0, env = "LLM_TEMPERATURE")
    LLM_MODEL: str = Field("llama-3.3-70b-versatile", env = "LLM_MODEL")
    GROQ_API_KEY: SecretStr = Field(..., env = "GROQ_API_KEY")

    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", env = "EMBEDDING_MODEL_NAME")
    EMBEDDING_DEVICE: str = Field("cpu", env="EMBEDDING_DEVICE")
    EMBEDDING_NORMALIZE: bool = Field(False, env="EMBEDDING_NORMALIZE")
    EMBEDDING_BATCH_SIZE: int = Field(1, env="EMBEDDING_BATCH_SIZE")


    class Config():
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# LLM_PROVIDER: str = Settings.LLM_PROVIDER
# LLM_TEMPERATURE: float = Settings.LLM_TEMPERATURE
# LLM_MODEL: str = Settings.LLM_MODEL
# GROQ_API_KEY: SecretStr = Settings.GROQ_API_KEY
# EMBEDDING_MODEL_NAME: str = Settings.EMBEDDING_MODEL_NAME
# EMBEDDING_DEVICE: str = Settings.EMBEDDING_DEVICE
# EMBEDDING_NORMALIZE: bool = Settings.EMBEDDING_NORMALIZE
# EMBEDDING_BATCH_SIZE: int = Settings.EMBEDDING_BATCH_SIZE
