from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from langchain_core.language_models.chat_models import BaseChatModel

class LLMSettings(BaseSettings):
    LLM_PROVIDER: str = Field("groq", env="LLM_PROVIDER")
    LLM_TEMPERATURE: float = Field(0.0, env = "LLM_TEMPERATURE")
    LLM_MODEL: str = Field("llama-3.3-70b-versatile", env = "LLM_MODEL")
    GROQ_API_KEY: SecretStr = Field(..., env = "GROQ_API_KEY")

    class Config():
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

def getLLM() -> BaseChatModel:
    try:
        settings = LLMSettings()
    except Exception as e:
        print(f"There was an error with setting up your .env: {e}")
        raise

    provider = settings.LLM_PROVIDER.lower()
    model_name = settings.LLM_MODEL
    temperature = settings.LLM_TEMPERATURE

    llm = None


    if provider == "groq":
        api_key = settings.GROQ_API_KEY.get_secret_value()
        llm = ChatGroq(api_key=api_key, model=model_name, temperature=temperature)
    
    return llm

llm : BaseChatModel = getLLM()
