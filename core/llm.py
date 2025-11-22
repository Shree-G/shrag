from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from app.settings import Settings

def getLLM() -> BaseChatModel:
    try:
        settings = Settings()
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
