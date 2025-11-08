from core.llm import llm
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from core.loaders import load_all_documents
from app.settings import Settings

# template = "what is the capital of india? what is the population?"

# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#     prompt | llm | StrOutputParser()
# )

# print(chain.invoke({}))

## ----------------

# settings = Settings()

# jsonObj = get_public_repo_urls(settings.GITHUB_USERNAME)

# print(jsonObj)

## ----------------

all_docs = load_all_documents()
print(all_docs)