from core.llm import llm
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

template = "what is the capital of india? what is the population?"

prompt = ChatPromptTemplate.from_template(template)

chain = (
    prompt | llm | StrOutputParser()
)

print(chain.invoke({}))

